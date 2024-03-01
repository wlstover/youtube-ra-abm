from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

import random
import numpy as np
import pandas as pd
import itertools
from scipy.optimize import minimize_scalar

from watcher import Watcher
from video import Video
from recommender import Recommender
        
class VideoRecommendationsModel(Model):
    def __init__(self, width, height, num_agents, agent_acuity_floor, recommender_acuity, recommender_trust_step, agent_search_cost):
        super().__init__()
        self.num_agents = num_agents
        self.recommender_acuity = recommender_acuity
        self.agent_search_cost = agent_search_cost
        self.recommender_trust_step = recommender_trust_step
        self.agent_acuity_floor = agent_acuity_floor
        self.search_quality = 0
        self.searcher_search_quality = 0
        self.mimic_search_quality = 0
        self.percent_recommended = 0
        self.recommender_trust = 0
        self.video_boxes = self.genVideoBoxes(400)
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.final_payoffs = []
        self.running = True
        self.recommender = Recommender("recommender", self, self.recommender_acuity)
        # self.datacollector = DataCollector(
        #      {"search_quality": "search_quality",
        #       "searcher_search_quality": "searcher_search_quality",
        #       "mimic_search_quality": "mimic_search_quality",
        #       "percent_recommended": "percent_recommended",
        #       "recommender_trust": "recommender_trust"})
        self.datacollector = DataCollector(
        model_reporters={"mimic_search_quality": self.compute_average_mimic_search_quality,
                            "searcher_search_quality": self.compute_average_searcher_search_quality, 
                            "percent_recommended": self.compute_percent_recommended,
                            "percent_videos_recommended_chosen": self.compute_percent_videos_recommended_chosen,
                            "recommender_trust": self.compute_average_recommender_trust},
    )
        
        
        for i in range(num_agents):
            uid = f"watcher_{i}"
            agent = Watcher(uid, self, self.agent_acuity_floor, self.recommender_trust_step, self.agent_search_cost)
            self.schedule.add(agent)

            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))
        
       # self.video_boxes
        video_boxes_zip = list(zip(self.video_boxes['prize_mean'], self.video_boxes['prize_std_dev']))
        
       # print(video_boxes_zip)
        x_locations = range(self.grid.width)
        y_locations = range(self.grid.height)
        
        grid_locations = list(itertools.product(x_locations, y_locations))
        
        #print(grid_locations)
        
        for i in range(1, (len(video_boxes_zip) + 1)):
            
            box_values = random.sample(video_boxes_zip, 1)[0]
            uid = (f"box_{i}")
            video = Video(uid, self, box_values[0], box_values[1])
            self.schedule.add(video)
            
            coords = random.sample(grid_locations, 1)[0] 
            x = coords[0]
            y = coords[1]    
            self.grid.place_agent(video, (x,y))
            
            video_boxes_zip.remove(box_values)
            grid_locations.remove(coords)
        
           # print(len(video_boxes_zip))

       # print(self.video_boxes)
        self.optimal_payoff = self.calculateOptimalSearchValue(self.video_boxes) 
        #print('Optimal payoff is', self.optimal_payoff)
        self.datacollector.collect(self)
        
    def calculateOptimalSearchValue(self, video_boxes):
        return self.solveVideoBoxes(video_boxes)
        
    def genVideoBoxes(self, n_boxes):

        import numpy as np
        import pandas as pd

        seed = random.randint(0, 1000)
        np.random.seed(seed)  # For reproducibility

        # Define number of boxes

        # Generate 10 mean values that increase exponentially from 20 to 110
        prize_means = np.logspace(np.log10(20), np.log10(110), num=n_boxes)

        # Generate 10 standard deviations that range from 5 to 50
        prize_stds = np.linspace(1, 3, num=n_boxes)
        
        #   # Generate 10 mean values that increase exponentially from 20 to 110
        # cost_means = np.logspace(np.log10(20), np.log10(110), num=n_boxes)

        # # Generate 10 standard deviations that range from 5 to 50
        # cost_stds = np.linspace(1, 3, num=n_boxes)

        # Combine the mean values and standard deviations into a list of boxes
        boxes = list(zip(prize_means, prize_stds))

        box_df = pd.DataFrame(boxes, columns=['prize_mean', 'prize_std_dev'])
        
        #print(df)
        return box_df
    
    
    
    def utility(self, x_S, y):
        return y + sum(x_S)

    def expected_utility(self, x_S, mu_i, sigma_i):
        samples = np.random.normal(mu_i, sigma_i, 10000)
        utilities = [self.utility(x_S, sample) for sample in samples]
        return np.mean(utilities)

    def find_reservation_price(self, c_i, mu_i, sigma_i, x_S):
        def to_minimize(y):
            return abs(self.utility(x_S, y) + c_i - self.expected_utility(x_S, mu_i, sigma_i))
        result = minimize_scalar(to_minimize)
        if result.success:
            return result.x
        else:
            return None

    def solveVideoBoxes(self, vdf):
        # Calculate indices
                # Initialize list of already opened boxes
        x_S = []

        # Initialize total cost
        total_cost = 0

        # Calculate initial reservation prices
        vdf['reservation_price'] = vdf.apply(lambda row: self.find_reservation_price(self.agent_search_cost, row['prize_mean'], row['prize_std_dev'], x_S), axis=1)

        while not vdf.empty:
            # Find the box with the lowest reservation price
            min_price_row = vdf.loc[vdf['reservation_price'].idxmin()]

            # If the reservation price is less than or equal to the mean value in the box, open the box
            if min_price_row['reservation_price'] <= min_price_row['prize_mean']:
                # Add value in box to list of already opened boxes
                x_S.append(min_price_row['prize_mean'])

                # Add cost of opening box to total cost
                total_cost += self.agent_search_cost

                # Remove box from dataframe
                vdf.drop(min_price_row.name, inplace=True)

                # Update reservation prices for remaining boxes
                vdf['reservation_price'] = vdf.apply(lambda row: self.find_reservation_price(self.agent_search_cost, row['prize_mean'], row['prize_std_dev'], x_S), axis=1)
            else:
                # If the reservation price is greater than the mean value in the box, don't open the box
                vdf.drop(min_price_row.name, inplace=True)
            
        total_payoff = sum(x_S) - total_cost
        return total_payoff

            # print(box_df)
                

    def step(self):
        self.schedule.step()
        watcher_payoffs = [sum(agent.payoffs) for agent in self.schedule.agents if isinstance(agent, Watcher) and agent.payoffs]
        # for agent in self.schedule.agents:
        #     if isinstance(agent, Watcher):
        #          agent.calculate_search_quality()
        # if watcher_payoffs:
        #     self.sum_payoff = sum(watcher_payoffs) / len(watcher_payoffs)
        
    
        self.datacollector.collect(self)
     #   print(self.datacollector.get_agent_vars_dataframe())

    def report_Agent_Locations(self):
        for agent in self.schedule.agents:
            agent_pos = agent.__dict__['pos']
            agent_id = agent.__dict__['unique_id']
            #print(f'Agent {agent_id} is at {agent_pos}')
           # print(type(agent), isinstance(agent, Video))
           
    def report_Watcher_Final_Payoffs(self):
        return self.final_payoffs
        
    def report_Video_Box_Values(self):
        for a in self.schedule.agents:
            if isinstance(a, Video):
                agent_pos = a.__dict__['pos']
                agent_id = a.__dict__['unique_id']
              #  print(agent_id, agent_pos, a.prize, a.cost)

    def compute_average_mimic_search_quality(model):
        mimic_search_qualities = [agent.search_quality for agent in model.schedule.agents if isinstance(agent, Watcher) and agent.type == 'mimic']
        mimic_pop = len([agent for agent in model.schedule.agents if isinstance(agent, Watcher) and agent.type == 'mimic'])
        if mimic_pop == 0:
            return 0
        else:
            return sum(mimic_search_qualities) / mimic_pop
    
    def compute_average_searcher_search_quality(model):
        searcher_search_qualities = [agent.search_quality for agent in model.schedule.agents if isinstance(agent, Watcher) and agent.type == 'searcher']
        searcher_pop = len([agent for agent in model.schedule.agents if isinstance(agent, Watcher) and agent.type == 'searcher'])
        if searcher_pop == 0:
            return 0
        else:
            return sum(searcher_search_qualities) / searcher_pop
    
    def compute_percent_recommended(model):
        recommended_videos = [agent for agent in model.schedule.agents if isinstance(agent, Video) and agent.recommended == True]
        number_of_videos = len([agent for agent in model.schedule.agents if isinstance(agent, Video)])
        return len(recommended_videos) / number_of_videos
    
    def compute_average_recommender_trust(model):
        recommender_trust = [agent.recommender_trust for agent in model.schedule.agents if isinstance(agent, Watcher)]
        return (sum(recommender_trust) / len(recommender_trust)) / 100
    
    def compute_percent_videos_recommended_chosen(model):
        recommended_videos = [agent for agent in model.schedule.agents if isinstance(agent, Watcher)]
        total_videos_chosen = sum([agent.videos_chosen_count for agent in model.schedule.agents if isinstance(agent, Watcher)])
        total_recommended_videos_chosen = sum([agent.recommended_videos_chosen_count for agent in model.schedule.agents if isinstance(agent, Watcher)])
        if total_videos_chosen == 0:
            return 0
        else:
            return total_recommended_videos_chosen / total_videos_chosen