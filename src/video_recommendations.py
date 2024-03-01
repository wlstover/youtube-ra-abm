from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

import random
import numpy as np
import pandas as pd
import itertools

from watcher import Watcher
from video import Video
from recommender import Recommender
        
class VideoRecommendationsModel(Model):
    def __init__(self, width, height, num_agents, recommender_acuity, recommender_trust_step):
        super().__init__()
        self.num_agents = num_agents
        self.recommender_acuity = recommender_acuity
        self.recommender_trust_step = recommender_trust_step
        self.search_quality = 0
        self.searcher_search_quality = 0
        self.mimic_search_quality = 0
        self.percent_recommended = 0
        self.video_boxes = self.genVideoBoxes(400)
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.final_payoffs = []
        self.running = True
        self.recommender = Recommender("recommender", self, self.recommender_acuity)
        self.datacollector = DataCollector(
             {"search_quality": "search_quality",
              "searcher_search_quality": "searcher_search_quality",
              "mimic_search_quality": "mimic_search_quality"})
        
        for i in range(num_agents):
            uid = f"watcher_{i}"
            agent = Watcher(uid, self, self.recommender_trust_step)
            self.schedule.add(agent)

            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))
        
       # self.video_boxes
        video_boxes_zip = list(zip(self.video_boxes['prize'], self.video_boxes['cost']))
        
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

        # Define mean and standard deviation for prize and cost distributions
        prize_mean = 100
        prize_std = 20
        cost_mean = 10
        cost_std = 2

        # Generate prizes and costs
        prizes = np.random.normal(prize_mean, prize_std, n_boxes)
        costs = np.random.normal(cost_mean, cost_std, n_boxes)

        # Make sure that costs and prizes are not negative
        costs = np.abs(costs)
        prizes = np.abs(prizes)

        # Create a DataFrame for better data management
        df = pd.DataFrame({
            'prize': prizes,
            'cost': costs
        })
        
        #print(df)
        return df

    def solveVideoBoxes(self, vdf):
        # Calculate indices
        vdf['index'] = vdf['prize'] - vdf['cost']

        total_prize = 0
        total_cost = 0

        while len(vdf) > 0:
            # Find the box with the maximum index
            max_index_box = vdf['index'].idxmax()

            # If the index is positive, open the box
            if vdf.loc[max_index_box, 'index'] > 0:
                total_prize += vdf.loc[max_index_box, 'prize']
                total_cost += vdf.loc[max_index_box, 'cost']
                vdf.drop(max_index_box, inplace=True)
            else:
                break

        # print("Total prize:", total_prize)
        # print("Total cost:", total_cost)
        
        net_gain = total_prize - total_cost
        return net_gain

        # print("Total prize:", total_prize)
        # print("Total cost:", total_cost)
        # print("Net gain:", total_prize - total_cost)
        

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
