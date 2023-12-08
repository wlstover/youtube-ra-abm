from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import random
import numpy as np
import pandas as pd
import itertools


class Watcher(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.pos = None
        self.past_videos = []
        self.payoffs = []
        self.average_payoff = 0
        self.payoff_direction = 0
        self.step_number = 0
        self.acuity = random.choice(range(30,101))
        self.recommender_trust = random.choice(range(0,101))
        self.type = random.choice(['searcher', 'mimic'])
        self.patience = 100
        self.search_quality = 0
        
        
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def choose_based_on_acuity(self, choice_set, acuity):
        #print(f'Calculating probabilities for {choice_set}')
        
        if len(choice_set) > 1:
        
            # Normalize the choice set
            normalized_choices = (choice_set - np.min(choice_set)) / (np.max(choice_set) - np.min(choice_set))

            # Multiply by the acuity
            acuity_adjusted_choices = normalized_choices * (acuity / 100)
            #print(acuity_adjusted_choices)

            # Calculate probabilities using softmax
            probabilities = self.softmax(acuity_adjusted_choices)

            # Choose an element from the choice set based on the probabilities
            choice = np.random.choice(choice_set, p=probabilities)

        elif len(choice_set) == 1:
            choice = choice_set[0]
            probabilities = [1]
            
        else:
            choice = []
            probabilities = []
            
        return choice, probabilities
    
    def move(self):
        steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        
        possible_steps = [step for step in steps if step not in self.past_videos]
       # print(f'{len(possible_steps)} possible steps are available in my neighborhood: {possible_steps}')
        if self.type == 'searcher':
            if possible_steps != []:
            
                possible_videos = [a for a in self.model.schedule.agents if isinstance(a, Video) and a.pos in possible_steps]
                
            #  print(f'Videos in my neighborhood: {possible_videos}')
                video_payoffs = [(v.prize - v.cost) for v in possible_videos]
                watcher_video_choice, watcher_weights = self.choose_based_on_acuity(video_payoffs, self.acuity)
            # print(watcher_video_choice)
                
                video_choice = [v for v in possible_videos if (v.prize - v.cost) == watcher_video_choice][0]
                step_choice = [step for step in possible_steps if step == video_choice.pos][0]
                
                if self.model.recommender.recommender_type != 'no_recommend':
                    self.get_recommendation(possible_steps)
                    recommended_video = [v for v in possible_videos if v.recommended == True][0]
                    
                    # for v in possible_videos:
                    #     if v.recommended == True:
                    #         print(f'{v.unique_id} at {v.pos} has been recommended by the algorithm')
                            
                    if random.choice(range(1,101)) < self.recommender_trust:
                        new_position = recommended_video.pos
                    # print('Following algorithm recommendation')
                    else:
                        new_position = step_choice
                    # print('Going with my choice')
                
                else:
                    new_position = step_choice
                # print('Going with my choice')
                            
                # new_position = random.choice(possible_steps)
                self.model.grid.move_agent(self, new_position)
                video = [a for a in self.model.schedule.agents if isinstance(a, Video) and a.pos == new_position][0]
                video.opened = True
            #   print(f'Moving agent {self} to {new_position}')
                self.past_videos.append(new_position)
                
            else:
            # print('Run out of videos to search and will remove myself from schedule')
                #new_position = self.pos
                pass
        else:
            if possible_steps != []:
                possible_videos = [a for a in self.model.schedule.agents if isinstance(a, Video) and a.pos in possible_steps]
                video_payoffs = [(v.prize - v.cost) for v in possible_videos]
                watcher_video_choice, watcher_weights = self.choose_based_on_acuity(video_payoffs, self.acuity)
                
                video_likes = [v.likes for v in possible_videos]
                if sum(video_likes) == 0:
                    video_choice = random.choice(possible_videos)
                else:
                    max_likes = max(video_likes)
                    max_liked_video = [v for v in possible_videos if v.likes == max_likes][0]
                    video_choice = max_liked_video
                
                step_choice = [step for step in possible_steps if step == video_choice.pos][0]
                new_position = step_choice
                self.model.grid.move_agent(self, new_position)
                video = [a for a in self.model.schedule.agents if isinstance(a, Video) and a.pos == new_position][0]
                video.opened = True
                self.past_videos.append(new_position)
            else:
                pass
            
        self.step_number += 1
        
       
    def open_video_box(self):
        x,y = self.pos
        agent_counter = 0
        for agent in self.model.schedule.agents:
            if isinstance(agent, Video):
                if agent.pos == self.pos:
                    
                    # Checking to make sure each cell is popualted only with one video
                 #   print(f"This is box {agent_counter} that i have found here at {agent.pos}")
                    agent_counter += 1
                    
                   # print(f"Box {agent.unique_id} has prize value {agent.prize}, at cost {agent.cost}")
                    
                    prize = agent.prize
                    cost = agent.cost
                    payoff = prize - cost
                    self.payoffs.append(payoff)
                  #  print(self.payoffs[-1])
                
    def calculate_average_payoff(self):
        self.past_average_payoff = self.average_payoff
        self.average_payoff = np.mean(self.payoffs)
        
    def calculate_search_quality(self):
        self.search_quality = sum(self.payoffs) / self.model.optimal_payoff
        self.model.search_quality = self.search_quality
     #   print(self.search_quality)
        
    def calculate_payoff_direction(self):
        
            if self.average_payoff > self.past_average_payoff and self.payoff_direction < 0:
                self.payoff_direction = 1
            elif self.average_payoff < self.past_average_payoff and self.payoff_direction > 0:
                self.payoff_direction = -1  
            elif self.average_payoff > self.past_average_payoff and self.payoff_direction >= 0:
                self.payoff_direction += 1
                self.like_last_video()
            elif self.average_payoff < self.past_average_payoff and self.payoff_direction <= 0:
                self.payoff_direction -= 1
                
    def like_last_video(self):
        last_video = [a for a in self.model.schedule.agents if isinstance(a, Video) and a.pos == self.past_videos[-1]][0]
        last_video.likes += 1
                
    def calculate_stopping_point(self):
        
        if self.model.recommender.recommender_type == 'no_recommend':
            recommender_hv = 0
            recommender_random = 0
        elif self.model.recommender.recommender_type == 'high_value':
            recommender_hv = 1
            recommender_random = 0
        else:
            recommender_hv = 0
            recommender_random = 1
            
        steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        possible_steps = [step for step in steps if step not in self.past_videos]
        
        if possible_steps == []:
            #print("I've run out of videos in my neighborhood to search, so I'm all done.")
            self.model.final_payoffs.append([sum(self.payoffs), self.unique_id, self.patience, self.step_number, self.acuity, self.recommender_trust, recommender_hv, recommender_random, self.type, self.search_quality])
            self.model.schedule.remove(self)
        
        elif self.payoff_direction == self.patience:
            #print("I have done really well so far and think it is a good time to stop.")
            self.model.final_payoffs.append([sum(self.payoffs), self.unique_id, self.patience, self.step_number, self.acuity, self.recommender_trust, recommender_hv, recommender_random, self.type, self.search_quality])
            self.model.schedule.remove(self)
            
        elif self.payoff_direction == -self.patience:
            self.model.final_payoffs.append([sum(self.payoffs), self.unique_id, self.patience, self.step_number, self.acuity, self.recommender_trust, recommender_hv, recommender_random, self.type, self.search_quality])
            self.model.schedule.remove(self)
          #  print("Time to cut my losses and stop watching stuff")
        
        
    def report_payoffs(self):
        print(f"My position is {self.pos}, with avg_payoff {self.average_payoff}, and direction {self.payoff_direction}")
    
    def get_recommendation(self, possible_steps):
        self.model.recommender.generate_recommendation(possible_steps)
    
    def step(self):
        self.calculate_stopping_point()
        self.move()
        self.open_video_box()
        self.calculate_average_payoff()
        self.calculate_payoff_direction()
        self.calculate_search_quality()
      #  self.report_payoffs()
        
class Video(Agent):
    def __init__(self, unique_id, model, prize, cost):
        super().__init__(unique_id, model)
        self.pos = None
        self.prize = prize
        self.cost = cost
        self.opened = False
        self.recommended = False
        self.likes = 0
        
class Recommender(Agent):
    def __init__(self, unique_id, model, recommender_type):
        super().__init__(unique_id, model)
        self.pos = None
        
        if recommender_type not in ['random', 'no_recommend', 'high_value']:
            raise Exception('Invalid recommender type')
        else:
            self.recommender_type = recommender_type
        
    def generate_recommendation(self, possible_steps):
        #for coords in agent_neighborhood:
        
        videos = [a for a in self.model.schedule.agents if isinstance(a, Video) and a.pos in possible_steps]
        video_payoffs = [(video.prize - video.cost) for video in videos]
        
        if self.recommender_type == 'random':
            random_video_choice = random.choice(videos)
            random_video_choice.recommended = True
            
        elif self.recommender_type == 'high_value':
            max_payoff = max(video_payoffs)
            max_payoff_video = [video for video in videos if (video.prize - video.cost) == max_payoff][0]
            max_payoff_video.recommended = True
            
        else:
            pass
        
        # print(f'Total possible payoffs of {video_payoffs}')
        # print(f'Videos {videos} have video max payoff of {max_payoff} for video {max_payoff_video.unique_id} at {max_payoff_video.pos}')
                
                
    def step(self):
        self.generate_recommendation()

class VideoRecommendationsModel(Model):
    def __init__(self, width, height, num_agents, recommender_type):
        self.num_agents = num_agents
        self.search_quality = 0
        self.percent_recommended = 0
        self.video_boxes = self.genVideoBoxes(400)
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.final_payoffs = []
        self.running = True
        self.recommender = Recommender("recommender", self, recommender_type)
        self.random_recommendation_treatment = False
        self.highest_value_recommendation_treatment = True
        self.datacollector = DataCollector(
             {"search_quality": "search_quality"})
        
        for i in range(num_agents):
            uid = f"watcher_{i}"
            agent = Watcher(uid, self)
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
        watcher_payoffs =[sum(agent.payoffs) for agent in self.schedule.agents if isinstance(agent, Watcher) and agent.payoffs]
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
