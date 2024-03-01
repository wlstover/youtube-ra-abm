
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

from scipy.optimize import minimize_scalar

from video import Video

import random
import numpy as np
import pandas as pd

class Watcher(Agent):
    def __init__(self, unique_id, model, acuity_floor, recommender_trust_step, search_cost):
        super().__init__(unique_id, model)
        self.pos = None
        self.past_videos = []
        self.payoffs = []
        self.average_payoff = 0
        self.payoff_direction = 0
        self.step_number = 0
        self.search_cost = search_cost
        self.acuity_floor = acuity_floor
        self.acuity = random.choice(range(self.acuity_floor,101))
        self.recommender_trust = random.choice(range(0,101))
        self.recommender_trust_step = recommender_trust_step
        self.recommended_videos_chosen_count = 0
        self.videos_chosen_count = 0
        self.type = random.choice(['searcher', 'mimic'])
        self.patience = 100
        self.search_quality = 0
        self.searcher_search_quality = 0
        self.mimic_search_quality = 0
        self.watched_videos = []
        
        
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

            expected_value = np.sum(acuity_adjusted_choices)

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
        
    def set_reservation_prices(self, box_df):
        box_means = box_df['mean']
        box_stds = box_df['std_dev']
        x_S = []
        reservation_prices = []
        for mean, std_dev in zip(box_means, box_stds):
            reservation_price = self.find_reservation_price(self.search_cost, mean, std_dev, x_S)
            reservation_prices.append(reservation_price)
            x_S.append(mean)
        self.box_df['reservation_price'] = reservation_prices
        
    def move(self):
        steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        
        unwatched_videos = [a for a in self.model.schedule.agents if isinstance(a, Video) and a.unique_id not in self.watched_videos]
        unwatched_video_means = [v.prize_mean for v in unwatched_videos]
        unwatched_video_stds = [v.prize_std_dev for v in unwatched_videos]
        unwatched_video_coords = [v.pos for v in unwatched_videos]
        
        unwatched_video_df = pd.DataFrame({'mean': unwatched_video_means, 
                                           'std_dev': unwatched_video_stds, 
                                           'pos': unwatched_video_coords})
        
        video_rps = unwatched_video_df.apply(lambda row: self.find_reservation_price(self.search_cost, row['mean'], row['std_dev'], self.payoffs), axis=1)
        
        possible_steps = [step for step in steps if step not in self.past_videos]
        
        adjacent_video_df = unwatched_video_df.loc[unwatched_video_df['pos'].isin(possible_steps)]
        adjacent_video_rps = adjacent_video_df.apply(lambda row: self.find_reservation_price(self.search_cost, row['mean'], row['std_dev'], self.payoffs), axis=1)
        
       # print(f'{len(possible_steps)} possible steps are available in my neighborhood: {possible_steps}')
        if self.type == 'searcher':
            if possible_steps != []:
            
                possible_videos = [a for a in self.model.schedule.agents if isinstance(a, Video) and a.pos in possible_steps]
                
            #  print(f'Videos in my neighborhood: {possible_videos}')
                # video_rps = [unwatched_video_df.loc[unwatched_video_df['pos'] == v.pos, 'reservation_price'] for v in possible_videos]
                watcher_video_choice, watcher_weights = self.choose_based_on_acuity(adjacent_video_rps, self.acuity)
            # print(watcher_video_choice)
                min_price_row = adjacent_video_df.loc[adjacent_video_rps['reservation_price'].idxmin()]
                
                video_choice = [v for v in possible_videos if v.pos == watcher_video_choice][0]
                step_choice = [step for step in possible_steps if step == video_choice.pos][0]
                
                self.get_recommendation(possible_steps, video_rps)
                recommended_video = [v for v in possible_videos if v.recommended == True][0]
                
                # for v in possible_videos:
                #     if v.recommended == True:
                #         print(f'{v.unique_id} at {v.pos} has been recommended by the algorithm')
                        
                if random.choice(range(1,101)) < self.recommender_trust:
                    new_position = recommended_video.pos
                    self.recommended_videos_chosen_count += 1
                # print('Following algorithm recommendation')
                else:
                    new_position = step_choice
                    
                if min_price_row['reservation_price'] <= min_price_row['prize_mean']:
                        
                    self.videos_chosen_count += 1
                    # print('Going with my choice')
                    
                    # new_position = random.choice(possible_steps)
                    self.model.grid.move_agent(self, new_position)
                    video = [a for a in self.model.schedule.agents if isinstance(a, Video) and a.pos == new_position][0]
                    video.opened = True
                #   print(f'Moving agent {self} to {new_position}')
                    self.past_videos.append(new_position)
                    self.watched_videos.append(video.unique_id)
                    
                else:
                    # Invoking stopping rule, removing self from schedule
                    pass
                
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
                self.watched_videos.append(video.unique_id)
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
                    
                    payoff = agent['prize_mean']
                    
                    if payoff > 0 and agent.recommended == True:
                        self.recommender_trust += self.recommender_trust_step
                    elif payoff < 0 and agent.recommended == True:
                        self.recommender_trust -= self.recommender_trust_step
                        
                    if self.recommender_trust >= 100:
                        self.recommender_trust = 100

                    self.payoffs.append(payoff)
                  #  print(self.payoffs[-1])
                
    def calculate_average_payoff(self):
        self.past_average_payoff = self.average_payoff
        self.average_payoff = np.mean(self.payoffs)
        
    def calculate_search_quality(self):
        self.search_quality = sum(self.payoffs) / self.model.optimal_payoff
        self.model.search_quality = self.search_quality

        if self.type == 'searcher':
            self.searcher_search_quality = self.search_quality
            self.model.searcher_search_quality = self.search_quality

        else:
            self.mimic_search_quality = self.search_quality
            self.model.mimic_search_quality = self.search_quality
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
            
        steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        possible_steps = [step for step in steps if step not in self.past_videos]
        
        if possible_steps == []:
            #print("I've run out of videos in my neighborhood to search, so I'm all done.")
            self.model.final_payoffs.append([sum(self.payoffs), self.unique_id, self.patience, self.step_number, self.acuity, self.recommender_trust, self.model.recommender_acuity, self.type, self.search_quality, self.searcher_search_quality, self.mimic_search_quality])
            self.model.schedule.remove(self)
        
        elif self.payoff_direction == self.patience:
            #print("I have done really well so far and think it is a good time to stop.")
            self.model.final_payoffs.append([sum(self.payoffs), self.unique_id, self.patience, self.step_number, self.acuity, self.recommender_trust, self.model.recommender_acuity, self.type, self.search_quality, self.searcher_search_quality, self.mimic_search_quality])
            self.model.schedule.remove(self)
            
        elif self.payoff_direction == -self.patience:
            self.model.final_payoffs.append([sum(self.payoffs), self.unique_id, self.patience, self.step_number, self.acuity, self.recommender_trust, self.model.recommender_acuity, self.type, self.search_quality, self.searcher_search_quality, self.mimic_search_quality])
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