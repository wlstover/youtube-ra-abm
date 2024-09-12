
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

from box import Box

import random
import numpy as np

class Watcher(Agent):
    def __init__(self, unique_id, model, acuity_floor, recommender_trust_step):
        super().__init__(unique_id, model)
        self.pos = None
        self.past_boxs = []
        self.payoffs = []
        self.average_payoff = 0
        self.payoff_direction = 0
        self.step_number = 0
        self.acuity_floor = acuity_floor
        self.acuity = random.choice(range(self.acuity_floor,101))
        self.recommender_trust = random.choice(range(0,101))
        self.recommender_trust_step = recommender_trust_step
        self.recommended_boxs_chosen_count = 0
        self.boxs_chosen_count = 0
        self.type = random.choice(['searcher', 'mimic'])
        self.patience = 100
        self.search_quality = 0
        self.searcher_search_quality = 0
        self.mimic_search_quality = 0
        
        
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
    
    def move(self):
        steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        
        possible_steps = [step for step in steps if step not in self.past_boxs]
       # print(f'{len(possible_steps)} possible steps are available in my neighborhood: {possible_steps}')
        if self.type == 'searcher':
            if possible_steps != []:
            
                possible_boxs = [a for a in self.model.schedule.agents if isinstance(a, Box) and a.pos in possible_steps]
                
            #  print(f'Boxs in my neighborhood: {possible_boxs}')
                box_payoffs = [(v.prize - v.cost) for v in possible_boxs]
                watcher_box_choice, watcher_weights = self.choose_based_on_acuity(box_payoffs, self.acuity)
            # print(watcher_box_choice)
                
                box_choice = [v for v in possible_boxs if (v.prize - v.cost) == watcher_box_choice][0]
                step_choice = [step for step in possible_steps if step == box_choice.pos][0]
                
                self.get_recommendation(possible_steps)
                recommended_box = [v for v in possible_boxs if v.recommended == True][0]
                
                # for v in possible_boxs:
                #     if v.recommended == True:
                #         print(f'{v.unique_id} at {v.pos} has been recommended by the algorithm')
                        
                if random.choice(range(1,101)) < self.recommender_trust:
                    new_position = recommended_box.pos
                    self.recommended_boxs_chosen_count += 1
                # print('Following algorithm recommendation')
                else:
                    new_position = step_choice
                    
                self.boxs_chosen_count += 1
                # print('Going with my choice')
                            
                # new_position = random.choice(possible_steps)
                self.model.grid.move_agent(self, new_position)
                box = [a for a in self.model.schedule.agents if isinstance(a, Box) and a.pos == new_position][0]
                box.opened = True
            #   print(f'Moving agent {self} to {new_position}')
                self.past_boxs.append(new_position)
                
            else:
            # print('Run out of boxs to search and will remove myself from schedule')
                #new_position = self.pos
                pass
        else:
            if possible_steps != []:
                possible_boxs = [a for a in self.model.schedule.agents if isinstance(a, Box) and a.pos in possible_steps]
                box_payoffs = [(v.prize - v.cost) for v in possible_boxs]
                watcher_box_choice, watcher_weights = self.choose_based_on_acuity(box_payoffs, self.acuity)
                
                box_likes = [v.likes for v in possible_boxs]
                if sum(box_likes) == 0:
                    box_choice = random.choice(possible_boxs)
                else:
                    max_likes = max(box_likes)
                    max_liked_box = [v for v in possible_boxs if v.likes == max_likes][0]
                    box_choice = max_liked_box
                
                step_choice = [step for step in possible_steps if step == box_choice.pos][0]
                new_position = step_choice
                self.model.grid.move_agent(self, new_position)
                box = [a for a in self.model.schedule.agents if isinstance(a, Box) and a.pos == new_position][0]
                box.opened = True
                self.past_boxs.append(new_position)
            else:
                pass
            
        self.step_number += 1
        
       
    def open_box_box(self):
        x,y = self.pos
        agent_counter = 0
        for agent in self.model.schedule.agents:
            if isinstance(agent, Box):
                if agent.pos == self.pos:
                    
                    # Checking to make sure each cell is popualted only with one box
                 #   print(f"This is box {agent_counter} that i have found here at {agent.pos}")
                    agent_counter += 1
                    
                   # print(f"Box {agent.unique_id} has prize value {agent.prize}, at cost {agent.cost}")
                    
                    prize = agent.prize
                    cost = agent.cost
                    payoff = prize - cost
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
                self.like_last_box()
            elif self.average_payoff < self.past_average_payoff and self.payoff_direction <= 0:
                self.payoff_direction -= 1
                
    def like_last_box(self):
        last_box = [a for a in self.model.schedule.agents if isinstance(a, Box) and a.pos == self.past_boxs[-1]][0]
        last_box.likes += 1
                
    def calculate_stopping_point(self):
            
        steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        possible_steps = [step for step in steps if step not in self.past_boxs]
        
        if possible_steps == []:
            #print("I've run out of boxs in my neighborhood to search, so I'm all done.")
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
        self.open_box_box()
        self.calculate_average_payoff()
        self.calculate_payoff_direction()
        self.calculate_search_quality()
      #  self.report_payoffs()