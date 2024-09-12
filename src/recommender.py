from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

import random
import numpy as np
from box import Box

class Recommender(Agent):
    def __init__(self, unique_id, model, recommender_acuity):
        super().__init__(unique_id, model)
        self.pos = None
        self.recommender_accuity = recommender_acuity
        
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
        
    def generate_recommendation(self, possible_steps):
        #for coords in agent_neighborhood:
        
        boxs = [a for a in self.model.schedule.agents if isinstance(a, Box) and a.pos in possible_steps]
        box_payoffs = [(box.prize - box.cost) for box in boxs]
        
        box_payoff_dict = dict(zip(box_payoffs, boxs))
        
        choice_set = box_payoffs
        
        if len(choice_set) > 1:
        
            # Normalize the choice set
            normalized_choices = (choice_set - np.min(choice_set)) / (np.max(choice_set) - np.min(choice_set))

            # Multiply by the acuity
            acuity_adjusted_choices = normalized_choices * (self.recommender_accuity / 100)
            #print(acuity_adjusted_choices)

            expected_value = np.sum(acuity_adjusted_choices)

            # Calculate probabilities using softmax
            probabilities = self.softmax(acuity_adjusted_choices)

            # Choose an element from the choice set based on the probabilities
            choice = np.random.choice(choice_set, p=probabilities)
            box_choice = box_payoff_dict[choice]
            box_choice.recommended = True

        elif len(choice_set) == 1:
            choice = choice_set[0]
            box_choice = box_payoff_dict[choice]
            box_choice.recommended = True
            
        else:
            choice = []
            probabilities = []
        
        # print(f'Total possible payoffs of {box_payoffs}')
        # print(f'Boxs {boxs} have box max payoff of {max_payoff} for box {max_payoff_box.unique_id} at {max_payoff_box.pos}')
                
                
    def step(self):
        self.generate_recommendation()