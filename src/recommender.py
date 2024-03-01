from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

import random
import numpy as np
from video import Video

class Recommender(Agent):
    def __init__(self, unique_id, model, recommender_acuity):
        super().__init__(unique_id, model)
        self.pos = None
        self.recommender_accuity = recommender_acuity
        
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
        
    def generate_recommendation(self, possible_steps, video_rps):
        #for coords in agent_neighborhood:
        
        videos = [a for a in self.model.schedule.agents if isinstance(a, Video) and a.pos in possible_steps]
        
        
        video_payoff_dict = dict(zip(video_rps, videos))
        
        choice_set = video_rps
        
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
            video_choice = video_payoff_dict[choice]
            video_choice.recommended = True

        elif len(choice_set) == 1:
            choice = choice_set[0]
            video_choice = video_payoff_dict[choice]
            video_choice.recommended = True
            
        else:
            choice = []
            probabilities = []
        
        # print(f'Total possible payoffs of {video_payoffs}')
        # print(f'Videos {videos} have video max payoff of {max_payoff} for video {max_payoff_video.unique_id} at {max_payoff_video.pos}')
                
                
    def step(self):
        self.generate_recommendation()