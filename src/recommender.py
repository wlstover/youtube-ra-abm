from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

import random
from video import Video

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