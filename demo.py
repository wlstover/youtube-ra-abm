from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule

from video_recommendations import VideoRecommendationsModel, Watcher
import pandas as pd

# Define the visualization elements
# Define the visualization elements
def agent_portrayal(agent):
    portrayal = {"Shape": "circle",
                 "Filled": "true",
                 "r": 0.5,
                 "Layer": 0}

    if isinstance(agent, Watcher):
        portrayal["Color"] = "red"
    else:
        portrayal["Color"] = "blue"
        portrayal["shape"] = "triangle"

    return portrayal

# Set the parameters for a single run
width = 20
height = 20
num_agents = 1
num_steps = 20

video_values = [i for i in range(1, 201)]
search_costs = [(101 - i) for i in video_values]
video_boxes = list(zip(video_values, search_costs))

treatment = 'high_value'

# Create a single model instance 
model = VideoRecommendationsModel(width, height, num_agents, treatment)

chart = ChartModule([{"Label": "payoff_sum",
                      "Color": "Black"}],
                data_collector_name='datacollector')

# Create the visualization
grid = CanvasGrid(agent_portrayal, width, height, 500, 500)
server = ModularServer(VideoRecommendationsModel,
                       [grid, chart],
                       "Video Recommendations Model",
                       {"width": width, "height": height, "num_agents": num_agents, "recommender_type": treatment})

# Start the server
server.launch()