from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule

from model import BoxRecommendationsModel, Watcher
import solara

# Define the visualization elements
# Define the visualization elements
def agent_portrayal(agent):
    portrayal = {"Shape": "circle",
                 "Filled": "true",
                 "r": 0.5,
                 "Layer": 0}

    if isinstance(agent, Watcher):
        if agent.type == 'searcher':
            portrayal["Color"] = "red"
        else:
            portrayal["Color"] = "black"
    else:
        if agent.opened == True:
            portrayal["Color"] = "gray"
        elif agent.recommended == True:
            portrayal["Color"] = "green"
        else:
            portrayal["Color"] = "blue"
       #portrayal["shape"] = "triangle"

    return portrayal

# Set the parameters for a single run
width = 20
height = 20
num_agents = 10
num_steps = 20
agent_search_cost = 10
recommender_trust_step = 1

box_values = [i for i in range(1, 201)]
search_costs = [(101 - i) for i in box_values]
box_boxes = list(zip(box_values, search_costs))

treatment = 'high_value'

# Create a single model instance 
# model = BoxRecommendationsModel(width, height, num_agents, treatment)

# Create the visualization
grid = CanvasGrid(agent_portrayal, width, height, 500, 500)

chart = ChartModule([{"Label": "searcher_search_quality",
                      "Color": "Red"},
                      {"Label": "mimic_search_quality",
                       "Color": "Black"}],
                    data_collector_name='datacollector')

model_params = {
"num_agents": {
    "type": "SliderInt",
    "value": 10,
    "label": "Number of searchers",
    "min": 10,
    "max": 100,
    "step": 1
},
"agent_acuity_floor": {
    "type": "SliderInt",
    "label": "Agent Acuity Floor",
    "value": 10,
    "min": 1,
    "max": 100,
    "step": 1
},
"recommender_acuity": {
    "type": "SliderFloat",
    "label": "Recommender System Accuity",
    "value": 0.5,
    "min": 0,
    "max": 1,
    "step": 0.01
},
"recommender_trust_step": {
    "type": "SliderInt",
    "label": "Recommender System Trust Step",
    "value": 1,
    "min": 1,
    "max": 10,
    "step": 1
},
"width": 20,
"height": 20
}

server = ModularServer(BoxRecommendationsModel,
                       [grid, chart],
                       "Box Recommendations Model",
                       model_params=model_params)

# Start the server
server.launch()