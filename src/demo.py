from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule

from box_recommendations import BoxRecommendationsModel, Watcher
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

box_values = [i for i in range(1, 201)]
search_costs = [(101 - i) for i in box_values]
box_boxes = list(zip(box_values, search_costs))

treatment = 'high_value'

# Create a single model instance 
model = BoxRecommendationsModel(width, height, num_agents, treatment)



# Create the visualization
grid = CanvasGrid(agent_portrayal, width, height, 500, 500)

chart = ChartModule([{"Label": "searcher_search_quality",
                      "Color": "Red"},
                      {"Label": "mimic_search_quality",
                       "Color": "Black"}],
                    data_collector_name='datacollector')

server = ModularServer(BoxRecommendationsModel,
                       [grid, chart],
                       "Box Recommendations Model",
                       {"width": width, "height": height, "num_agents": num_agents, "recommender_type": treatment})

# Start the server
server.launch()