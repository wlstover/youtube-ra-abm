# Directory Structure

`video_recommendations.py` is the main codebase that defines and instantiates agents, steps them through the model, and runs analyis/generates reports. Look here for anything to do with the main model itself.
`run_model.py` is the auxiliary Mesa file that is essentially the controller for the model - this is the file used to actually launch and start running the VideoRecommendations model.

## Operations

Right now, it's pretty straightforward - navigate to this project directory and in a console, run `python run_model.py`. This will generate some printouts in the console, but the main results will be output in a `recommender_abm_results` Excel file in the root directory.
