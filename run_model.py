from video_recommendations import VideoRecommendationsModel, Watcher
import pandas as pd

if __name__ == "__main__":
    width = 20
    height = 20
    num_agents = 1
    num_steps = 20
    
    video_values = [i for i in range(1, 201)]
    search_costs = [(101 - i) for i in video_values]
    video_boxes = list(zip(video_values, search_costs))
    
    agent_payoff_df = pd.DataFrame(columns=['final_payoff', 'uid', 'patience', 'step_number', 'acuity', 'recommender_trust', 'recommender_hv', 'recommender_random',  'optimal_search_value'])
    
    for treatment in ['no_recommend', 'high_value', 'random']:
        
        for i in range(1,101):

            model = VideoRecommendationsModel(width, height, num_agents, treatment)

            while model.running == True:
                for i in range(num_steps):
                    if any(isinstance(a, Watcher) for a in model.schedule.agents):
                        #model.report_Video_Box_Values()
                        #model.report_Agent_Locations()
                    # print(f'Stepping model to step {i}')
                        model.step()
                    else:
                        #model.report_Watcher_Final_Payoffs()
                        #print('Agent has stopped searching, breaking the model')
                        model.running = False
        
            optimal_search_value = model.calculateOptimalSearchValue(model.video_boxes)
            agent_payoffs = model.report_Watcher_Final_Payoffs()
            agent_payoff_treatment_df = pd.DataFrame(agent_payoffs, columns=['final_payoff', 'uid', 'patience', 'step_number', 'acuity', 'recommender_trust', 'recommender_hv', 'recommender_random'])
            agent_payoff_treatment_df['optimal_search_value'] = optimal_search_value
            agent_payoff_df = pd.concat([agent_payoff_df, agent_payoff_treatment_df])
            
    agent_payoff_df['search_quality'] = agent_payoff_df['final_payoff'] / agent_payoff_df['optimal_search_value']
    agent_payoff_df.to_csv('recommender_abm_results.csv')
                
            #print(f'Iterating model to step {i}')
    
   
        


