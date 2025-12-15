import pandas as pd
import pickle
import matplotlib.pyplot as plt 
import os
from network_functions import check_right_winner
from network_functions import check_iteration_difference
from network_functions import prepare_df
import config_Diane


def explore_valid_h_t():
    # load parquet
    simulation_df = pd.read_parquet(f'../data/simulation_data_all_biases_{config_Diane.self_excitation}.parquet')

    sim_data = prepare_df(simulation_df)
   
    # prepare dict
    possible_combinations_dict = {}
            
    # loop through bias combinations
    bias_combinations = sim_data.groupby(['bias'])
    for bias, df_bias in bias_combinations:

        possible_combinations = []
        
        # loop trough h/t parameter combinations
        h_t_combinations = df_bias.groupby(['h/t'])
        for h_t, df_h_t in h_t_combinations:
            
            # check for right winner, plausible iteration count, iteration difference
            right_winner = False
            iterations_plausible = False
            iteration_difference = False

            # check for plausible iterations
            max_iterations = df_h_t['iteration count'].max()
            iterations_plausible = max_iterations < 2000

            # check for right winner
            right_winner_series = df_h_t.apply(check_right_winner, axis=1)
            right_winner = right_winner_series.all()

            # check for iteration differences         
            iteration_difference = check_iteration_difference(df_h_t)

            # combine all criteria            
            if right_winner and iterations_plausible and iteration_difference:
                possible_combinations.append(h_t[0])

        if possible_combinations:
            possible_combinations_dict[bias[0]] = possible_combinations
    
    
    # print number of possible h/t against bias set
    bias_values = list(possible_combinations_dict.keys())
    counts = [len(possible_combinations_dict[b]) for b in bias_values]
    directory = '../figures/sensitivity'
    os.makedirs(directory, exist_ok=True)
    path = f'{directory}/bias_h_t.png'

    plt.figure()
    plt.bar(bias_values, counts)
    plt.xlabel('Bias')
    plt.ylabel('number of valid h/t combinations')
    plt.title('Valid h/t combinations per bias')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


    # save dictionary
    with open('log_files/valid_h_t_per_bias.pkl', 'wb') as f:
        pickle.dump(possible_combinations_dict, f)

    # save updated simulation data
    sim_data.to_parquet('../data/simulation_data_processed.parquet', index=False)

    print('Determined valid h/t combinations per bias, saved them and processed data frame successfully.')