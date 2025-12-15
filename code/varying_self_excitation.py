from network_functions import prepare_df
from network_functions import keep_valid_h_t
from network_functions import check_right_winner
from network_functions import check_iteration_difference
import pandas as pd
import config_Diane

def explore_self_excitations(self_excitations):

    sim_data = [pd.read_parquet(f'../data/simulation_data_90_10_{self_ex}.parquet') for self_ex in self_excitations]
    sim_data_merged = pd.concat(sim_data, ignore_index=True)

    sim_data_se = prepare_df(sim_data_merged)

    # standard self-ex: 3.5
    standard_self_ex = config_Diane.self_excitation
    standard_set = sim_data_se.loc[sim_data_se['self-excitation'] == standard_self_ex]
    medians_standard_set = standard_set.groupby(['h/t', 'condition'])['iteration count'].median()

    # look for valid h/t
    possible_combinations_dict = {}
            
    # loop through self_ex
    self_ex_subsets = sim_data_se.groupby(['self-excitation'])
    for self_ex, df_self_ex in self_ex_subsets:

        possible_combinations = []
        
        # loop trough h/t parameter combinations
        h_t_combinations = df_self_ex.groupby(['h/t'])
        for h_t, df_h_t in h_t_combinations:
            
            # check for right winner, plausible iteration count, iteration difference
            right_winner = False
            iterations_plausible = False
            iteration_difference = False
            valid_self_ex = False

            # check for plausible iterations
            max_iterations = df_h_t['iteration count'].max()
            iterations_plausible = max_iterations < 2000

            # check for right winner
            right_winner_series = df_h_t.apply(check_right_winner, axis=1)
            right_winner = right_winner_series.all()

            # check for iteration differences         
            iteration_difference = check_iteration_difference(df_h_t)

            # check for difference to standard set
            medians_subset = df_h_t.groupby('condition')['iteration count'].median()
            median_difference = medians_subset - medians_standard_set[h_t[0]].reindex(medians_subset.index) # positive if subset medians higher than standard medians

            if self_ex[0] < standard_self_ex: # if self-excitation lower than 3.5 -> iteration count must be higher
                valid_self_ex = (median_difference > 0).all()
            elif self_ex[0] > standard_self_ex: # if self-excitation higher than 3.5 -> iteration cound must be lower
                valid_self_ex = (median_difference < 0).all()
            else:
                valid_self_ex = True

            # combine all criteria          
            if right_winner and iterations_plausible and iteration_difference and valid_self_ex:
                possible_combinations.append(h_t[0])

        if possible_combinations:
            possible_combinations_dict[self_ex[0]] = possible_combinations
    
    # keep only valid h/t
    sim_data_se_valid, _ = keep_valid_h_t(sim_data_se, possible_combinations_dict, ['self-excitation', 'h/t'])

    # save whole and valid data frame
    sim_data_se.to_parquet('../data/simulation_data_all_self_ex.parquet', index=False)
    sim_data_se_valid.to_parquet('../data/simulation_data_all_self_ex_valid.parquet', index=False)

    print('Explored valid self-excitation values and saved line plots and updated data frame successully.')
