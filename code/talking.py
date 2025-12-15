from Teresa_Bob_Diane import Turn_Taker
import os
import pandas as pd

def simulate(self_excitation, all_biases, h_t_pairs):
    def talking(parameter_files, heard, thought, self_excitation):

        sentences = 50

        metrics = []

        # loop through paramter sets
        for n, file in enumerate(parameter_files):

            for i in range(sentences):
                file_Teresa = f'log_files/{file}'
                file_Bob = file_Teresa.replace('Teresa', 'Bob')
                # instantiate Turn_Taker
                Dialogue = Turn_Taker(parameter_file_Teresa=file_Teresa, parameter_file_Bob=file_Bob)
                # let Teresa begin 50 sentences
                metrics.extend(Dialogue.talk(bob_begins=True, heard_factor=heard, thought_factor=thought, self_excitation=self_excitation))
                # instantiate Turn_Taker
                Dialogue = Turn_Taker(parameter_file_Teresa=file_Teresa, parameter_file_Bob=file_Bob)
                # let Bob begin 50 sentences
                metrics.extend(Dialogue.talk(bob_begins=False, heard_factor=heard, thought_factor=thought, self_excitation=self_excitation))

        return metrics

    # extract parameter files
    if all_biases:
        dir = 'log_files'
        files_Teresa = [f for f in os.listdir(dir) if f.endswith('.pkl') and 'parameters' in f and 'Teresa' in f and '64' in f]
    else:
        files_Teresa = ['parameters_Teresa_90_10_64.pkl']

    # make container for metrics
    metrics = []

    # loop through all possible combinations, log metrics, collect them all in one gigantic data frame

    for h, t in h_t_pairs:
        metrics_per_combo = talking(files_Teresa, h, t, self_excitation)
        metrics.extend(metrics_per_combo)
        print(f'Processed parameter combination h={h}/t={t}.')


    # transform into pandas
    simulation_df = pd.DataFrame(metrics)

    # save
    if all_biases:
        path_file = f'../data/simulation_data_all_biases_{self_excitation}.parquet'
    else:
        path_file = f'../data/simulation_data_90_10_{self_excitation}.parquet'
    
    simulation_df.to_parquet(path_file, index=False)

    print('Completed simulation and saved pandas data frame successfully.')





