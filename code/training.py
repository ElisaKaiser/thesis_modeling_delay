from Trainer import ElmanNetwork
from network_functions import split_file_name
import config
import os

def train_all_biases():
    # take all filenames from the training data files
    data_dir = '../data/training_data'
    training_files = [f for f in os.listdir(data_dir)]

    for training_file in training_files:

        # create save file name
        split = split_file_name(training_file)
        agent = 'Teresa' if 'Teresa' in training_file else 'Bob'
        save_file = f'parameters_{agent}_{split[-2]}_{split[-1]}_{config.hidden_units}'

        # instantiate the training network
        Speaker = ElmanNetwork(save_file, training_data=f'{data_dir}/{training_file}')

        # train
        Speaker.train_in_epochs()

    print('Trained Teresa and Bob with all biases successfully.')

