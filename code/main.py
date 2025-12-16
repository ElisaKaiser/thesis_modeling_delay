import random
import numpy as np
from network_functions import set_wd
from training_data_generator import create_training_data
from validating import validate
from testing import test
from training import train_all_biases
from talking import simulate
from exploring import explore_valid_h_t
from analyzing import analyze
from analyzing_sensitivity import analyze_sensitivity
from varying_self_excitation import explore_self_excitations
from analyzing_self_ex import analyze_self_ex
from finding_parameter_sets import find_parameter_sets
from showing_Diane import show_Diane
import config
import time
import pickle

set_wd()
random.seed(config.seed)
np.random.seed(config.seed)

start = time.time()

# generate training data for different biases
create_training_data() # generates training data with biases for Teresa and Bob and saves them to /data/training_data

# train Teresa with the neutral training data, validate for different hidden units
validate() # saves a markdown file to data/, saves training loss figures to figures/training_figures

# give Teresa_64 test set and report test loss, perplexity, accuracy and stuff
test() # saves markdown with metrics to /data

# show off Diane
show_Diane() # saves figures for different support vectors, self-excitations, and inhibitions to figures/figures_Diane

# train Teresa and Bob with all biases and save parameters
train_all_biases() # saves parameters to /code/log_files

# run simulation for all biases and all h/t combinations
h_t_pairs = [(h/10, t/10) for h in range(1,10) for t in range(1,10) if h >= t]
simulate(self_excitation=3.5, all_biases=True, h_t_pairs=h_t_pairs)

# determine valid bias-h/t combinations
explore_valid_h_t() # saves dictionary to /code/log_files, updated simulation data to /data, and figure to /figures/sensitivity

# make descriptive analyses, histograms, line plots, summary for all of them
analyze() # saves descriptive stats to /data/bias_h_t_stats, histograms to figures/bias_h_t_histograms, and line plots to /figures/line_plots

# make sensitivity analysis (bias-h/t)
analyze_sensitivity() # saves heat maps to /figures/heat_maps

# run simulation for the 90/10 bias and different (higher) self-excitations and only valid h/t
with open('log_files/valid_h_t_per_bias.pkl', 'rb') as f:
    valid_h_t_dict = pickle.load(f)
valid_h_t_str = valid_h_t_dict['90/10']
valid_h_t = [tuple(map(float, s.split('/'))) for s in valid_h_t_str]
self_excitations = [round(s/10, 1) for s in range(35, 41)]
for self_ex in self_excitations:
    simulate(self_excitation=self_ex, all_biases=False, h_t_pairs=valid_h_t)

# explore valid self-excitation values
explore_self_excitations(self_excitations) # saves updated df to /data and line plots to /figures/self_ex_exploration

# sensitivity analysis with self-ex and h/t
analyze_self_ex() # saves line plots to /figures/self_ex_interaction_plots and heat maps to /figures/heat_maps

# find improv player parameter set
find_parameter_sets() # saves interaction plots to /figures/interaction_plots, and md table to /data


duration = time.time() - start
print(f'Running main script took {int(duration)} seconds.')
print('Done.')