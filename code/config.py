import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# config file

seed = 42

split = 0.7

hidden_units = 64

batch_size = 16

epochs = 1000

learning_rate = 0.001

gradient_clipping = False

gradient_threshold = 5

tanh_activation = True

training_loss_threshold = 0.001

training_loss_patience = 10

validation_loss_patience = 5

top_n = 5

bias_commited = 'parameters_Teresa_90_10_64.pkl'