import numpy as np
import random
from Preparer import MotherTeresa
from network_functions import sigmoid_activation
from network_functions import softmax_activation
from network_functions import cross_entropy
from network_functions import split_file_name
import matplotlib.pyplot as plt
import pickle
import time
import os
import config

# i = input, o = output, h = hidden, c = context

class ElmanNetwork(MotherTeresa):

    def __init__(self, save_path, **preparer_kwargs):
        super().__init__(**preparer_kwargs)
        self.save_path = save_path

    def train_in_epochs(self, epochs=config.epochs):
        self.epochs = config.epochs
        self.epoch_losses = []
        self.validation_loss = []
        training_loss_patience = 0
        validation_loss_patience = 0
        self.min_val_loss = float('inf')
        for epoch in range(self.epochs):
            start = time.time()

            # run training set
            epoch_loss_per_token = self.train(self.input, training=True)
            self.epoch_losses.append(epoch_loss_per_token)
            
            # run validation set
            epoch_validation_loss_per_token = self.train(self.validation_set, training=False)
            self.validation_loss.append(epoch_validation_loss_per_token)
            
            if self.validation_loss[-1] < self.min_val_loss:
                self.min_val_loss = self.validation_loss[-1]
                best_parameters = self.save_parameters()

            duration = time.time() - start
            
            if epoch > 0:
                print(f'Processed epoch {epoch+1} in {duration:.2f} seconds, training loss = {self.epoch_losses[-1]:.4f}, delta = {self.epoch_losses[-1] - self.epoch_losses[-2]:.4f}, validation loss = {self.validation_loss[-1]:.4f}')
            
            # shuffle training data after every epoch
            random.shuffle(self.input)
            self.input = self.batch(self.seq_train, config.batch_size)
            
            # determine if the training should stop
            if epoch > 0 and self.validation_loss[-1] > self.min_val_loss:
                validation_loss_patience += 1
                if validation_loss_patience >= config.validation_loss_patience:
                    print(f'Stopping training after {epoch+1} epochs because validation loss was minimal at {self.min_val_loss}.')
                    self.write_parameters(best_parameters)
                    break
            else:
                validation_loss_patience = 0
                if epoch > 0 and np.abs(self.epoch_losses[-1] - self.epoch_losses[-2]) < config.training_loss_threshold:
                    training_loss_patience += 1
                    if training_loss_patience >= config.training_loss_patience:
                        print(f'Stopping training after {epoch+1} epochs because training loss did not change more than {config.training_loss_threshold} for {config.training_loss_patience} epochs.')
                        self.write_parameters(best_parameters)
                        break
                    else:
                        training_loss_patience = 0

        self.visualize_loss()
        return epoch+1

        
    def visualize_loss(self):
        log_dir = '../figures/training_figures'
        os.makedirs(log_dir, exist_ok=True)
        file_path = self.save_path.replace('parameters', 'loss')
        split = split_file_name(file_path)

        # visualize the loss
        plt.plot(self.epoch_losses, label="Train Loss")
        plt.plot(self.validation_loss, label="Validation Loss")
        plt.scatter(np.argmin(self.validation_loss), self.min_val_loss, color="red", zorder=5, label="Best Loss")
        plt.title(f"Loss over Epochs (Bias: {split[-3]}/{split[-2]}, Hidden units: {split[-1]})")
        plt.xlabel("Epoch")
        plt.ylabel("Network Loss")
        plt.legend()
        plt.savefig(f'{log_dir}/{file_path}.png', dpi=300)
        plt.close()


    def train(self, input, training): # input shape: (sequence_length, batch_size, vocab_size)
        # store batch_losses
        epoch_loss = 0
        epoch_tokens = 0

        # loop through the elements of the input list
        for batch in input:

            # extract batch_size and sequence length for every batch
            self.batch_size = batch.shape[1]
            sequence_length = batch.shape[0]
            
            # initialize context units and set them to zero for every batch
            self.context_units = np.zeros((self.batch_size, self.n_h_c)) 
            
            # initialize hidden states, ouputs and losses for every batch
            hidden_states_activated = []
            outputs = []
            
            for t in range(sequence_length - 1):
                # slice the input for the current time step from the 3D input array
                input_t = batch[t,:,:] # batch_size x self.n_i_o, teacher forcing 
                
                # run the prediction method
                if training:
                    self.predict(input_t, add_noise=True)
                else:
                    self.predict(input_t, add_noise=False)
                
                # calclulate the loss
                next_word = batch[t+1, :, :]
                loss = cross_entropy(self.output_activated, next_word) # (batch_size, )
                epoch_loss += np.sum(loss)
                epoch_tokens += loss.size

                # collect the hidden states and outputs from all time steps
                hidden_states_activated.append(self.hidden_state_activated) # batch_size x n_h_c as list items
                outputs.append(self.output_activated) # batch_size x self.n_i_o as list items
            
            # put hidden states and outputs in the right shape for the backward pass
            hidden_states_activated.insert(0, np.zeros_like(self.hidden_state_activated)) # insert zeros for the first time step to ensure correspondence with the input
            self.hidden_states_activated = np.stack(hidden_states_activated, axis = 0) # (sequence_length, batch_size, n_h_c)
            outputs.insert(0, np.zeros_like(self.output_activated)) # insert zeros to ensure correspondence with the input
            self.outputs = np.stack(outputs, axis = 0) # sequence_length, batch_size, self.n_i_o

            # run backward pass
            if training:
                self.backward(batch, config.learning_rate)

        return epoch_loss/epoch_tokens

    
    def predict(self, input_t, add_noise):
        hidden_state = np.dot(input_t, self.weights_i_h) + self.biases_h # batch_size x self.n_i_o * self.n_i_o x n_h_c = batch_size x n_h_c
        hidden_state_with_context = hidden_state + (np.dot(self.context_units, self.weights_c_h)) # batch_size x n_h_c * n_h_c x n_h_c = batch_size x n_h_c
        
        if add_noise:
            hidden_state_with_context += 0.01 * np.random.randn(*hidden_state_with_context.shape)

        if config.tanh_activation:
            # tanh activation
            self.hidden_state_activated = np.tanh(hidden_state_with_context) # batch_size x n_h_c
        else:
            # sigmoid activation
            self.hidden_state_activated = sigmoid_activation(hidden_state_with_context) # batch_size x n_h_c
        
        self.context_units = self.hidden_state_activated # batch_size x n_h_c
        self.output_state = np.dot(self.hidden_state_activated, self.weights_h_o) + self.biases_o # batch_size x n_h_c * n_h_c x self.n_i_o = batch_size x self.n_i_o
        self.output_activated = softmax_activation(self.output_state) # batch_size x self.n_i_o
                    

    def backward(self, input_batch, learning_rate):

        for t in reversed(range(1, np.shape(self.outputs)[0])): # for loop over all but the first time step (because the first word was given and not predicted)
            output_t = self.outputs[t, :, :] # (batch_size x self.n_i_o) 
            input_t = input_batch[t, :, :] # (batch_size x self.n_i_o)
            hidden_t = self.hidden_states_activated[t, :, :] # (batch_size x n_h_c)

            # calculating the partial derivative with respect to z (= output values before softmax = output_state)
            pd_output_state = output_t - input_t # batch_size x self.n_i_o // y_pred (= output_activated) - y-true (= next input)

            # calculate the gradient for weights_h_o
            delta_weights_h_o = np.dot(hidden_t.T, pd_output_state) # n_h_c x batch_size * batch_size x self.n_i_o = n_h_c x self.n_i_o

            # calculate the gradient for biases_o
            delta_biases_o = pd_output_state # batches x self.n_i_o
                       
            # calculate the partial derivative with respect to the hidden state at timestep t
            pd_hidden_state = np.dot(pd_output_state, self.weights_h_o.T) # batch_size x self.n_i_o * self.n_i_o x n_h_c = batch_size x n_h_c

            # calculate the activation derivative
            if config.tanh_activation:
                # tanh activation
                pd_activation = pd_hidden_state * (1 - hidden_t ** 2) # batch_size x n_h_c
            else:
                # sigmoid activation
                pd_activation = pd_hidden_state * hidden_t * (1 - hidden_t) # batch_size x n_h_c
            
            # calculate the gradient for the weights_i_h
            delta_weights_i_h = np.dot(input_t.T, pd_activation) # self.n_i_o x batch_size * batch_size x n_h_c = self.n_i_o x n_h_c

            # calculate the gradient for the biases_h
            delta_biases_h = pd_activation # batch_size x n_h_c
                      
            # calculate the gradient for the weights_c_h
            previous_word = self.hidden_states_activated[t-1, :, :]
            delta_weights_c_h = np.dot(previous_word.T, pd_activation) # n_h_c x batch_size * batch_size x n_h_c = n_h_c x n_h_c           

            # monitor gradients
            gradient_vector = np.concatenate([
                delta_weights_h_o.flatten(), 
                delta_weights_i_h.flatten(), 
                delta_weights_c_h.flatten(),
                delta_biases_o.flatten(),
                delta_biases_h.flatten()])
            gradient_norm = np.linalg.norm(gradient_vector)
            # print(f'Gradient norm: {gradient_norm}')

            scale_factor = self.clip_gradients(gradient_norm, config.gradient_threshold)

            # updating weights and biases
            self.weights_h_o -= learning_rate * (delta_weights_h_o * scale_factor) # n_h_c x self.n_i_o
            self.biases_o -= learning_rate * np.sum((delta_biases_o * scale_factor), axis = 0, keepdims= True) # 1, self.n_i_o
            self.weights_i_h -= learning_rate * (delta_weights_i_h * scale_factor) # self.n_i_o x n_h_c
            self.biases_h -= learning_rate * np.sum((delta_biases_h * scale_factor), axis=0, keepdims=True) # 1 x n_h_c
            self.weights_c_h -= learning_rate * (delta_weights_c_h * scale_factor) # n_h_c x n_h_c


    def clip_gradients(self, gradient_norm, threshold):
        if config.gradient_clipping and gradient_norm > threshold:
            scale_factor = threshold / gradient_norm

        else:
            scale_factor = 1

        return scale_factor


    def save_parameters(self):

        parameters = {
            'weights_i_h': self.weights_i_h,
            'weights_h_o': self.weights_h_o,
            'weights_c_h': self.weights_c_h,
            'biases_h': self.biases_h,
            'biases_o': self.biases_o,
            'first_words': self.first_words,
            'unique_words': self.unique_words,
            'n_i_o': self.n_i_o,
            'n_h_c': config.hidden_units,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'split': config.split,
            'epochs': config.epochs,
            'gradient_clipping': config.gradient_clipping,
            'gradient_threshold': config.gradient_threshold,
            'tanh_activation': config.tanh_activation,
            'training_loss_threshold': config.training_loss_threshold,
            'training_loss_patience': config.training_loss_patience,
            'validation_loss_patience': config.validation_loss_patience,
            'top_n': config.top_n,
            'seed': config.seed}
        
        return parameters
        

    def write_parameters(self, parameter_dict):
        # save a human readable txt with all parameters in a subdirectory
        log_dir = 'log_files'
        os.makedirs(log_dir, exist_ok=True)  
        filename = f'{log_dir}/{self.save_path}'


        with open(f'{filename}.pkl', 'wb') as f:
            pickle.dump(parameter_dict, f)

        with open(f'{filename}.txt', 'w') as f:
            for key, value in parameter_dict.items():
                f.write(f'{key}: {value}\n')
