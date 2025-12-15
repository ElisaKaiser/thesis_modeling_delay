from Trainer import ElmanNetwork
from network_functions import one_hot
from network_functions import select_first_word
from network_functions import translate_prediction
from network_functions import set_wd
import random
import numpy as np
import pickle
import config


class Generator(ElmanNetwork):
    def __init__(self, parameters_file):
        set_wd()

        # load parameters from pickle file
        with open(parameters_file, 'rb') as f:
            parameters = pickle.load(f)

        self.weights_i_h = parameters['weights_i_h']
        self.weights_h_o = parameters['weights_h_o']
        self.weights_c_h = parameters['weights_c_h']
        self.biases_h = parameters['biases_h']
        self.biases_o = parameters['biases_o']
        self.n_i_o = parameters['n_i_o']
        self.n_h_c = parameters['n_h_c']
        self.first_words = parameters['first_words']
        self.unique_words = parameters['unique_words']
                

    def prepare_to_generate(self):
        # empty context units before generating a sequence
        self.context_units = np.zeros((1, self.n_h_c))

        # initialize prediction output as a list for the indices
        self.probabilities_candidates = []
        self.prediction_output = []
        
        
    def generate_next_word(self, previous_output):

        # predict the next word
        self.predict(previous_output, add_noise=False)

        # from acitvated output
        top_idx = np.argsort(self.output_activated, axis=1)[:, -config.top_n:]
        probabilities_top = np.take_along_axis(self.output_activated, top_idx, axis=1)

        next_words = []

        for row in range(probabilities_top.shape[0]):
            next_words.append(random.choices(top_idx[row,:], weights=probabilities_top[row,:])[0])

        if len(next_words) == 1:
            next_word = next_words[0]
        else:
            next_word = next_words
 
        return next_word, probabilities_top, top_idx
    

    def generate_sentence(self, sequence_length):
        self.prepare_to_generate()

        # define first word
        first_word = int(select_first_word(self.first_words))
        first_word_one_hot = np.array(one_hot(first_word, self.n_i_o))
            
        # write the first word in in the prediction output list
        self.prediction_output.append(first_word)

        self.probabilities = []
        self.candidates = []

        # run generate_next_word() pass from the second to the last word
        # collect next word, candidates, and probabilities
        for word_prediction in range(1, sequence_length):
           
            # predict the second word from the first word
            if word_prediction == 1:
                next_word, prob, cand = self.generate_next_word(first_word_one_hot)
                
            # predict the next words from the previous output
            else:
                next_word, prob, cand = self.generate_next_word(prediction_output_one_hot)

            self.probabilities.append(prob)
            self.candidates.append(cand)

            # transform the next word index into a one-hot coded vector
            prediction_output_one_hot = np.array(one_hot(next_word, self.n_i_o))
            
            # append the predicted word to the list
            self.prediction_output.append(next_word)

        # translate prediction output into a sentence
        self.predicted_sentence = [translate_prediction(word, self.unique_words) for word in self.prediction_output]

        predicted_sentence_words = f'The {self.predicted_sentence[0]} {self.predicted_sentence[1]} with the {self.predicted_sentence[2]} to the {self.predicted_sentence[3]} and {self.predicted_sentence[4]} {self.predicted_sentence[5]}'

        return self.predicted_sentence, predicted_sentence_words
    

    def generate_to_test(self, test_set, teacher_forcing):
        # extract sequence length from test_set that comes as a list of 3D arrays, shape of batch: (sequence_length, batch_size, vocab_size)
        with open(test_set, 'rb') as f:
            seq_test = pickle.load(f)

        sequence_length = seq_test[0].shape[0]

        predicted_sentences_in_batches = [] # list of lists of lists, inner lists = batches of predicted words, middle lists = word in the sentence, outer lists = batches

        total_token = 0
        total_hits = 0
        hits_per_word = {
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0
        }
        token_per_word = {
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0
        }

        for batch in seq_test:

            predicted_sentence = [] # list of lists, inner lists = batches of predicted words, outer lists = word in the sentence
            # extract batch size from every batch (in case of incomplete last batch)
            batch_size = batch.shape[1]
            self.context_units = np.zeros((batch_size, self.n_h_c))

            for w in range(sequence_length):
                if w == 0:
                    # select first words from pool according to batch size -> list
                    prediction_output = [int(select_first_word(self.first_words)) for _ in range(batch_size)]
                    target = np.array([one_hot(word, self.n_i_o) for word in prediction_output])
                      
                else:
                    # predict the next five words based on the first word, no teacher forcing
                    prediction_output, prob, cand = self.generate_next_word(prediction_output_one_hot)

                    target = batch[w,:,:]
                    target_idx = np.argmax(target, axis=1)

                    # find out if true next word is in top 5
                    hits = (cand == target_idx[:, None]).any(axis=1)
                    total_hits += np.sum(hits)
                    total_token += hits.size

                    hits_per_word[w+1] += np.sum(hits)
                    token_per_word[w+1] += hits.size

                    target_words = [translate_prediction(index, self.unique_words) for index in target_idx.tolist()]
                    # print(f'{w+1}. target words: {target_words}')


                prediction_output_word = [translate_prediction(index, self.unique_words) for index in prediction_output]
                # print(f'{w+1}. words: {prediction_output_word}')
                
                if teacher_forcing:
                    prediction_output_one_hot = target
                else: # free run   
                    prediction_output_one_hot = np.array([one_hot(word, self.n_i_o) for word in prediction_output])

                # append predicted word-batch to predicted_sentence
                predicted_sentence.append(prediction_output)

            predicted_sentences_in_batches.append(predicted_sentence)

        # calculate accuracy per word
        accuracy_per_word = []
        for key in range(2, 7):
            accuracy_per_word.append(hits_per_word[key]/token_per_word[key])

        # calculate total accuracy
        total_accuracy = total_hits/total_token

        return total_accuracy, accuracy_per_word 

        
        
        

      