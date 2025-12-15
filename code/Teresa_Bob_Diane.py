from Generator import Generator
import random
import numpy as np
from network_functions import one_hot
from network_functions import translate_prediction
from network_functions import split_file_name
from Diane import Resolver
import config
import config_Diane


class Turn_Taker():
    def __init__(self, parameter_file_Teresa, parameter_file_Bob):

        # instantiate Teresa and Bob with their parameters
        self.Teresa = Generator(parameter_file_Teresa)
        self.Bob = Generator(parameter_file_Bob)

        self.predicted_words = []
        self.unique_words = self.Teresa.unique_words
        self.vocabulary_size = self.Teresa.n_i_o
        self.iterations = []

        # extract metrics from file name
        split = split_file_name(parameter_file_Teresa)
        self.bias_Teresa = int(split[-3])
        self.bias_Bob = int(split[-2])


    def talk(self, bob_begins, heard_factor, thought_factor, self_excitation):

        self.Teresa.prepare_to_generate()
        self.Bob.prepare_to_generate()

        metrics_list = []

        # begin loop at word 0 with Bob
        for word in range(6):

            # prepare metrics dict
            metrics = {
                'predicted index': None,
                'predicted word': None,
                'iteration count': None,
                'winner Diane': None,
                'word position': word+1,
                'match': None,
                'confidence': None,
                'turn': None,
                'internal winner': None,
                'top 5 internal prediction': None,
                'top 5 words': None,
                'top 5 probabilities': None,
                'bias Teresa': self.bias_Teresa,
                'bias Bob': self.bias_Bob,
                'bob begins': bob_begins,
                'heard factor': heard_factor,
                'thought factor': thought_factor,
                'inhibition': config_Diane.inhibition,
                'self-excitation': self_excitation
            }

            if bob_begins: # Bob takes equal turns (0, 2, 4)
                if word % 2 == 0:
                    bobs_turn = True
                else:
                    bobs_turn = False
            else: # Teresa takes equal turns
                if word % 2 == 0:
                    bobs_turn = False
                else:
                    bobs_turn =True
            
            metrics['turn'] = 'Bob' if bobs_turn else 'Teresa'

            # check, if it's the first turn
            if word == 0:
                if bob_begins: # Bob draws first word
                    first_word = random.choice(self.Bob.first_words)
                else:
                    first_word = random.choice(self.Teresa.first_words)
                self.predicted_words.append(first_word)
                output_one_hot = np.array(one_hot(first_word, self.vocabulary_size))
                # print first word
                # if bob_begins:
                #     print(f'Bob says {word+1}. word: {translate_prediction(first_word, self.unique_words)}')
                # else:
                #     print(f'Teresa says {word+1}. word: {translate_prediction(first_word, self.unique_words)}')

                metrics['predicted index'] = first_word
                metrics['predicted word'] = translate_prediction(first_word, self.unique_words)
                metrics_list.append(metrics)

            else:
                # check if it's Teresa's or Bob's turn
                if bobs_turn: # it's Bob's turn
                    # predict a word based on the previous word (probabilistic from top 5)
                    next_word, _, _ = self.Bob.generate_next_word(output_one_hot)
                    # print(f'Bob says {word+1}. word: {translate_prediction(next_word, self.unique_words)}')

                    # Teresa makes internal prediction
                    _, probabilities, idx = self.Teresa.generate_next_word(output_one_hot)
                    idx = np.squeeze(idx)
                    internal_value_next_word = self.Teresa.output_activated[0, next_word]
                    internal_prediction_words = [translate_prediction(word, self.unique_words) for word in idx]
                    # print(f"Teresa's internally predicted words are: {internal_prediction_words} with probabilities: {probabilities}")
                            
                else: # it's Teresa's turn
                    # predict the next word
                    next_word, _, _ = self.Teresa.generate_next_word(output_one_hot)
                    # print(f'Teresa says {word+1}. word: {translate_prediction(next_word, self.unique_words)}')

                    # Bob makes internal prediction
                    _, probabilities, idx = self.Bob.generate_next_word(output_one_hot)
                    idx = np.squeeze(idx)
                    internal_value_next_word = self.Bob.output_activated[0, next_word]
                    internal_prediction_words = [translate_prediction(word, self.unique_words) for word in idx]
                    # print(f"Bob's internally predicted words are: {internal_prediction_words} with probabilities: {probabilities}")            
                
                metrics['predicted index'] = next_word
                metrics['predicted word'] = translate_prediction(next_word, self.unique_words)
                metrics['top 5 internal prediction'] = idx
                metrics['top 5 words'] = internal_prediction_words
                metrics['top 5 probabilities'] = np.squeeze(probabilities)
                
                # build combined support vector
                
                # check if said word is in internal prediction
                if next_word in idx:
                    metrics['confidence'] = True
                    heard_word = np.where(idx == next_word)[0][0]
                    heard_word_one_hot = np.array(one_hot(heard_word, config.top_n))
                    
                else:
                    metrics['confidence'] = False
                    # find minimal value
                    least_likely_word_idx = np.argmin(probabilities)
                    # replace with heard word value
                    probabilities[0][least_likely_word_idx] = internal_value_next_word
                    heard_word_one_hot = np.array(one_hot(least_likely_word_idx, config.top_n))

                support = heard_factor * heard_word_one_hot + thought_factor * np.array(probabilities)
                internal_max = np.argmax(probabilities)
                internal_winner = idx[internal_max]
                metrics['internal winner'] = internal_winner
                if next_word == internal_winner:
                    metrics['match'] = True
                    # print('Said word matched internal prediction.')
                else:
                    # print('Said word did not match internal prediction.')
                    metrics['match'] = False

                # instantiate Diane with the support vector
                Diane = Resolver(support, self_excitation)
                Diane.prepare()
                # let Diane resolve
                iteration_count, winner_diane = Diane.iterate()

                metrics['iteration count'] = iteration_count
                metrics['winner Diane'] = winner_diane
                metrics_list.append(metrics)

                # collect the iteration count in a list
                self.iterations.append(iteration_count)
                # show Diane's graphics
                # Diane.visualize()

                # return the index of the predicted word and add it to a shared list of predicted words
                self.predicted_words.append(next_word)
                # set the heard word as new input
                output_one_hot = np.array(one_hot(next_word, self.vocabulary_size))

        # translate predicted indices into words
        predicted_sentence = [translate_prediction(index, self.unique_words) for index in self.predicted_words]
        self.predicted_sentence_words = f'The {predicted_sentence[0]} {predicted_sentence[1]} with the {predicted_sentence[2]} to the {predicted_sentence[3]} and {predicted_sentence[4]} {predicted_sentence[5]}.'


        return metrics_list
    


