from Diane import Resolver
import numpy as np
import pandas as pd
from itertools import product

def show_Diane():
    support = [[0.01, 0.19, 0.005, 0.79, 0.005], [0.01, 0.48, 0.005, 0.5, 0.005], [0.9, 0.01, 0.03, 0.04, 0.02]]
    support = [np.array(s) for s in support]

    self_excitation = [round(s/10, 1) for s in range(35, 41)]

    iteration_counts = []

    for s, e in product(support, self_excitation):
        
        Diane = Resolver(support=s, self_excitation=e)
        iterations, winner_out = Diane.iterate()
        Diane.visualize()

        iter = {
        'support': list(s),
        'winner in': np.argmax(s),
        'winner out': winner_out,
        'self-excitation': e,
        'iterations': iterations}

        iteration_counts.append(iter)

    s_i_e_df = pd.DataFrame(iteration_counts)

    s_i_e_df['right winner'] = s_i_e_df['winner in'] == s_i_e_df['winner out']

    print('Presented Diane successfully.')
