from network_functions import sigmoid_activation
import numpy as np
import matplotlib.pyplot as plt
import config_Diane
import os

class Resolver:
    def __init__(self, support, self_excitation):
        self.support = support.reshape(-1)
        self.support_original = self.support
        self.prepare()
        units = len(self.support)

        # parameters
        self.inhibition = config_Diane.inhibition
        self.self_excitation = self_excitation
        self.gamma = config_Diane.gamma
        self.alpha = config_Diane.delta_t / config_Diane.tau
        self.b = config_Diane.b

        # initialize units
        self.activations = np.zeros((units))

        # weight matrix
        self.weights = self.inhibition + np.zeros((units, units))
        np.fill_diagonal(self.weights, self.self_excitation)

        self.steps = 0

        # dominance criterion
        self.dominance = False
        self.patience_dominance = 0
        self.patience_limit_dominance = config_Diane.patience_limit_dominance
        self.threshold_dominance = config_Diane.threshold_dominance

        # convergence criterion
        self.convergence = False
        self.patience_convergence = 0
        self.patience_limit_convergence = config_Diane.patience_limit_convergence
        self.threshold_activations = config_Diane.threshold_activations
        self.threshold_energy = config_Diane.threshold_energy

        # visualizing
        self.e = [0]
        self.trace = []
        self.trace.append(self.activations)

    def prepare(self):
        # center support input at zero
        self.support = self.support - 1/len(self.support)


    def update(self):
        self.steps += 1

        # leaky integrator update rule: calculate activation in next time step
        delta_next_step = (
            - self.gamma * self.activations 
            + np.dot(self.weights, sigmoid_activation(self.activations)) 
            + self.support 
            + self.b)
        next_activations = self.activations + self.alpha * delta_next_step

        changes = np.abs(self.activations - next_activations)
        self.activations = next_activations.reshape(-1)

        # collect values
        self.e.append(self.energy())
        self.trace.append(self.activations.copy())

        # check dominance
        if not self.dominance:
            a = sigmoid_activation(self.activations)
            idx_top_2 = np.argpartition(a, -2)[-2:]
            top_2 = a[idx_top_2]
            a_top_1 = top_2.max()
            a_top_2 = top_2.min()
            dominance_level = (a_top_1 - a_top_2) / (a_top_1 + a_top_2 + 1e-12)
            if dominance_level >= self.threshold_dominance:
                self.patience_dominance += 1
            else:
                self.patience_dominance = 0

            if self.patience_dominance >= self.patience_limit_dominance:
                self.dominance = True
                # print(f'Dominance reached after {self.steps} steps.')

        # check convergence
        if self.dominance:
            energy_change = np.abs(self.e[-1] - self.e[-2])
            if np.max(changes) <= self.threshold_activations and energy_change < self.threshold_energy:
                self.patience_convergence += 1
            else:
                self.patience_convergence = 0

            if self.patience_convergence >= self.patience_limit_convergence:
                self.convergence = True
                # print(f'Convergence reached after {self.steps} steps.')


    def iterate(self):
        while self.steps < 2000 and not self.convergence:
            self.update()
            
        return self.steps, np.argmax(self.activations)
    
    def energy(self):
        e = -np.dot(self.activations, np.dot(self.weights, self.activations))
        return e
    
    def visualize(self):
        dir = '../figures/Diane_figures'
        os.makedirs(dir, exist_ok=True)
        file = f'Diane_{self.support_original}_{self.inhibition}_{self.self_excitation}.png'
        file_path = f'{dir}/{file}'

        fig, axs = plt.subplots(2, 2)

        # traces
        trace_array = np.array(self.trace)  # shape: [timesteps, units]
        for i in range(trace_array.shape[1]):
            axs[0, 0].plot(trace_array[:, i], label=f'Unit {i}')
        axs[0, 0].set_xlabel('Timestep')
        axs[0, 0].set_ylabel('Activation')
        axs[0, 0].set_title('Activation Traces over Time')
        axs[0, 0].grid(True)
        axs[0, 0].legend()

        # energy
        axs[0, 1].plot(self.e)
        axs[0, 1].set_xlabel('Iteration')
        axs[0, 1].set_ylabel('Energy')
        axs[0, 1].set_title('Energy over iterations')
        axs[0, 1].grid(True)

        # combined support distribution
        axs[1, 0].scatter(range(len(self.support_original)), self.support_original)
        axs[1, 0].set_xlabel("Index")
        axs[1, 0].set_ylabel("Value")
        axs[1, 0].set_title("Support Distribution")
        axs[1, 0].grid(True)

        # show final activation
        axs[1, 1].scatter(range(len(self.activations)), self.activations)
        axs[1, 1].set_xlabel("Index")
        axs[1, 1].set_ylabel("Value")
        axs[1, 1].set_title("Final Activations")
        axs[1, 1].grid(True)

        fig.suptitle(f"Activation & Energy - Self-excitation: {self.self_excitation:.2f}, Inhibition: {self.inhibition:.2f}, Steps: {self.steps}")
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

    
