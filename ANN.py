import random
import math


class ANN:
    def __init__(self, momentum: float, learning_rate: float, layers: int, neurons: list[int]) -> None:
        try:
            assert layers == len(neurons)
            self.momentum = momentum  # Setting the momentum term (γ)
            self.learning_rate = learning_rate  # Setting the learning rate (λ)
            self.weights = [[] for _ in range(layers)]  # Neuron layers and weights (w)
            for n_layer in range(layers):
                if n_layer == 0:
                    # Maintain a unity value weight for all input layer neurons.
                    # (Assuming "unity value" means 1)
                    self.weights[0] = [1 for _ in range(neurons[0])]
                else:
                    # Generate arbitrary weights within an interval [-1, 1] and assign
                    # it to the hidden layer neurons and output layer neurons.
                    self.weights[n_layer] = [random.uniform(-1, 1) for _ in range(neurons[n_layer])]
        except AssertionError:
            raise ValueError("'layers' and 'neurons' sizes do not match.")

    def _output_error_rate(self, predicted, actual):
        """Calculates the error rate of the output (δ_op_r)."""
        # Equation 7: δ_op_r = p_op_r * (1 − p_op_r ) * (t_op_r − p_op_r )
        return predicted * (1 - predicted) * (actual - predicted)

    def _hidden_error_rate(self, output, this_neuron, this_layer):
        """Calculates the error rate of a hidden neuron (δ_op_q)."""
        # Equation 8: δ_op_r = p_op_q * (1 - p_op_q) * Σ (w_qr * δ_op_r)
        return this_neuron * (1 - this_neuron) * sum([weight * output for weight in this_layer])

    def _sigmoid_activation(self, weight: float, inputs: list[float]) -> float:
        """Returns the output of a neuron's activation function(p_op)."""
        # Equations 9/10: p_op_p = 1 / ( 1 + e^-Σ(w_pq * ip_p))
        return 1 / (1 + math.e ** -sum([_input * weight for _input in inputs]))

    def _new_weight(self, current_weight, weight_change, previous_weight_change):
        """Calculates the new weight of a neurone (w_rq)."""
        # Equation 11/13: w_rq = w_rq + Δ w_rq + (γ * Δ (t - 1))
        return current_weight + weight_change + (self.momentum * previous_weight_change)

    def _weight_change(self, error_rate, _input):
        """Calculates the weight change for a given node and the input going into it (Δ w_rq)."""
        # Equation 12/14: Δ w_rq = λ * δ_op_rq * ip_rq
        return self.learning_rate * error_rate * _input
