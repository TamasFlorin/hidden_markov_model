from typing import List
from typing import Any
from typing import Tuple
from math import isclose
from copy import deepcopy


class HiddenMarkovModel(object):
    """
    A probabilistic sequence classifier - given a sequence of units,
    computes a probability distribution over possible labels and chooses the
    best label.
    """

    def __init__(self, states: List[Any], vocabulary: List[Any], transition_probabilities: Any, emission_probabilities: Any, initial_probabilities: Any):
        self.transition_probabilities = transition_probabilities
        self.emission_probabilities = emission_probabilities
        self.initial_probabilities = initial_probabilities
        self.states = states
        self.vocabulary = vocabulary

    def forward(self, observations: List[Any]) -> Tuple[float, List[Tuple[Any, float]]]:
        """
        Returns the probability of seeing the given `observations` sequence,
        using the Forward algorithm.

        :param observations: The list of observed values.
        :return: The probability of seeing the given observations along with the forward states
                 for each iteration.
        """
        assert len(
            observations) > 0, "Observations list must contain at least one element."

        f = [[0.0 for _ in observations] for _ in self.states]
        forwards = []
        # initialization step
        for i, state in enumerate(self.states):
            transition = self.initial_probabilities[state]
            emission = self.emission_probabilities[state][observations[0]]
            result = transition * emission
            f[i][0] = result

            forwards.append((state, result))

        # recursion step
        for t, observation in enumerate(observations[1:], start=1):
            for s in range(len(self.states)):
                current_sum = 0
                for i in range(len(self.states)):
                    f_value = f[i][t - 1]
                    transition = self.transition_probabilities[self.states[i]
                                                               ][self.states[s]]
                    current_sum += (f_value * transition)

                emission = self.emission_probabilities[self.states[s]
                                                       ][observation]
                f[s][t] = current_sum * emission
                forwards.append((self.states[s], current_sum))

        # termination step
        T = len(observations) - 1
        P = sum(f[s][T] for s in range(len(self.states)))

        return P, f

    def backward(self, observations: List[Any]) -> Tuple[float, List[Tuple[Any, float]]]:
        """
        Returns the probability of seeing the given `observations` sequence,
        using the Backward algorithm.

        :param observations: The list of observed values.
        :return: The probability of seeing the given observations along with the backward states
                 for each iteration.
        """
        assert len(
            observations) > 0, "Observations list must contain at least one element."

        b = [[0.0 for _ in observations] for _ in self.states]
        backwards = []
        T = len(observations) - 1
        for i, state in enumerate(self.states):
            b[i][T] = 1.0
            backwards.append((state, b[i][T]))

        # recursion step
        for t in range(T - 1, -1, -1):
            next_observation = observations[t + 1]
            for i, state in enumerate(self.states):
                current_sum = 0
                for j, next_state in enumerate(self.states):
                    transition = self.transition_probabilities[state][next_state]
                    emission = self.emission_probabilities[next_state][next_observation]
                    next_value = b[j][t + 1]
                    current_sum += (transition * emission * next_value)
                b[i][t] = current_sum
                backwards.append((state, b[i][t]))

        # termination step
        P = 0.0
        first_obs = observations[0]
        for j, state in enumerate(self.states):
            transition = self.initial_probabilities[state]
            emission = self.emission_probabilities[state][first_obs]
            P += (transition * emission * b[j][0])

        return P, b

    def viterbi(self, observations: List[Any]) -> Tuple[float, List[Any]]:
        """
        Returns the best path of states and the probability that the returned states
        will lead to the given observations.

        :param observations: The list of observed values.
        :return: The list of the most likely states and the probability.
        """
        v = [[0.0 for _ in observations] for _ in self.states]
        backpoints = [[None] for _ in self.states]
        # Initialization step
        for s in range(0, len(self.states)):
            transition = self.initial_probabilities[self.states[s]]
            emission = self.emission_probabilities[self.states[s]
                                                   ][observations[0]]
            result = transition * emission
            v[s][0] = result
            backpoints[s] = [s]

        # recursion step
        for t in range(1, len(observations)):
            current_backpoints = [0 for _ in self.states]
            for s in range(len(self.states)):
                values = []
                for i in range(len(self.states)):
                    v_v = v[i][t - 1]
                    transition = self.transition_probabilities[self.states[i]
                                                               ][self.states[s]]
                    emission = self.emission_probabilities[self.states[s]
                                                           ][observations[t]]
                    result = v_v * transition * emission
                    values.append(result)

                max_value, max_index = max((v, idx)
                                           for idx, v in enumerate(values))
                current_backpoints[s] = backpoints[max_index] + [s]
                v[s][t] = max_value
            backpoints = current_backpoints

        # termination step
        T = len(observations) - 1
        max_value, max_index = max((v[i][T], i)
                                   for i in range(len(self.states)))
        path = [self.states[index] for index in backpoints[max_index]]
        return max_value, path

    def forward_backward(self, observations: List[Any]):
        assert len(
            observations) > 0, "Observations list must contain at least one element."

        forward_probability, forwards = self.forward(observations)
        backward_probability, backwards = self.backward(observations)

        # probabilities should be roughly the same
        assert isclose(forward_probability, backward_probability)
        assert forward_probability > 0

        observations_probability = forward_probability

        gamma = [[0.0 for _ in observations] for _ in self.states]
        for t in range(len(observations)):
            for y in range(len(self.states)):
                gamma[y][t] = forwards[y][t] \
                    * backwards[y][t] \
                    / observations_probability

        xi = [[[0 for _ in observations] for _ in self.states]
              for _ in self.states]

        for t in range(0, len(observations) - 1):
            for s1, state1 in enumerate(self.states):
                for s2, state2 in enumerate(self.states):
                    f_value = forwards[s1][t]
                    transition = self.transition_probabilities[state1][state2]
                    emission = self.emission_probabilities[state2][observations[t + 1]]
                    b_value = backwards[s2][t + 1]
                    xi[s1][s2][t] = f_value * transition * emission * b_value \
                        / observations_probability

        first_state = self.states[0]
        last_state = self.states[-1]

        # compute new transition probabilities
        transition_prob = deepcopy(self.transition_probabilities)
        for i, state in enumerate(self.states):
            state_probability = sum(gamma[i])
            transition_prob[first_state][state] = gamma[i][0]
            transition_prob[state][last_state] = gamma[i][-1] / \
                state_probability

            for j, other_state in enumerate(self.states):
                transition_prob[state][other_state] = sum(
                    xi[i][j]) / state_probability

        # compute new emission probabilities
        emission_prob = deepcopy(self.emission_probabilities)
        for i, state in enumerate(self.states):
            for j, symbol in enumerate(self.vocabulary):
                current_sum = sum(gamma[i][t] for t in range(
                    len(observations)) if observations[t] == symbol)
                gamma_sum = sum(gamma[i])
                gamma_sum = gamma_sum if gamma_sum > 0 else 1.0
                emission_prob[state][symbol] = current_sum / gamma_sum

        return HiddenMarkovModel(self.states, self.vocabulary, transition_prob, emission_prob, self.initial_probabilities)
