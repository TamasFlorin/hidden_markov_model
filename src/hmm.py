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
            print(state)
            emission = self.emission_probabilities[state][observations[0]]
            result = transition * emission
            f[i][0] = result
            forwards.append((state, result))

        # recursion step
        for t, observation in enumerate(observations[1:], start=1):
            for s in range(len(self.states)):
                current_sum = 0
                for i in range(len(self.states)):
                    if i == s:
                        continue
                    f_value = f[i][t - 1]
                    # print(self.transition_probabilities[self.states[i]])
                    transition = self.transition_probabilities[self.states[i]
                                                               ][self.states[s]]
                    print("From {} to {} with t={} f={}".format(
                        self.states[i], self.states[s], transition, f_value))
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
                    if i == j:
                        continue
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

        T = len(observations) - 1
        N = len(self.states)
        gamma = [[0.0 for _ in range(T + 1)]
                 for _ in self.states]
        xi = [[[0 for _ in range(T)] for _ in self.states]
              for _ in self.states]

        for t in range(T):
            s = 0
            next_obs = observations[t + 1]
            for i, state1 in enumerate(self.states):
                for j, state2 in enumerate(self.states):
                    xi[i][j][t] = forwards[i][t] * self.transition_probabilities[state1][state2] * \
                        self.emission_probabilities[state2][next_obs] * \
                        backwards[j][t+1]
                    s += xi[i][j][t]

            # Normalize
            for i in range(N):
                for j in range(N):
                    xi[i][j][t] *= 1 / s

        # Now calculate the gamma table
        for t in range(T):
            for i in range(N):
                s = 0
                for j in range(N):
                    s += xi[i][j][t]
                gamma[i][t] = s

        # update the initial probabilities
        new_initial = deepcopy(self.initial_probabilities)
        for i, state in enumerate(self.states):
            new_initial[state] = gamma[i][0]

        # update transition probabilities
        new_transition = deepcopy(self.transition_probabilities)
        for i, state1 in enumerate(self.states):
            for j, state2 in enumerate(self.states):
                numerator = 0
                denominator = 0
                for t in range(T):
                    numerator += xi[i][j][t]
                    denominator += gamma[i][t]
                new_transition[state1][state2] = numerator / denominator

        # update emission probabilities
        new_emission = deepcopy(self.emission_probabilities)
        for j, state in enumerate(self.states):
            for value in self.vocabulary:
                numerator = 0
                denominator = 0
                for t in range(T):
                    if observations[t] == value:
                        numerator += gamma[j][t]
                    denominator += gamma[j][t]
                new_emission[state][value] = numerator / denominator

        return HiddenMarkovModel(self.states, self.vocabulary, new_transition, new_emission, new_initial)
