from typing import List
from typing import Any
from typing import Tuple


class HiddenMarkovModel(object):
    """
    A probabilistic sequence classifier - given a sequence of units,
    computes a probability distribution over possible labels and chooses the
    best label.
    """

    def __init__(self, states: List[Any], transition_probabilities: Any, emission_probabilities: Any, initial_probabilities: Any):
        self.transition_probabilities = transition_probabilities
        self.emission_probabilities = emission_probabilities
        self.initial_probabilities = initial_probabilities
        self.states = states

    def forward(self, observations: List[Any]) -> Tuple[float, List[Tuple[Any, float]]]:
        """
        Returns the probability of seeing the given `observations` sequence,
        using the Forward algorithm.

        :param observations: The list of observed values.
        :return: The probability of seeing the given observations along with the forward states
                 for each iteration.
        """
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
            for s in range(0, len(self.states)):
                current_sum = 0
                for i in range(0, len(self.states)):
                    f_value = f[i][t - 1]
                    transition = self.transition_probabilities[self.states[i]
                                                               ][self.states[s]]
                    emission = self.emission_probabilities[self.states[s]
                                                           ][observation]
                    current_sum += (f_value * transition * emission)
                f[s][t] = current_sum
                forwards.append((self.states[s], current_sum))

        # termination step
        T = len(observations) - 1
        P = sum(f[s][T] for s in range(len(self.states)))

        return P, forwards

    def backward(self, observations: List[Any]) -> Tuple[float, List[Tuple[Any, float]]]:
        """
        Returns the probability of seeing the given `observations` sequence,
        using the Backward algorithm.

        :param observations: The list of observed values.
        :return: The probability of seeing the given observations along with the backward states
                 for each iteration.
        """

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

        return P, backwards

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
