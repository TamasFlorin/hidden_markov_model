import unittest
from hmm import HiddenMarkovModel


class HmmWeatherTest(unittest.TestCase):
    def setUp(self):
        self.states = ["rainy", "sunny"]
        self.vocabulary = ["waslk", "shop", "clean"]

        self.initial_probabilities = {
            "rainy": 0.6,
            "sunny": 0.4,
        }

        # The probability of moving from state a to state b
        # such that the sum of the all the probabilities = 1
        self.transition_probabilities = {
            "rainy": {
                "rainy": 0.7,
                "sunny": 0.3,
            },
            "sunny": {
                "rainy": 0.4,
                "sunny": 0.6,
            },
        }

        # the probability of the observation O being generated from the state q
        self.emission_probabilities = {
            "rainy": {
                "walk": 0.1,
                "shop": 0.4,
                "clean": 0.5,
            },
            "sunny": {
                "walk": 0.6,
                "shop": 0.3,
                "clean": 0.1,
            }
        }

        self.hmm = HiddenMarkovModel(
            self.states,  # all the possible hidden states
            self.vocabulary,  # all possible observation types
            self.transition_probabilities,
            self.emission_probabilities,
            self.initial_probabilities
        )

    def test_forward(self):
        observations = ["walk", "shop", "clean"]
        P, forwards = self.hmm.forward(observations)
        self.assertEqual(P, 0.033611999999999996)
        self.assertEqual(forwards[0], [0.06, 0.0552, 0.029039999999999996])
        self.assertEqual(forwards[1], [0.24, 0.0486, 0.004572])

    def test_backward(self):
        observations = ["walk", "shop", "clean"]
        P, backwards = self.hmm.backward(observations)
        self.assertEqual(P, 0.033612)
        self.assertEqual(backwards[0], [0.1298, 0.38, 1.0])
        self.assertEqual(backwards[1], [0.10760000000000002, 0.26, 1.0])

    def test_viterbi(self):
        observations = ["walk", "shop", "clean"]
        P, backpoints = self.hmm.viterbi(observations)
        self.assertEqual(P, 0.01344)
        self.assertEqual(backpoints, ['sunny', 'rainy', 'rainy'])

    def test_forward_backward(self):
        observations = ["walk", "shop", "clean"]
        new_hmm = self.hmm.forward_backward(observations)
        prediction = new_hmm.viterbi(observations[:3])
        self.assertEqual(prediction, (0.010992253093590539,
                                      ['sunny', 'rainy', 'rainy']))
