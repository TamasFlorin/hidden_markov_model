import unittest
from hmm import HiddenMarkovModel


class HmmWeatherTest(unittest.TestCase):
    def setUp(self):
        self.states = ["rainy", "sunny"]

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
            self.states, self.transition_probabilities, self.emission_probabilities, self.initial_probabilities)

    def test_forward(self):
        observations = ["walk", "shop", "clean"]
        P, forwards = self.hmm.forward(observations)
        self.assertEqual(P, 0.033611999999999996)
        # day 1
        self.assertEqual(forwards[0], ('rainy', 0.06))
        self.assertEqual(forwards[1], ('sunny', 0.24))

        # day 2
        self.assertEqual(forwards[2], ('rainy', 0.0552))
        self.assertEqual(forwards[3], ('sunny', 0.0486))

        # day 3
        self.assertEqual(forwards[4], ('rainy', 0.029039999999999996))
        self.assertEqual(forwards[5], ('sunny', 0.004572))

    def test_backward(self):
        observations = ["walk", "shop", "clean"]
        P, backwards = self.hmm.backward(observations)
        self.assertEqual(P, 0.033612)
        # day 1
        self.assertEqual(backwards[0], ('rainy', 1.0))
        self.assertEqual(backwards[1], ('sunny', 1.0))

        # day 2
        self.assertEqual(backwards[2], ('rainy', 0.38))
        self.assertEqual(backwards[3], ('sunny', 0.26))

        # day 3
        self.assertEqual(backwards[4], ('rainy', 0.1298))
        self.assertEqual(backwards[5], ('sunny', 0.10760000000000002))

    def test_viterbi(self):
        observations = ["walk", "shop", "clean"]
        P, backpoints = self.hmm.viterbi(observations)
        self.assertEqual(P, 0.01344)
        self.assertEqual(backpoints, ('rainy', [0, 1, 0]))
