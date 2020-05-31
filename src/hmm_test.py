import unittest
from hmm import HiddenMarkovModel


class HmmWeatherTest(unittest.TestCase):
    def setUp(self):
        self.states = ["rainy", "sunny"]
        self.vocabulary = ["walk", "shop", "clean"]

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
        self.assertEqual(P, 0.033612)
        self.assertEqual(forwards[0], [0.06, 0.055200000000000006, 0.02904])
        self.assertEqual(forwards[1], [0.24, 0.04859999999999999, 0.004572])

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
        self.assertEqual(prediction, (0.010994296643152459,
                                      ['sunny', 'rainy', 'rainy']))


class HmmCoinsTest(unittest.TestCase):
    def setUp(self):
        self.states = ["Coin 1", "Coin 2", "Coin 3"]
        self.vocabulary = ["Heads", "Tails"]

        self.initial_probabilities = {
            "Coin 1": 0.4,
            "Coin 2": 0.3,
            "Coin 3": 0.3
        }

        # The probability of moving from state a to state b
        # such that the sum of the all the probabilities = 1
        self.transition_probabilities = {
            "Coin 1": {"Coin 1": 0.6, "Coin 2": 0.3, "Coin 3": 0.1},
            "Coin 2": {"Coin 1": 0.2, "Coin 2": 0.5, "Coin 3": 0.3},
            "Coin 3": {"Coin 1": 0.3, "Coin 2": 0.2, "Coin 3": 0.5}
        }

        # the probability of the observation O being generated from the state q
        self.emission_probabilities = {
            "Coin 1": {"Heads": 0.7, "Tails": 0.3},
            "Coin 2": {"Heads": 0.3, "Tails": 0.7},
            "Coin 3": {"Heads": 0.5, "Tails": 0.5}
        }

        self.hmm = HiddenMarkovModel(
            self.states,  # all the possible hidden states
            self.vocabulary,  # all possible observation types
            self.transition_probabilities,
            self.emission_probabilities,
            self.initial_probabilities
        )

    def test_forward(self):
        observations = ["Heads", "Heads", "Heads"]
        P, forwards = self.hmm.forward(observations)
        self.assertEqual(P, 0.14533999999999997)
        self.assertEqual(
            forwards[0], [0.27999999999999997, 0.16169999999999998, 0.08824199999999997])
        self.assertEqual(forwards[1], [0.09, 0.0477, 0.025607999999999995])

    def test_backward(self):
        observations = ["Heads", "Heads", "Heads"]
        P, backwards = self.hmm.backward(observations)
        self.assertEqual(P, 0.14534)
        self.assertEqual(backwards[0],  [0.30080000000000007, 0.56, 1.0])
        self.assertEqual(backwards[1], [0.2224, 0.43999999999999995, 1.0])

    def test_viterbi(self):
        observations = ["Heads", "Heads", "Heads"]
        P, backpoints = self.hmm.viterbi(observations)
        self.assertEqual(P, 0.049391999999999985)
        self.assertEqual(backpoints, ['Coin 1', 'Coin 1', 'Coin 1'])

    def test_forward_backward(self):
        observations = ["Heads", "Heads", "Heads"]
        old_prediction = self.hmm.viterbi(observations)
        new_hmm = self.hmm.forward_backward(observations)
        prediction = new_hmm.viterbi(observations[:3])

        # we should see an increase in the probability output
        assert old_prediction[0] < prediction[0]
        self.assertEqual(prediction, (0.10346810696178374,
                                      ['Coin 1', 'Coin 1', 'Coin 1']))
