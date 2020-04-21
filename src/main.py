from hmm import HiddenMarkovModel

if __name__ == "__main__":
    states = ["start", "rainy", "sunny", "end"]
    observations = ["walk", "shop", "clean"]

    # The probability of moving from state a to state b
    # such that the sum of the all the probabilities = 1
    transition_probabilities = {
        "start": {
            "rainy": 0.6,
            "sunny": 0.4,
            "end": 0.0,
            "start": 0.0
        },
        "rainy": {
            "rainy": 0.7,
            "sunny": 0.3,
            "end": 0.0,
            "start": 0.0
        },
        "sunny": {
            "rainy": 0.4,
            "sunny": 0.6,
            "end": 0.0,
            "start": 0.0
        },
        "end": {
            "end": 0.0,
            "rainy": 0.0,
            "sunny": 0.0,
            "start": 0.0,
        }
    }

    # the probability of the observation O being generated from the state q
    emission_probabilities = {
        "start": {
            "walk": 0.0,
            "shop": 0.0,
            "clean": 0.0,
        },
        "end": {
            "walk": 0.0,
            "shop": 0.0,
            "clean": 0.0,
        },
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

    model = HiddenMarkovModel(
        states, observations, transition_probabilities, emission_probabilities
    )
    P, forwards = model.forward()
    print("Overall Probability: {0}".format(P))
    print("Forwards: {0}".format(forwards))
