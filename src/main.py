from hmm import HiddenMarkovModel
from argparse import ArgumentParser


def load_json_data(file_path):
    from json import load
    with open(file_path) as json_file:
        data = load(json_file)
        return data


if __name__ == "__main__":
    parser = ArgumentParser(description='Hidden Markov Model')
    parser.add_argument('json_file', type=str,
                        help='the json file that contains the model data')
    parser.add_argument('-op', '--operation_name', required=True, type=str,
                        help='the operation that should be executed on the model')
    parser.add_argument('-obs', '--observations', nargs='+', required=True,
                        type=str, help='the list of observations that should be used for the operation')

    args = parser.parse_args()
    data = load_json_data(args.json_file)
    operation_name = args.operation_name
    observations = args.observations

    states = data['states']
    vocabulary = data['vocabulary']
    initial_probabilities = data['initial_probabilities']
    transition_probabilities = data['transition_probabilities']
    emission_probabilities = data['emission_probabilities']
    hmm = HiddenMarkovModel(states, vocabulary, transition_probabilities,
                            emission_probabilities, initial_probabilities)

    operations = {
        'viterbi': hmm.viterbi,
        'forward': hmm.forward,
        'backward': hmm.backward,
        'forward_backward': hmm.forward_backward,
    }

    operation = operations[operation_name]
    print(operation(observations))
