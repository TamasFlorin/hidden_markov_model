from os import listdir
from os.path import isfile, join
from hmm import HiddenMarkovModel
from PIL import Image
from util import image_to_array
from util import get_blocks
from util import observations_from_blocks
from util import VOCABULARY, START_ID, END_ID


def is_image_file(filename):
    return ".jpeg" in filename or ".png" in filename


def process_image(filename, slices=5):
    print("Parsing image file: " + filename)
    image = Image.open(filename)
    image_pixels = image_to_array(image, new_size=(28, 28))
    block_height, block_width = 7, 7
    blocks = get_blocks(image_pixels, block_size=(block_height, block_width))
    observation = observations_from_blocks(blocks)
    return observation


def load_data(data_path):
    data_files = [join(data_path, f) for f in listdir(data_path) if isfile(
        join(data_path, f)) and is_image_file(f)]

    data = {}
    for image_file in data_files:
        current_char = image_file.split("/")[-1]
        current_char = current_char.split(".")[0].split("_")[0]
        features = process_image(image_file)
        if current_char not in data:
            data[current_char] = [features]
        else:
            data[current_char].append(features)

    return data


def train(data_path):
    states = [i for i in range(0, 27)]

    # special start and end state
    start_state = -1337
    end_state = 1337

    vocabulary = VOCABULARY
    initial_probabilities = {}
    transition_probabilities = {}
    emission_probabilities = {}

    # empty dicitonaries
    transition_probabilities[start_state] = {}
    transition_probabilities[end_state] = {}
    emission_probabilities[start_state] = {}
    emission_probabilities[end_state] = {}

    # compute initial probabilities and transition probabilities
    for state in states:
        initial_probabilities[state] = 0.0
        transition_probabilities[state] = {}
        emission_probabilities[state] = {}

        emission_probabilities[start_state][state] = 0.0
        emission_probabilities[end_state][state] = 0.0
        transition_probabilities[start_state][state] = 0.0
        transition_probabilities[end_state][state] = 0.0
        transition_probabilities[state][end_state] = 0.0
        transition_probabilities[state][start_state] = 0.0
        # special states

        for other in states:
            if other == state:
                transition_probabilities[state][other] = 0.0
            else:
                transition_probabilities[state][other] = 1.0 / len(states)

    training_data = load_data(data_path)
    for state in states:
        for item in vocabulary:
            if item != START_ID and item != END_ID:
                emission_probabilities[state][item] = 1.0 / len(vocabulary)
            else:
                emission_probabilities[state][item] = 0.0

    initial_probabilities[start_state] = 1.0
    initial_probabilities[end_state] = 0.0

    # we can directly go to the first state
    transition_probabilities[start_state][states[0]] = 1.0
    transition_probabilities[start_state][end_state] = 0.0

    transition_probabilities[end_state][start_state] = 1.0
    transition_probabilities[end_state][end_state] = 0.0

    # emission init
    emission_probabilities[start_state][vocabulary[0]] = 1.0
    emission_probabilities[start_state][vocabulary[-1]] = 0.0
    emission_probabilities[end_state][vocabulary[-1]] = 1.0
    emission_probabilities[end_state][vocabulary[0]] = 0.0

    states = [start_state] + states + [end_state]
    for item in vocabulary:
        if item != START_ID and item != END_ID:
            emission_probabilities[start_state][item] = 0.0
            emission_probabilities[end_state][item] = 0.0

    model = HiddenMarkovModel(states=states, vocabulary=vocabulary, transition_probabilities=transition_probabilities,
                              emission_probabilities=emission_probabilities, initial_probabilities=initial_probabilities)

    for i in range(1):
        for name in training_data:
            for item in training_data[name]:
                item = [START_ID] + item + [END_ID]
                print(item)
                model = model.forward_backward(item)
    return model


def test_model(data_path, model):
    observations = load_data(data_path)

    for observation in observations['A']:
        print(observation)
        print(model.viterbi(observation))


if __name__ == "__main__":
    model = train("../dataset/train/")
    test_model("../dataset/test/", model)
