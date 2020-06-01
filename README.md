# Hidden Markov Model Project

This repository contains a from-scratch implementation of a Hidden Markov model.
The model uses the Forward-Backward algorithm and the Expectation-Maximization algorithm for the optimization of the probabilities.

## Running the model

`python3 main.py --json_file ../data/coins.json --operation_name viterbi --observations 'Heads' Heads' 'Tails'`\
`python3 main.py -j ../data/ice_cream.json -op viterbi -obs '2' '3' '3' '2' '3' '2' '3'`

## Testing the model

Simply execute `./test.sh`
