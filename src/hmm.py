class HiddenMarkovModel(object):
    """ A probabilistic sequence classifier - given a sequence of units,
        computes a probability distribution over possible labels and chooses the
        best label."""

    def __init__(self, states, observations, transition_probabilities, emission_probabilities):
        self.t_p = transition_probabilities
        self.e_p = emission_probabilities
        self.st = states
        self.obs = observations

    def forward(self):
        f = {}
        forwards = []

        # initialize forward map
        for i in range(0, len(self.st)):
            f[self.st[i]] = [0.0 for _ in range(len(self.st))]

        # initialization step
        for i in range(1, len(self.st) - 1):
            current_state = self.st[i]
            transition = self.t_p[self.st[0]][current_state]
            emission = self.e_p[current_state][self.obs[0]]
            result = transition * emission
            f[current_state][0] = result
            forwards.append((current_state, result))

        # recursion step
        for t in range(1, len(self.obs)):
            for s in range(1, len(self.st) - 1):
                current_sum = 0
                for i in range(1, len(self.st) - 1):
                    f_value = f[self.st[i]][t - 1]
                    transition = self.t_p[self.st[i]][self.st[s]]
                    emission = self.e_p[self.st[s]][self.obs[t]]
                    current_sum += (f_value * transition * emission)
                f[self.st[s]][t] = current_sum
                forwards.append((self.st[s], current_sum))

        # termination step
        P = 0.0
        for s in range(1, len(self.st)):
            P = P + f[self.st[s]][len(self.obs) - 1]

        return P, forwards
