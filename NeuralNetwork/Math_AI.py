from scipy.special import expit

class Math_AI:
    def __init__(self):
        pass

    def sigmoid(x):
        return expit(x)

    def sigmoid_derivative(x):
        return x * (1.0 - x)