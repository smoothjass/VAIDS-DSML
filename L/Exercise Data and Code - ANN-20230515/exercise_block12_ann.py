########################################################################################################################
# REMARKS
########################################################################################################################
'''
## Coding
- please note, this no SE course and much of the code in ML is more akin to executing workflows
- please try to use the scripts as a documentation of your analysis, including comments, results and interpretations

## GRADING
- Please refer to the moodle course for grading information

## UPLOAD
- upload your solution on Moodle as: "yourLASTNAME_yourFIRSTNAME_yourMOODLE-UID___exercise_blockX.py"
- please no non-ascii characters on last/first name :)
- NO zipfile, NO data!, ONLY the .py file!

## PRE-CODED Parts
- all exercise might contain parts which where not yet discussed in the course
- these sections are pre-coded then and you are encouraged to research their meaning ahead of the course
- so, if you find something in the exercise description you haven´t heard about, look at the ode section and check if this is pre-coded

## ERRORS
- when you open exercise files you'll see error (e.g. unassigned variables)
- this is due to the fact that some parts are missing, and you should fill them in
'''

########################################################################################################################
# IMPORTS
########################################################################################################################


import numpy as np
import pandas as pd
import math

########################################################################################################################
# PART 1 // IMPLEMENT PCA (2 components) using ONLY scikit learn
########################################################################################################################


'''
Implement a simple feed forward neural network based ont he code below (copy & paste is not enough!)
-There should be: 2 inputs, a first hidden layer with 3 nodes, a second hidden layer with 4 nodes and one output node.
-Use random weights and biases.
-Make up a fixed target (e.g. 0.7) and inputs (e.g. 9 and 3) and calculate the loss 
    between the output of your network and the target 
    (in this case a simple difference is enough, e.g. 0.8 – 0.7 = 0.1)
-Run it multiple times and note down the weights and biases for the lowest loss. 
    (Next time we will learn how to use better-than-random weights and biases!)
'''

'''
# HELPFER:

def activation_sigmoid(t):
    s = 1 / (1 + math.exp(-t))
    return s

def neuron_output(weights, inputs):
    d = np.dot(weights, inputs)
    s = activation_sigmoid(d)
    return s

def forward_propagation (neural_network, input_vector):
    # our ANN is a list of (non-input) layers; each layer is a list of neuros in that layer:
    # list (layers) of lists (neuros) of lists (weights)
    output_all = []
    for layers in neural_network:
        input_with_bias = input_vector + [1]  # add the bias
        output_one_layer = []
        for neuron in layers:
            output_one_neuron = neuron_output(neuron, input_with_bias)
            output_one_layer.append(output_one_neuron)
        output_all.append(output_one_layer)

        # then the input to the next layer is the output of this one
        input_vector = output_one_layer

    return output_all
'''


# ----------------------------------------------------------------------------------------------------------------------
# adapt the above code/input s.t. a forward pass of a NN with a 2-3-4-1 architecture can be calculated
# - student work - #


def activation_sigmoid(val):
    '''
    :param val: float (here a linear combination of neuron inputs, weights and biases) that should be activated
    :return: float
    '''

    # https://www.digitalocean.com/community/tutorials/sigmoid-activation-function-python
    return 1 / (1 + np.exp(-val))


def neuron_output(incoming_neuron_weights_and_bias_weight, incoming_values_and_bias):
    pass
    '''
    :param incoming_neuron_weights_and_bias: list of floats
    :param incoming_values_and_bias: list of floats
    '''

    # it felt easier to not use this function, I hope that's ok
    # return s


def forward_propagation(neural_network, input_vector, verbosity=2):
    '''
    :param neural_network: list of neurons, neurons are lists of weights and biases, weights and biases are floats
    :param input_vector: list of floats
    :param verbosity: controls print
    :return:
    '''
    count = 1
    for layer in neural_network:
        output_vector = []
        if verbosity >= 1:
            print("\nProcessing layer: ", count, layer)
            print("incoming values: ", input_vector)
        for node in layer:
            # get bias from index 0 and remove it from there so that node and input_vector have the same len
            bias = out = node.pop(0)
            if verbosity >= 2:
                print("Processing neuron with weights and bias:", node, bias)
            # while indexes are left, multiply and add
            for i in range(len(node)):
                out = out + node[i] * input_vector[i]
            out = activation_sigmoid(out)
            output_vector.append(out)
            node.insert(0, bias)
        input_vector = output_vector
    # at the end, input_vector should hold the final output, which is an unfortunate naming but then again,
    # it makes sense, as each output had been used as new input soooooo that's ok

    return input_vector[0]


def randomArchitecture(architecture):
    nn = []
    for i in range(1, len(architecture)):  # the network
        layer = []
        for j in range(architecture[i]):  # the layer
            node = []
            for k in range(architecture[i - 1] + 1):  # the node
                # (we need as many weights as the number of nodes in the layer before this one plus 1 for the bias
                node.append(np.random.randint(1, 10) / 10)
            layer.append(node)
        nn.append(layer)

    return nn


'''
# list of neurons, neurons are lists of weights and biases, weights and biases are floats
# list (layers) of lists (neuros) of lists (weights)
# list of layers:
# [
# (1),      layer 1
# (2),      layer 2
# [o]       layer out
# ]

# list of lists of neurons:
# [
# [b1, b2, b3],         layer 1
# [b1, b2, b3, b4],     layer 2
# [b1]                  layer out
# ]

# list of lists of neurons of lists of biases and weights:
# [
# [[b1, w11, w12], [b2, w21, w22], [b3, w31, w32]],                                         layer 1
# [[b1, w11, w12, w13], [b2, w21, w22, w23], [b3, w31, w32, w33], [b4, w41, w42, w43]],     layer 2
# [b1, w11, w12, w13, w14]                                                                  layer out
# ]
'''

# ----------------------------------------------------------------------------------------------------------------------
# run the forward pass with random weights/biases, calculate the error
# - student work - #
output = 0.7
input = [9, 3]
architecture = [2, 3, 4, 1]

results = []
for i in np.arange(10):
    nn = randomArchitecture(architecture)
    out = forward_propagation(nn, input, verbosity=2)
    error = abs(output - out)
    print("error: ", error)
    result = [nn, out, error]
    results.append(result)

results = pd.DataFrame(results, columns=['network', 'output', 'error'])
