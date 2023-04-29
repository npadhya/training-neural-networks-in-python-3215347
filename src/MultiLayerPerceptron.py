import numpy as np
from SingleLayerPerceptron import SingleLayerPerceptron as singleLayerPerceptron


class MultiLayerPerceptron:
    """A Multilayer perceptron class that used the Perceptron class above.
        Attributes:
            layers: A python list with the number of elements per layer.
            bias:   The bias term. The same bias is used for all neurons.
            eta:    The learning rate."""

    def __init__(self,layers,bias = 1.0):
        """Return a new MLP object with the specified parameters."""
        self.bias = bias

        self.layers = np.array(layers, dtype = object) # List of integers, representing no on neurons per layer
        self.network = [] # The list of lists of neurons (will become numpy array of numpy arrays)
        self.values = []  # The list of lists of output values (will become numpy array of numpy arrays)

        # Create neurons layer by layer
        for i in range(len(self.layers)):
            self.values.append([])
            self.network.append([])

            self.values[i] = [0.0 for j in range(self.layers[i])]
            if i > 0: # 0th layer is the input so keep it empty
                for j in range(self.layers[i]):
                    # Create each layer's perceptron by calling SingleLayerPerceptron
                    # Pass the Previous layer's neuron (i-1) and the bias to current layer
                    self.network[i].append(singleLayerPerceptron(inputs=self.layers[i-1], bias=self.bias))

        # Convert the list into numpy array
        self.network = np.array([np.array(x) for x in self.network], dtype=object)
        self.values = np.array([np.array(x) for x in self.values], dtype=object)

    def set_weights(self,w_init):
        """Set the weights.
            w_init: is a list of lists with weights for all but the input layer"""
        for i in range(len(w_init)): # Itirates each layers
            for j in range(len(w_init[i])): # Iterates each neurons in the ith layer
                #Start setting weights for each layer excpt the input layer, Hence 'self.network[i+1]'
                self.network[i+1][j].set_weights(w_init[i][j])

    def printWeights(self):
        print()
        for i in range(1,len(self.network)):
            for j in range(self.layers[i]):
                print("Layer: ", i+1," Neuron: ",j,self.network[i][j].weights)
        print()

    def run(self,x):
        """ Feed a sample x into the Multilayer Perception"""
        x = np.array(x,dtype=object)
        self.values[0] = x
        print(self.values[0])
        for i in range(1,len(self.network)):
            for j in range(self.layers[i]):
                self.values[i][j] = self.network[i][j].run(self.values[i-1])
        return self.values[-1] # Return the last values's first item which is the output layer.
    
mlp = MultiLayerPerceptron(layers=[2,2,1]) # MLP
mlp.set_weights(([[-10,-10,15],[15,15,-10]],[[10,10,-15]]))  #XOR Gate
mlp.printWeights()
print("MLP:")
print("0 0 = {0:.10f}".format(mlp.run([0, 0])[0]))
print("0 1 = {0:.10f}".format(mlp.run([0, 1])[0]))
print("1 0 = {0:.10f}".format(mlp.run([1, 0])[0]))
print("1 1 = {0:.10f}".format(mlp.run([1, 1])[0]))

    
mlp = MultiLayerPerceptron(layers=[2,2,1]) # MLP
mlp.set_weights(([[-10,-10,15],[15,15,-10]],[[-10,-10,15]]))  #X-NOR Gate
mlp.printWeights()
print("MLP:")
print("0 0 = {0:.10f}".format(mlp.run([0, 0])[0]))
print("0 1 = {0:.10f}".format(mlp.run([0, 1])[0]))
print("1 0 = {0:.10f}".format(mlp.run([1, 0])[0]))
print("1 1 = {0:.10f}".format(mlp.run([1, 1])[0]))
        
