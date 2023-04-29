import numpy as np
from SingleLayerPerceptron import SingleLayerPerceptron as singleLayerPerceptron



class BackPropagation:
    """A Back Propogation perceptron class that used the Perceptron class above.
        Attributes:
            layers: A python list with the number of elements per layer.
            bias:   The bias term. The same bias is used for all neurons.
            eta:    The learning rate."""

    def __init__(self,layers,bias = 1.0):
        """Return a new MLP object with the specified parameters."""
        self.bias = bias
        self.eta = 0.5
        self.layers = np.array(layers, dtype = object) # List of integers, representing no on neurons per layer
        self.network = [] # The list of lists of neurons (will become numpy array of numpy arrays)
        self.values = []  # The list of lists of output values (will become numpy array of numpy arrays)
        self.d = []       # The list of lists of error terms (lowercase deltas)

        # Create neurons layer by layer
        for i in range(len(self.layers)):
            self.values.append([])
            self.network.append([])
            self.d.append([])

            self.values[i] = [0.0 for j in range(self.layers[i])]
            self.d[i] = [0.0 for j in range(self.layers[i])]
            if i > 0: # 0th layer is the input so keep it empty
                for j in range(self.layers[i]):
                    # Create each layer's perceptron by calling SingleLayerPerceptron
                    # Pass the Previous layer's neuron (i-1) and the bias to current layer
                    self.network[i].append(singleLayerPerceptron(inputs=self.layers[i-1], bias=self.bias))

        # Convert the list into numpy array
        self.network = np.array([np.array(x) for x in self.network], dtype=object)
        self.values = np.array([np.array(x) for x in self.values], dtype=object)
        self.d = np.array([np.array(x) for x in self.d], dtype=object)

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
        for i in range(1,len(self.network)):
            for j in range(self.layers[i]):
                self.values[i][j] = self.network[i][j].run(self.values[i-1])
        return self.values[-1] # Return the last values's first item which is the output layer.

    def bp(self, x,y):
        """Run a single (x,y) pair with the backpropagation algorithm"""

        x = np.array(x,dtype=object)
        y = np.array(y,dtype=object)

        #Backpropagation Step by step

        # STEP 1: Feed a sample to the network

        outputs = self.run(x)

        # STEP 2: Calculate the MSE
        error = (y - outputs)  # numpy vector operations
        MSE = sum(error ** 2) / self.layers[-1] 

        #STEP 3: Calculate teh output error terms
        self.d[-1] = outputs * (1-outputs) * (error) # numpy vector operations

        # STEP 4: Calculate theerror term of each unit on each layer
        for i in reversed(range(1, len(self.network)-1 )):
            for h in range(len(self.network[i])):
                fwd_error = 0.0
                for k in range(self.layers[i+1]):
                    fwd_error += self.network[i+1][k].weights[h]*self.d[i+1][k]
                self.d[i][h] = self.values[i][h] * (1-self.values[i][h]) * fwd_error

        # STEP 5 & 6: Calculate the deltas and update the weights
        for i in range(1, len(self.network)):
            for j in range(self.layers[i]):
                for k in range(self.layers[i-1]+1):
                    if k == self.layers[i-1]:
                        delta = self.eta * self.d[i][j] * self.bias
                    else:
                        delta = self.eta * self.d[i][j] * self.values[i-1][k]
                    self.network[i][j].weights[k] += delta

        return MSE


# test code
mlp = BackPropagation(layers=[3, 4, 1])
print("\nTraining Neural Network as an XOR Gate...\n")
for i in range(10000):
    mse = 0.0
    mse += mlp.bp([0, 0, 0], [0])
    mse += mlp.bp([0, 0, 1], [1])
    mse += mlp.bp([0, 1, 0], [1])
    mse += mlp.bp([0, 1, 1], [0])
    mse += mlp.bp([1, 0, 0], [1])
    mse += mlp.bp([1, 0, 1], [0])
    mse += mlp.bp([1, 1, 0], [0])
    mse += mlp.bp([1, 1, 1], [1])
    
    mse = mse / 8
    if (i % 100 == 0):
        print(mse)

mlp.printWeights()

print("MLP:")
print("0 0 0 = {0:.10f}".format(mlp.run([0, 0, 0])[0]))
print("1 0 1 = {0:.10f}".format(mlp.run([1, 0, 1])[0]))
print("1 1 1 = {0:.10f}".format(mlp.run([1, 1, 1])[0]))
print("1 0 0 = {0:.10f}".format(mlp.run([1, 0, 0])[0]))
#print("1 1 = {0:.10f}".format(mlp.run([1, 1])[0]))

mlp = BackPropagation(layers=[3, 4, 2])
print("\nTraining Neural Network as an XOR Gate...\n")
for i in range(10000):
    mse = 0.0
    mse += mlp.bp([0, 0, 0], [0,0])
    mse += mlp.bp([0, 0, 1], [1,0])
    mse += mlp.bp([0, 1, 0], [1,0])
    mse += mlp.bp([0, 1, 1], [0,1])
    mse += mlp.bp([1, 0, 0], [1,0])
    mse += mlp.bp([1, 0, 1], [0,1])
    mse += mlp.bp([1, 1, 0], [0,1])
    mse += mlp.bp([1, 1, 1], [1,1])
    
    mse = mse / 8
    if (i % 100 == 0):
        print(mse)

mlp.printWeights()

print("MLP:")

print("The number recognized by network is:", mlp.run([0, 0, 0]))
print("The number recognized by network is:", mlp.run([1, 0, 1]))
print("The number recognized by network is:", mlp.run([1, 1, 0]))
print("The number recognized by network is:", mlp.run([1, 1, 1]))


"""
mlp = BackPropagation(layers=[2, 2, 1])
print("\nTraining Neural Network as an XOR Gate...\n")
for i in range(3000):
    mse = 0.0
    mse += mlp.bp([0, 0], [0])
    mse += mlp.bp([0, 1], [1])
    mse += mlp.bp([1, 0], [1])
    mse += mlp.bp([1, 1], [0])
    mse = mse / 4
    if (i % 100 == 0):
        print(mse)

mlp.printWeights()

print("MLP:")
print("0 0 = {0:.10f}".format(mlp.run([0, 0])[0]))
print("0 1 = {0:.10f}".format(mlp.run([0, 1])[0]))
print("1 0 = {0:.10f}".format(mlp.run([1, 0])[0]))
print("1 1 = {0:.10f}".format(mlp.run([1, 1])[0]))

epochs = int(input("How many epochs?"))

mlp1 = BackPropagation(layers=[7,7,10])

print("Training 7 to 10 networks...")

for i in range(epochs):
    mse = 0.0
    mse += mlp1.bp([1,1,1,1,1,1,0],[1,0,0,0,0,0,0,0,0,0]) # 0
    mse += mlp1.bp([0,1,1,0,0,0,0],[0,1,0,0,0,0,0,0,0,0]) # 1
    mse += mlp1.bp([1,1,0,1,1,0,1],[0,0,1,0,0,0,0,0,0,0]) # 2
    mse += mlp1.bp([1,1,1,1,0,0,1],[0,0,0,1,0,0,0,0,0,0]) # 3
    mse += mlp1.bp([0,1,1,0,0,1,1],[0,0,0,0,1,0,0,0,0,0]) # 4
    mse += mlp1.bp([1,0,1,1,0,1,1],[0,0,0,0,0,1,0,0,0,0]) # 5
    mse += mlp1.bp([1,0,1,1,1,1,1],[0,0,0,0,0,0,1,0,0,0]) # 6
    mse += mlp1.bp([1,1,1,0,0,0,0],[0,0,0,0,0,0,0,1,0,0]) # 7
    mse += mlp1.bp([1,1,1,1,1,1,0],[0,0,0,0,0,0,0,0,1,0]) # 8
    mse += mlp1.bp([1,1,1,1,0,1,1],[0,0,0,0,0,0,0,0,0,1]) # 9
    mse = mse/10.0
    
print(" Training Done !!! \n")
pattern = [1.2]
while(pattern[0] >= 0.0):
    pattern = list(map(float, input("Input pattern 'a b c d e f g':").strip().split()))
    if pattern[0] < 0.0:
        break
    print()
    print("The number recognized by network is:", np.argmax(mlp1.run(pattern)))
    """