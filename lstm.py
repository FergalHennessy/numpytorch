#we are implementing mini-batch gradient descent. For each mini-batch:
#1. Perform a forward pass for each data in mini-batch and calculate avg loss.
#2. Calculate the gradients of the loss function based on average mini-batch loss.
#3. update parameters

#this network has no bias term, so a fictional dimension needs to be added to input data
import numpy as np
from tqdm import tqdm

#should i calculate the mean gradient (per point) or the total gradient?
#A layer has 1.state 2.activated state 3. weights. 4. activation function. 5.learning rate 6.nw
class Layer:
    #let input size of the layer be d, let output size of the layer be n. Assume that our batch has m sample points.
    #self.weights is a dxm random matrix converting input matrix of shape mxd to output matrix of shape mxn.
    #xavier initialization of weights matrix, zero initialization of bias
    def __init__(self, input_size, output_size, activation_function, learning_rate=0.05, nw=None, output_layer=False):
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.nw = nw
        self.bias = np.zeros((output_size, ), dtype=float)

        if isinstance(self.activation_function, Relu):      # he initialization for relu
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        else:                                               # Xavier initialization for others
            self.weights = np.random.randn(input_size, output_size) * \
                  np.sqrt(1. / (input_size + output_size))

    #forward propagation
    #step 1: H = XW
    #step 2: Z = activation(H)
    #setp 3: set dW and dB grad loss wrt weights and biases to 0
    def forward(self, input_data):
        self.input = input_data
        self.h = np.dot(input_data, self.weights) + self.bias
        self.output = self.activation_function.forward(self.h)
        self.dW = np.zeros(self.weights.shape)
        self.dB = np.zeros(self.bias.shape)
        return self.output
    
    #backwawrd propagation
    #step 1: dz/dh  =  activation.backward(h)
    #step 2: dL/dh  =  dL/dz  @ dz/dh
    #step 3: dL/dx  =  dL/dh @ dh/dx = dL/d
    #step 4: dL/dWi =  dL/dhi @ dhi/dWi
    #step 5: accumulate gradients that were zeroed in forward
    #step 6: return dL/dx
    def backward(self, output_derivative):

        if self.nw and isinstance(self.activation_function, Softmax) \
            and isinstance(self.nw.loss, CrossEntropy):
            #print("shortcutting softmax gradient")
            dLdh = output_derivative
        else:
            #nxmxm such that ith element is dZi/dHi(mxm)
            dzdh = self.activation_function.backward(self.h)
            
            #dLdh = np.dot(output_derivative, dzdh)
            dLdh = np.empty_like(output_derivative)
            for i in range(output_derivative.shape[0]):
                dLdh[i] = np.dot(output_derivative[i], dzdh[i].T)

        dLdX = np.dot(dLdh, self.weights.T)
        
        dLdW = np.dot(self.input.T, dLdh)
        dLdB = np.sum(dLdh, axis=0)

        self.dW += dLdW
        self.dB += dLdB

        return dLdX

#all my activation derivatives have been symmetric so some confusion abt dzdh not resolved
class Activation:
    def init(self, x):
        pass
    def forward(self, x):
        pass
    def backward(self, x):
        pass

# batch forward: given H(nxm) return Z(nxm)
# batch backward: given H(nxm), return matrix dZdH(nxmxm) such that
# dZdH[i] = dZi/dHi where the ith element is an mxm matrix. For relu, only diagonal maybe nonzero
class Relu(Activation):
    def forward(self, H):
        return np.maximum(0, H)
    def backward(self, H):
        dZdH = np.zeros((H.shape[0], H.shape[1], H.shape[1]))
        #only the diagonal elements of each submatrix may be nonzero. einsum better for this
        for row in range(H.shape[0]):
            for column in range(H.shape[1]):
                dZdH[row, column, column] = 1.0 if H[row, column] > 0 else 0.0
        return dZdH

# batch forward: given H(nxm) return Z(nxm)
# batch backward: given H(nxm) return matrix dZdH(nxmxm) such that
# dZdH[i] = dZi/dHi where the ith element is an mxm matrix. For softmax, each i diagonal.
class Softmax(Activation):
    def forward(self, H):
        shift_H = H - np.max(H, axis=1, keepdims=True)
        exps = np.exp(shift_H)
        return exps / np.sum(exps, axis=1, keepdims=True)       #keepdims makes matrix stay 2d here
    def backward(self, H):

        Z = self.forward(H)
        
        diag_Z = np.zeros((Z.shape[0], Z.shape[1], Z.shape[1]))
        for i in range(Z.shape[0]):
            np.fill_diagonal(diag_Z[i], Z[i])

        outer_Z = np.einsum('bi, bj->bij', Z, Z)
    
        dZdH = diag_Z - outer_Z

        return dZdH
    
class Loss:
    def init(self):
        self.nw = None
    def calculate(self, z, y):
        pass
    def backward(self, z, y):
        pass

# batched. Expects output Z(nxm) as a  design matrix and labels Y(nxm) as a design matrix.
# forward: returns a scalar 0.5 sum(||z-y||^2) / n the mean squared error
# backward: returns nxm matrix (Z - Y) / n representing the derivative of the loss for each Z
class LeastSquares(Loss):
    def calculate(self, Z, Y):
        return 0.5 * np.mean((Z - Y) ** 2)
    def backward(self, Z, Y):
        return (Z - Y) / (Z.shape[0])

# cross entropy simplifies calculation of the output layer.
# batched. Expects nw output Z(nxm) and one-hot labels Y(nxm) where each row sums to 1
# forward: elementwise multiply log(Z) * one-hot encoded Y matrix and take the negative mean
# backward: Z-Y if output softmax
class CrossEntropy(Loss):
    def calculate(self, Z, Y, epsilon = 1e-10):
        Z = np.clip(Z, epsilon, 1-epsilon)
        weighted_log_probs = np.log(Z) * Y
        sample_losses = -np.sum(weighted_log_probs, axis=1)
        loss = np.mean(sample_losses)
        return loss
    def backward(self, Z, Y, epsilon=1e-10):
        if self.nw and isinstance( self.nw.layers[-1].activation_function, Softmax ):
            return (Z - Y)
        print("self.nw status is:", 1 if self.nw else 0 , \
              "self.nw.layers[-1] is softmax status: ", \
              1 if isinstance(self.nw.layers[-1].activation_function, Softmax) else 0, \
              "the last layer is not softmax, backprop failed")
        
# the idea for cell:
# a python array represents cell inputs
# layers are initialized with an input pointer and an output pointer
# The output of a layer in the network connects with other layers
# according to a series of elementwise operations
# inbetween each layer, intermediate calculations
# computation is done according to [ops, layer, ops, layer]
# an intermediate generally 
# data is a np array (Txd) of time data where th element is time-series input at time t
class Cell:
    def __init__(self, layers, intermediates, inputs, outputs, data, cachesize):

        # each element of layers is a tuple (arg, layer, output)
        self.layers = layers
        # each element of intermediates is a list of tuples 
        # (arg, arg, func, output). 
        # Execute self.cache[output] = func(self.cache[arg], self.cache[arg])
        # for backward, self.cache[arg1, arg2] = self.backward(self.cache[output])
        self.intermediates = intermediates
        # a python list of the locations of the cell input. 
        # these must be set before running forward()
        self.inputs = inputs
        # a python list of the locations of the cell output.
        # in backward pass, we set these locations to specified error,
        # then apply the backward methods in the reverse order of forward.
        self.outputs = outputs
        # a list with cachesize + datasize elements for storing computation results.
        # data is a Txd arra
        self.cache = ([0] * cachesize) + [data[t] for t in range(data.shape[0])]

    # forward propagation of a cell. takes inputs to the cell and time
    # saves the inputs to their tensors and returns the output of cell
    # savedinputs is python array of np tensors input over time
    def forward(self, inputs, time, savedinputs):
        assert len(self.intermediates) == len(self.layers) + 1
        assert len(inputs) == len(self.inputs)
        for i in range(len(self.inputs)):
            self.cache[self.inputs[i]] = inputs[i]
        
        n = len(self.layers)
        for i in range(n+1):
            for j in range(len(self.intermediates[i])):
                self.cache[self.intermediates[i][j][3]] = \
                    self.intermediates[i][j][2].forward(
                    self.cache[self.intermediates[i][j][0]], 
                    self.cache[self.intermediates[i][j][1]])
            if i < n:    
                self.cache[self.layers[i][2]] = \
                    self.layers[i][1].forward(self.cache[self.layers[i][0]])

        return [self.cache[i] for i in self.outputs]

    def backward(self, error):
        assert len(error) == len(self.outputs) 
        self.cache = [0] * len(self.cache)
        for i in range(len(error)):
            self.cache[self.outputs[i]] = error[i]

        n = len(self.layers)
        for i in list(reversed(range(n+1))):
            for j in range(len(self.intermediates[i])):
                grad1, grad2 = self.intermediates[i][j][2].backward(
                    self.cache[self.intermediates[i][j][0]], 
                    self.cache[self.intermediates[i][j][1]], 
                    self.cache[self.intermediates[i][j][3]])
                self.cache[self.intermediates[i][j][0]] += grad1
                self.cache[self.intermediates[i][j][1]] += grad2
            if i > 0:    
                self.cache[self.layers[i][0]] += \
                    self.layers[i][1].backward(
                        self.cache[self.intermediates[i][j][3]])
        

# intermediate functions used in a neural net. multiple input forward, multiple output backward
# backward method takes original input of the vector x1 and x2 and dL/dz err
class identity:
    def forward(self, x1, x2):
        return x1
    def backward(self, x1, x2, err): 
        return err, None
class hadamard: #receives strictly 1d vectors
    def forward(self, x1, x2):
        assert x1.shape == x2.shape
        return x1 * x2
    def backward(self, x1, x2, err):
        assert x1.shape == x2.shape
        return x2 * err, x1 * err



class FCNetwork:
    def __init__(self, lossfn=LeastSquares()):
        self.layers = []
        self.loss = lossfn
        self.loss.nw = self
        self.output_layer = None
    def forward(self, x, label):
        for layer in self.layers:
            x = layer.forward(x)
        self.prediction = x
        return self.loss.calculate(x, label)
    
    def backward(self, label):
        saved_grad = self.loss.backward(self.prediction, label)
        for layer in reversed(self.layers):
            saved_grad = layer.backward(saved_grad)
        for layer in self.layers:
            layer.weights = layer.weights - layer.dW * layer.learning_rate
            layer.bias = layer.bias - layer.dB * layer.learning_rate
        return
    
def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)
def one_hot_encode(labels, num_classes):
    num_samples = labels.shape[0]
    one_hot_encoded = np.zeros((num_samples, num_classes), dtype=float)
    one_hot_encoded[np.arange(num_samples), labels] = 1.0
    return one_hot_encoded
# train mini batches of size batch_size and train on the whole dataset a total of num_epochs times
def train_mini_batches(nw, X, y, batch_size, num_epochs):
    num_samples = X.shape[0]
    num_batches = num_samples // batch_size
    for epoch in range(num_epochs):
        #begin epoch. shuffle the data at the beginning of each epoch
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        start_index = None
        end_index = None

        print("current total loss is: ", nw.forward(X, y))
        print("current prediction is: ", nw.prediction)
        print("training iteration: " + str(epoch * batch_size) + "...")

        with tqdm(total=num_batches, desc=f'Epoch {epoch + 1}/{num_epochs}') as pbar:
            for batch_index in range(num_batches):
                start_index = batch_index * batch_size
                end_index = start_index + batch_size
                nw.forward(X_shuffled[start_index : end_index], y_shuffled[start_index : end_index])
                nw.backward(y_shuffled[start_index : end_index])
                
                # print("finished a batch! we are", \
                #     str((epoch * num_batches + batch_index) * 100 / (num_epochs * num_batches)),
                #     r"% done training.")

                pbar.update(1)

            if num_samples % batch_size != 0:
                nw.forward(X_shuffled[end_index : ], y_shuffled[end_index : ])
                nw.backward(y_shuffled[end_index : ])

#x_train shape: (60000, 28, 28) y_train shape: (60000,) x_test shape: (10000, 28, 28) y_test shape: (10000,) 
(x_train, y_train), (x_test, y_test) = load_data('datasets/mnist.npz')
xflat_train = x_train.reshape(x_train.shape[0], 28*28)
xflat_test  =  x_test.reshape(x_test.shape[0], 28*28)
print("y_train shape is", y_train.shape, "x_train shape is", x_train.shape)
print("y_test shape is", y_test.shape, "x_test shape is", x_test.shape)

A = FCNetwork(lossfn=CrossEntropy())
A.layers.append(Layer(784, 784, Relu(), learning_rate=0.0005, nw=A))
A.layers.append(Layer(784, 250, Relu(), learning_rate = 0.0005, nw=A))
A.layers.append(Layer(250, 10, Softmax(), learning_rate=0.0005, nw=A))
print("A is a network with 4 layers only, with weights: ", A.layers[0].weights)

X = xflat_train / 255
y = one_hot_encode(y_train, 10)

print("the one-hot-encoding of y is", y)


train_mini_batches(A, X, y, 32, 1)


A.forward(X, y)
y_train_pred = np.argmax(A.prediction, axis=1)
correct_predictions = np.sum(y_train_pred == y_train)
print("the training accuracy of the network is:", correct_predictions / y_train.shape[0])


A.forward(xflat_test / 255, one_hot_encode(y_test, 10))
print("final predictions: ", A.prediction)
y_test_pred = np.argmax(A.prediction, axis=1)
correct_predictions = np.sum(y_test_pred == y_test)
print("the accuracy of the network is", correct_predictions / y_test.shape[0])


