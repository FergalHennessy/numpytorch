import numpy as np
from tqdm import tqdm

#A layer has 1.state 2.activated state 3. weights. 4. activation function. 5.learning rate 6.nw
class Layer:
    #let input size of the layer be d, let output size of the layer be n. Assume that our batch has m sample points.
    #self.weights is a dxm random matrix converting input matrix of shape mxd to output matrix of shape mxn.
    #xavier initialization of weights matrix, zero initialization of bias
    def __init__(self, input_size, output_size, activation_function, learning_rate=0.05, output_layer=False):
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.bias = np.zeros((output_size, ), dtype=float)

        if isinstance(self.activation_function, Relu):      # he initialization for relu
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        else:                                               # Xavier initialization for others
            self.weights = np.random.randn(input_size, output_size) * \
                  np.sqrt(1. / (input_size + output_size))
        
        #if this layer is used to generate many outputs, for example in LSTM,
        #the gradients of loss wrt each output for this layer need to be collected in dW and dB
        #dW and dB should be reset for each separate call to backward() of the entire network.
        self.dW = np.zeros(self.weights.shape)
        self.dB = np.zeros(self.bias.shape)

    #forward propagation
    #step 1: H = XW
    #step 2: Z = activation(H)
    #setp 3: set dW and dB grad loss wrt weights and biases to 0
    def forward(self, input_data):
        #print(f"the inputs to forward for this layer are: ", input_data)
        #print(f"the weights that will dot the input layer are: ", self.weights)
        self.input = input_data
        self.h = np.dot(input_data, self.weights) + self.bias
        self.output = self.activation_function.forward(self.h)

        # this ensures that weights for each layer in the cell are always zero after the cell output is calculated.
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
    def backward(self, X, err):

        if isinstance(self.activation_function, Softmax) and self.activation_function.crossEntropyFollows:     # softmax shortcut
            dLdh = err
        else:
            #nxmxm such that ith element is dZi/dHi(mxm)
            dzdh = self.activation_function.backward(self.h)
            
            #dLdh = np.dot(output_derivative, dzdh)
            dLdh = np.empty_like(err)
            for i in range(err.shape[0]):
                dLdh[i] = np.dot(err[i], dzdh[i].T)

        dLdX = np.dot(dLdh, self.weights.T)
        
        dLdW = np.dot(X.T, dLdh)
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
    def __init__(self, crossEntropyFollows = True):
        super().__init__()
        self.crossEntropyFollows = crossEntropyFollows
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
        if Z.ndim == 2:
            return (Z - Y) / (Z.shape[0])
        elif Z.ndim == 3:
            return (Z - Y) / (Z.shape[1])

# cross entropy simplifies calculation of the output layer. (must have input rows sum to 1)
# batched. Expects nw output Z(nxm) and one-hot labels Y(nxm) where each row sums to 1
# forward: elementwise multiply log(Z) * one-hot encoded Y matrix and take the negative mean
# backward: Z-Y if output softmax
# this has been changed so that it now expects to be used only in combination with softmax. if not, flag False.
class CrossEntropy(Loss):
    def calculate(self, Z, Y, epsilon = 1e-10):
        Z = np.clip(Z, epsilon, 1-epsilon)
        weighted_log_probs = np.log(Z) * Y
        sample_losses = -np.sum(weighted_log_probs, axis=1)
        loss = np.mean(sample_losses)
        return loss
    def backward(self, Z, Y, epsilon=1e-10, loss_is_softmax = True):
        if loss_is_softmax:
            return (Z - Y)
        print("cross entropy used when loss is not softmax, currently unhandled")
        
# the idea for cell:
# a python array represents cell inputs
# layers are initialized with an input pointer and an output pointer
# The output of a layer in the network connects with other layers
# according to a series of elementwise operations
# inbetween each layer, intermediate calculations
# computation is done according to [ops, layer, ops, layer]
# an intermediate generally 
# data is a np array (Txnxd) of time data where th element is time-series input at time t
# note that Cell.predictions is a python list of numpy outputs (nxm per item if batched)
# when you're writing a cell, each operation MUST have its own separate err location. 
class Cell:
    def __init__(self, layers, intermediates, inputs, outputs, data, prediction_cache_index = -1, loss=CrossEntropy()):

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
        # cache has space for inputs, intermediates, outputs, and one object of time data.
        # data is a Txnxd np array
        self.oplength = sum(len(ilist) for ilist in intermediates) + len(layers) + len(inputs) +len(outputs)
        self.cache = [0] * (self.oplength + 1)
        self.data = data
        # the grad of loss wrt each element of the cell. indexes correspond to cache
        self.err = [0] * (self.oplength + 1)
        # there are layer_len + intermediate_len lists inside this list
        # the tth element of the list corresponding to layer is the input at time t
        # for intermediate, allinputs[cellop][time] is a 2 element list of np arrays , such that self.allinputs[cellop][time][0] holds a copy of the first input
        self.allinputs = [[] for _ in range(self.oplength - len(self.outputs))]
        #the predictions of the cell. tth element represents the prediction at time t
        self.predictions = [None for _ in range(data.shape[0])]
        #which element of the *output list* should be saved to predictions list? -1 by default.
        self.prediction_cache_index = prediction_cache_index
        self.loss = loss

    # forward propagation of a cell for one time step. takes inputs to the cell and time, uses data from time t.
    # saves all op inputs to allinpits and returns the output of cell
    # savedinputs is python array of np tensors input over time
    # note that some of our outputs should be saved in self.predictions, some passed to next cell.
    def forward_one(self, inputs, time):
        assert len(self.intermediates) == len(self.layers) + 1
        assert len(inputs) == len(self.inputs)

        self.cache = [0] * (self.oplength + 2)
        for i in range(len(self.inputs)):
            self.cache[self.inputs[i]] = inputs[i]
        self.cache[-1] = self.data[time, :, :]

        n = len(self.layers)
        cellop = 0
        for i in range(n+1):
            for j in range(len(self.intermediates[i])):

                #save in1, in2 of cellop to allinputs[cellop][time]
                self.allinputs[cellop].append(
                    [self.cache[self.intermediates[i][j][0]].copy(), 
                     self.cache[self.intermediates[i][j][1]].copy()]) 

                self.cache[self.intermediates[i][j][3]] = \
                    self.intermediates[i][j][2].forward(
                    self.cache[self.intermediates[i][j][0]], 
                    self.cache[self.intermediates[i][j][1]])
                
                #print(f"self.cache after intermediate {i}: ", self.cache)
                
                cellop += 1
            if i < n:

                self.allinputs[cellop].append(self.cache[self.layers[i][0]].copy())
                self.cache[self.layers[i][2]] = self.layers[i][1].forward(self.cache[self.layers[i][0]])
                #print(f"self.cache after layer {i}: ", self.cache)
                
                cellop += 1

        self.predictions[time] = self.cache[self.outputs[self.prediction_cache_index]]
        return [self.cache[i] for i in self.outputs]
    
    #forward propagate cell for all time steps, and return an array of all predictions made.
    #remember, by default, the output saved as prediction is outputs[-1]
    def forward(self, inputs, total_time = None):
        self.allinputs = [[] for _ in range(self.oplength - len(self.outputs))]
        if total_time == None:
            total_time = self.data.shape[0]
        assert len(self.intermediates) == len(self.layers) + 1
        assert len(inputs) == len(self.inputs)
        temp_inputs = inputs
        for t in range(total_time):
            temp_inputs = self.forward_one(temp_inputs, t)

        return self.predictions

        

    # error is the derivative of loss wrt cell output. combines raw output loss and future loss.
    # if we know cache[0] causes loss at rate 2, then we can infer about operations that feed it.
    # this returns a vector of errors for all the inputs.
    def backward_one(self, error, time):
        assert len(error) == len(self.outputs)
        # errors are stored and collected in self.err. error of cellop[i] is stored in err[i]
        # the output error is pulled from the last argument of layer/intermediate
        # err length is one for each operator, input, output, one for data, and one at the end for garbage operations like identity
        self.err = [None] * (self.oplength + 2)
        for i in range(len(error)):
            self.err[self.outputs[i]] = error[i]

        n = len(self.layers)
        cellop = self.oplength - len(self.inputs) - len(self.outputs) - 1
        for i in list(reversed(range(n+1))):
            for j in list(reversed(range(len(self.intermediates[i])))):
                # grad1 is the derivative of loss wrt input 1 of the intermediate
                # grad2 is the derivative of loss wrt input 2 of the intermediate
                grad1, grad2 = self.intermediates[i][j][2].backward(
                    self.allinputs[cellop][time][0], 
                    self.allinputs[cellop][time][1], 
                    self.err[self.intermediates[i][j][3]])
                #print(f"intermediate {i} grad1 is: ", grad1)
                #print(f"intermediate {i} grad2 is: ", grad2)

                if self.err[self.intermediates[i][j][0]] is None:
                    self.err[self.intermediates[i][j][0]] = grad1
                else:
                    self.err[self.intermediates[i][j][0]] += grad1
                
                if self.err[self.intermediates[i][j][1]] is None:
                    self.err[self.intermediates[i][j][1]] = grad2
                else:
                    self.err[self.intermediates[i][j][1]] += grad2
                cellop -= 1

            #layer backwards
            if i > 0:
                grad1 = self.layers[i-1][1].backward(
                    self.allinputs[cellop][time],
                    self.err[self.layers[i-1][2]]
                )
                if self.err[self.layers[i-1][0]] is None:
                    self.err[self.layers[i-1][0]] = grad1
                else:
                    self.err[self.layers[i-1][0]] += grad1
                cellop -= 1
                #print(f"layer {i-1} grad1 is: ", grad1)
        return [self.err[i] for i in self.inputs]

    # given the input data, true output labels, and some predictions,
    # accumulate the gradient of the loss with respect to all parameters.
    # gradients should be accumulated in the layer class instances.
    # *NOTE: we assume that the first n outputs are the inputs to the next layer, where n is len(inputs)
    def backward(self, labels, total_time = None, predictions = None):
        if total_time is None:
            total_time = self.data.shape[0]
        if predictions is None:
            predictions = self.predictions
        
        # output errors measures the error wrt each output.
        # in general, error for outputs should come from higher-t layer OR from direct output.
        output_errors = [None for _ in range(len(self.outputs))]
        
        for t in list(reversed(range(total_time))):
            # the error for the prediction at time t. only affects the output prediction.
            error_for_prediction_t = self.loss.backward(
                self.predictions[t], labels[t])
            
            # what is the error of the prediction? add to the error vector.
            if output_errors[self.prediction_cache_index] is None:
                output_errors[self.prediction_cache_index] = error_for_prediction_t
            else:
                output_errors[self.prediction_cache_index] += error_for_prediction_t

            input_errors = self.backward_one(output_errors, t)

            for i in range(len(input_errors)):
                output_errors[i] = input_errors[i]

# intermediate functions used in a neural net. multiple input forward, multiple output backward
# backward method takes original input of the vector x1 and x2 and dL/dz err
# identity should only be passed dupes
class identity:
    def forward(self, x1, x2):
        return x1
    def backward(self, x1, x2, err): 
        return err, 0
class hadamard: # elementwise prod 2 2d arrays
    def forward(self, x1, x2):
        return x1 * x2
    def backward(self, x1, x2, err):
        return x2 * err, x1 * err
class addition: # elementwise add 2 2d arrays
    def forward(self, x1, x2):
        return x1 + x2
    def backward(self, x1, x2, err):
        return err, err
    
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

# for an rnn, train mini batches of size batch_size and train on the whole dataset num_epochs times.
# Note that X is input in the shape (T, n, d) for data with T time steps, n data points, d dimensions.
# When we run nw.forward(data), data should similarly be in the shape (T, b, d) for batch size b
def rnn_train_mini_batches(nw : Cell, X, y, batch_size, num_epochs):
    num_samples = X.shape[1]
    num_batches = num_samples // batch_size
    for epoch in range(num_epochs):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X_shuffled = X[:, indices, :]
        y_shuffled = y[:, indices, :]
        start_index = None
        end_index = None
        
        nw.forward(X)
        epoch_beginning_loss = nw.loss.calculate(nw.predictions, y)
        print(f"total loss at the start of epoch {epoch + 1} is {epoch_beginning_loss}")
        
        with tqdm(total=num_batches, desc= f'Epoch {epoch + 1}/{num_epochs}') as pbar:
            for batch_index in range(num_batches):
                start_index = batch_index * batch_size
                end_index = start_index + batch_size

                nw.forward(X_shuffled[:, start_index : end_index, :])
                nw.backward(y_shuffled[:, start_index : end_index, :])

                #gradients are going to collect batch_size times in the layer variables, we should normalize it.
                for layer_tuple in B.layers:
                    layer_tuple[1].bias -= (layer_tuple[1].dB * layer_tuple[1].learning_rate) / batch_size
                    layer_tuple[1].weights -= (layer_tuple[1].dW * layer_tuple[1].learning_rate) / batch_size

                pbar.update(1)

            if num_samples % batch_size != 0:
                nw.forward(X_shuffled[:, end_index : , :])
                nw.backward(y_shuffled[:, end_index : , :])



#x_train shape: (60000, 28, 28) y_train shape: (60000,) x_test shape: (10000, 28, 28) y_test shape: (10000,) 
(x_train, y_train), (x_test, y_test) = load_data('datasets/mnist.npz')
xflat_train = x_train.reshape(x_train.shape[0], 28*28)
xflat_test  =  x_test.reshape(x_test.shape[0], 28*28)

# B is a cell with 
# 3 layers, 2 of which are ReLU and 1 of which is Softmax.
# nodes in the graph: 3 layers, input, output have 1, one for garbage
# -> len(self.err) = self.oplength + len(self.outputs) + 1
# data has 1 time and 60000 number and 784 dimensions
# input is at index 0 of cache and output is at index 3 of cache
# in this test, we will not use the time series data. some filler is there.
layer1 = Layer(784, 784, Relu(), learning_rate = 0.015)
layer2 = Layer(784, 250, Relu(), learning_rate = 0.015)
layer3 = Layer(250, 10, Softmax(crossEntropyFollows=True), learning_rate = 0.015)
B = Cell([(0, layer1, 1), 
          (1, layer2, 2),
          (2, layer3, 3)], 
        [[], 
         [], 
         [],
         []], 
         [0], [3], np.array([[[8., 8.], [8., 8.]]]), loss=CrossEntropy())

X = xflat_train / 255           # scale input data
y = one_hot_encode(y_train, 10) # generate one-hot vectors representing output data

X = X[np.newaxis, :]
y = y[np.newaxis, :]

# each time we run B.backward(y), the errors for y are collected in layer.dW, layer.dB.
rnn_train_mini_batches(B, X, y, 32, 1)

B.forward(X)
print("B.PREDICTIONS: ", B.predictions)
y_train_pred = np.argmax(B.predictions, axis=2)
print(f"y_train_pred has shape {y_train_pred.shape} and contents {y_train_pred}")
correct_predictions = np.sum(y_train_pred == y_train)
print(f"At the end of our training loop, the loss is: {B.loss.calculate(B.predictions, y)}")
print(f"The training accuracy of the network is : {correct_predictions / y_train.shape[0]}")

B.forward(xflat_test[np.newaxis, :] / 255)
y_test_pred = np.argmax(B.predictions, axis=2)
correct_predictions = np.sum(y_test_pred == y_test)
print(f" The test accuracy of the network is : {correct_predictions / y_test.shape[0]}")
    
