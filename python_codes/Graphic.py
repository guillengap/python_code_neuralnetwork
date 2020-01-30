import numpy as np

# We create the class 
class NeuralNetwork:

    def __init__(self, layers, activation='tanh'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_derivada
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_derivada

        # Initialize the weights
        self.weights = []
        self.deltas = []
        # Assign random values to input layer and hidden layer
        for i in range(1, len(layers) - 1):
            r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
            self.weights.append(r)
        # Assigned random to output layer
        r = 2*np.random.random( (layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r)

    def fit(self, X, y, learning_rate=0.2, epochs=100000):
        # I add column of ones to the X inputs. With this we add the Bias unit to the input layer
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
        
        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                    dot_value = np.dot(a[l], self.weights[l])
                    activation = self.activation(dot_value)
                    a.append(activation)
            #Calculate the difference in the output layer and the value obtained
            error = y[i] - a[-1]
            deltas = [error * self.activation_prime(a[-1])]
            
            # We start in the second layer until the last one (A layer before the output one)
            for l in range(len(a) - 2, 0, -1): 
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(a[l]))
            self.deltas.append(deltas)

            # Reverse
            deltas.reverse()

            # Backpropagation
            # 1. Multiply the output delta with the input activations to obtain the weight gradient.             
            # 2. Updated the weight by subtracting a percentage of the gradient
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

            if k % 10000 == 0: print('epochs:', k)

    def predict(self, x): 
        ones = np.atleast_2d(np.ones(x.shape[0]))
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=0)
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

    def print_weights(self):
        print("LIST OF CONNECTION WEIGHTS")
        for i in range(len(self.weights)):
            print(self.weights[i])

    def get_weights(self):
        return self.weights
    
    def get_deltas(self):
        return self.deltas

# When creating the network, we can choose between using the sigmoid or tanh function
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_derivada(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_derivada(x):
    return 1.0 - x**2

########## CAR NETWORK

nn = NeuralNetwork([2,2,4],activation ='tanh') # no incluir la bias aqui porque si la esta en los calculos

X = np.array([[1,1],   # light_Current >= light_Before & light_Current <= 3750
              [-1,1],   # light_Current < light_Before & light_Current <= 3750
              [1,-1],   # light_Current >= light_Before & light_Current > 3750
              [-1,-1],   # light_Current < light_Before & light_Current > 3750
             ])
# the outputs correspond to starting (or not) the motors
y = np.array([[1,0,1,0], # go forward 
              [0,1,1,0], # turn to the left
              [0,0,0,0], # stop
              [0,0,0,0], # stop
             ])
nn.fit(X, y, learning_rate=0.03,epochs=15001)
 
def valNN(x):
    return (int)(abs(round(x)))
 
index=0
for e in X:
    prediccion = nn.predict(e)
    print("X:",e,"expected:",y[index],"obtained:", valNN(prediccion[0]),valNN(prediccion[1]),valNN(prediccion[2]),valNN(prediccion[3]))
    index=index+1

########## WE GRAPH THE COST FUNCTION
    
import matplotlib.pyplot as plt

deltas = nn.get_deltas()
valores=[]
index=0
for arreglo in deltas:
    valores.append(arreglo[1][0] + arreglo[1][1])
    index=index+1

plt.plot(range(len(valores)), valores, color='b')
plt.ylim([0, 1])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.tight_layout()
plt.show()
