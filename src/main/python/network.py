import numpy as np
import random

class Network(object):
    def __init__(self,sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x)
                        for x,y in zip(sizes[:-1], sizes[1:])]
        
        self.irmeli=1
       
  


    def feedforward(self,a):
       # if self.irmeli ==1:
             #zipped = zip(self.biases,self.weights)
             #print(list(zipped))
        
        for b, w in(zip(self.biases,self.weights)):
                #if self.irmeli ==1:
                    #print(f'Bias:  {b}')
                    #print(f'Weight: {w}')
                #Note internal loop!!! a updated its own value!!!
                a = sigmoid(np.dot(w,a)+b)
        
        self.irmeli=0
        return a

    def SGD(self,training_data, epochs, mini_batch_size, eta, test_data= None):
        
        listTrainingdata = list(training_data)
        lit = list(test_data)
        
        if test_data: n_test = len(lit)
        n = len(listTrainingdata)
       
        for j in range(epochs):
            random.shuffle(listTrainingdata)
            mini_batches = [
                listTrainingdata[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if lit:
                    print ('Epoch {0}: {1} / {2}'.format(
                        j, self.evaluate(lit), n_test))
            else:
                print ("Epoch {0} complete".format(j))
            

    def update_mini_batch(self, mini_batch, eta):
        #initializing empty nablas
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                    for w, nw in zip(self.weights, nabla_w)] #eq 20 http://neuralnetworksanddeeplearning.com/chap1.html
        
        self.biases = [b-(eta/len(mini_batch))*nb
                        for b, nb in zip(self.biases, nabla_b)] # eq 21 http://neuralnetworksanddeeplearning.com/chap1.html
            

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)),y)
                            for(x,y) in test_data]
            
        return sum(int(x==y) for (x,y) in test_results)

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        
        activations = [x] #list to store all the activations layer by layer
        #print(activations)
        zs = [] # list to all the z vectors, layer by layer
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b # actually w.x +y
            zs.append(z)
            activation = sigmoid(z)
            #print(activation)
            activations.append(activation)
            
        #last layer
        delta = self.cost_derivate(activations[-1], y)  * \
                sigmoid_prime(zs[-1]) #BP1
        nabla_b[-1] = delta #BP3
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())#BP4
        
        #layer by layer calculate bias and weight
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta)*sp #BP2  see eq 34 http://neuralnetworksanddeeplearning.com/chap2.html
            nabla_b[-l] = delta #BP3
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())  #PB4
        return (nabla_b, nabla_w)
    
    def cost_derivate(self, ouput_activations, y):
        return (ouput_activations-y)
    


def sigmoid_prime(z):
    "Derivate sigmoid function"
    return sigmoid(z)*(1-sigmoid(z))

  
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
