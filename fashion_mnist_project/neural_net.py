import numpy as np
import DnnLib

class ManualDenseLayer:
    """Capa densa con backpropagation"""
    
    def __init__(self, input_dim, output_dim, activation_type=DnnLib.ActivationType.RELU):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_type = activation_type
        
       
        self.weights = np.random.randn(output_dim, input_dim) * np.sqrt(2.0 / input_dim)
        self.bias = np.zeros((output_dim, 1))
        
    
        self.dweights = np.zeros_like(self.weights)
        self.dbias = np.zeros_like(self.bias)
        
      
        self.last_input = None
        self.last_output = None
        self.last_z = None
    
    def forward(self, x):
        self.last_input = x
        
        
        self.last_z = np.dot(x, self.weights.T) + self.bias.T
        
      
        if self.activation_type == DnnLib.ActivationType.RELU:
            self.last_output = self.relu(self.last_z)
        elif self.activation_type == DnnLib.ActivationType.SOFTMAX:
            self.last_output = self.softmax(self.last_z)
        else:
            self.last_output = self.last_z
        
        return self.last_output
    
    def backward(self, doutput):
        
        if self.activation_type == DnnLib.ActivationType.RELU:
            dactivation = self.relu_derivative(self.last_z)
        elif self.activation_type == DnnLib.ActivationType.SOFTMAX:
            dactivation = 1  
        else:
            dactivation = 1
        
        dz = doutput * dactivation
        
        
        batch_size = self.last_input.shape[0]
        self.dweights = np.dot(dz.T, self.last_input) / batch_size
        self.dbias = np.sum(dz.T, axis=1, keepdims=True) / batch_size
        
       
        dinput = np.dot(dz, self.weights)
        
        return dinput
    
    def update(self, learning_rate):
        """Actualizar pesos"""
        self.weights -= learning_rate * self.dweights
        self.bias -= learning_rate * self.dbias
    
   
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(np.float64)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)