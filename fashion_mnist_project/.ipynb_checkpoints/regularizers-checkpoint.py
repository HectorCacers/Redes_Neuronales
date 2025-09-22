import numpy as np

class L2Regularizer:
    """Regularización L2"""
    
    def __init__(self, lambda_val=0.001):
        self.lambda_val = lambda_val
    
    def apply(self, layer):
        """Aplicar regularización L2 a una capa"""
        reg_loss = 0.5 * self.lambda_val * np.sum(layer.weights ** 2)
        layer.dweights += self.lambda_val * layer.weights
        return reg_loss

class Dropout:
    """Capa de Dropout"""
    
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training = True
    
    def forward(self, x):
        if self.training:
            self.mask = (np.random.rand(*x.shape) > self.dropout_rate).astype(np.float64)
            return x * self.mask / (1.0 - self.dropout_rate)
        return x
    
    def backward(self, doutput):
        if self.training:
            return doutput * self.mask / (1.0 - self.dropout_rate)
        return doutput