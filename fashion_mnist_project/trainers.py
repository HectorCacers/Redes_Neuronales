import numpy as np
from neural_net import ManualDenseLayer
from regularizers import L2Regularizer, Dropout
from utils import cross_entropy_loss, cross_entropy_gradient

class SGDMomentumOptimized:
    """Optimizador SGD con momentum optimizado para mejor convergencia"""
    
    def __init__(self, learning_rate=0.005, momentum=0.95):  
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}
    
    def update_layer(self, layer, layer_idx):
        if layer_idx not in self.velocities:
            self.velocities[layer_idx] = {
                'weights': np.zeros_like(layer.dweights),
                'bias': np.zeros_like(layer.dbias)
            }
        
       
        self.velocities[layer_idx]['weights'] = (
            self.momentum * self.velocities[layer_idx]['weights'] + 
            (1 - self.momentum) * layer.dweights
        )
        self.velocities[layer_idx]['bias'] = (
            self.momentum * self.velocities[layer_idx]['bias'] + 
            (1 - self.momentum) * layer.dbias
        )
        
        layer.weights -= self.learning_rate * self.velocities[layer_idx]['weights']
        layer.bias -= self.learning_rate * self.velocities[layer_idx]['bias']

def train_model_optimized(layers, optimizer, X_train, y_train, X_test, y_test_labels,
                         epochs=15, l2_lambda=0.0, dropout_layers=None):
    """Función de entrenamiento optimizada para mayor accuracy"""
    
    n_samples = X_train.shape[0]
    batch_size = 128
    train_losses = []
    test_accuracies = []
    l2_regularizer = L2Regularizer(l2_lambda) if l2_lambda > 0 else None
    
    for epoch in range(epochs):
    
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        epoch_loss = 0.0
        n_batches = 0
        
      
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
      
            activations = [X_batch]
            for j, layer in enumerate(layers):
                activation = layer.forward(activations[-1])
                if dropout_layers and j < len(dropout_layers):
                    activation = dropout_layers[j].forward(activation)
                activations.append(activation)
            
            output = activations[-1]
            
       
            loss = cross_entropy_loss(output, y_batch)
            
         
            reg_loss = 0
            if l2_regularizer:
                for layer in layers:
                    reg_loss += l2_regularizer.apply(layer)
            
            total_loss = loss + reg_loss
            
       
            grad = cross_entropy_gradient(output, y_batch)
            
            
            for j in reversed(range(len(layers))):
                if dropout_layers and j < len(dropout_layers):
                    grad = dropout_layers[j].backward(grad)
                grad = layers[j].backward(grad)
            
       
            for j, layer in enumerate(layers):
                optimizer.update_layer(layer, j)
            
            epoch_loss += total_loss
            n_batches += 1
      
        test_output = X_test
        for j, layer in enumerate(layers):
            test_output = layer.forward(test_output)
            if dropout_layers and j < len(dropout_layers):
                dropout_layers[j].training = False
        
        predicted = np.argmax(test_output, axis=1)
        accuracy = np.mean(predicted == y_test_labels)
        
      
        if dropout_layers:
            for dropout in dropout_layers:
                dropout.training = True
        
        avg_loss = epoch_loss / n_batches
        train_losses.append(avg_loss)
        test_accuracies.append(accuracy)
        
        print(f"Epoch {epoch+1:2d}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    return train_losses, test_accuracies


def train_model(layers, optimizer, X_train, y_train, X_test, y_test_labels,
               epochs=10, l2_lambda=0.0, dropout_layers=None):
    return train_model_optimized(layers, optimizer, X_train, y_train, X_test, 
                               y_test_labels, epochs, l2_lambda, dropout_layers)
