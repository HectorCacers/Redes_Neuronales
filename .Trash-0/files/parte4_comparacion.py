import DnnLib
import numpy as np
import json

class ManualOptimizer:
    """Optimizadores implementados manualmente"""
    def __init__(self, optimizer_type, learning_rate, **kwargs):
        self.type = optimizer_type
        self.lr = learning_rate
        self.momentum = kwargs.get('momentum', 0.0)
        self.decay_rate = kwargs.get('decay_rate', 0.9)
        self.beta1 = kwargs.get('beta1', 0.9)
        self.beta2 = kwargs.get('beta2', 0.999)
        self.epsilon = kwargs.get('epsilon', 1e-8)
        
        # Estados para optimizadores adaptativos
        self.reset_state()
    
    def reset_state(self):
        self.v_w1 = None  # Momentum para layer1 weights
        self.v_b1 = None  # Momentum para layer1 bias
        self.v_w2 = None  # Momentum para layer2 weights
        self.v_b2 = None  # Momentum para layer2 bias
        
        # Para Adam
        self.m_w1 = None
        self.m_b1 = None
        self.m_w2 = None
        self.m_b2 = None
        self.t = 0
    
    def update(self, layer1, layer2, dW1, db1, dW2, db2):
        if self.type == "SGD":
            layer1.weights = layer1.weights - self.lr * dW1
            layer1.bias = layer1.bias - self.lr * db1
            layer2.weights = layer2.weights - self.lr * dW2
            layer2.bias = layer2.bias - self.lr * db2
            
        elif self.type == "SGD+Momentum":
            # Inicializar momentum si es necesario
            if self.v_w1 is None:
                self.v_w1 = np.zeros_like(dW1)
                self.v_b1 = np.zeros_like(db1)
                self.v_w2 = np.zeros_like(dW2)
                self.v_b2 = np.zeros_like(db2)
            
            # Actualizar momentum
            self.v_w1 = self.momentum * self.v_w1 + self.lr * dW1
            self.v_b1 = self.momentum * self.v_b1 + self.lr * db1
            self.v_w2 = self.momentum * self.v_w2 + self.lr * dW2
            self.v_b2 = self.momentum * self.v_b2 + self.lr * db2
            
            # Actualizar pesos
            layer1.weights = layer1.weights - self.v_w1
            layer1.bias = layer1.bias - self.v_b1
            layer2.weights = layer2.weights - self.v_w2
            layer2.bias = layer2.bias - self.v_b2
            
        elif self.type == "RMSprop":
            # Inicializar si es necesario
            if self.v_w1 is None:
                self.v_w1 = np.zeros_like(dW1)
                self.v_b1 = np.zeros_like(db1)
                self.v_w2 = np.zeros_like(dW2)
                self.v_b2 = np.zeros_like(db2)
            
            # Actualizar moving average de gradientes cuadrados
            self.v_w1 = self.decay_rate * self.v_w1 + (1 - self.decay_rate) * dW1**2
            self.v_b1 = self.decay_rate * self.v_b1 + (1 - self.decay_rate) * db1**2
            self.v_w2 = self.decay_rate * self.v_w2 + (1 - self.decay_rate) * dW2**2
            self.v_b2 = self.decay_rate * self.v_b2 + (1 - self.decay_rate) * db2**2
            
            # Actualizar pesos
            layer1.weights = layer1.weights - self.lr * dW1 / (np.sqrt(self.v_w1) + self.epsilon)
            layer1.bias = layer1.bias - self.lr * db1 / (np.sqrt(self.v_b1) + self.epsilon)
            layer2.weights = layer2.weights - self.lr * dW2 / (np.sqrt(self.v_w2) + self.epsilon)
            layer2.bias = layer2.bias - self.lr * db2 / (np.sqrt(self.v_b2) + self.epsilon)
            
        elif self.type == "Adam":
            self.t += 1
            
            # Inicializar si es necesario
            if self.m_w1 is None:
                self.m_w1 = np.zeros_like(dW1)
                self.m_b1 = np.zeros_like(db1)
                self.m_w2 = np.zeros_like(dW2)
                self.m_b2 = np.zeros_like(db2)
                self.v_w1 = np.zeros_like(dW1)
                self.v_b1 = np.zeros_like(db1)
                self.v_w2 = np.zeros_like(dW2)
                self.v_b2 = np.zeros_like(db2)
            
            # Actualizar momentos
            self.m_w1 = self.beta1 * self.m_w1 + (1 - self.beta1) * dW1
            self.m_b1 = self.beta1 * self.m_b1 + (1 - self.beta1) * db1
            self.m_w2 = self.beta1 * self.m_w2 + (1 - self.beta1) * dW2
            self.m_b2 = self.beta1 * self.m_b2 + (1 - self.beta1) * db2
            
            self.v_w1 = self.beta2 * self.v_w1 + (1 - self.beta2) * dW1**2
            self.v_b1 = self.beta2 * self.v_b1 + (1 - self.beta2) * db1**2
            self.v_w2 = self.beta2 * self.v_w2 + (1 - self.beta2) * dW2**2
            self.v_b2 = self.beta2 * self.v_b2 + (1 - self.beta2) * db2**2
            
            # Bias correction
            m_w1_hat = self.m_w1 / (1 - self.beta1**self.t)
            m_b1_hat = self.m_b1 / (1 - self.beta1**self.t)
            m_w2_hat = self.m_w2 / (1 - self.beta1**self.t)
            m_b2_hat = self.m_b2 / (1 - self.beta1**self.t)
            
            v_w1_hat = self.v_w1 / (1 - self.beta2**self.t)
            v_b1_hat = self.v_b1 / (1 - self.beta2**self.t)
            v_w2_hat = self.v_w2 / (1 - self.beta2**self.t)
            v_b2_hat = self.v_b2 / (1 - self.beta2**self.t)
            
            # Actualizar pesos
            layer1.weights = layer1.weights - self.lr * m_w1_hat / (np.sqrt(v_w1_hat) + self.epsilon)
            layer1.bias = layer1.bias - self.lr * m_b1_hat / (np.sqrt(v_b1_hat) + self.epsilon)
            layer2.weights = layer2.weights - self.lr * m_w2_hat / (np.sqrt(v_w2_hat) + self.epsilon)
            layer2.bias = layer2.bias - self.lr * m_b2_hat / (np.sqrt(v_b2_hat) + self.epsilon)

def train_with_optimizer(optimizer_name, optimizer, X, y, X_test, y_test, epochs=3):
    """Entrenar con un optimizador específico"""
    print(f"\n=== Training with {optimizer_name} ===")
    
    # Crear capas nuevas para cada optimizador
    layer1 = DnnLib.DenseLayer(784, 128, DnnLib.ActivationType.RELU)
    layer2 = DnnLib.DenseLayer(128, 10, DnnLib.ActivationType.SOFTMAX)
    
    optimizer.reset_state()
    
    results = {
        'optimizer': optimizer_name,
        'losses': [],
        'accuracies': []
    }
    
    for epoch in range(epochs):
        # Forward pass
        h1 = layer1.forward(X)
        output = layer2.forward(h1)
        
        # Compute loss manualmente
        epsilon = 1e-15
        output_clipped = np.clip(output, epsilon, 1 - epsilon)
        loss = -np.mean(np.sum(y * np.log(output_clipped), axis=1))
        
        # Backward pass manual
        batch_size = X.shape[0]
        dz2 = (output - y) / batch_size
        
        # Gradientes para layer2
        dW2 = np.dot(h1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        # Propagación hacia layer1
        dh1 = np.dot(dz2, layer2.weights.T)
        dz1 = dh1 * (h1 > 0)  # ReLU derivative
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Actualizar con optimizador
        optimizer.update(layer1, layer2, dW1, db1, dW2, db2)
        
        # Calculate test accuracy
        h1_test = layer1.forward(X_test)
        output_test = layer2.forward(h1_test)
        predictions = np.argmax(output_test, axis=1)
        test_acc = np.mean(predictions == y_test)
        
        results['losses'].append(loss)
        results['accuracies'].append(test_acc)
        
        print(f"Epoch {epoch+1:02d} | Loss: {loss:.4f} | Test Acc: {test_acc:.2%}")
    
    # Evaluación final completa
    final_accuracy = evaluate_full_test(layer1, layer2, X_test, y_test)
    results['final_accuracy'] = final_accuracy
    results['layers'] = (layer1, layer2)
    
    print(f"Final Test Accuracy: {final_accuracy:.2%}")
    
    return results

def evaluate_full_test(layer1, layer2, X_test, y_test):
    """Evaluación completa en test set"""
    h1 = layer1.forward(X_test)
    output = layer2.forward(h1)
    predictions = np.argmax(output, axis=1)
    accuracy = np.mean(predictions == y_test)
    return accuracy

def save_best_model(best_result, filename="best_optimizer_model.json"):
    """Guardar el mejor modelo"""
    layer1, layer2 = best_result['layers']
    
    model_config = {
        "optimizer_used": best_result['optimizer'],
        "final_accuracy": best_result['final_accuracy'],
        "input_shape": [28, 28, 1],
        "preprocess": {"scale": 255.0},
        "layers": [
            {
                "units": 128,
                "activation": "relu",
                "W": layer1.weights.tolist(),
                "b": layer1.bias.tolist()
            },
            {
                "units": 10,
                "activation": "softmax",
                "W": layer2.weights.tolist(),
                "b": layer2.bias.tolist()
            }
        ]
    }
    
    with open(filename, 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print(f"Best model ({best_result['optimizer']}) saved to {filename}")

def main():
    print("OPTIMIZER COMPARISON ON MNIST")
    print("=" * 50)
    
    # Cargar datos
    print("Loading MNIST data...")
    train_data = np.load("mnist_train.npz")
    train_images = train_data["images"]
    train_labels = train_data["labels"]
    
    test_data = np.load("mnist_test.npz")
    test_images = test_data["images"]
    test_labels = test_data["labels"]
    
    # Preprocesar
    X = train_images.reshape(-1, 784).astype(np.float64) / 255.0
    y = np.eye(10)[train_labels].astype(np.float64)
    X_test = test_images.reshape(-1, 784).astype(np.float64) / 255.0
    
    print(f"Training set: {X.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Definir optimizadores manuales
    optimizers = [
        ("SGD", ManualOptimizer("SGD", learning_rate=0.1)),
        ("SGD+Momentum", ManualOptimizer("SGD+Momentum", learning_rate=0.05, momentum=0.9)),
        ("RMSprop", ManualOptimizer("RMSprop", learning_rate=0.001, decay_rate=0.9)),
        ("Adam", ManualOptimizer("Adam", learning_rate=0.001))
    ]
    
    results = []
    
    # Probar cada optimizador
    for opt_name, optimizer in optimizers:
        result = train_with_optimizer(opt_name, optimizer, X, y, X_test, test_labels, epochs=3)
        results.append(result)
    
    # Mostrar comparación final
    print("\n" + "=" * 50)
    print("FINAL COMPARISON RESULTS")
    print("=" * 50)
    print(f"{'Optimizer':<15} {'Final Loss':<12} {'Test Accuracy':<15}")
    print("-" * 50)
    
    for result in results:
        final_loss = result['losses'][-1]
        final_acc = result['final_accuracy']
        print(f"{result['optimizer']:<15} {final_loss:<12.4f} {final_acc:<15.2%}")
    
    # Encontrar mejor resultado
    best_result = max(results, key=lambda x: x['final_accuracy'])
    print(f"\nBEST OPTIMIZER: {best_result['optimizer']} ({best_result['final_accuracy']:.2%})")
    
    # Guardar mejor modelo
    save_best_model(best_result)
    
    print("\nOptimizer comparison completed!")
    return results

if __name__ == "__main__":
    main()