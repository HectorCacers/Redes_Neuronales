import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import json
import os

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def initialize(self, params):
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0
    
    def update(self, params, grads):
        if self.m is None:
            self.initialize(params)
        
        self.t += 1
        updated_params = []
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            param_update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            updated_param = param - param_update
            updated_params.append(updated_param)
        
        return updated_params

class MNISTTrainer:
    def __init__(self):
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        
       
        self.input_size = 784
        self.hidden_size = 128  
        self.output_size = 10
        self.learning_rate = 0.001 
        self.epochs = 3  
        self.batch_size = 128  
        
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        
        self.optimizer = AdamOptimizer(learning_rate=self.learning_rate)
    
    def load_data(self):
        """Cargar datos MNIST"""
        train_data = np.load("mnist_train.npz")
        self.train_images = train_data["images"]
        self.train_labels = train_data["labels"]
        
        test_data = np.load("mnist_test.npz")
        self.test_images = test_data["images"]
        self.test_labels = test_data["labels"]
        
        print(" Datos MNIST cargados:")
        print(f"   Entrenamiento: {self.train_images.shape} imágenes")
        print(f"   Prueba: {self.test_images.shape} imágenes")
    
    def preprocess_data(self):
        """Preprocesar datos"""
        self.X_train = self.train_images.reshape(-1, 784).astype(np.float64) / 255.0
        self.X_test = self.test_images.reshape(-1, 784).astype(np.float64) / 255.0
        self.y_train = np.eye(10, dtype=np.float64)[self.train_labels]
        self.y_test = self.test_labels
        
        print(" Datos preprocesados (/255.0 y one-hot encoding)")
    
    def initialize_weights(self):
        """Inicializar pesos con He Initialization"""
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2.0 / self.input_size)
        self.b1 = np.zeros((1, self.hidden_size))
        
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2.0 / self.hidden_size)
        self.b2 = np.zeros((1, self.output_size))
        
        print(" Pesos inicializados con He Initialization")
        self.optimizer.initialize([self.W1, self.b1, self.W2, self.b2])
    
    def relu(self, x):
        """Función de activación ReLU"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivada de ReLU"""
        return (x > 0).astype(float)
    
    def softmax(self, x):
        """Función Softmax"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward_propagation(self, X):
        """Forward pass"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def backward_propagation(self, X, y, output):
        """Backward pass"""
        m = X.shape[0]
        
       
        dz2 = output - y
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
       
        dz1 = np.dot(dz2, self.W2.T) * self.relu_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        return dW1, db1, dW2, db2
    
    def update_parameters_adam(self, grads):
        """Actualizar parámetros con Adam"""
        params = [self.W1, self.b1, self.W2, self.b2]
        updated_params = self.optimizer.update(params, grads)
        self.W1, self.b1, self.W2, self.b2 = updated_params
    
    def compute_loss(self, y_true, y_pred):
        """Calcular pérdida cross-entropy"""
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
        return loss
    
    def train(self):
        """Entrenar el modelo"""
        print(" Iniciando entrenamiento (3 épocas)")
        print("=" * 50)
        print(f"Arquitectura: {self.input_size} -> {self.hidden_size} -> {self.output_size}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print("Optimizador: Adam")
        print("=" * 50)
        
        losses = []
        accuracies = []
        
        n_samples = self.X_train.shape[0]
        num_batches = n_samples // self.batch_size
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            
            
            indices = np.random.permutation(n_samples)
            X_shuffled = self.X_train[indices]
            y_shuffled = self.y_train[indices]
            
            for i in range(num_batches):
                
                start = i * self.batch_size
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
               
                output = self.forward_propagation(X_batch)
                
                
                loss = self.compute_loss(y_batch, output)
                epoch_loss += loss
                
                
                dW1, db1, dW2, db2 = self.backward_propagation(X_batch, y_batch, output)
                
                
                grads = [dW1, db1, dW2, db2]
                self.update_parameters_adam(grads)
            
         
            train_acc = self.evaluate_accuracy(self.X_train, self.train_labels)
            test_acc = self.evaluate_accuracy(self.X_test, self.test_labels)
            
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            accuracies.append(test_acc)
            
            print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
        
        return losses, accuracies
    
    def evaluate_accuracy(self, X, y):
        """Evaluar precisión"""
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
    def predict(self, X):
        """Hacer predicciones"""
        output = self.forward_propagation(X)
        return np.argmax(output, axis=1)
    
    def save_model(self, filename):
        """Guardar modelo en formato JSON"""
        model_config = {
            "input_shape": [28, 28, 1],
            "preprocess": {"scale": 255.0},
            "layers": [
                {
                    "units": self.hidden_size,
                    "activation": "relu",
                    "W": self.W1.tolist(),
                    "b": self.b1.tolist()
                },
                {
                    "units": self.output_size,
                    "activation": "softmax", 
                    "W": self.W2.tolist(),
                    "b": self.b2.tolist()
                }
            ]
        }
        
       
        os.makedirs("exercise4models", exist_ok=True)
        
        with open(f"exercise4models/{filename}", 'w') as f:
            json.dump(model_config, f, indent=2)
        
        print(f" Modelo exportado a: exercise4models/{filename}")
    
    def visualize_training(self, losses, accuracies):
        """Visualizar progreso del entrenamiento"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(losses, 'b-', linewidth=2)
        plt.title('Pérdida durante entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(accuracies, 'g-', linewidth=2)
        plt.title('Precisión durante entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    print(" ENTRENAMIENTO CON ADAM OPTIMIZER")
    print("=" * 50)
    print("Misma configuración")
    print("=" * 50)
    
    
    trainer = MNISTTrainer()
    
 
    trainer.load_data()
    
   
    trainer.preprocess_data()
    
   
    trainer.initialize_weights()
    
  
    losses, accuracies = trainer.train()
    
   
    final_accuracy = trainer.evaluate_accuracy(trainer.X_test, trainer.test_labels)
    print("=" * 50)
    print(f" TEST ACCURACY: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    

    trainer.save_model("mnist_mlp_own_model.json")
    
  
    print("\n" + "=" * 50)
    print(" RESUMEN DEL ENTRENAMIENTO:")
    print("=" * 50)
    for epoch, (loss, acc) in enumerate(zip(losses, accuracies)):
        print(f"Epoch {epoch+1:02d} | Loss: {loss:.4f} | Test Acc: {acc:.4f}")
    
    print("=" * 50)
    print(f"Test Accuracy: {final_accuracy:.4f}")
    print("Model exported to exercise4models/mnist_mlp_own_model.json")
    
   
    trainer.visualize_training(losses, accuracies)
    
    print(" Entrenamiento completado exitosamente!")

if __name__ == "__main__":
    main()