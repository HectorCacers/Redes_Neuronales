import DnnLib
import numpy as np
import json

def simple_sgd_update(layer, learning_rate=0.001):
    """Actualización SGD simple sin optimizador"""
    # Actualizar pesos y sesgos directamente
    layer.weights = layer.weights - learning_rate * layer.weight_gradients
    layer.bias = layer.bias - learning_rate * layer.bias_gradients

def train_network(layer1, layer2, X, y, X_test, y_test, epochs=3, learning_rate=0.001):
    """
    Función de entrenamiento como la mostrada por el instructor
    Usando SGD manual ya que no hay optimizadores en tu DnnLib
    """
    loss_history = []
    val_acc_history = []
    
    print("Training with manual SGD")
    
    for epoch in range(epochs):
        # Forward pass
        h1 = layer1.forward(X)
        output = layer2.forward(h1)
        
        # Compute loss manualmente (cross entropy)
        epsilon = 1e-15
        output_clipped = np.clip(output, epsilon, 1 - epsilon)
        loss = -np.mean(np.sum(y * np.log(output_clipped), axis=1))
        
        # Backward pass manual
        # Gradient de cross entropy + softmax
        batch_size = X.shape[0]
        dz2 = (output - y) / batch_size
        
        # Gradientes para layer2
        dW2 = np.dot(h1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        # Propagación hacia layer1
        dh1 = np.dot(dz2, layer2.weights.T)
        # Gradiente de ReLU
        dz1 = dh1 * (h1 > 0)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Actualizar parámetros manualmente
        layer2.weights = layer2.weights - learning_rate * dW2
        layer2.bias = layer2.bias - learning_rate * db2
        layer1.weights = layer1.weights - learning_rate * dW1
        layer1.bias = layer1.bias - learning_rate * db1
        
        # Calculate test accuracy
        h1_test = layer1.forward(X_test)
        output_test = layer2.forward(h1_test)
        predictions = np.argmax(output_test, axis=1)
        test_acc = np.mean(predictions == y_test)
        
        loss_history.append(loss)
        val_acc_history.append(test_acc)
        
        print(f"Epoch {epoch+1:02d} | Loss: {loss:.4f} | Test Acc: {test_acc:.2%}")
    
    return loss_history, val_acc_history

def evaluate_accuracy(test_images, test_labels, layer1, layer2):
    """Evaluar precisión final"""
    X_test = test_images.reshape(-1, 784).astype(np.float64) / 255.0
    h1 = layer1.forward(X_test)
    output = layer2.forward(h1)
    predictions = np.argmax(output, axis=1)
    accuracy = np.mean(predictions == test_labels)
    return accuracy

def main():
    print("Train MNIST MLP model")
    print("=" * 40)
    
    # Cargar datos
    print("Loading data...")
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
    
    # Crear capas como el instructor
    layer1 = DnnLib.DenseLayer(784, 128, DnnLib.ActivationType.RELU)
    layer2 = DnnLib.DenseLayer(128, 10, DnnLib.ActivationType.SOFTMAX)
    
    # Entrenar por 3 épocas como mostró el instructor
    loss_history, accuracy_history = train_network(layer1, layer2, X, y, X_test, test_labels, epochs=3, learning_rate=0.001)
    
    # Evaluación final
    final_accuracy = evaluate_accuracy(test_images, test_labels, layer1, layer2)
    print(f"\nTest Accuracy: {final_accuracy:.2%}")
    
    # Guardar modelo en el formato mostrado
    model_config = {
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
    
    with open("mnist_mlp_own_model.json", 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print("Model exported to mnist_mlp_own_model.json")

if __name__ == "__main__":
    main()