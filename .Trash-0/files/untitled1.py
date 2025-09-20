import DnnLib
import numpy as np
import json

def train_network(layer1, layer2, optimizer, X, y, X_test, y_test, epochs=3):
    """
    Función de entrenamiento como la mostrada por el instructor
    """
    loss_history = []
    val_acc_history = []
    
    print(f"Training with {type(optimizer).__name__}")
    
    for epoch in range(epochs):
        # Forward pass
        h1 = layer1.forward(X)
        output = layer2.forward(h1)
        
        # Compute loss
        loss = DnnLib.cross_entropy(output, y)
        
        # Backward pass
        loss_grad = DnnLib.cross_entropy_gradient(output, y)
        grad2 = layer2.backward(loss_grad)
        grad1 = layer1.backward(grad2)
        
        # Update parameters
        optimizer.update(layer2)
        optimizer.update(layer1)
        
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
    
    # Crear optimizador Adam como en la presentación
    optimizer = DnnLib.Adam(learning_rate=0.001)
    
    # Entrenar por 3 épocas como mostró el instructor
    loss_history, accuracy_history = train_network(layer1, layer2, optimizer, X, y, X_test, test_labels, epochs=3)
    
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