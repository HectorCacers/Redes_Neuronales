import numpy as np
import json

def cross_entropy_loss(y_pred, y_true, epsilon=1e-15):
    """Pérdida de entropía cruzada optimizada"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def cross_entropy_gradient(y_pred, y_true):
    """Gradiente de entropía cruzada optimizada"""
    return (y_pred - y_true) / y_pred.shape[0]

def save_model(layers, filename, accuracy):
    """Guardar modelo"""
    model_config = {
        "input_shape": [28, 28, 1],
        "preprocess": {
            "scale": 255.0,
            "reshape": [784]
        },
        "layers": [],
        "final_accuracy": float(accuracy)
    }
    
    for i, layer in enumerate(layers):
        layer_config = {
            "type": "Dense",
            "units": layer.weights.shape[0],
            "activation": "relu" if i < len(layers)-1 else "softmax",
            "W": layer.weights.tolist(),
            "b": layer.bias.flatten().tolist()  
        }
        model_config["layers"].append(layer_config)
    
    with open(filename, 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print(f" Modelo guardado en {filename} con accuracy: {accuracy:.4f}")
