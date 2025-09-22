import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_fashion_data
from neural_net import ManualDenseLayer
from regularizers import Dropout
from trainers import SGDMomentum, train_model
from utils import save_model
import DnnLib

def main():
    print("FASHION MNIST - COMPARACIÓN DE REGULARIZADORES")
    print("=" * 60)
    
    
    X_train, y_train, X_test, y_test, y_test_labels = load_fashion_data()
    
   
    optimizer = SGDMomentum(learning_rate=0.01, momentum=0.9)
    
    # 1. Entrenar SIN regularización
    print("\n" + "="*50)
    print("1. ENTRENANDO SIN REGULARIZACIÓN")
    print("="*50)
    
    layers_no_reg = [
        ManualDenseLayer(784, 256, DnnLib.ActivationType.RELU),
        ManualDenseLayer(256, 128, DnnLib.ActivationType.RELU),
        ManualDenseLayer(128, 10, DnnLib.ActivationType.SOFTMAX)
    ]
    
    losses_no_reg, accuracies_no_reg = train_model(
        layers_no_reg, optimizer, X_train, y_train, X_test, y_test_labels,
        epochs=10
    )
    
   
    print("\n" + "="*50)
    print("2. ENTRENANDO CON L2 REGULARIZATION (lambda=0.001)")
    print("="*50)
    
    layers_l2 = [
        ManualDenseLayer(784, 256, DnnLib.ActivationType.RELU),
        ManualDenseLayer(256, 128, DnnLib.ActivationType.RELU),
        ManualDenseLayer(128, 10, DnnLib.ActivationType.SOFTMAX)
    ]
    
    losses_l2, accuracies_l2 = train_model(
        layers_l2, optimizer, X_train, y_train, X_test, y_test_labels,
        epochs=10, l2_lambda=0.001
    )
    
    
    print("\n" + "="*50)
    print("3. ENTRENANDO CON DROPOUT (rate=0.5)")
    print("="*50)
    
    layers_dropout = [
        ManualDenseLayer(784, 256, DnnLib.ActivationType.RELU),
        ManualDenseLayer(256, 128, DnnLib.ActivationType.RELU),
        ManualDenseLayer(128, 10, DnnLib.ActivationType.SOFTMAX)
    ]
    
    dropout_layers = [Dropout(0.5), Dropout(0.5)]
    
    losses_dropout, accuracies_dropout = train_model(
        layers_dropout, optimizer, X_train, y_train, X_test, y_test_labels,
        epochs=10, dropout_layers=dropout_layers
    )
    

    print("\n" + "="*60)
    print("RESULTADOS FINALES")
    print("="*60)
    
    results = {
        "Sin Regularización": accuracies_no_reg[-1],
        "L2 Regularization": accuracies_l2[-1],
        "Dropout": accuracies_dropout[-1]
    }
    
    for method, acc in results.items():
        print(f"{method}: {acc:.4f} ({acc*100:.2f}%)")
    
   
    best_method = max(results, key=results.get)
    best_accuracy = results[best_method]
    
    if best_method == "Sin Regularización":
        best_layers = layers_no_reg
    elif best_method == "L2 Regularization":
        best_layers = layers_l2
    else:
        best_layers = layers_dropout
    
    save_model(best_layers, "fashion_mnist_best_model.json", best_accuracy)
    
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses_no_reg, label='Sin Regularización', linewidth=2)
    plt.plot(losses_l2, label='L2 Regularization', linewidth=2)
    plt.plot(losses_dropout, label='Dropout', linewidth=2)
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.title('Pérdida de Entrenamiento')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies_no_reg, label='Sin Regularización', linewidth=2)
    plt.plot(accuracies_l2, label='L2 Regularization', linewidth=2)
    plt.plot(accuracies_dropout, label='Dropout', linewidth=2)
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.title('Accuracy en Prueba')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('fashion_mnist_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n ENTRENAMIENTO COMPLETADO!")
    print(f" Gráficos guardados en 'fashion_mnist_results.png'")
    print(f" Modelo guardado en 'fashion_mnist_best_model.json'")

if __name__ == "__main__":
    main()