import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib
import json

# ==================== CARGAR DATOS ====================
def load_data():
    
    train = np.load('fashion_mnist_train.npz')
    test = np.load('fashion_mnist_test.npz')
    
    X_train, y_train = train['images'], train['labels']
    X_test, y_test = test['images'], test['labels']
    

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test, scaler

# ==================== CREAR MODELO ====================
def create_model():
  
    model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation='relu',
        solver='adam',
        alpha=0.001, 
        batch_size=128,
        learning_rate_init=0.001,
        max_iter=15,  
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=5,
        random_state=42
    )
    return model

# ==================== ENTRENAR MODELO ====================
def train_model(model, X_train, y_train):
    print("\n==================================================")
    print("ENTRENANDO MODELO (15 iteraciones máx)")
    print("==================================================")
    
    model.fit(X_train, y_train)
    
    loss_curve = model.loss_curve_
    
    return model, loss_curve

# ==================== EVALUAR MODELO ====================
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"\n ACCURACY EN TEST: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    return test_accuracy

# ==================== GUARDAR MODELO ====================
def save_model(model, scaler, accuracy, filename='fashion_mnist_best_model.joblib'):
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'accuracy': accuracy
    }
    
    joblib.dump(model_data, filename)
    print(f" Modelo guardado en {filename} con accuracy: {accuracy:.4f}")

# ==================== VISUALIZAR RESULTADOS ====================
def plot_training_history(loss_curve):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_curve)
    plt.title('Pérdida durante el entrenamiento')
    plt.ylabel('Pérdida')
    plt.xlabel('Iteración')
    plt.grid(True)
    plt.savefig('fashion_training_history.png')
    plt.show()

# ==================== FUNCIÓN PRINCIPAL ====================
def main():
    print("=== FASHION MNIST - MODELO CON SCIKIT-LEARN ===")
    print("============================================================")
    print(" Objetivo: >80% accuracy")
    print("============================================================")
    

    X_train, y_train, X_test, y_test, scaler = load_data()
    print(f" Datos cargados: {X_train.shape[0]} entrenamiento, {X_test.shape[0]} prueba")
    
    model = create_model()
    print(" Modelo MLP creado con regularización")
    
    trained_model, loss_curve = train_model(model, X_train, y_train)

    test_accuracy = evaluate_model(trained_model, X_test, y_test)
    
    save_model(trained_model, scaler, test_accuracy)

    plot_training_history(loss_curve)
    
    print("\n ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
    print(f" Accuracy alcanzado: {test_accuracy*100:.2f}%")

if __name__ == "__main__":
    main()
