import numpy as np

def load_fashion_data():
    """Cargar y preprocesar datos de Fashion MNIST"""
    try:
        train_data = np.load('fashion_mnist_train.npz')
        test_data = np.load('fashion_mnist_test.npz')

        X_train = train_data['images'].reshape(-1, 784).astype(np.float64) / 255.0
        y_train_labels = train_data['labels']
        
        y_train = np.zeros((len(y_train_labels), 10), dtype=np.float64)
        y_train[np.arange(len(y_train_labels)), y_train_labels] = 1.0

        X_test = test_data['images'].reshape(-1, 784).astype(np.float64) / 255.0
        y_test_labels = test_data['labels']
        
        y_test = np.zeros((len(y_test_labels), 10), dtype=np.float64)
        y_test[np.arange(len(y_test_labels)), y_test_labels] = 1.0

        print(f" Datos cargados: {X_train.shape[0]} entrenamiento, {X_test.shape[0]} prueba")
        return X_train, y_train, X_test, y_test, y_test_labels
        
    except Exception as e:
        print(f" Error cargando datos: {e}")
        raise