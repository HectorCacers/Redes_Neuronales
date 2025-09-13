import numpy as np
import matplotlib.pyplot as plt
import json

class MNISTClassifier:
    def __init__(self):
        self.model_config = None
        self.weights = []
        self.biases = []
    
    def load_model(self, model_file):
        """Cargar el modelo desde el archivo JSON"""
        try:
            with open(model_file, 'r') as f:
                self.model_config = json.load(f)
            
            print(" Modelo cargado exitosamente")
            print(f"Forma de entrada: {self.model_config['input_shape']}")
            print(f"Preprocesamiento - escala: {self.model_config['preprocess']['scale']}")
            print(f"Número de capas: {len(self.model_config['layers'])}")
            
            # Extraer pesos y sesgos de cada capa
            for i, layer in enumerate(self.model_config['layers']):
                if 'W' in layer and 'b' in layer:
                    weights = np.array(layer['W'])
                    biases = np.array(layer['b'])
                    self.weights.append(weights)
                    self.biases.append(biases)
                    print(f"Capa {i+1}: {layer['units']} neuronas, activación: {layer['activation']}")
                    print(f"  - Pesos: {weights.shape}")
                    print(f"  - Sesgos: {biases.shape}")
            
        except Exception as e:
            print(f" Error cargando el modelo: {e}")
    
    def load_data(self, train_file="mnist_train.npz", test_file="mnist_test.npz"):
        """Cargar los datos MNIST"""
        try:
            # Cargar datos de entrenamiento
            train_data = np.load(train_file)
            self.train_images = train_data["images"]
            self.train_labels = train_data["labels"]
            
            # Cargar datos de prueba
            test_data = np.load(test_file)
            self.test_images = test_data["images"]
            self.test_labels = test_data["labels"]
            
            print(" Datos cargados exitosamente")
            print(f"Entrenamiento - Imágenes: {self.train_images.shape}, Etiquetas: {self.train_labels.shape}")
            print(f"Prueba - Imágenes: {self.test_images.shape}, Etiquetas: {self.test_labels.shape}")
            
        except Exception as e:
            print(f" Error cargando los datos: {e}")
    
    def preprocess_image(self, image):
        """Preprocesar una imagen para el modelo"""
       
        scale = self.model_config['preprocess']['scale']
        normalized = image.astype(np.float32) / scale
        
       
        flattened = normalized.reshape(-1)
        return flattened
    
    def relu(self, x):
        """Función de activación ReLU"""
        return np.maximum(0, x)
    
    def softmax(self, x):
        """Función de activación Softmax"""
        exp_x = np.exp(x - np.max(x))  # Estabilidad numérica
        return exp_x / np.sum(exp_x)
    
    def predict_single(self, image):
        """Hacer predicción para una sola imagen"""
    
        x = self.preprocess_image(image)
        
      
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
         
            x = np.dot(x, W) + b
            
       
            layer_config = self.model_config['layers'][i]
            if layer_config['activation'] == 'relu':
                x = self.relu(x)
            elif layer_config['activation'] == 'softmax':
                x = self.softmax(x)
        
        return x
    
    def predict_digit(self, image):
        """Predecir el dígito de una imagen"""
        probabilities = self.predict_single(image)
        predicted_digit = np.argmax(probabilities)
        confidence = probabilities[predicted_digit]
        return predicted_digit, confidence, probabilities
    
    def visualize_samples(self, images, labels, num_samples=9, title="Muestras MNIST"):
        """Visualizar muestras del dataset"""
        plt.figure(figsize=(10, 10))
        
        for i in range(num_samples):
            plt.subplot(3, 3, i+1)
            plt.imshow(images[i], cmap="gray")
            plt.title(f"Etiqueta: {labels[i]}")
            plt.axis("off")
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def visualize_test_grid(self, num_images=16):
        """Mostrar las primeras 16 imágenes de prueba en una cuadrícula 4x4"""
        plt.figure(figsize=(12, 12))
        
        for i in range(num_images):
            plt.subplot(4, 4, i+1)
            plt.imshow(self.test_images[i], cmap="gray")
            plt.title(f"Etiqueta real: {self.test_labels[i]}")
            plt.axis("off")
        
        plt.suptitle("Primeras 16 imágenes del conjunto de prueba", fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def test_predictions(self, num_samples=10):
        """Probar predicciones con muestras aleatorias"""
      
        indices = np.random.choice(len(self.test_images), num_samples, replace=False)
        
        plt.figure(figsize=(15, 8))
        
        for i, idx in enumerate(indices):
            image = self.test_images[idx]
            true_label = self.test_labels[idx]
            
           
            predicted_digit, confidence, probabilities = self.predict_digit(image)
            
           
            plt.subplot(2, 5, i+1)
            plt.imshow(image, cmap="gray")
            
            # Color del título: verde si correcto, rojo si incorrecto
            color = 'green' if predicted_digit == true_label else 'red'
            plt.title(f"Real: {true_label}\nPred: {predicted_digit}\nConf: {confidence:.3f}", 
                     color=color, fontsize=10)
            plt.axis("off")
        
        plt.suptitle("Predicciones del Modelo (Verde=Correcto, Rojo=Incorrecto)", fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def evaluate_accuracy(self, num_samples=1000):
        """Evaluar la precisión del modelo"""
        print(f"\n Evaluando modelo con {num_samples} muestras...")
        
        correct = 0
        indices = np.random.choice(len(self.test_images), num_samples, replace=False)
        
        for idx in indices:
            image = self.test_images[idx]
            true_label = self.test_labels[idx]
            predicted_digit, _, _ = self.predict_digit(image)
            
            if predicted_digit == true_label:
                correct += 1
        
        accuracy = correct / num_samples
        print(f" Precisión: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f" Predicciones correctas: {correct}/{num_samples}")
        
        return accuracy


def main():
    print(" Iniciando clasificador MNIST")
    print("=" * 50)
    
    # Crear instancia del clasificador
    classifier = MNISTClassifier()
    
   
    print("\n Paso 1: Cargando datos...")
    classifier.load_data("mnist_train.npz", "mnist_test.npz")
   
    print("\n Paso 2: Cargando modelo...")
    classifier.load_model("mnist_mlp_pretty")
    

    print("\n Paso 3: Visualizando muestras de entrenamiento...")
    classifier.visualize_samples(classifier.train_images, classifier.train_labels, 
                               title="Muestras del conjunto de entrenamiento")
    
    print("\n Paso 4: Ejercicio - Primeras 16 imágenes de prueba...")
    classifier.visualize_test_grid(16)
    
 
    print("\n Paso 5: Probando predicciones del modelo...")
    classifier.test_predictions(10)
    
    print("\n Paso 6: Evaluando precisión del modelo...")
    accuracy = classifier.evaluate_accuracy(1000)
    
    print(f"\n ¡Proceso completado! Precisión final: {accuracy*100:.2f}%")


if __name__ == "__main__":
    main()