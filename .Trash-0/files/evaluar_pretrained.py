import DnnLib
import numpy as np
import json
import matplotlib.pyplot as plt

class MNISTPretrainedEvaluator:
    def __init__(self):
        self.layers = []
        self.class_names = [str(i) for i in range(10)]
    
    def load_pretrained_model(self, model_path):
        """Cargar modelo pre-entrenado del Ing. Deras"""
        with open(model_path, 'r') as f:
            model_config = json.load(f)
        
        print("📊 Arquitectura del modelo pre-entrenado:")
        print(f"   Input shape: {model_config['input_shape']}")
        print(f"   Preprocess: scale={model_config['preprocess']['scale']}")
        
        # Crear capas según la arquitectura del JSON
        for i, layer_config in enumerate(model_config['layers']):
            units = layer_config['units']
            activation = layer_config['activation']
            
            # Mapear activación a enum de DnnLib
            if activation == 'relu':
                act_type = DnnLib.ActivationType.RELU
            elif activation == 'softmax':
                act_type = DnnLib.ActivationType.SOFTMAX
            else:
                act_type = DnnLib.ActivationType.RELU
            
            # Crear capa con las dimensiones CORRECTAS
            if i == 0:
                # Primera capa: 784 inputs (28*28)
                layer = DnnLib.DenseLayer(784, units, act_type)
            else:
                # Capas siguientes
                prev_units = model_config['layers'][i-1]['units']
                layer = DnnLib.DenseLayer(prev_units, units, act_type)
            
            # CARGAR PESOS CORRECTAMENTE (transponer si es necesario)
            weights = np.array(layer_config['W'])
            bias = np.array(layer_config['b'])
            
            # Verificar y ajustar forma de los pesos
            if weights.shape == (units, layer.weights.shape[1]):
                # Los pesos ya están en la forma correcta (output_dim, input_dim)
                layer.weights = weights
            else:
                # Transponer pesos a la forma que espera DnnLib (input_dim, output_dim)
                layer.weights = weights.T
            
            layer.bias = bias
            
            self.layers.append(layer)
            print(f"   Capa {i+1}: {units} neuronas, activación: {activation}")
            print(f"     Pesos: {layer.weights.shape}, Sesgos: {layer.bias.shape}")
        
        print("✅ Modelo pre-entrenado cargado exitosamente")
    
    def load_test_data(self):
        """Cargar datos de prueba MNIST"""
        test_data = np.load("mnist_test.npz")
        self.test_images = test_data["images"]
        self.test_labels = test_data["labels"]
        
        # Preprocesar correctamente
        self.X_test = self.test_images.reshape(-1, 784).astype(np.float64) / 255.0
        
        print(f"✅ Datos de prueba cargados: {self.test_images.shape}")
        print(f"   X_test shape: {self.X_test.shape}")
    
    def evaluate_accuracy(self):
        """Evaluar precisión del modelo pre-entrenado"""
        print("\n🧮 Evaluando precisión del modelo pre-entrenado...")
        
        try:
            # Forward pass
            activations = [self.X_test]
            for i, layer in enumerate(self.layers):
                print(f"   Capa {i+1} - Input: {activations[-1].shape}")
                activations.append(layer.forward(activations[-1]))
                print(f"   Capa {i+1} - Output: {activations[-1].shape}")
            
            output = activations[-1]
            predictions = np.argmax(output, axis=1)
            
            accuracy = np.mean(predictions == self.test_labels)
            print(f"🎯 PRECISIÓN DEL MODELO PRE-ENTRENADO: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            return accuracy
            
        except Exception as e:
            print(f"❌ Error durante forward pass: {e}")
            print("🔍 Debug info:")
            for i, layer in enumerate(self.layers):
                print(f"   Capa {i+1}: weights {layer.weights.shape}, bias {layer.bias.shape}")
            raise
    
    def visualize_predictions(self, num_samples=12):
        """Visualizar predicciones del modelo"""
        indices = np.random.choice(len(self.X_test), num_samples, replace=False)
        
        plt.figure(figsize=(15, 10))
        for i, idx in enumerate(indices):
            image = self.test_images[idx]
            true_label = self.test_labels[idx]
            
            # Predecir muestra individual
            sample = self.X_test[idx:idx+1]
            
            activations = [sample]
            for layer in self.layers:
                activations.append(layer.forward(activations[-1]))
            
            output = activations[-1][0]
            predicted_label = np.argmax(output)
            confidence = output[predicted_label]
            
            plt.subplot(3, 4, i+1)
            plt.imshow(image, cmap="gray")
            color = 'green' if predicted_label == true_label else 'red'
            plt.title(f"Real: {true_label} | Pred: {predicted_label}\nConf: {confidence:.3f}", 
                     color=color, fontsize=10)
            plt.axis("off")
        
        plt.suptitle("Predicciones del Modelo Pre-entrenado (Ing. Deras)", fontsize=16)
        plt.tight_layout()
        plt.show()

def main():
    print("🔍 PARTE 3: EVALUACIÓN MODELO PRE-ENTRENADO")
    print("=" * 55)
    
    evaluator = MNISTPretrainedEvaluator()
    
    try:
        # 1. Cargar modelo pre-entrenado
        evaluator.load_pretrained_model("mnist_mlp_pretty.json")
        
        # 2. Cargar datos de prueba
        evaluator.load_test_data()
        
        # 3. Evaluar precisión (debe ser >90%)
        accuracy = evaluator.evaluate_accuracy()
        
        # 4. Visualizar predicciones
        evaluator.visualize_predictions()
        
        print(f"\n✅ Parte 3 completada. Precisión: {accuracy*100:.2f}%")
        
    except Exception as e:
        print(f"❌ Error crítico: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()