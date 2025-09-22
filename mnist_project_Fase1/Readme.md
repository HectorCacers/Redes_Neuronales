#  MNIST - Entrenamiento y Evaluación de Red Neuronal

## Descripción
Proyecto completo de red neuronal para clasificación de dígitos handwritten, incluyendo entrenamiento, monitoreo y evaluación.

## 📁 Archivos y su propósito:
- `mnist_mlp_pretty.json` - Modelo pre-entrenado (arquitectura)
- `mnist_train.npz` - Dataset de entrenamiento (60,000 imágenes)
- `mnist_test.npz` - Dataset de prueba (10,000 imágenes)
- `parte3_evaluar_pretrained.py` - **Parte 3: Evaluación del modelo**
- `parte4_monitoreo_entrenamiento.py` - **Parte 4: Entrenamiento y debugging**

##  Partes del Proyecto:

### Parte 3: Evaluación del Modelo
- Carga del modelo pre-entrenado
- Evaluación de accuracy en test set
- Análisis de métricas de performance
- Matriz de confusión

### Parte 4: Entrenamiento y Debugging
- **Training:** Monitorización de loss durante entrenamiento
- **Learning rates:** Experimentación con tasas de aprendizaje (0.01-0.001)
- **Optimizers:** Comparación de diferentes optimizadores (Adam, SGD, etc.)
- **Mini-batch training:** Implementación para grandes datasets
- **Debugging:** 
  - Acceso a gradientes después de backward()
  - Detección de vanishing/exploding gradients
  - Verificación de disminución de loss
  - Validación de predicciones

## Como Ejecutar:
```bash
# Evaluar modelo pre-entrenado (Parte 3)
$run parte3_evaluar_pretrained.py

# Entrenar y monitorear modelo (Parte 4) 
$run parte4_monitoreo_entrenamiento.py
