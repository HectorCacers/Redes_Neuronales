# Redes Neuronales - Proyectos de Clasificación

Repositorio con implementaciones completas de redes neuronales para clasificación de imágenes.

## ⚠️ IMPORTANTE: Archivos grandes excluidos
Los archivos `.npz` (datasets) han sido excluidos del repositorio debido a su tamaño grande. Intente y me decia que sobrepasa lo permitido.

## 📂 Estructura del repositorio:

###  [MNIST_Pre-Trained_Model_Fase1](/MNIST_Pre-Trained_Model_Fase1/)
**Clasificación de dígitos handwritten** - Modelo pre-entrenado y sistema completo de evaluación y debugging.

### 📁 [fashion_mnist_project](/fashion_mnist_project/)  
**Clasificación de prendas de vestir** - Alcanzando **88.41% accuracy** con técnicas avanzadas de regularización.

### 📁 NO mnist_project (NO ABRIR)
Carpeta intermedia - Contiene archivos duplicados o en proceso. Usar las carpetas específicas de cada proyecto.

## 🚀 Cómo usar los proyectos:

### Para MNIST:
1. Descargar MNIST dataset
2. Colocar los datos en `MNIST_Pre-Trained_Model_Fase1/`
3. Ejecutar: `python parte3_evaluar_pretrained.py`

### Para Fashion MNIST:
1. Descargar Fashion MNIST
2. Colocar los datos en `fashion_mnist_project/`
3. Ejecutar: `python main_optimized.py`


## Nota sobre archivos grandes:
Los archivos `.npz` originales no están incluidos en el repositorio. Debes:
1. Descargar los datasets desde las fuentes oficiales
2. Ejecutar los scripts de preprocesamiento incluidos
3. Los modelos pre-entrenados SÍ están incluidos (.json, .joblib)

---

 **Autor:** Hector Caceres  
 **Contacto:** [GitHub Profile](https://github.com/HectorCacers)

⭐ ¡Si este repositorio te resulta útil, déjale una estrella!
