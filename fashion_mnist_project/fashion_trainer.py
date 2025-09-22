import DnnLib
import numpy as np

layer = DnnLib.DenseLayer(10, 5, DnnLib.ActivationType.RELU)
print("Métodos de DenseLayer:")
for method in dir(layer):
    if not method.startswith('_'):
        print(f"  {method}")


x = np.random.randn(3, 10).astype(np.float64)
print(f"\nInput shape: {x.shape}")
try:
    output = layer.forward(x)
    print(f"Forward pass exitoso, output shape: {output.shape}")
except Exception as e:
    print(f"Error en forward: {e}")

if hasattr(layer, 'backward'):
    print("✓ backward() existe")
else:
    print("✗ backward() NO existe")