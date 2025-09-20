#!/usr/bin/env python3
"""
PROYECTO REDES NEURONALES - MNIST
Archivo principal que replica la presentación del instructor

Uso:
    python main_proyecto.py [opcion]
    
Opciones:
    --parte3      : Solo entrenamiento básico (como mostró el instructor)
    --parte4      : Solo comparación de optimizadores
    --completo    : Ambas partes (por defecto)
    --ayuda       : Mostrar esta ayuda
"""

import sys
import os
import time
import numpy as np
from datetime import datetime

# Importar nuestros módulos
try:
    import parte3_entrenamiento as parte3
    import parte4_comparacion as parte4
    import DnnLib
except ImportError as e:
    print(f"Error al importar: {e}")
    print("Verificar que estén disponibles:")
    print("- parte3_entrenamiento.py")
    print("- parte4_comparacion.py") 
    print("- DnnLib")
    sys.exit(1)

def print_header():
    """Cabecera del proyecto"""
    print("=" * 60)
    print("    PROYECTO REDES NEURONALES - CLASIFICACIÓN MNIST")
    print("      Replicando la presentación del instructor")
    print("=" * 60)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def check_requirements():
    """Verificar archivos necesarios"""
    required_files = ["mnist_train.npz", "mnist_test.npz"]
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("ERROR: Archivos faltantes:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    print("Archivos de datos: OK")
    
    # Verificar DnnLib
    try:
        import DnnLib
        print("DnnLib: OK")
    except ImportError:
        print("ERROR: DnnLib no disponible")
        return False
    
    return True

def run_parte3():
    """Ejecutar Parte 3: Entrenamiento básico"""
    print("\n" + "=" * 40)
    print("   PARTE 3: ENTRENAMIENTO BÁSICO")
    print("=" * 40)
    print("Replicando el entrenamiento mostrado en clase...")
    
    try:
        parte3.main()
        print("\nParte 3 completada exitosamente")
        return True
    except Exception as e:
        print(f"Error en Parte 3: {e}")
        return False

def run_parte4():
    """Ejecutar Parte 4: Comparación de optimizadores"""
    print("\n" + "=" * 40)
    print("  PARTE 4: COMPARACIÓN DE OPTIMIZADORES")
    print("=" * 40)
    print("Comparando SGD, SGD+Momentum, RMSprop, y Adam...")
    
    try:
        results = parte4.main()
        print("\nParte 4 completada exitosamente")
        return True, results
    except Exception as e:
        print(f"Error en Parte 4: {e}")
        return False, None

def show_help():
    """Mostrar ayuda"""
    print(__doc__)

def demonstrate_dnnlib():
    """Demostración rápida de DnnLib como en clase"""
    print("\nDemostración DnnLib:")
    print("-" * 30)
    
    # Crear ejemplo simple como mostró el instructor
    out_logits = np.array([1, 2, 3, 3.2, 3.3, 4, 2.4, 1.7])
    out_probs = DnnLib.softmax(out_logits)
    
    print(">>> import DnnLib")
    print(f">>> out_logits = {list(out_logits)}")
    print(">>> out_probs = DnnLib.softmax(out_logits)")
    print(">>> out_probs")
    print(f"array({out_probs})")
    print(">>> sum(out_probs)")
    print(f"np.float32({np.sum(out_probs)})")

def main():
    """Función principal"""
    print_header()
    
    # Procesar argumentos
    if len(sys.argv) > 1:
        option = sys.argv[1].lower()
    else:
        option = "--completo"
    
    if option in ["--ayuda", "--help", "-h"]:
        show_help()
        return
    
    # Verificar requisitos
    print("Verificando requisitos...")
    if not check_requirements():
        sys.exit(1)
    
    print("Todos los requisitos OK\n")
    
    # Demostración DnnLib como en clase
    import numpy as np
    demonstrate_dnnlib()
    
    # Ejecutar según opción
    if option == "--parte3":
        print("\nModo: Solo Parte 3 (Entrenamiento básico)")
        if not run_parte3():
            sys.exit(1)
            
    elif option == "--parte4":
        print("\nModo: Solo Parte 4 (Comparación optimizadores)")
        success, results = run_parte4()
        if not success:
            sys.exit(1)
            
    elif option == "--completo":
        print("\nModo: Proyecto completo (Parte 3 + Parte 4)")
        
        # Ejecutar Parte 3
        if not run_parte3():
            sys.exit(1)
        
        # Pausa antes de Parte 4
        print("\nPresiona Enter para continuar con Parte 4...")
        input()
        
        # Ejecutar Parte 4
        success, results = run_parte4()
        if not success:
            sys.exit(1)
    
    else:
        print(f"Opción desconocida: {option}")
        show_help()
        sys.exit(1)
    
    # Mensaje final
    print("\n" + "=" * 60)
    print("        PROYECTO COMPLETADO EXITOSAMENTE")
    print("=" * 60)
    print("Archivos generados:")
    if os.path.exists("mnist_mlp_own_model.json"):
        print("  ✓ mnist_mlp_own_model.json - Modelo básico (Parte 3)")
    if os.path.exists("best_optimizer_model.json"):
        print("  ✓ best_optimizer_model.json - Mejor optimizador (Parte 4)")
    
    print(f"\nTiempo: {datetime.now().strftime('%H:%M:%S')}")
    print("Proyecto terminado correctamente.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEjecución interrumpida")
        sys.exit(0)
    except Exception as e:
        print(f"\nError inesperado: {e}")
        sys.exit(1)