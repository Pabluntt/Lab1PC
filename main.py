#import extraccion_dataset
import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import funciones as fn
import textos_propios

# Los textos propios fueron creados a base de IA, y en inglés para poder ser consistentes con el dataset y realizar una correcta comparación.

opcion = 0
while opcion != 5:
    print("""
          1.- Entrenar el modelo con los datos del data set
          2.- Cargar el modelo previamente entrenado
          3.- Guardar el nuevo modelo entrenado
          4.- Predecir la categoria de un texto con el modelo entrenado
          5.- Salir
            """)
    try:
        opcion = int(input("Seleccione una opción (1-5): ").strip())
    except ValueError:
        print("Opción inválida, ingrese un número entre 1 y 5.")
        continue

    if (opcion==1):
        print(f"Soy la opcion {opcion}")
        resultados = fn.calcular_tfidf_por_subcategoria()
        print(resultados["sci.space"])
    elif (opcion==2):
        print(f"Soy la opcion {opcion}")

    elif (opcion==3):
        print(f"Soy la opcion {opcion}")

    elif(opcion==4):
        print(f"Soy la opcion {opcion}")
