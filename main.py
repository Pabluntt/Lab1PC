#import extraccion_dataset
import math as math
import re as re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import funciones as fn
import textos_propios as txtp

# Los textos propios fueron creados a base de IA, y en inglés para poder ser consistentes con el dataset y realizar una correcta comparación.

opcion = 0
# variables para modelos en memoria
modelo_entrenado = None   # dict devuelto por funciones.calcular_tfidf_por_subcategoria()
modelo_cargado = None     # dict cargado desde datos.txt

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
        print("Entrenando modelo (calculando TF-IDF por subcategoría)...")
        modelo_entrenado = fn.calcular_tfidf_por_subcategoria("dataset")
        total = sum(len(info.get("paths", [])) for info in modelo_entrenado.values()) if modelo_entrenado else 0
        print(f"Entrenamiento completado. Categorías: {len(modelo_entrenado)}. Documentos totales: {total}")

    elif (opcion==2):
        print("Cargando modelo desde 'datos.txt'...")
        modelo_cargado = fn.cargar_modelo_txt("datos.txt")
        if modelo_cargado:
            print("Modelo cargado. Resumen por categoría:")
            for cat, info in modelo_cargado.items():
                print(f"{cat}: n_docs={info.get('n_docs',0)} tokens_centroid={len(info.get('centroid',{}))}")
        else:
            print("No se encontró o no se pudo cargar 'datos.txt'.")

    elif (opcion==3):
        if modelo_entrenado is None:
            print("No hay modelo entrenado en memoria. Ejecute opción 1 primero.")
        else:
            ruta = fn.guardar_modelo_txt(modelo_entrenado, "datos.txt")
            print(f"Modelo guardado en {ruta}")

    elif(opcion==4):
        print(f"Soy la opion {opcion}")
