import extraccion_dataset
import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

#son algunas nomas, ya que podemos ampliar esto de ser necesario xd
stopwords = {
    'de','la','que','el','en','y','a','los','del','se','las','por','un','para',
    'con','no','una','su','al','lo','como','más','pero','sus','le','ya','o',
    'este','sí','porque','esta','entre','cuando','muy','sin','sobre','también',
    'me','hasta','hay','donde','quien','desde','todo','nos','durante','todos',
    'uno','les','ni','contra','otros','ese','eso','ante','ellos','e'
}

def normalizar_texto(texto: str) -> str:
    texto = texto.lower()
    # quitamos tildes y signos
    reemplazos = {
        'á':'a','é':'e','í':'i','ó':'o','ú':'u','ü':'u','ñ':'n'
    }
    for a,b in reemplazos.items():
        texto = texto.replace(a,b)
    # mantener letras y números y espacios
    texto = re.sub(r'[^a-z0-9\s]', ' ', texto)
    # reducir espacios
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

# Separa el texto en palabras individuales
def tokenizar(texto: str) -> List[str]:
    if not texto:
        return []
    return texto.split()

# Elimina stopwords
def remover_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in stopwords and len(t) > 1]

# Ejecuta el preprocesamiento, quitando stopwords, tokenizando y normalizando
def preprocesamiento(texto: str) -> List[str]:
    return remover_stopwords(tokenizar(normalizar_texto(texto)))

# Calcula la frecuencia por termino, devuelve un diccionario con cada string y su frecuencia
def calcular_tf(tokens: List[str]) -> Dict[str, float]:
    cnt = Counter(tokens)
    n = len(tokens) or 1
    return {t: c / n for t, c in cnt.items()}

# Calcula el IDF de cada término
def calcular_idf(corpus_tokens: List[List[str]]) -> Dict[str, float]:
    N = len(corpus_tokens)
    df = Counter()
    for tokens in corpus_tokens:
        df.update(set(tokens))
    return {t: math.log((N + 1) / (df_t + 1)) + 1 for t, df_t in df.items()}

# Calcula el vector, recibiendo los tokens y el diccionario con los IDF
def vector_tfidf(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf = calcular_tf(tokens)
    return {t: tf_val * idf.get(t, 0.0) for t, tf_val in tf.items()}

# Calcula el módulo de un vector
def _l2(v: Dict[str, float]) -> float:
    return math.sqrt(sum(val * val for val in v.values()))

#Calcula  el producto punto entre 2 vectores
def _dot(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    if len(v1) > len(v2):
        v1, v2 = v2, v1
    return sum(val * v2.get(term, 0.0) for term, val in v1.items())

# Calcula la similitud de coseno entre 2 vectores de los textos, utilizando el producto punto y los modulos
def similitud_coseno(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    num = _dot(vec1, vec2)
    den = _l2(vec1) * _l2(vec2)
    return num / den if den else 0.0

#Calcula la similitud de coseno entre 2 textos, preprocesandolos y calculando los vectores TF-IDF
def similitud_coseno_textos(t1: str, t2: str, idf: Dict[str, float]) -> float:
    tokens1 = preprocesamiento(t1)
    tokens2 = preprocesamiento(t2)
    vec1 = vector_tfidf(tokens1, idf)
    vec2 = vector_tfidf(tokens2, idf)
    return similitud_coseno(vec1, vec2)



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
        print(f"Soy la opion {opcion}")
    elif (opcion==2):
        print(f"Soy la opion {opcion}")

    elif (opcion==3):
        print(f"Soy la opion {opcion}")

    elif(opcion==4):
        print(f"Soy la opion {opcion}")
