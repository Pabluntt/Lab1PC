#import extraccion_dataset
import math
import re
import json
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
from collections import Counter

# Algunas Stopwords
# Se incluyen también en inglés ya que los textos estan en inglés
stopwords = {
    'de','la','que','el','en','y','a','los','del','se','las','por','un','para',
    'con','no','una','su','al','lo','como','más','pero','sus','le','ya','o',
    'este','sí','porque','esta','entre','cuando','muy','sin','sobre','también',
    'me','hasta','hay','donde','quien','desde','todo','nos','durante','todos',
    'uno','les','ni','contra','otros','ese','eso','ante','ellos','e', 'a','an','the','and','or','but','if',
    'then','else','for','to','from','of','in','on','at','by','with','about',
    'as','into','through','over','after','before','between','under','again','further','once','here','there',
    'all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same',
    'so','than','too','very','can','will','just','also','this','that','these','those',
    'i','you','he','she','it','we','they','me','him','her','us','them','my','your','his','its','our','their',
    'myself','yourself','himself','herself','itself','ourselves','yourselves','themselves',
    'am','is','are','was','were','be','been','being','do','does','did','doing','have','has','had','having',
    'im','ive','id','youre','youve','youd','hes','shes','its','were','weve','wed','theyre','theyve','theyd',
    'cant','couldnt','dont','doesnt','didnt','hasnt','havent','hadnt','isnt','arent','wasnt','werent','wont',
    'wouldnt','shouldnt','mightnt','mustnt','neednt','shan',
    're','ve','ll','d','s','m','don','t','doesn','didn','hasn','haven','hadn','isn','aren','wasn','weren','won',
    'wouldn','shouldn','mightn','mustn','needn','ma'
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
    # Conteo manual: por cada documento, considerar términos únicos y aumentar su document frequency en 1
    df = {}  # tipo: Dict[str, int]
    for tokens in corpus_tokens:
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1

    return {t: math.log((N) / (df_t + 1)) for t, df_t in df.items()}

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

def calcular_tfidf_por_subcategoria(dataset_dir: str = "dataset", min_df: int = 1):
    
    #Recorre cada subcarpeta de dataset_dir y calcula manualmente:
    #  - idf: mapping token -> idf_value (calculado por conteo de documentos)
    #  - paths: lista de archivos
    #  - matrix: lista de dicts (tf-idf por documento) -> cada elemento es {token: tfidf_val, ...}
    
    import os
    import glob

    results = {}
    if not os.path.isdir(dataset_dir):
        return results

    for categoria in sorted(os.listdir(dataset_dir)):
        cat_path = os.path.join(dataset_dir, categoria)
        if not os.path.isdir(cat_path):
            continue

        paths = sorted(glob.glob(os.path.join(cat_path, "*.txt")))
        corpus_tokens = []  # lista de listas de tokens por documento
        textos_raw = []

        for p in paths:
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
            except Exception:
                txt = ""
            textos_raw.append(txt)
            corpus_tokens.append(preprocesamiento(txt))

        # filtrar documentos vacíos
        non_empty = [toks for toks in corpus_tokens if toks]
        if not non_empty:
            results[categoria] = {"vectorizer": None, "matrix": [], "paths": paths, "idf": {}}
            continue

        # calcular idf usando la función existente (conteo de documentos donde aparece el término)
        idf_map = calcular_idf(corpus_tokens)

        # calcular tf-idf por documento (lista de dicts)
        tfidf_docs = []
        for toks in corpus_tokens:
            if not toks:
                tfidf_docs.append({})
                continue
            tfidf_docs.append(vector_tfidf(toks, idf_map))

        results[categoria] = {
            "vectorizer": None,
            "matrix": tfidf_docs,   # lista de dicts tf-idf por documento
            "paths": paths,
            "idf": idf_map
        }

    return results

def guardar_modelo_txt(model_dict: dict, filepath: str = "datos.txt"):
    
    #Guarda una versión serializable del modelo por subcategoría en 'filepath'.
    #Ahora solo se almacena por categoría:
    #  - idf: mapping token -> idf_value
    #  - paths: lista de archivos
    #  - n_docs: número de documentos

    serial = {}
    for categoria, info in (model_dict or {}).items():
        # preferir idf ya calculado; si no existe, obtenerlo desde el vectorizer
        idf_map = info.get("idf")
        if not idf_map:
            vec = info.get("vectorizer")
            if vec is not None:
                feats = vec.get_feature_names_out()
                idf_values = getattr(vec, "idf_", None)
                if idf_values is not None:
                    idf_map = {feats[i]: float(idf_values[i]) for i in range(len(feats))}
                else:
                    idf_map = {}
            else:
                idf_map = {}

        paths = info.get("paths", []) or []
        serial[categoria] = {
            "idf": idf_map,
            "paths": paths,
            "n_docs": len(paths)
        }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serial, f, ensure_ascii=False, indent=2)
    return filepath

def cargar_modelo_txt(filepath: str = "datos.txt") -> dict:
    
    #Lee el JSON guardado en 'filepath' y lo devuelve como dict.
    #Estructura esperada por categoría:
    #  { "idf": {token: idf_val, ...}, "paths": [...], "n_docs": N }
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        # garantizar formato mínimo
        if not isinstance(data, dict):
            return {}
        return data
    except FileNotFoundError:
        return {}
    except Exception:
        return {}

def predict_category_from_model(query_text: str, model_dict: dict, top_k: int = 5):
    
    #Dado un texto de consulta y un modelo TF-IDF por subcategoría,
    #calcula la categoría predicha aplicando los 5 pasos del procedimiento:
    
    #1. Preprocesa la consulta y calcula su vector TF-IDF.
    #2. Calcula la similitud de coseno entre la consulta y todos los documentos del modelo.
    #3. Ordena los documentos por similitud descendente.
    #4. Selecciona los K documentos más similares.
    #5. Aplica una votación mayoritaria entre las categorías de los K vecinos.
    
    #Retorna:
    #  (categoria_predicha, vecinos)
    #  donde vecinos es una lista [(categoria, path, similitud)] ordenada por similitud.
    
    if not query_text or not model_dict:
        return None, []

    q_tokens = preprocesamiento(query_text)
    if not q_tokens:
        return None, []

    # Lista global de todos los documentos comparados
    all_docs = []  # [(categoria, path, similitud)]

    for cat, info in model_dict.items():
        idf_map = info.get("idf", {}) or {}
        paths = info.get("paths", []) or []
        matrix = info.get("matrix")

        # Construir los vectores TF-IDF de cada documento (si no hay en memoria)
        docs_tfidf = []
        if matrix:
            docs_tfidf = matrix
        else:
            for p in paths:
                try:
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        txt = f.read()
                except Exception:
                    txt = ""
                toks = preprocesamiento(txt)
                docs_tfidf.append(vector_tfidf(toks, idf_map) if toks else {})

        # Calcular vector de la consulta (usando el IDF de la categoría)
        query_vec = vector_tfidf(q_tokens, idf_map)

        # Calcular similitud con cada documento
        for i, doc_vec in enumerate(docs_tfidf):
            if not doc_vec:
                continue
            sim = similitud_coseno(query_vec, doc_vec)
            doc_path = paths[i] if i < len(paths) else f"{cat}_doc{i}"
            all_docs.append((cat, doc_path, sim))

    # Si no hay documentos válidos
    if not all_docs:
        return None, []

    # Ordenar por similitud descendente
    all_docs.sort(key=lambda x: x[2], reverse=True)

    # Seleccionar los K documentos más similares
    vecinos = all_docs[:top_k]

    # Votación mayoritaria
    categorias = [c for c, _, _ in vecinos]
    conteo = Counter(categorias)
    categoria_predicha = conteo.most_common(1)[0][0]

    return categoria_predicha, vecinos