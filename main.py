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

def tokenizar(texto: str) -> List[str]:
    if not texto:
        return []
    return texto.split()

def remover_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in stopwords and len(t) > 1]

def preprocesamiento(texto: str) -> List[str]:
    return remover_stopwords(tokenizar(normalizar_texto(texto)))