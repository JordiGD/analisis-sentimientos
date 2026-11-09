"""
Configuración del proyecto de análisis de sentimientos en texto.

Aquí se definen las emociones, el sentimiento (polaridad) asociado a cada
una, las fuentes de datos y los parámetros del modelo.
"""

from pathlib import Path

# Carpetas del proyecto
RAIZ = Path(__file__).resolve().parent
CARPETA_DATOS = RAIZ / "datos"
CARPETA_PROCESADOS = CARPETA_DATOS / "procesados"
CARPETA_MODELO = RAIZ / "modelo"
CARPETA_RESULTADOS = RAIZ / "resultados"
RUTA_MODELO = CARPETA_MODELO / "modelo.joblib"
RUTA_META = CARPETA_MODELO / "meta.json"

# Emociones que reconoce el modelo (las 6 básicas de Ekman + neutral)
EMOCIONES = ["Alegría", "Tristeza", "Enojo", "Miedo", "Sorpresa", "Asco", "Neutral"]

# Traducción de las etiquetas originales de los datasets (en inglés)
TRADUCCION = {
    "joy": "Alegría",
    "sadness": "Tristeza",
    "anger": "Enojo",
    "fear": "Miedo",
    "surprise": "Sorpresa",
    "disgust": "Asco",
    "others": "Neutral",   # EmoEvent: "neutral u otras emociones"
    "neutral": "Neutral",
}

# Polaridad del sentimiento asociada a cada emoción.
# La sorpresa es ambigua (puede ser buena o mala), por eso cuenta como neutral.
SENTIMIENTO = {
    "Alegría": "Positivo",
    "Sorpresa": "Neutral",
    "Neutral": "Neutral",
    "Tristeza": "Negativo",
    "Enojo": "Negativo",
    "Miedo": "Negativo",
    "Asco": "Negativo",
}

EMOJIS = {
    "Alegría": "😄", "Tristeza": "😢", "Enojo": "😠", "Miedo": "😨",
    "Sorpresa": "😮", "Asco": "🤢", "Neutral": "😐",
    "Positivo": "👍", "Negativo": "👎",
}

# Datasets públicos en español (se descargan con preparar_datos.py)
URL_EMOEVENT = "https://raw.githubusercontent.com/fmplaza/EmoEvent/master/splits/es/{particion}.tsv"
# Corpus completo de EmoEvent: tiene las etiquetas correctas de todos los tuits
URL_EMOEVENT_COMPLETO = "https://raw.githubusercontent.com/fmplaza/EmoEvent/master/emoevent_es.csv"
URL_BRIGHTER = (
    "https://raw.githubusercontent.com/emotion-analysis-project/SemEval2025-task11/main/"
    "task-dataset/semeval-2025-task11-dataset/track_a/{particion}/esp.csv"
)

# Parámetros del modelo
MAX_PALABRAS = 50_000        # vocabulario de palabras y pares de palabras
MAX_CARACTERES = 100_000     # fragmentos de 2 a 5 letras
VALORES_C = [0.5, 1, 2, 4, 8]  # regularización de la regresión logística a probar
BALANCEO = 0.5               # 0 = sin compensar clases, 1 = balanceo completo
SEMILLA = 42


def sentimiento_de(emocion: str) -> str:
    return SENTIMIENTO.get(emocion, "Desconocido")
