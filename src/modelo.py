"""
Modelo de análisis de sentimientos: TF-IDF + Regresión Logística.

¿Cómo funciona?
1. Limpieza: el texto se normaliza (menciones, enlaces, emojis, risas...).
2. TF-IDF convierte el texto en un vector de números. Cada posición
   corresponde a una característica y su valor indica qué tan importante es
   en ese texto: alto si aparece en él y es poco común en el resto.
   Se usan dos tipos de características:
     - palabras y pares de palabras: "feliz", "qué miedo", "no_contento"...
     - fragmentos de 2 a 5 letras: "feli", "eliz"... toleran errores de
       ortografía y variantes ("felíz", "felicidad", "felices").
3. La regresión logística aprende un peso por característica y emoción, y
   devuelve la probabilidad de cada emoción (función softmax).
4. El sentimiento (positivo/negativo/neutral) se obtiene sumando las
   probabilidades de las emociones de cada tipo.
"""

import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    MAX_CARACTERES, MAX_PALABRAS, RUTA_META, RUTA_MODELO, SENTIMIENTO, sentimiento_de,
)
from src.texto import dividir_oraciones, limpiar_texto  # noqa: E402

# --------------------------------------------------------------------------- #
# Manejo de negaciones
# --------------------------------------------------------------------------- #
NEGACIONES = {"no", "nunca", "jamás", "jamas", "ni", "tampoco", "sin", "nadie", "ningún", "ninguna"}
_FIN_NEGACION = re.compile(r"^[.,;:!?¡¿()\"«»…]+$|^(pero|aunque|sino)$")


def marcar_negaciones(texto: str) -> str:
    """
    Las palabras que siguen a una negación (hasta el siguiente signo de
    puntuación o "pero") se marcan con "no_". Así el modelo distingue
    "contento" de "no ... contento":
        "no estoy nada contento ." -> "no no_estoy no_nada no_contento ."
    """
    salida, negado = [], False
    for palabra in texto.split():
        if _FIN_NEGACION.match(palabra):
            negado = False
            salida.append(palabra)
        elif palabra in NEGACIONES:
            negado = True
            salida.append(palabra)
        else:
            salida.append(f"no_{palabra}" if negado else palabra)
    return " ".join(salida)


# --------------------------------------------------------------------------- #
# Construcción del modelo
# --------------------------------------------------------------------------- #
def crear_modelo(C: float = 1.0, pesos_clase: dict | None = None) -> Pipeline:
    """Pipeline de scikit-learn: recibe texto ya limpio y devuelve la emoción."""
    caracteristicas = FeatureUnion([
        ("palabras", TfidfVectorizer(
            analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2),
            preprocessor=marcar_negaciones,
            min_df=2, max_features=MAX_PALABRAS, sublinear_tf=True,
        )),
        ("caracteres", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5),
            min_df=2, max_features=MAX_CARACTERES, sublinear_tf=True,
        )),
    ])
    clasificador = LogisticRegression(C=C, max_iter=2000, class_weight=pesos_clase)
    return Pipeline([("tfidf", caracteristicas), ("clasificador", clasificador)])


def palabras_clave(modelo: Pipeline, n: int = 12) -> dict[str, list[tuple[str, float]]]:
    """Palabras (o pares de palabras) con más peso a favor de cada emoción."""
    vocabulario = modelo.named_steps["tfidf"].transformer_list[0][1].get_feature_names_out()
    coeficientes = modelo.named_steps["clasificador"].coef_[:, : len(vocabulario)]
    resultado = {}
    for i, emocion in enumerate(modelo.classes_):
        mejores = np.argsort(coeficientes[i])[::-1]
        lista = [(vocabulario[j], float(coeficientes[i, j])) for j in mejores
                 if "@usuario" not in vocabulario[j] and "url" not in vocabulario[j].split()][:n]
        resultado[emocion] = lista
    return resultado


# --------------------------------------------------------------------------- #
# Uso del modelo entrenado
# --------------------------------------------------------------------------- #
class AnalizadorSentimiento:
    """
    Carga el modelo entrenado y analiza textos.

        analizador = AnalizadorSentimiento()
        r = analizador.analizar("¡Por fin aprobé el examen!")
        print(r["emocion"], r["confianza"], r["sentimiento"])
    """

    def __init__(self, ruta_modelo: str | Path = RUTA_MODELO):
        ruta_modelo = Path(ruta_modelo)
        if not ruta_modelo.is_file():
            raise FileNotFoundError(
                f"No se encontró el modelo en {ruta_modelo}. "
                "Entrénalo con: python preparar_datos.py && python entrenar.py"
            )
        self.modelo: Pipeline = joblib.load(ruta_modelo)
        self.etiquetas: list[str] = list(self.modelo.classes_)
        ruta_meta = ruta_modelo.parent / RUTA_META.name
        self.metadatos = json.loads(ruta_meta.read_text(encoding="utf-8")) if ruta_meta.is_file() else {}

    def probabilidades(self, textos: list[str]) -> np.ndarray:
        """Matriz (n_textos x n_emociones) con la probabilidad de cada emoción."""
        limpios = [limpiar_texto(t) or "." for t in textos]
        return self.modelo.predict_proba(limpios)

    def _resultado(self, texto: str, probs: np.ndarray) -> dict:
        i = int(np.argmax(probs))
        emocion = self.etiquetas[i]
        por_sentimiento = {"Positivo": 0.0, "Negativo": 0.0, "Neutral": 0.0}
        for e, p in zip(self.etiquetas, probs):
            if SENTIMIENTO.get(e) in por_sentimiento:
                por_sentimiento[SENTIMIENTO[e]] += float(p)
        return {
            "texto": texto,
            "emocion": emocion,
            "confianza": float(probs[i]),
            "sentimiento_emocion": sentimiento_de(emocion),
            # Sentimiento global: el tipo cuyas emociones suman más probabilidad
            "sentimiento": max(por_sentimiento, key=por_sentimiento.get),
            "prob_sentimiento": por_sentimiento,
            "probabilidades": {e: float(p) for e, p in zip(self.etiquetas, probs)},
        }

    def analizar(self, texto: str) -> dict:
        """Emoción principal, confianza, sentimiento y probabilidades de un texto."""
        if not str(texto).strip():
            raise ValueError("El texto está vacío.")
        return self._resultado(texto, self.probabilidades([texto])[0])

    def analizar_lote(self, textos: list[str]) -> list[dict]:
        textos = [str(t) for t in textos]
        return [self._resultado(t, p) for t, p in zip(textos, self.probabilidades(textos))]

    def analizar_por_oraciones(self, texto: str) -> list[dict]:
        """Divide un párrafo en oraciones y analiza cada una por separado."""
        return self.analizar_lote(dividir_oraciones(texto) or [texto])

    def explicar(self, texto: str, emocion: str | None = None, max_palabras: int = 60):
        """
        Qué palabras influyeron en la predicción (técnica de "oclusión"): se
        quita cada palabra y se mide cuánto baja (o sube) la probabilidad de la
        emoción. Peso positivo = la palabra apoya esa emoción.
        """
        palabras = re.findall(r"\S+", str(texto))[:max_palabras]
        if not palabras:
            return []
        variantes = [" ".join(palabras)] + [
            " ".join(palabras[:i] + palabras[i + 1:]) for i in range(len(palabras))
        ]
        probs = self.probabilidades(variantes)
        j = self.etiquetas.index(emocion) if emocion else int(np.argmax(probs[0]))
        return [(p, float(probs[0, j] - probs[i + 1, j])) for i, p in enumerate(palabras)]
