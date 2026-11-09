"""
Limpieza y normalización de texto.

Los textos de redes sociales tienen enlaces, menciones, emojis, risas
("jajajaja") y letras repetidas ("siiiii"). Este módulo los deja en una forma
uniforme para que el modelo vea siempre el mismo tipo de entrada, tanto al
entrenar como al predecir.

Ejemplo:
    >>> limpiar_texto("@ana ¡¡No puedo creerlo!!! 😭😭 https://t.co/x #triste")
    '@usuario ¡ ¡ no puedo creerlo ! ! emoji_cara_llorando_fuerte emoji_cara_llorando_fuerte url triste'
"""

import html
import re

import emoji

_URL = re.compile(r"https?://\S+|www\.\S+|\bURL\b")
_MENCION = re.compile(r"@<username>|@\w+|\bUSER\b")
_HASHTAG_ANONIMO = re.compile(r"\bHASHTAG\b")       # así vienen en EmoEvent
_HASHTAG = re.compile(r"#(\w+)")
_RISA = re.compile(
    r"\b(?:a?j+[aeiou]+){2,}j?\b|\b(?:h+a+){2,}h?\b|\bj+s+j[js]*\b|\bxd+\b", re.IGNORECASE
)
_REPETIDAS = re.compile(r"(\w)\1{2,}")
_SIGNOS_REPETIDOS = re.compile(r"([¡!¿?.])\1{2,}")
_PUNTUACION = re.compile(r"([¡!¿?.,;:()\"«»…])")
_ESPACIOS = re.compile(r"\s+")
# Emoticones de texto -> palabras (si no, ":(" se partiría en ":" y "(")
_EMOTICONES = [
    (re.compile(r"(?<!\S)[:;=]-?[)\]]+|(?<!\S)\(+-?[:;=]"), " emoticono_feliz "),
    (re.compile(r"(?<!\S)[:;=]-?D+\b"), " emoticono_risa "),
    (re.compile(r"(?<!\S)[:;=]'?-?[(\[]+|(?<!\S)\)+-?[:;=]"), " emoticono_triste "),
    (re.compile(r"(?<!\S)[:;=]-?[pP]\b"), " emoticono_burla "),
    (re.compile(r"(?<!\S)[:;=]-?[oO0]\b"), " emoticono_sorpresa "),
    (re.compile(r"<3+"), " emoticono_corazon "),
]
_ORACIONES = re.compile(r"(?<=[.!?…])\s+|\n+")


def limpiar_texto(texto: str) -> str:
    """Normaliza un texto para el modelo (se aplica siempre antes de predecir)."""
    t = html.unescape(str(texto))
    t = _URL.sub(" url ", t)
    t = _MENCION.sub(" @usuario ", t)
    t = _HASHTAG_ANONIMO.sub(" hashtag ", t)
    t = _HASHTAG.sub(r" \1 ", t)                    # "#feliz" -> "feliz"
    for patron, palabra in _EMOTICONES:              # ":(" -> emoticono_triste
        t = patron.sub(palabra, t)
    # Cada emoji se convierte en una palabra: 😭 -> emoji_cara_llorando_fuerte
    t = emoji.demojize(t, language="es", delimiters=(" emoji_", " "))
    t = t.lower()
    t = _RISA.sub(" jaja ", t)                      # "jajajaja", "jsjs"... -> "jaja"
    t = _REPETIDAS.sub(r"\1\1", t)                  # "siiiii" -> "sii"
    t = _SIGNOS_REPETIDOS.sub(r"\1\1", t)           # "!!!!" -> "!!"
    t = _PUNTUACION.sub(r" \1 ", t)                 # separa "!" y "?" como tokens
    return _ESPACIOS.sub(" ", t).strip()


def dividir_oraciones(texto: str) -> list[str]:
    """Separa un párrafo en oraciones (por puntos, signos o saltos de línea)."""
    partes = [p.strip() for p in _ORACIONES.split(str(texto))]
    return [p for p in partes if len(p) > 1]
