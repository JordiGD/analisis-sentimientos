"""
Descarga y prepara los datos de entrenamiento (textos en español etiquetados
con emociones).

Fuentes:
  1. EmoEvent (Plaza-del-Arco et al., 2020): ~8.400 tuits en español sobre
     eventos de abril de 2019, una emoción por tuit (o "otras/neutral").
  2. BRIGHTER / SemEval-2025 Tarea 11 (Muhammad et al., 2025), parte en
     español: ~3.900 textos de redes sociales; un texto puede tener varias
     emociones a la vez.

Resultado (columnas: texto, emocion, fuente):
    datos/procesados/train.csv   -> para entrenar
    datos/procesados/valid.csv   -> para elegir hiperparámetros / detener el entrenamiento
    datos/procesados/test.csv    -> para la evaluación final (no se toca al entrenar)

Uso:
    python preparar_datos.py
    python preparar_datos.py --extra mis_textos.csv     # añade tus propios datos a train

Tus propios datos deben ser un CSV con columnas "texto" y "emocion", usando
las mismas etiquetas: Alegría, Tristeza, Enojo, Miedo, Sorpresa, Asco, Neutral.
"""

import argparse
import io
import re
import urllib.request

import pandas as pd

from config import (
    CARPETA_DATOS,
    CARPETA_PROCESADOS,
    EMOCIONES,
    TRADUCCION,
    URL_BRIGHTER,
    URL_EMOEVENT,
    URL_EMOEVENT_COMPLETO,
)

CARPETA_ORIGINALES = CARPETA_DATOS / "originales"
EMOCIONES_BRIGHTER = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]


def descargar(url: str, destino) -> str:
    """Descarga un archivo (una sola vez) y devuelve su contenido."""
    if not destino.exists():
        print(f"  Descargando {url}")
        destino.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as r:
            destino.write_bytes(r.read())
    return destino.read_text(encoding="utf-8")


def _clave(texto: str) -> str:
    """Huella de un tuit sin enlaces, hashtags ni menciones (para cruzar archivos)."""
    t = re.sub(r"https?://\S+|\bURL\b|#\w+|\bHASHTAG\b|@\w+|\bUSER\b", "", str(texto))
    return re.sub(r"\W+", "", t.lower())[:60]


def etiquetas_emoevent() -> dict[str, str]:
    """Etiqueta correcta de cada tuit según el corpus completo de EmoEvent."""
    contenido = descargar(URL_EMOEVENT_COMPLETO, CARPETA_ORIGINALES / "emoevent" / "emoevent_es.csv")
    df = pd.read_csv(io.StringIO(contenido), sep="\t")
    return dict(zip(df["tweet"].map(_clave), df["emotion"]))


def cargar_emoevent(particion: str, etiquetas: dict[str, str]) -> pd.DataFrame:
    """
    Se usan los textos de las particiones oficiales (conservan los hashtags) y
    la etiqueta del corpus completo. En el repositorio, test.tsv trae etiquetas
    que no corresponden a cada tuit (solo ~35 % coincide con el corpus), así
    que se corrigen; en train y dev coinciden en más del 99,9 %.
    """
    contenido = descargar(
        URL_EMOEVENT.format(particion=particion),
        CARPETA_ORIGINALES / "emoevent" / f"{particion}.tsv",
    )
    df = pd.read_csv(io.StringIO(contenido), sep="\t")
    etiqueta = df["tweet"].map(_clave).map(etiquetas)
    corregidas = int((etiqueta.notna() & (etiqueta != df["emotion"])).sum())
    if corregidas:
        print(f"  EmoEvent {particion}: {corregidas} etiquetas corregidas con el corpus completo")
    return pd.DataFrame(
        {
            "texto": df["tweet"],
            "emocion": etiqueta.map(TRADUCCION),   # sin coincidencia -> se descarta
            "fuente": "EmoEvent",
        }
    )


def cargar_brighter(particion: str, duplicar_multiples: bool) -> pd.DataFrame:
    """
    En BRIGHTER un texto puede tener varias emociones. Para entrenar un
    clasificador de una sola etiqueta:
      - en entrenamiento, el texto se repite una vez por cada emoción
        (el modelo aprende que ambas son probables);
      - en validación y prueba se usan solo los textos con una única emoción,
        para poder medir aciertos sin ambigüedad.
    """
    archivo = "dev" if particion == "valid" else particion
    contenido = descargar(
        URL_BRIGHTER.format(particion=archivo),
        CARPETA_ORIGINALES / "brighter" / f"{archivo}.csv",
    )
    df = pd.read_csv(io.StringIO(contenido))
    filas = []
    for _, fila in df.iterrows():
        emociones = [e for e in EMOCIONES_BRIGHTER if fila[e] == 1]
        if not emociones:
            continue
        if len(emociones) > 1 and not duplicar_multiples:
            continue
        for e in emociones:
            filas.append({"texto": fila["text"], "emocion": TRADUCCION[e], "fuente": "BRIGHTER"})
    return pd.DataFrame(filas)


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["texto", "emocion"]).copy()
    df["texto"] = df["texto"].astype(str).str.strip().str.strip('"').str.strip()
    df = df[df["texto"].str.len() > 2]
    return df.drop_duplicates(subset=["texto", "emocion"])


def main():
    p = argparse.ArgumentParser(description="Descargar y preparar los datos")
    p.add_argument("--extra", nargs="*", default=[],
                   help="CSV propios (columnas texto, emocion) para añadir al entrenamiento")
    args = p.parse_args()

    print("Preparando datos...")
    particiones = {}
    etiquetas = etiquetas_emoevent()
    for nombre, emo, bri_dup in [("train", "train", True), ("valid", "dev", False), ("test", "test", False)]:
        df = pd.concat(
            [cargar_emoevent(emo, etiquetas), cargar_brighter(nombre, duplicar_multiples=bri_dup)],
            ignore_index=True,
        )
        particiones[nombre] = limpiar(df)

    for ruta in args.extra:
        extra = pd.read_csv(ruta)
        faltan = {"texto", "emocion"} - set(extra.columns)
        if faltan:
            raise SystemExit(f"{ruta}: faltan las columnas {faltan}")
        extra = extra[["texto", "emocion"]].assign(fuente="Propio")
        desconocidas = set(extra["emocion"]) - set(EMOCIONES)
        if desconocidas:
            print(f"  ⚠ {ruta}: etiquetas nuevas {desconocidas} (se usarán tal cual)")
        particiones["train"] = limpiar(pd.concat([particiones["train"], extra], ignore_index=True))
        print(f"  + {len(extra)} textos propios desde {ruta}")

    # Evita que un mismo texto quede en entrenamiento y en prueba
    vistos = set(particiones["test"]["texto"]) | set(particiones["valid"]["texto"])
    particiones["train"] = particiones["train"][~particiones["train"]["texto"].isin(vistos)]

    CARPETA_PROCESADOS.mkdir(parents=True, exist_ok=True)
    resumen = {}
    for nombre, df in particiones.items():
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df.to_csv(CARPETA_PROCESADOS / f"{nombre}.csv", index=False, encoding="utf-8")
        resumen[nombre] = df["emocion"].value_counts()

    tabla = pd.DataFrame(resumen).fillna(0).astype(int)
    tabla.loc["TOTAL"] = tabla.sum()
    print("\nTextos por emoción y partición:")
    print(tabla.to_string())
    print(f"\n✔ Datos guardados en {CARPETA_PROCESADOS}/")
    print("Siguiente paso: python entrenar.py")


if __name__ == "__main__":
    main()
