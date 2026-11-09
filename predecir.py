"""
Analiza el sentimiento de textos desde la consola.

Ejemplos:
    python predecir.py --texto "¡Por fin aprobé el examen!"
    python predecir.py --texto "Me da miedo la tormenta" --explicar
    python predecir.py --texto "Fue un gran día. Pero perdí el bus." --oraciones
    python predecir.py --interactivo
    python predecir.py --archivo comentarios.csv --columna texto
    python predecir.py --archivo frases.txt          # una frase por línea

Con --archivo se guarda un CSV con los resultados en resultados/.
"""

import argparse
from pathlib import Path

import pandas as pd

from config import CARPETA_RESULTADOS, EMOJIS
from src.modelo import AnalizadorSentimiento


def barra(p: float, ancho: int = 20) -> str:
    llenos = round(p * ancho)
    return "█" * llenos + "░" * (ancho - llenos)


def mostrar(r: dict, detalle: bool = True) -> None:
    emo, sent = r["emocion"], r["sentimiento"]
    print(f"\n  «{r['texto']}»")
    print(f"  → Emoción: {EMOJIS.get(emo, '')} {emo} ({r['confianza']:.0%})"
          f"   |   Sentimiento: {EMOJIS.get(sent, '😐')} {sent} ({r['prob_sentimiento'][sent]:.0%})")
    if detalle:
        for e, p in sorted(r["probabilidades"].items(), key=lambda kv: kv[1], reverse=True):
            print(f"     {e:<9} {barra(p)} {p:5.1%}")


def mostrar_explicacion(analizador, texto: str, emocion: str) -> None:
    pesos = sorted(analizador.explicar(texto, emocion), key=lambda kv: abs(kv[1]), reverse=True)
    print(f"  Palabras más influyentes para «{emocion}»:")
    for palabra, peso in pesos[:6]:
        print(f"     {palabra:<18} {'apoya     ' if peso > 0 else 'contradice'} ({peso:+.2f})")


def analizar_archivo(analizador, ruta: Path, columna: str | None) -> None:
    if ruta.suffix.lower() == ".csv":
        df = pd.read_csv(ruta)
        columna = columna or ("texto" if "texto" in df.columns else df.columns[0])
        if columna not in df.columns:
            raise SystemExit(f"La columna '{columna}' no existe. Columnas: {list(df.columns)}")
    else:
        lineas = ruta.read_text(encoding="utf-8").splitlines()
        df, columna = pd.DataFrame({"texto": [l for l in lineas if l.strip()]}), "texto"

    resultados = analizador.analizar_lote(df[columna].fillna("").astype(str).tolist())
    df["emocion"] = [r["emocion"] for r in resultados]
    df["confianza"] = [round(r["confianza"], 3) for r in resultados]
    df["sentimiento"] = [r["sentimiento"] for r in resultados]
    for e in analizador.etiquetas:
        df[f"prob_{e}"] = [round(r["probabilidades"][e], 3) for r in resultados]

    CARPETA_RESULTADOS.mkdir(exist_ok=True)
    destino = CARPETA_RESULTADOS / f"{ruta.stem}_sentimientos.csv"
    df.to_csv(destino, index=False, encoding="utf-8-sig")  # utf-8-sig: se abre bien en Excel

    print(f"\nTextos analizados: {len(df)}")
    for titulo, col in [("Emociones", "emocion"), ("Sentimiento", "sentimiento")]:
        print(f"\n{titulo}:")
        for valor, n in df[col].value_counts().items():
            print(f"  {EMOJIS.get(valor, '😐')} {valor:<9} {n:>5} ({n / len(df):.0%})")
    print(f"\n✔ Resultados guardados en {destino}")


def main():
    p = argparse.ArgumentParser(description="Analizar el sentimiento de textos en español")
    grupo = p.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--texto", help="Texto a analizar")
    grupo.add_argument("--archivo", help="CSV o TXT con varios textos")
    grupo.add_argument("--interactivo", action="store_true", help="Escribir frases una tras otra")
    p.add_argument("--columna", default=None, help="Columna del CSV con los textos")
    p.add_argument("--oraciones", action="store_true", help="Analizar también cada oración")
    p.add_argument("--explicar", action="store_true", help="Mostrar qué palabras influyeron")
    args = p.parse_args()

    analizador = AnalizadorSentimiento()

    if args.archivo:
        analizar_archivo(analizador, Path(args.archivo), args.columna)
        return

    def procesar(texto: str):
        if args.oraciones:
            print("\n  Por oración:")
            for r in analizador.analizar_por_oraciones(texto):
                mostrar(r, detalle=False)
            print("\n  Texto completo:")
        r = analizador.analizar(texto)
        mostrar(r)
        if args.explicar:
            mostrar_explicacion(analizador, texto, r["emocion"])

    if args.texto:
        procesar(args.texto)
        return
    print("Escribe una frase y pulsa Enter (línea vacía o 'salir' para terminar).")
    while True:
        try:
            texto = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not texto or texto.lower() == "salir":
            break
        procesar(texto)


if __name__ == "__main__":
    main()
