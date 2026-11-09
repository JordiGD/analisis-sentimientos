"""
Entrena y evalúa el modelo de análisis de sentimientos.

Uso:
    python entrenar.py

Pasos:
  1. Carga los datos preparados (datos/procesados/train, valid y test).
  2. Prueba varios valores de regularización C y elige el de mejor F1 en validación.
  3. Reentrena con train + valid usando ese C.
  4. Evalúa con el conjunto de prueba (textos que el modelo nunca vio).

Guarda:
    modelo/modelo.joblib, modelo/meta.json
    resultados/reporte.txt               -> métricas por emoción
    resultados/matriz_confusion.png      -> qué emociones confunde
    resultados/busqueda_C.png            -> efecto de la regularización
    resultados/palabras_clave.png/.txt   -> palabras más asociadas a cada emoción
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score  # noqa: E402
from sklearn.utils.class_weight import compute_class_weight  # noqa: E402

from config import (  # noqa: E402
    BALANCEO, CARPETA_MODELO, CARPETA_PROCESADOS, CARPETA_RESULTADOS, EMOCIONES,
    RUTA_META, RUTA_MODELO, VALORES_C, sentimiento_de,
)
from src.modelo import crear_modelo, palabras_clave  # noqa: E402
from src.texto import limpiar_texto  # noqa: E402


def cargar(nombre: str, carpeta) -> pd.DataFrame | None:
    ruta = carpeta / f"{nombre}.csv"
    if not ruta.is_file():
        return None
    df = pd.read_csv(ruta).dropna(subset=["texto", "emocion"])
    df["limpio"] = df["texto"].map(limpiar_texto)
    return df


def pesos_de_clase(y, etiquetas) -> dict:
    """Compensa las emociones con pocos ejemplos (p. ej. Miedo)."""
    pesos = compute_class_weight("balanced", classes=np.array(etiquetas), y=y)
    return {e: float(p ** BALANCEO) for e, p in zip(etiquetas, pesos)}


# --------------------------------------------------------------------------- #
# Gráficas
# --------------------------------------------------------------------------- #
def graficar_busqueda(busqueda: list[dict], ruta):
    fig, eje = plt.subplots(figsize=(7, 4))
    C = [b["C"] for b in busqueda]
    eje.plot(C, [b["f1_macro"] for b in busqueda], "o-", label="F1 macro")
    eje.plot(C, [b["accuracy"] for b in busqueda], "s-", label="Accuracy")
    eje.set_xscale("log", base=2)
    eje.set_xticks(C, [str(c) for c in C])
    eje.set_xlabel("C (mayor = menos regularización)")
    eje.set_ylabel("Valor en validación")
    eje.set_title("Elección del hiperparámetro C")
    eje.grid(alpha=0.3)
    eje.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def graficar_matriz(matriz, etiquetas, ruta, titulo):
    normal = matriz / np.maximum(matriz.sum(axis=1, keepdims=True), 1)
    fig, eje = plt.subplots(figsize=(8, 7))
    im = eje.imshow(normal, cmap="Blues", vmin=0, vmax=1)
    eje.set_xticks(range(len(etiquetas)), etiquetas, rotation=45, ha="right")
    eje.set_yticks(range(len(etiquetas)), etiquetas)
    eje.set_xlabel("Predicción del modelo")
    eje.set_ylabel("Emoción real")
    for i in range(len(etiquetas)):
        for j in range(len(etiquetas)):
            v = normal[i, j]
            eje.text(j, i, f"{v:.0%}\n({matriz[i, j]})", ha="center", va="center",
                     fontsize=8, color="white" if v > 0.5 else "black")
    fig.colorbar(im, ax=eje, fraction=0.046)
    eje.set_title(titulo)
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def graficar_palabras(claves: dict, ruta):
    emociones = list(claves)
    columnas = 4
    filas = int(np.ceil(len(emociones) / columnas))
    fig, ejes = plt.subplots(filas, columnas, figsize=(4.2 * columnas, 3.6 * filas))
    for eje, emocion in zip(ejes.flat, emociones):
        palabras, pesos = zip(*claves[emocion][:10])
        eje.barh(range(len(palabras))[::-1], pesos, color="#3b7dd8")
        eje.set_yticks(range(len(palabras))[::-1], palabras, fontsize=9)
        eje.set_title(emocion)
        eje.tick_params(axis="x", labelsize=8)
    for eje in list(ejes.flat)[len(emociones):]:
        eje.axis("off")
    fig.suptitle("Palabras con más peso para cada emoción (coeficientes del modelo)")
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Entrenar y evaluar el modelo")
    p.add_argument("--datos", default=str(CARPETA_PROCESADOS),
                   help="Carpeta con train.csv, valid.csv y test.csv (columnas texto, emocion)")
    args = p.parse_args()
    carpeta = Path(args.datos)

    train, valid, test = (cargar(n, carpeta) for n in ("train", "valid", "test"))
    if train is None or valid is None:
        raise SystemExit("Faltan los datos. Ejecuta primero: python preparar_datos.py")
    presentes = set(train["emocion"])
    etiquetas = [e for e in EMOCIONES if e in presentes] + sorted(presentes - set(EMOCIONES))
    valid = valid[valid["emocion"].isin(etiquetas)]
    print(f"Textos: {len(train)} entrenamiento | {len(valid)} validación | "
          f"{0 if test is None else len(test)} prueba")

    inicio = time.time()
    pesos = pesos_de_clase(train["emocion"], etiquetas)

    # 1) Elegir C con el conjunto de validación
    print("\nBuscando el mejor valor de C...")
    busqueda = []
    for C in VALORES_C:
        modelo = crear_modelo(C, pesos).fit(train["limpio"], train["emocion"])
        pred = modelo.predict(valid["limpio"])
        f1 = f1_score(valid["emocion"], pred, average="macro")
        acc = accuracy_score(valid["emocion"], pred)
        busqueda.append({"C": C, "f1_macro": round(f1, 4), "accuracy": round(acc, 4)})
        print(f"  C={C:<4} -> F1 macro {f1:.3f} | accuracy {acc:.3f}")
    mejor = max(busqueda, key=lambda b: b["f1_macro"])
    print(f"Mejor C = {mejor['C']}")

    # 2) Modelo final con train + valid
    print("\nEntrenando el modelo final con entrenamiento + validación...")
    completo = pd.concat([train, valid], ignore_index=True)
    modelo = crear_modelo(mejor["C"], pesos_de_clase(completo["emocion"], etiquetas))
    modelo.fit(completo["limpio"], completo["emocion"])

    CARPETA_MODELO.mkdir(exist_ok=True)
    CARPETA_RESULTADOS.mkdir(exist_ok=True)
    joblib.dump(modelo, RUTA_MODELO, compress=3)
    graficar_busqueda(busqueda, CARPETA_RESULTADOS / "busqueda_C.png")

    claves = palabras_clave(modelo)
    graficar_palabras(claves, CARPETA_RESULTADOS / "palabras_clave.png")
    (CARPETA_RESULTADOS / "palabras_clave.txt").write_text(
        "\n".join(f"{e}: " + ", ".join(p for p, _ in lista) for e, lista in claves.items()),
        encoding="utf-8",
    )

    meta = {
        "modelo": "TF-IDF (palabras + caracteres) + Regresión Logística",
        "etiquetas": etiquetas,
        "C": mejor["C"],
        "busqueda_C": busqueda,
        "textos_entrenamiento": int(len(completo)),
        "fecha": datetime.now().isoformat(timespec="seconds"),
    }

    # 3) Evaluación final con el conjunto de prueba
    if test is not None:
        test = test[test["emocion"].isin(etiquetas)]
        pred = modelo.predict(test["limpio"])
        real = test["emocion"].to_numpy()
        acc = accuracy_score(real, pred)
        f1 = f1_score(real, pred, labels=etiquetas, average="macro", zero_division=0)
        acc_sent = float(np.mean([sentimiento_de(a) == sentimiento_de(b) for a, b in zip(real, pred)]))
        matriz = confusion_matrix(real, pred, labels=etiquetas)
        reporte = classification_report(real, pred, labels=etiquetas, digits=3, zero_division=0)

        por_fuente = ""
        if "fuente" in test.columns:
            aciertos = pd.Series(real == pred, index=test.index)
            for fuente, grupo in aciertos.groupby(test["fuente"]):
                por_fuente += f"  {fuente}: {grupo.mean():.1%} ({len(grupo)} textos)\n"

        texto = (
            f"Modelo: {meta['modelo']} (C={mejor['C']})\n"
            f"Textos de prueba: {len(real)}\n\n"
            f"Accuracy (emoción exacta):                  {acc:.2%}\n"
            f"F1 macro (promedio entre emociones):        {f1:.2%}\n"
            f"Accuracy del sentimiento (pos/neg/neutral): {acc_sent:.2%}\n"
            + (f"\nAccuracy por fuente de datos:\n{por_fuente}" if por_fuente else "")
            + f"\n{reporte}\n"
            "Matriz de confusión (filas = real, columnas = predicción):\n"
            f"{'':>10} " + " ".join(f"{e[:7]:>7}" for e in etiquetas) + "\n"
            + "\n".join(f"{e:>10} " + " ".join(f"{v:>7}" for v in fila)
                        for e, fila in zip(etiquetas, matriz))
        )
        (CARPETA_RESULTADOS / "reporte.txt").write_text(texto, encoding="utf-8")
        graficar_matriz(matriz, etiquetas, CARPETA_RESULTADOS / "matriz_confusion.png",
                        f"Matriz de confusión (accuracy {acc:.1%})")
        meta["prueba"] = {"accuracy": round(acc, 4), "f1_macro": round(f1, 4),
                          "accuracy_sentimiento": round(acc_sent, 4), "textos": int(len(real))}
        print("\n" + texto)

    meta["segundos_entrenamiento"] = round(time.time() - inicio)
    RUTA_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✔ Modelo guardado en {RUTA_MODELO}")
    print(f"✔ Reportes y gráficas en {CARPETA_RESULTADOS}/")
    print("\nSiguiente paso: streamlit run app.py   (o python predecir.py --texto \"...\")")


if __name__ == "__main__":
    main()
