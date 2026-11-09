"""
Aplicación web para analizar el sentimiento de un texto.

Ejecutar con:
    streamlit run app.py
Se abre en el navegador (normalmente http://localhost:8501).
"""

import html

import pandas as pd
import streamlit as st

from config import CARPETA_RESULTADOS, EMOJIS, SENTIMIENTO
from src.modelo import AnalizadorSentimiento

COLOR_SENTIMIENTO = {"Positivo": "green", "Negativo": "red", "Neutral": "gray"}
EJEMPLOS = [
    "¡Por fin me dieron la beca! No lo puedo creer, estoy feliz 😄",
    "Desde que se fue mi abuelita la casa se siente vacía 😢",
    "Estoy harto de que cancelen el vuelo por tercera vez, pésimo servicio",
    "Me da mucho miedo caminar sola por esa calle de noche",
    "Qué asco, encontré un pelo en la sopa del restaurante",
    "¿En serio ganamos en el último minuto? ¡Increíble! 😱",
    "No estoy nada contento con la compra, llegó dañada",
    "El viaje empezó genial y conocimos gente increíble. Pero el último día nos robaron "
    "las maletas y fue horrible.",
]

st.set_page_config(page_title="Análisis de sentimientos", page_icon="💬", layout="wide")


@st.cache_resource(show_spinner="Cargando el modelo...")
def cargar_analizador() -> AnalizadorSentimiento:
    return AnalizadorSentimiento()


def resaltar(pesos: list[tuple[str, float]]) -> str:
    """HTML con cada palabra coloreada: verde = apoya la emoción, rojo = la contradice."""
    maximo = max((abs(p) for _, p in pesos), default=0) or 1
    trozos = []
    for palabra, peso in pesos:
        alfa = min(1.0, abs(peso) / maximo) * 0.55
        color = "46,160,67" if peso > 0 else "218,54,51"
        trozos.append(
            f'<span title="{peso:+.3f}" style="background: rgba({color},{alfa:.2f}); '
            f'padding: 2px 4px; border-radius: 4px; line-height: 2">{html.escape(palabra)}</span>'
        )
    return " ".join(trozos)


def mostrar_resultado(analizador: AnalizadorSentimiento, texto: str) -> None:
    r = analizador.analizar(texto)
    emo, sent = r["emocion"], r["sentimiento"]

    col1, col2 = st.columns(2, gap="large")
    with col1:
        with st.container(border=True):
            st.markdown(f"### :{COLOR_SENTIMIENTO.get(sent, 'gray')}[Sentimiento {sent.lower()}]")
            c = st.columns(3)
            for i, s in enumerate(["Positivo", "Neutral", "Negativo"]):
                c[i].metric(f"{EMOJIS.get(s, '😐')} {s}", f"{r['prob_sentimiento'][s]:.0%}")
            st.caption("Cada valor es la suma de las probabilidades de las emociones de ese tipo.")
        with st.container(border=True):
            st.markdown(f"**¿Qué palabras influyeron en «{emo}»?**")
            st.markdown(resaltar(analizador.explicar(texto, emo)), unsafe_allow_html=True)
            st.caption("🟩 apoya la emoción · 🟥 la contradice. Se mide quitando cada palabra "
                       "y viendo cuánto cambia la probabilidad.")
    with col2:
        with st.container(border=True):
            st.markdown(f"### {EMOJIS.get(emo, '')} {emo}")
            st.caption(f"Emoción principal · confianza {r['confianza']:.0%}")
            for e, p in sorted(r["probabilidades"].items(), key=lambda kv: kv[1], reverse=True):
                st.progress(p, text=f"{EMOJIS.get(e, '')} {e}: {p:.1%}")

    oraciones = analizador.analizar_por_oraciones(texto)
    if len(oraciones) > 1:
        st.markdown("#### Análisis por oración")
        st.dataframe(
            pd.DataFrame([{
                "Oración": o["texto"],
                "Emoción": f"{EMOJIS.get(o['emocion'], '')} {o['emocion']}",
                "Confianza": o["confianza"],
                "Sentimiento": o["sentimiento"],
            } for o in oraciones]),
            hide_index=True, width="stretch",
            column_config={"Confianza": st.column_config.ProgressColumn(
                format="percent", min_value=0, max_value=1)},
        )


def pestaña_texto(analizador):
    if "texto" not in st.session_state:
        st.session_state.texto = EJEMPLOS[0]

    def usar_ejemplo():
        st.session_state.texto = st.session_state.ejemplo

    st.selectbox("Probar con un ejemplo", EJEMPLOS, key="ejemplo", on_change=usar_ejemplo)
    texto = st.text_area("Escribe o pega un texto en español", key="texto", height=120, max_chars=5000)
    if texto.strip():
        mostrar_resultado(analizador, texto)
    else:
        st.info("Escribe un texto para analizarlo.")


def pestaña_archivo(analizador):
    st.write("Sube un **CSV** (comentarios, reseñas, encuestas...) o un **TXT** con una frase por línea.")
    archivo = st.file_uploader("Archivo", type=["csv", "txt"])
    if archivo is None:
        return
    try:
        if archivo.name.lower().endswith(".csv"):
            df = pd.read_csv(archivo)
            columnas = list(df.columns)
            col = st.selectbox("Columna con los textos", columnas,
                               index=columnas.index("texto") if "texto" in columnas else 0)
        else:
            lineas = archivo.getvalue().decode("utf-8", errors="replace").splitlines()
            df, col = pd.DataFrame({"texto": [l for l in lineas if l.strip()]}), "texto"
    except Exception as e:
        st.error(f"No se pudo leer el archivo: {e}")
        return
    if df.empty:
        st.warning("El archivo no tiene textos.")
        return

    with st.spinner(f"Analizando {len(df)} textos..."):
        resultados = analizador.analizar_lote(df[col].fillna("").astype(str).tolist())
    df["emocion"] = [r["emocion"] for r in resultados]
    df["confianza"] = [round(r["confianza"], 3) for r in resultados]
    df["sentimiento"] = [r["sentimiento"] for r in resultados]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Emociones**")
        st.bar_chart(df["emocion"].value_counts(), horizontal=True, x_label="", y_label="Textos")
    with c2:
        st.markdown("**Sentimiento**")
        st.bar_chart(df["sentimiento"].value_counts(), horizontal=True, x_label="", y_label="Textos")
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button(
        "⬇️ Descargar resultados (CSV)",
        df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{archivo.name.rsplit('.', 1)[0]}_sentimientos.csv",
        mime="text/csv",
    )


def pestaña_modelo(analizador):
    meta = analizador.metadatos
    st.markdown(f"**Modelo:** {meta.get('modelo', 'TF-IDF + Regresión Logística')} · "
                f"C = {meta.get('C', '—')} · entrenado con {meta.get('textos_entrenamiento', 0):,} textos")
    prueba = meta.get("prueba")
    if prueba:
        st.markdown(f"Resultados con **{prueba['textos']:,} textos de prueba** que el modelo nunca vio:")
        c = st.columns(3)
        c[0].metric("Accuracy (emoción)", f"{prueba['accuracy']:.1%}")
        c[1].metric("F1 macro (emoción)", f"{prueba['f1_macro']:.1%}")
        c[2].metric("Accuracy (sentimiento)", f"{prueba['accuracy_sentimiento']:.1%}")
    st.markdown("**Emociones y su sentimiento:** " + " · ".join(
        f"{EMOJIS.get(e, '')} {e} → {SENTIMIENTO.get(e, '—')}" for e in analizador.etiquetas))
    for nombre, titulo in [("matriz_confusion.png", "Matriz de confusión"),
                           ("palabras_clave.png", "Palabras más asociadas a cada emoción")]:
        ruta = CARPETA_RESULTADOS / nombre
        if ruta.is_file():
            st.markdown(f"#### {titulo}")
            st.image(str(ruta), width=800 if "matriz" in nombre else "stretch")


def main():
    st.title("💬 Análisis de sentimientos en texto")
    st.write(
        "Escribe un texto en español y el modelo identifica **qué emoción expresa** "
        "(alegría, tristeza, enojo, miedo, sorpresa, asco o neutral) y si el "
        "**sentimiento** es positivo, negativo o neutral."
    )
    try:
        analizador = cargar_analizador()
    except FileNotFoundError:
        st.error("No se encontró el modelo entrenado. Ejecuta:\n\n"
                 "`python preparar_datos.py`\n\n`python entrenar.py`")
        st.stop()

    t1, t2, t3 = st.tabs(["✍️ Analizar texto", "📄 Analizar archivo", "📊 Sobre el modelo"])
    with t1:
        pestaña_texto(analizador)
    with t2:
        pestaña_archivo(analizador)
    with t3:
        pestaña_modelo(analizador)
    st.caption("El modelo interpreta cómo está escrito el texto; no conoce la intención real "
               "de quien lo escribe. La ironía y el sarcasmo suelen confundirlo.")


main()
