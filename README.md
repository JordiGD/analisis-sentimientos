# 💬 Análisis de sentimientos en texto

Modelo de **Machine Learning** que lee un texto en español e interpreta **qué emoción expresa** y si el **sentimiento** es positivo, negativo o neutral.

```
Entrada:  "No puedo creer que me dieron la beca, estoy feliz 😄"
Salida:   Emoción: Alegría (73 %)  ·  Sentimiento: Positivo (73 %)
```

| Emoción | Sentimiento |
|---|---|
| 😄 Alegría | Positivo |
| 😢 Tristeza · 😠 Enojo · 😨 Miedo · 🤢 Asco | Negativo |
| 😮 Sorpresa · 😐 Neutral | Neutral |


---

## 1. Inicio rápido

Requiere **Python 3.10 o superior**.

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt

python preparar_datos.py      # descarga y prepara los datos
python entrenar.py            # entrena, evalúa y guarda el modelo y las gráficas

streamlit run app.py              # aplicación web en http://localhost:8501
```

O desde la consola:
```bash
python predecir.py --texto "Me da miedo la tormenta"
python predecir.py --texto "No estoy nada contento con la compra" --explicar
python predecir.py --texto "Fue un gran día. Pero perdí el bus." --oraciones
python predecir.py --interactivo
python predecir.py --archivo comentarios.csv --columna texto      # analiza muchos textos y guarda un CSV
```

Para reentrenar desde cero (unos 2 minutos):
```bash
python preparar_datos.py      # descarga y prepara los datos
python entrenar.py            # entrena, evalúa y guarda el modelo y las gráficas
```

---

## 2. ¿Cómo funciona?

```
 Texto ─► 1. Limpieza ─► 2. TF-IDF ─► 3. Regresión logística ─► 4. Probabilidad ─► 5. Sentimiento
                            (texto →      (un peso por palabra       de cada            (suma por
                             números)      y emoción)                emoción             polaridad)
```

**1. Limpieza** (`src/texto.py`). Los textos de redes sociales se normalizan para que el modelo vea siempre el mismo formato:

| Original | Limpio |
|---|---|
| `@ana`, `https://t.co/...` | `@usuario`, `url` |
| `#felicidad` | `felicidad` |
| `😭` | `emoji_cara_llorando_fuerte` |
| `:(`  `:)`  `<3` | `emoticono_triste`, `emoticono_feliz`, `emoticono_corazon` |
| `jajajaja`, `jsjsjs`, `xD` | `jaja` |
| `siiiiii`, `!!!!` | `sii`, `!!` |
| `¡Qué miedo!` | `¡ qué miedo !` (los signos se conservan: indican sorpresa o enojo) |

**2. TF‑IDF** (`src/modelo.py`). Convierte cada texto en un vector de ~150.000 números. Cada posición es una característica, y su valor es alto si aparece en ese texto y es poco común en los demás (*Term Frequency – Inverse Document Frequency*). Se combinan dos tipos de características:
- **Palabras y pares de palabras**: `feliz`, `qué miedo`, `me encanta`...
- **Fragmentos de 2 a 5 letras**: `feli`, `eliz`... Así se toleran errores de ortografía y variantes (`felíz`, `felices`, `felicidad`).

**Manejo de negaciones**: las palabras que siguen a *no, nunca, ni, sin...* (hasta el siguiente signo de puntuación o "pero") se marcan, por ejemplo `no estoy contento` → `no no_estoy no_contento`. Así "contento" y "no contento" son características distintas.

**3. Regresión logística (multiclase)**. Aprende un peso por cada característica y emoción. Para un texto suma los pesos de sus características y, con la función *softmax*, obtiene la probabilidad de cada una de las 7 emociones. El hiperparámetro **C** (regularización) se elige probando varios valores con el conjunto de validación (`resultados/busqueda_C.png`). El mejor fue **C = 2**.

**4. Clases desbalanceadas**. Hay muchos más textos neutrales y alegres que de miedo, así que cada emoción recibe un peso inverso a su frecuencia (suavizado con raíz cuadrada) para que las emociones raras no se ignoren.

**5. Sentimiento**. Se suman las probabilidades de las emociones de cada tipo. Por ejemplo, si un texto da Alegría 30 %, Enojo 25 %, Tristeza 20 % y Asco 15 %, la emoción principal es *Alegría*, pero el sentimiento es **negativo** (60 %).

**Explicación de cada predicción**. Para mostrar qué palabras influyeron, se quita cada palabra del texto y se mide cuánto cambia la probabilidad (técnica de *oclusión*). En la app se resaltan en verde (apoya la emoción) o rojo (la contradice).

---

## 3. Datos

Dos corpus públicos en español, que `preparar_datos.py` descarga desde GitHub:

| Corpus | Textos | Descripción |
|---|---|---|
| **EmoEvent** (Plaza‑del‑Arco et al., 2020) | ~8.400 | Tuits sobre eventos de abril de 2019 (incendio de Notre Dame, Juego de Tronos, Champions League, elecciones en España...) |
| **BRIGHTER / SemEval‑2025 Tarea 11** (Muhammad et al., 2025) | ~3.900 | Comentarios de redes sociales en español con estilo más latinoamericano |

| Partición | Alegría | Tristeza | Enojo | Miedo | Sorpresa | Asco | Neutral | Total |
|---|---|---|---|---|---|---|---|---|
| Entrenamiento | 1.867 | 1.000 | 1.081 | 382 | 658 | 765 | 2.781 | **8.534** |
| Validación | 213 | 122 | 93 | 31 | 52 | 38 | 409 | **958** |
| Prueba | 712 | 355 | 240 | 141 | 276 | 204 | 794 | **2.722** |

**Decisiones de preparación:**
- En BRIGHTER un texto puede tener varias emociones. En entrenamiento se repite una vez por emoción; en validación y prueba solo se usan los textos de una única emoción.
- ⚠️ **Se corrigió un error del dataset original:** en el repositorio de EmoEvent, las etiquetas de `test.tsv` no corresponden a los tuits (solo el ~35 % coincide con el corpus completo, frente al 99,9 % en entrenamiento y validación). Con ellas, la precisión caía artificialmente al ~36 %. Se toman las etiquetas correctas del corpus completo (se corrigieron 1.050 tuits).
- Se eliminan duplicados y ningún texto de prueba aparece en entrenamiento.
- El modelo final se entrena con entrenamiento + validación, una vez elegido C.

**Agregar tus propios datos:** prepara un CSV con columnas `texto` y `emocion` (usando las mismas etiquetas) y ejecuta `python preparar_datos.py --extra mis_textos.csv` y luego `python entrenar.py`.

---

## 4. Limitaciones

- **Ironía y sarcasmo**: *"Uy sí, qué maravilla, otra vez se fue la luz 🙄"* sale como Alegría.
- **Sesgo temporal y temático** de los datos (tuits de 2019): *"Mañana tengo clase a las 8"* sale como Miedo, porque en los datos "mañana" aparece mucho junto a exámenes y nervios.
- **Bolsa de palabras**: el modelo casi no tiene en cuenta el orden de las palabras más allá de pares y del manejo de negaciones.
- Solo funciona en **español** y con textos cortos tipo redes sociales.
- El modelo clasifica **cómo está escrito** el texto, no lo que la persona realmente siente. No debe usarse para tomar decisiones sobre personas sin revisión humana.

## 5. Posibles mejoras

- Añadir datos propios del dominio de interés (reseñas, encuestas de satisfacción...).
- Clasificación **multietiqueta** (un texto con varias emociones a la vez).
- Probar un Transformer en español (BETO, RoBERTuito) con GPU, que maneja mejor el contexto y la negación.

---

## 6. Estructura del proyecto

```
analisis_sentimientos/
├── app.py               # Aplicación web (Streamlit)
├── predecir.py          # Uso por consola
├── entrenar.py          # Entrenamiento + evaluación + gráficas
├── preparar_datos.py    # Descarga, corrige y une los datasets
├── config.py            # Emociones, sentimientos, URLs y parámetros
├── requirements.txt
├── src/
│   ├── texto.py         # Limpieza del texto y división en oraciones
│   └── modelo.py        # TF-IDF + Regresión Logística, negaciones y clase AnalizadorSentimiento
├── modelo/              # modelo.joblib y meta.json (ya entrenado)
├── resultados/          # reporte.txt, matriz_confusion.png, palabras_clave.png, busqueda_C.png
└── datos/               # se llena con preparar_datos.py
```

Uso desde tu propio código:
```python
from src.modelo import AnalizadorSentimiento

analizador = AnalizadorSentimiento()
r = analizador.analizar("¡Ganamos la final! 🎉")
print(r["emocion"], r["confianza"], r["sentimiento"], r["probabilidades"])
```
