# Chatbot de Contabilidad Básica — RAG + FAISS + Streamlit

Chatbot conversacional para estudiantes universitarios basado en el libro **"Contabilidad Básica"** de Jorge Simaro y Omar Tonelli. Responde preguntas en lenguaje natural buscando en el contenido del libro mediante RAG (Retrieval-Augmented Generation).

Deployado en Streamlit Cloud con acceso protegido por contraseña.

---

## Decisiones de diseño

**RAG con FAISS** — el libro tiene 615 páginas. En lugar de pasarle todo el texto al modelo, el sistema indexa el libro en fragmentos, convierte las preguntas en vectores y busca los fragmentos más relevantes semánticamente. El modelo solo recibe el contexto necesario para responder.

**Filtro por umbral de distancia** — en lugar del retriever estándar de LangChain que devuelve siempre k fragmentos, el sistema usa `similarity_search_with_score` y filtra por distancia euclidiana. Solo pasan los fragmentos realmente relevantes.

**Routing por tipo de pregunta** — el sistema detecta si la pregunta es una comparación, una definición, un ejemplo o una lista, y ajusta k y umbral automáticamente. Las comparaciones necesitan más contexto; las definiciones necesitan más precisión.

**Normalización de vocabulario** — los estudiantes usan vocabulario informal ("plata", "guita", "PN") y cometen errores de tipeo. El sistema normaliza la pregunta antes de buscar usando un diccionario de sinónimos específico del dominio.

**Detección de preguntas fuera del dominio** — si la pregunta no tiene relación con contabilidad, el sistema responde que solo puede responder preguntas sobre el libro.

**Feedback de usuarios** — cada respuesta tiene botones 👍 / 👎. El feedback se guarda en Neon (PostgreSQL) con metadata completa: tipo de pregunta, tema detectado, configuración usada, fragmentos encontrados. Permite mejorar el sistema basándose en datos reales de uso.

**Evaluación con RAGAS** — dataset de preguntas y respuestas esperadas provisto por los autores del libro. Mide faithfulness, answer relevancy y context recall. La configuración actual fue optimizada iterativamente mediante estas métricas.

---

## Arquitectura

```
app.py                  ← interfaz Streamlit — login, chat, botones de feedback
config.py               ← configuración centralizada — K, umbral, modelo, rutas
chatbot/
    retriever.py        ← normalización + routing + búsqueda con umbral
    prompts.py          ← templates del prompt
    feedback.py         ← guardar feedback en Neon
indexar.py              ← indexa el PDF en FAISS (correr una sola vez)
evaluar.py              ← evaluación con RAGAS
faiss_db/               ← base vectorial (generada por indexar.py, no en git)
```

---

## Configuración RAG optimizada

Parámetros encontrados mediante evaluación iterativa con RAGAS:

```python
chunk_size    = 1000   # tamaño de fragmentos al indexar
chunk_overlap = 200    # superposición entre fragmentos
K             = 12     # fragmentos que busca FAISS
UMBRAL        = 1.5    # distancia máxima aceptada (score FAISS)
```

Configuraciones por tipo de pregunta (routing):

| Tipo | k | Umbral |
|---|---|---|
| Comparación | 12 | 1.5 |
| Definición | 6 | 1.0 |
| Ejemplo | 8 | 1.2 |
| Lista | 10 | 1.3 |
| General | 12 | 1.5 |

---

## Feedback y mejora continua

El sistema guarda cada interacción en Neon con:

- Pregunta original y normalizada
- Tipo de pregunta detectado y tema contable
- Configuración usada (k, umbral)
- Fragmentos encontrados y páginas consultadas
- Respuesta del modelo
- Voto del usuario (👍 / 👎)
- Versión del chatbot y chunk_size activo

Consultas útiles para analizar el feedback:

```sql
-- Temas con más feedback negativo
SELECT tema_detectado, COUNT(*) as negativos
FROM feedback WHERE voto = 'negativo'
GROUP BY tema_detectado ORDER BY negativos DESC;

-- Preguntas que fallan con más frecuencia
SELECT pregunta_original, COUNT(*) as negativos
FROM feedback WHERE voto = 'negativo'
GROUP BY pregunta_original ORDER BY negativos DESC LIMIT 10;

-- Impacto de cambios de configuración
SELECT chunk_size, umbral_usado,
       AVG(CASE WHEN voto = 'positivo' THEN 1.0 ELSE 0.0 END) as satisfaccion
FROM feedback GROUP BY chunk_size, umbral_usado;
```

---

## Evaluación con RAGAS

Dataset de 21 preguntas con ground truth provisto por los autores — 4 temas:

| Tema | Preguntas |
|---|---|
| Variaciones Patrimoniales | 9 |
| Principio de Devengado | 10 |
| Ejercicio Económico | 1 |
| Cuentas Corrientes | 1 |

Métricas con configuración actual (chunk_size=1000, umbral=1.5):

| Tema | Faithfulness | Answer Relevancy | Context Recall |
|---|---|---|---|
| Variaciones Patrimoniales | 0.92 | 0.84 | 1.00 |
| Devengado | 1.00 | 0.96 | 1.00 |
| Ejercicio Económico | 0.78 | 0.99 | 1.00 |
| Cuentas Corrientes | 0.88 | 0.66 | 1.00 |

---

## Técnicas aplicadas

- **RAG** — Retrieval-Augmented Generation con FAISS y HuggingFace Embeddings
- **Filtro por umbral** — `similarity_search_with_score` en lugar del retriever estándar
- **Routing por tipo de pregunta** — configuración dinámica de k y umbral según el tipo
- **Normalización de vocabulario** — diccionario de sinónimos para vocabulario universitario argentino informal
- **Detección fuera del dominio** — respuesta diferenciada para preguntas no contables
- **Feedback en producción** — 👍/👎 guardado en PostgreSQL con metadata completa
- **RAGAS** — evaluación con faithfulness, answer relevancy y context recall
- **LangSmith** — observabilidad de las llamadas al LLM

---

## Stack

- **LangChain + Groq** — modelo `llama-3.3-70b-versatile`
- **FAISS** — base vectorial liviana, compatible con Streamlit Cloud
- **HuggingFace Embeddings** — `all-MiniLM-L6-v2`
- **Streamlit** — interfaz web con login por contraseña
- **Neon (PostgreSQL)** — feedback de usuarios
- **RAGAS** — evaluación de calidad del RAG
- **LangSmith** — observabilidad

---

## Setup

```bash
pip install langchain langchain-community langchain-groq langchain-huggingface
pip install faiss-cpu sentence-transformers streamlit python-dotenv
pip install psycopg2-binary ragas datasets
```

Creá un archivo `.env`:

```
GROQ_API_KEY=tu-key           # gratis en console.groq.com
CHATBOT_PASSWORD=tu-contraseña
DATABASE_URL=postgresql://... # gratis en console.neon.tech
LANGCHAIN_API_KEY=tu-key      # gratis en smith.langchain.com
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=chatbot-contabilidad
```

---

## Correr el proyecto

```bash
# Solo la primera vez — indexa el PDF en FAISS
python indexar.py

# Chatbot
streamlit run app.py

# Evaluación con RAGAS (ajustar TEMA_HOY en el archivo)
python evaluar.py
```