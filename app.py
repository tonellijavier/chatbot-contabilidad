# ==============================================================================
# APP.PY — Chatbot de Contabilidad Básica
# ==============================================================================
#
# Interfaz web para hacer preguntas sobre el libro
# "Contabilidad Básica" de Jorge Simaro y Omar Tonelli.
#
# Acceso protegido por contraseña.
# Usa FAISS como base vectorial — liviano y compatible con Streamlit Cloud.
#
# BÚSQUEDA CON FILTRO POR UMBRAL:
# En lugar de usar RetrievalQA con el retriever estándar, usamos
# similarity_search_with_score directamente para filtrar fragmentos
# por distancia euclidiana. Esto garantiza que el modelo solo recibe
# fragmentos realmente relevantes — igual que en evaluar.py.
# Así la evaluación y producción usan exactamente la misma lógica.
#
# PARA CORRERLO:
#   streamlit run app.py
# ==============================================================================

import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ── CONFIGURACIÓN ──────────────────────────────────────────────────────────────

PASSWORD = os.getenv("CHATBOT_PASSWORD")

K = 12       # cuántos fragmentos busca FAISS antes de filtrar
UMBRAL = 1.5 # distancia máxima aceptada
             # score bajo = muy similar, score alto = poco similar
             # 1.0 es más permisivo que 0.8 — captura más contexto
             # para preguntas sobre conceptos distribuidos en el libro

TEMPLATE = """Sos un asistente especializado en contabilidad que responde
preguntas basándote en el libro "Contabilidad Básica" de Jorge Simaro y Omar Tonelli.

Contexto del libro:
{context}

Pregunta: {question}

Instrucciones:
- Todos los términos deben interpretarse en el contexto de la contabilidad
- Usá el contexto del libro como base principal para tu respuesta
- Si el contexto contiene una definición formal del término preguntado,
  citala antes de explicarla con tus palabras
- Si la respuesta está explícita en el contexto, explicala con claridad
- Si no está explícita pero podés deducirla razonando sobre el contexto,
  hacelo y aclará que es una deducción basada en el libro
- Si la pregunta involucra comparar conceptos, explicá cada uno por 
  separado antes de compararlos
- Si la información genuinamente no está en el contexto, decilo — no inventes
- Respondé en español, de forma clara y didáctica, como si el lector 
  fuera un estudiante universitario de primer año
- Sé conciso — máximo 3 o 4 párrafos
- Si hay ejemplos numéricos en el contexto, incluilos — son muy útiles
- Mencioná la página cuando sea útil para que el lector pueda profundizar

    Respuesta:"""

# ── PÁGINA ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Contabilidad Básica — Asistente",
    page_icon="📊",
    layout="centered"
)

st.markdown("""
<style>
    .titulo { font-size: 1.8rem; font-weight: bold; color: #1a3a5c; }
    .subtitulo { font-size: 1rem; color: #666; margin-bottom: 2rem; }
    .fuente { font-size: 0.85rem; color: #888; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ── LOGIN ──────────────────────────────────────────────────────────────────────

def mostrar_login():
    st.markdown('<div class="titulo">📊 Contabilidad Básica</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitulo">Asistente del libro de Simaro y Tonelli</div>', unsafe_allow_html=True)
    st.markdown("---")

    with st.form("login"):
        contrasena = st.text_input("Contraseña", type="password")
        ingresar = st.form_submit_button("Ingresar")

        if ingresar:
            if contrasena == PASSWORD:
                st.session_state.autenticado = True
                st.rerun()
            else:
                st.error("Contraseña incorrecta.")


# ── CARGA DE RECURSOS ──────────────────────────────────────────────────────────

@st.cache_resource
def cargar_recursos():
    """
    Carga embeddings, FAISS y LLM — una sola vez cuando arranca la app.
    @st.cache_resource evita recargar en cada pregunta.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(
        "./faiss_db",
        embeddings,
        allow_dangerous_deserialization=True
    )
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    return embeddings, vector_store, llm


# ── BÚSQUEDA CON FILTRO POR UMBRAL ────────────────────────────────────────────

def buscar_con_umbral(vector_store, pregunta: str) -> list:
    """
    Busca fragmentos relevantes y filtra por umbral de distancia.

    similarity_search_with_score devuelve (documento, score) donde:
    - score bajo  → fragmento MUY similar a la pregunta
    - score alto  → fragmento POCO similar

    Solo pasan los fragmentos con score < UMBRAL.
    Si ninguno pasa (umbral muy estricto), devuelve el mejor disponible.

    Esta función es idéntica a la de evaluar.py — garantiza que
    el chatbot y la evaluación usen exactamente la misma lógica.
    """
    docs_con_score = vector_store.similarity_search_with_score(pregunta, k=K)
    fragmentos = [doc for doc, score in docs_con_score if score < UMBRAL]
    if not fragmentos:
        fragmentos = [docs_con_score[0][0]]
    return fragmentos


# ── CHATBOT ────────────────────────────────────────────────────────────────────

def mostrar_chatbot():
    st.markdown('<div class="titulo">📊 Contabilidad Básica</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitulo">Asistente basado en el libro de Simaro y Tonelli.<br>Hacé tu pregunta sobre contabilidad.</div>', unsafe_allow_html=True)

    if st.button("Cerrar sesión", type="secondary"):
        st.session_state.autenticado = False
        st.session_state.historial = []
        st.rerun()

    with st.spinner("Cargando el sistema..."):
        _, vector_store, llm = cargar_recursos()

    if "historial" not in st.session_state:
        st.session_state.historial = []

    for item in st.session_state.historial:
        with st.chat_message("user"):
            st.write(item["pregunta"])
        with st.chat_message("assistant"):
            st.write(item["respuesta"])
            if item["paginas"]:
                st.markdown(
                    f'<div class="fuente">📄 {item["paginas"]}</div>',
                    unsafe_allow_html=True
                )

    pregunta = st.chat_input("Escribí tu pregunta sobre contabilidad...")

    if pregunta:
        with st.chat_message("user"):
            st.write(pregunta)

        with st.chat_message("assistant"):
            with st.spinner("Buscando en el libro..."):

                # Buscamos fragmentos con filtro por umbral
                fragmentos = buscar_con_umbral(vector_store, pregunta)

                # Armamos el contexto con los fragmentos que pasaron el filtro
                contexto = "\n\n".join([f.page_content for f in fragmentos])

                # Llamamos al LLM directamente — sin RetrievalQA
                # Misma lógica que evaluar.py para consistencia
                prompt_final = TEMPLATE.replace("{context}", contexto).replace("{question}", "")
                respuesta = llm.invoke([
                    SystemMessage(content=prompt_final),
                    HumanMessage(content=pregunta)
                ])
                respuesta_texto = respuesta.content

            st.write(respuesta_texto)

            # Páginas consultadas
            paginas = sorted(set(
                doc.metadata.get("page", 0) + 1
                for doc in fragmentos
                if doc.metadata.get("page") is not None
            ))

            paginas_str = ""
            if paginas:
                paginas_str = f"Páginas consultadas: {', '.join(map(str, paginas))}"
                st.markdown(
                    f'<div class="fuente">📄 {paginas_str}</div>',
                    unsafe_allow_html=True
                )

        st.session_state.historial.append({
            "pregunta": pregunta,
            "respuesta": respuesta_texto,
            "paginas": paginas_str
        })


# ── PUNTO DE ENTRADA ───────────────────────────────────────────────────────────

if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

if not st.session_state.autenticado:
    mostrar_login()
else:
    mostrar_chatbot()