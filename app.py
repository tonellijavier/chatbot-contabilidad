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
# PARA CORRERLO:
#   streamlit run app.py
# ==============================================================================

import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

load_dotenv()

PASSWORD = os.getenv("CHATBOT_PASSWORD")

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
def cargar_sistema():
    """
    Carga FAISS desde disco — no necesita el PDF original.
    Se ejecuta una sola vez cuando arranca la app.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # FAISS carga los vectores desde los archivos index.faiss e index.pkl
    # allow_dangerous_deserialization=True es necesario para cargar el .pkl
    vector_store = FAISS.load_local(
        "./faiss_db",
        embeddings,
        allow_dangerous_deserialization=True
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 12}
    )

    template = """Sos un asistente especializado en contabilidad que responde
preguntas basándote en el libro "Contabilidad Básica" de Jorge Simaro y Omar Tonelli.

Contexto del libro:
{context}

Pregunta: {question}

Instrucciones:
- Usá el contexto del libro como base principal para tu respuesta
- Si la respuesta está explícita en el contexto, explicala con claridad
- Si no está explícita pero podés deducirla razonando sobre el contexto,
  hacelo y aclará que es una deducción basada en el libro
- Si el tema genuinamente no está en el contexto proporcionado, decilo
- Respondé en español, de forma clara y didáctica
- Si hay ejemplos numéricos en el contexto, incluilos — son muy útiles
- Mencioná la página cuando sea útil para que el lector pueda profundizar

Respuesta:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    cadena = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    return cadena


# ── CHATBOT ────────────────────────────────────────────────────────────────────

def mostrar_chatbot():
    st.markdown('<div class="titulo">📊 Contabilidad Básica</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitulo">Asistente basado en el libro de Simaro y Tonelli.<br>Hacé tu pregunta sobre contabilidad.</div>', unsafe_allow_html=True)

    if st.button("Cerrar sesión", type="secondary"):
        st.session_state.autenticado = False
        st.session_state.historial = []
        st.rerun()

    with st.spinner("Cargando el sistema..."):
        cadena = cargar_sistema()

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
                resultado = cadena.invoke({"query": pregunta})
                respuesta = resultado["result"]
                fuentes = resultado["source_documents"]

            st.write(respuesta)

            paginas = sorted(set(
                doc.metadata.get("page", 0) + 1
                for doc in fuentes
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
            "respuesta": respuesta,
            "paginas": paginas_str
        })


# ── PUNTO DE ENTRADA ───────────────────────────────────────────────────────────

if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

if not st.session_state.autenticado:
    mostrar_login()
else:
    mostrar_chatbot()