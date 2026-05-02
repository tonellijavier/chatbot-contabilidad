# ==============================================================================
# APP.PY — Interfaz Streamlit del chatbot de contabilidad
# ==============================================================================
#
# Solo maneja la interfaz de usuario.
# La lógica de búsqueda está en chatbot/retriever.py
# Los prompts están en chatbot/prompts.py
# La configuración está en config.py
# El feedback está en chatbot/feedback.py
#
# PARA CORRERLO:
#   streamlit run app.py
# ==============================================================================

import uuid
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

from config import PASSWORD, LLM_MODEL, LLM_TEMPERATURE, EMBEDDINGS_MODEL, FAISS_PATH
from chatbot.retriever import buscar_fragmentos, normalizar_pregunta
from chatbot.prompts import TEMPLATE_PRINCIPAL, TEMPLATE_FUERA_DOMINIO, construir_prompt
from chatbot.feedback import guardar_feedback, detectar_tema

# ── PÁGINA ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Contabilidad Básica — Asistente",
    page_icon="📊",
    layout="centered"
)

st.markdown("""
<style>
    .titulo    { font-size: 1.8rem; font-weight: bold; color: #1a3a5c; }
    .subtitulo { font-size: 1rem; color: #666; margin-bottom: 2rem; }
    .fuente    { font-size: 0.85rem; color: #888; margin-top: 0.5rem; }
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
    Carga embeddings, FAISS y LLM una sola vez cuando arranca la app.
    @st.cache_resource evita recargar en cada pregunta.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    vector_store = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    llm = ChatGroq(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    return vector_store, llm


# ── CHATBOT ────────────────────────────────────────────────────────────────────

def mostrar_chatbot():
    st.markdown('<div class="titulo">📊 Contabilidad Básica</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitulo">Asistente basado en el libro de Simaro y Tonelli.<br>'
        'Hacé tu pregunta sobre contabilidad.</div>',
        unsafe_allow_html=True
    )

    if st.button("Cerrar sesión", type="secondary"):
        st.session_state.autenticado = False
        st.session_state.historial = []
        st.rerun()

    with st.spinner("Cargando el sistema..."):
        vector_store, llm = cargar_recursos()

    # Inicializar estado de sesión
    if "historial" not in st.session_state:
        st.session_state.historial = []

    if "sesion_id" not in st.session_state:
        st.session_state.sesion_id = str(uuid.uuid4())[:8]

    if "feedback_dado" not in st.session_state:
        st.session_state.feedback_dado = {}

    # Mostrar historial con botones de feedback
    for i, item in enumerate(st.session_state.historial):
        with st.chat_message("user"):
            st.write(item["pregunta"])

        with st.chat_message("assistant"):
            st.write(item["respuesta"])

            if item["paginas"]:
                st.markdown(
                    f'<div class="fuente">📄 {item["paginas"]}</div>',
                    unsafe_allow_html=True
                )

            # Botones de feedback
            if i not in st.session_state.feedback_dado:
                col1, col2, col3 = st.columns([1, 1, 8])

                with col1:
                    if st.button("👍", key=f"positivo_{i}"):
                        ok = guardar_feedback(
                            sesion_id=st.session_state.sesion_id,
                            pregunta_original=item["pregunta"],
                            pregunta_normalizada=item["pregunta_normalizada"],
                            tipo_pregunta=item["tipo"],
                            tema_detectado=detectar_tema(item["pregunta"]),
                            k_usado=item["config"]["k"],
                            umbral_usado=item["config"]["umbral"],
                            fragmentos_usados=item["fragmentos_usados"],
                            paginas=item["paginas"],
                            respuesta=item["respuesta"],
                            voto="positivo",
                            modelo=LLM_MODEL,
                        )
                        if ok:
                            st.session_state.feedback_dado[i] = "positivo"
                            st.rerun()

                with col2:
                    if st.button("👎", key=f"negativo_{i}"):
                        ok = guardar_feedback(
                            sesion_id=st.session_state.sesion_id,
                            pregunta_original=item["pregunta"],
                            pregunta_normalizada=item["pregunta_normalizada"],
                            tipo_pregunta=item["tipo"],
                            tema_detectado=detectar_tema(item["pregunta"]),
                            k_usado=item["config"]["k"],
                            umbral_usado=item["config"]["umbral"],
                            fragmentos_usados=item["fragmentos_usados"],
                            paginas=item["paginas"],
                            respuesta=item["respuesta"],
                            voto="negativo",
                            modelo=LLM_MODEL,
                        )
                        if ok:
                            st.session_state.feedback_dado[i] = "negativo"
                            st.rerun()

            else:
                voto = st.session_state.feedback_dado[i]
                if voto == "positivo":
                    st.caption("✓ Gracias por tu feedback 👍")
                else:
                    st.caption("✓ Gracias por tu feedback 👎 — lo usamos para mejorar")

    # Input de nueva pregunta
    pregunta = st.chat_input("Escribí tu pregunta sobre contabilidad...")

    if pregunta:
        with st.chat_message("user"):
            st.write(pregunta)

        with st.chat_message("assistant"):
            with st.spinner("Buscando en el libro..."):

                fragmentos, tipo, config = buscar_fragmentos(vector_store, pregunta)
                template = TEMPLATE_FUERA_DOMINIO if tipo == "fuera_dominio" else TEMPLATE_PRINCIPAL
                contexto = "\n\n".join([f.page_content for f in fragmentos])
                prompt_final = construir_prompt(template, contexto, pregunta)

                respuesta = llm.invoke([
                    SystemMessage(content=prompt_final),
                    HumanMessage(content=pregunta)
                ])
                respuesta_texto = respuesta.content

            st.write(respuesta_texto)

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
            "pregunta":             pregunta,
            "pregunta_normalizada": normalizar_pregunta(pregunta),
            "respuesta":            respuesta_texto,
            "paginas":              paginas_str,
            "tipo":                 tipo,
            "config":               config,
            "fragmentos_usados":    len(fragmentos),
        })

        st.rerun()


# ── PUNTO DE ENTRADA ───────────────────────────────────────────────────────────

if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

if not st.session_state.autenticado:
    mostrar_login()
else:
    mostrar_chatbot()