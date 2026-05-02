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
# MEMORIA CONVERSACIONAL:
# Manda los últimos mensajes al LLM limitados por tokens, no por cantidad fija.
# La búsqueda en FAISS se enriquece con palabras clave del historial.
# El modelo decide si la pregunta es de seguimiento o independiente.
#
# MANEJO DE ERRORES:
# Rate limit de Groq → mensaje amigable al usuario
# Error de Neon (feedback) → el chatbot sigue funcionando
# Error de FAISS → mensaje de sistema no disponible
#
# PARA CORRERLO:
#   streamlit run app.py
# ==============================================================================

import uuid
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from groq import RateLimitError

from config import PASSWORD, LLM_MODEL, LLM_TEMPERATURE, EMBEDDINGS_MODEL, FAISS_PATH
from chatbot.retriever import buscar_fragmentos, normalizar_pregunta, extraer_contexto_conversacion
from chatbot.prompts import TEMPLATE_PRINCIPAL, TEMPLATE_FUERA_DOMINIO, construir_prompt
from chatbot.feedback import guardar_feedback, detectar_tema

# ── CONFIGURACIÓN DE MEMORIA ───────────────────────────────────────────────────

MAX_MENSAJES_HISTORIAL = 6      # máximo de mensajes a considerar
MAX_TOKENS_HISTORIAL = 500      # límite de tokens del historial

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
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
        vector_store = FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        llm = ChatGroq(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
        return vector_store, llm, None
    except Exception as e:
        return None, None, str(e)


# ── MEMORIA CONVERSACIONAL ─────────────────────────────────────────────────────

def preparar_historial(historial: list) -> list:
    """
    Prepara los mensajes del historial para mandar al LLM.

    Estrategia profesional: siempre mandamos historial pero limitado por tokens,
    no por número fijo de mensajes. El modelo decide si es relevante.

    Empieza desde el mensaje más reciente y va hacia atrás hasta
    alcanzar el límite de tokens.
    """
    mensajes = []
    tokens_usados = 0

    for item in reversed(historial[-MAX_MENSAJES_HISTORIAL:]):
        # Aproximación de tokens: palabras * 1.3
        tokens_pregunta  = len(item["pregunta"].split()) * 1.3
        tokens_respuesta = len(item["respuesta"].split()) * 1.3
        tokens_item = tokens_pregunta + tokens_respuesta

        if tokens_usados + tokens_item > MAX_TOKENS_HISTORIAL:
            break

        mensajes.insert(0, item)
        tokens_usados += tokens_item

    return mensajes


def construir_mensajes_llm(system_prompt: str, historial_filtrado: list, pregunta: str) -> list:
    """
    Construye la lista de mensajes para el LLM incluyendo el historial.

    Formato:
        SystemMessage (prompt con contexto del libro)
        HumanMessage  (pregunta anterior)
        AIMessage     (respuesta anterior)
        ...
        HumanMessage  (pregunta actual)
    """
    mensajes = [SystemMessage(content=system_prompt)]

    for item in historial_filtrado:
        mensajes.append(HumanMessage(content=item["pregunta"]))
        mensajes.append(AIMessage(content=item["respuesta"]))

    mensajes.append(HumanMessage(content=pregunta))
    return mensajes


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

    # Cargar recursos con manejo de error
    with st.spinner("Cargando el sistema..."):
        vector_store, llm, error_carga = cargar_recursos()

    if error_carga:
        st.error("El sistema no está disponible temporalmente. Intentá en unos minutos.")
        return

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
                        try:
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
                        except Exception:
                            ok = False  # si falla Neon, el chatbot sigue funcionando

                        if ok:
                            st.session_state.feedback_dado[i] = "positivo"
                        else:
                            st.session_state.feedback_dado[i] = "positivo"  # igual registramos visualmente
                        st.rerun()

                with col2:
                    if st.button("👎", key=f"negativo_{i}"):
                        try:
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
                        except Exception:
                            ok = False

                        if ok:
                            st.session_state.feedback_dado[i] = "negativo"
                        else:
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
                try:
                    # Enriquecer búsqueda con contexto de la conversación
                    contexto_conversacion = extraer_contexto_conversacion(
                        st.session_state.historial
                    )
                    pregunta_enriquecida = (
                        f"{contexto_conversacion} {pregunta}".strip()
                        if contexto_conversacion else pregunta
                    )

                    # Pipeline de búsqueda con pregunta enriquecida
                    fragmentos, tipo, config = buscar_fragmentos(
                        vector_store, pregunta_enriquecida
                    )

                    # Elegir template
                    template = (
                        TEMPLATE_FUERA_DOMINIO
                        if tipo == "fuera_dominio"
                        else TEMPLATE_PRINCIPAL
                    )

                    # Construir contexto y prompt
                    contexto = "\n\n".join([f.page_content for f in fragmentos])
                    prompt_final = construir_prompt(template, contexto, pregunta)

                    # Preparar historial limitado por tokens
                    historial_filtrado = preparar_historial(st.session_state.historial)

                    # Llamar al LLM con historial
                    mensajes = construir_mensajes_llm(prompt_final, historial_filtrado, pregunta)
                    respuesta = llm.invoke(mensajes)
                    respuesta_texto = respuesta.content

                except RateLimitError:
                    respuesta_texto = (
                        "El sistema está recibiendo muchas consultas en este momento. "
                        "Por favor, intentá de nuevo en unos minutos. 🙏"
                    )
                    fragmentos = []
                    tipo = "error"
                    config = {"k": 0, "umbral": 0}

                except Exception as e:
                    respuesta_texto = (
                        "Ocurrió un error inesperado. "
                        "Por favor, intentá de nuevo o recargá la página."
                    )
                    fragmentos = []
                    tipo = "error"
                    config = {"k": 0, "umbral": 0}

            st.write(respuesta_texto)

            # Páginas consultadas
            paginas_str = ""
            if fragmentos:
                paginas = sorted(set(
                    doc.metadata.get("page", 0) + 1
                    for doc in fragmentos
                    if doc.metadata.get("page") is not None
                ))
                if paginas:
                    paginas_str = f"Páginas consultadas: {', '.join(map(str, paginas))}"
                    st.markdown(
                        f'<div class="fuente">📄 {paginas_str}</div>',
                        unsafe_allow_html=True
                    )

        # Guardar en historial
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