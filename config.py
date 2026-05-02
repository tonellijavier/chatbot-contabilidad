# ==============================================================================
# CONFIG.PY — Configuración global del chatbot de contabilidad
# ==============================================================================
#
# Centraliza todos los parámetros del sistema.
# Para cambiar el comportamiento del chatbot, modificar solo este archivo.
# ==============================================================================

import os
from dotenv import load_dotenv

load_dotenv()

# ── ACCESO ─────────────────────────────────────────────────────────────────────

PASSWORD = os.getenv("CHATBOT_PASSWORD")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── MODELO ─────────────────────────────────────────────────────────────────────

LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.2

# ── RAG ────────────────────────────────────────────────────────────────────────

EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
FAISS_PATH = "./faiss_db"

# k = cuántos fragmentos busca FAISS antes de filtrar
# umbral = distancia máxima aceptada (score bajo = muy similar, score alto = poco similar)
# Configuración optimizada mediante RAGAS con chunk_size=1000

K = 12
UMBRAL_DEFAULT = 1.5

# Configuraciones por tipo de pregunta (routing)
# Comparaciones necesitan más contexto — k alto, umbral alto
# Definiciones necesitan precisión — k bajo, umbral más estricto
CONFIGURACIONES_ROUTING = {
    "comparacion": {"k": 12, "umbral": 1.5},
    "definicion":  {"k": 6,  "umbral": 1.0},
    "ejemplo":     {"k": 8,  "umbral": 1.2},
    "lista":       {"k": 10, "umbral": 1.3},
    "fuera_dominio": {"k": 4, "umbral": 0.8},
    "general":     {"k": 12, "umbral": 1.5},
}