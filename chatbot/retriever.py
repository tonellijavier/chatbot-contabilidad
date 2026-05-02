# ==============================================================================
# CHATBOT/RETRIEVER.PY — Lógica de búsqueda y recuperación de fragmentos
# ==============================================================================
#
# Incluye:
#   - Normalización de preguntas (sinónimos informales, vocabulario universitario)
#   - Routing por tipo de pregunta (comparaciones, definiciones, ejemplos)
#   - Detección de preguntas fuera del dominio
#   - Búsqueda con filtro por umbral de distancia
# ==============================================================================

from config import CONFIGURACIONES_ROUTING, K, UMBRAL_DEFAULT

# ── SINÓNIMOS Y VOCABULARIO INFORMAL ──────────────────────────────────────────
#
# Mapea vocabulario informal de estudiantes universitarios argentinos (18-25 años)
# al vocabulario técnico del libro.
# Incluye: jerga argentina, abreviaciones, errores comunes, términos informales.

SINONIMOS = {
    # Jerga argentina
    "plata":            "patrimonio",
    "guita":            "activo",
    "deuda":            "pasivo",
    "ganancias":        "resultados positivos",
    "pérdidas":         "resultados negativos",
    "perdidas":         "resultados negativos",
    "lo que tengo":     "activo",
    "lo que debo":      "pasivo",
    "lo que me queda":  "patrimonio neto",

    # Abreviaciones comunes
    "pn":               "patrimonio neto",
    "rc":               "resultado corriente",
    "ee":               "estados contables",
    "eeff":             "estados financieros",

    # Términos informales
    "plata en caja":    "activo corriente",
    "bienes":           "activo",
    "lo que vale":      "valor",

    # Errores de tipeo comunes
    "devengamiento":    "devengado",
    "devengacion":      "devengado",
    "permutativa":      "variación permutativa",
    "modificativa":     "variación modificativa",
    "patrimonial":      "patrimoniales",
    "contabilida":      "contabilidad",
    "contabiliad":      "contabilidad",
    "asientoo":         "asiento",
    "balancee":         "balance",

    # Sinónimos técnicos que los estudiantes usan
    "debe y haber":     "debe y haber contable",
    "t account":        "cuenta t",
    "cuenta t":         "debe y haber",
    "entrada":          "ingreso",
    "salida":           "egreso",
    "cobro":            "percepción",
    "pago":             "erogación",
}

# ── PALABRAS CLAVE FUERA DEL DOMINIO ──────────────────────────────────────────
#
# Si la pregunta contiene estas palabras y NO contiene palabras de contabilidad,
# se considera fuera del dominio.

PALABRAS_FUERA_DOMINIO = [
    "fútbol", "futbol", "música", "musica", "película", "pelicula",
    "cocina", "receta", "clima", "tiempo", "deporte", "política", "politica",
    "historia", "geografía", "geografia", "física", "fisica", "química", "quimica",
    "biología", "biologia", "matemática", "matematica", "capital", "país", "pais",
    "presidente", "gobierno", "partido", "elecciones",
]

PALABRAS_CONTABILIDAD = [
    "activo", "pasivo", "patrimonio", "cuenta", "balance", "asiento",
    "debe", "haber", "contabilidad", "ejercicio", "devengado", "variación",
    "variacion", "resultado", "ingreso", "egreso", "capital", "saldo",
    "transferencia", "deudor", "acreedor", "inventario", "depreciación",
    "depreciacion", "amortización", "amortizacion", "costo", "gasto",
]


# ── NORMALIZACIÓN ──────────────────────────────────────────────────────────────

def normalizar_pregunta(pregunta: str) -> str:
    """
    Normaliza la pregunta del usuario reemplazando vocabulario informal
    por términos técnicos del libro.

    No modifica la pregunta original — trabaja sobre una copia en minúsculas
    para la búsqueda, pero la interfaz muestra la pregunta original.
    """
    pregunta_normalizada = pregunta.lower()
    for informal, formal in SINONIMOS.items():
        if informal in pregunta_normalizada:
            pregunta_normalizada = pregunta_normalizada.replace(informal, formal)
    return pregunta_normalizada


# ── DETECCIÓN DE DOMINIO ───────────────────────────────────────────────────────

def es_fuera_del_dominio(pregunta: str) -> bool:
    """
    Detecta si la pregunta no tiene relación con contabilidad.

    Lógica: si contiene palabras fuera del dominio Y no contiene
    palabras de contabilidad, se considera fuera del dominio.
    """
    pregunta_lower = pregunta.lower()

    tiene_fuera = any(palabra in pregunta_lower for palabra in PALABRAS_FUERA_DOMINIO)
    tiene_contabilidad = any(palabra in pregunta_lower for palabra in PALABRAS_CONTABILIDAD)

    return tiene_fuera and not tiene_contabilidad


# ── ROUTING POR TIPO DE PREGUNTA ───────────────────────────────────────────────

def detectar_tipo_pregunta(pregunta: str) -> str:
    """
    Detecta el tipo de pregunta para elegir la configuración óptima de búsqueda.

    Tipos:
    - comparacion: preguntas que comparan dos conceptos
    - definicion:  preguntas que piden definir un término
    - ejemplo:     preguntas que piden ejemplos
    - lista:       preguntas que piden enumerar conceptos
    - fuera_dominio: preguntas no relacionadas con contabilidad
    - general:     cualquier otra pregunta
    """
    pregunta_lower = pregunta.lower()

    if es_fuera_del_dominio(pregunta):
        return "fuera_dominio"

    if any(p in pregunta_lower for p in [
        "diferencia", "diferencia entre", "comparar", "versus", "vs",
        "distinto", "distinción", "distincion", "igual", "similar"
    ]):
        return "comparacion"

    if any(p in pregunta_lower for p in [
        "qué es", "que es", "definí", "defini", "concepto", "significa",
        "significado", "definición", "definicion", "explicá", "explica"
    ]):
        return "definicion"

    if any(p in pregunta_lower for p in [
        "ejemplo", "ejemplificá", "ejemplifica", "ilustrá", "ilustra",
        "caso", "supongamos", "imaginate"
    ]):
        return "ejemplo"

    if any(p in pregunta_lower for p in [
        "listá", "lista", "enumerá", "enumera", "cuáles son", "cuales son",
        "qué tipos", "que tipos", "clasificación", "clasificacion"
    ]):
        return "lista"

    return "general"


def obtener_configuracion(tipo: str) -> dict:
    """Devuelve la configuración de k y umbral para el tipo de pregunta."""
    return CONFIGURACIONES_ROUTING.get(tipo, {"k": K, "umbral": UMBRAL_DEFAULT})


# ── BÚSQUEDA CON FILTRO POR UMBRAL ────────────────────────────────────────────

def buscar_fragmentos(vector_store, pregunta: str) -> tuple:
    """
    Pipeline completo de búsqueda:
    1. Normaliza la pregunta (sinónimos informales)
    2. Detecta el tipo de pregunta (routing)
    3. Elige la configuración óptima de k y umbral
    4. Busca con FAISS y filtra por umbral

    Devuelve: (fragmentos, tipo_pregunta, config_usada)
    """
    # Paso 1 — normalizar
    pregunta_normalizada = normalizar_pregunta(pregunta)

    # Paso 2 — detectar tipo
    tipo = detectar_tipo_pregunta(pregunta_normalizada)

    # Paso 3 — elegir configuración
    config = obtener_configuracion(tipo)
    k = config["k"]
    umbral = config["umbral"]

    # Paso 4 — buscar con FAISS
    docs_con_score = vector_store.similarity_search_with_score(
        pregunta_normalizada, k=k
    )

    # Filtrar por umbral
    fragmentos = [doc for doc, score in docs_con_score if score < umbral]

    # Si ninguno pasó el filtro, devolvemos el mejor disponible
    if not fragmentos:
        fragmentos = [docs_con_score[0][0]]

    return fragmentos, tipo, config