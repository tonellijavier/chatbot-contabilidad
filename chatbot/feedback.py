# ==============================================================================
# CHATBOT/FEEDBACK.PY — Registro de feedback de usuarios
# ==============================================================================
#
# Guarda el feedback (👍 / 👎) de cada respuesta en Neon (PostgreSQL).
# Permite analizar qué preguntas responde mal el chatbot y mejorar
# iterativamente basándose en datos reales de uso.
#
# CONSULTAS ÚTILES PARA ANALIZAR EL FEEDBACK:
#
#   -- Temas con más feedback negativo
#   SELECT tema_detectado, COUNT(*) as negativos
#   FROM feedback WHERE voto = 'negativo'
#   GROUP BY tema_detectado ORDER BY negativos DESC;
#
#   -- Preguntas específicas que fallan
#   SELECT pregunta_original, COUNT(*) as negativos
#   FROM feedback WHERE voto = 'negativo'
#   GROUP BY pregunta_original ORDER BY negativos DESC LIMIT 10;
#
#   -- Impacto de cambios de configuración
#   SELECT chunk_size, umbral_usado,
#          AVG(CASE WHEN voto = 'positivo' THEN 1.0 ELSE 0.0 END) as satisfaccion
#   FROM feedback GROUP BY chunk_size, umbral_usado;
# ==============================================================================

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Versión actual del chatbot — incrementar al deployar cambios significativos
VERSION_APP = "1.1.0"
CHUNK_SIZE_ACTUAL = 1000  # chunk_size con el que está indexado el libro actualmente


def get_conn():
    """Abre una conexión a Neon."""
    return psycopg2.connect(os.getenv("DATABASE_URL"))


def guardar_feedback(
    sesion_id: str,
    pregunta_original: str,
    pregunta_normalizada: str,
    tipo_pregunta: str,
    tema_detectado: str,
    k_usado: int,
    umbral_usado: float,
    fragmentos_usados: int,
    paginas: str,
    respuesta: str,
    voto: str,
    modelo: str,
    comentario: str = None,
) -> bool:
    """
    Guarda un registro de feedback en la base de datos.

    Parámetros:
        voto: 'positivo' o 'negativo'
        comentario: opcional, texto libre del usuario

    Devuelve True si se guardó correctamente, False si hubo error.
    """
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO feedback (
                sesion_id, pregunta_original, pregunta_normalizada,
                tipo_pregunta, tema_detectado,
                k_usado, umbral_usado, fragmentos_usados, paginas,
                respuesta, voto, comentario,
                modelo, chunk_size, version_app
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s
            )
        """, (
            sesion_id, pregunta_original, pregunta_normalizada,
            tipo_pregunta, tema_detectado,
            k_usado, umbral_usado, fragmentos_usados, paginas,
            respuesta, voto, comentario,
            modelo, CHUNK_SIZE_ACTUAL, VERSION_APP
        ))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error guardando feedback: {e}")
        return False


def detectar_tema(pregunta: str) -> str:
    """
    Detecta el tema contable de la pregunta para análisis de feedback.
    Complementa el routing por tipo — este detecta el TEMA, no el tipo.
    """
    TEMAS = {
        "devengado":     ["devengado", "percibido", "hecho sustancial", "hecho generador"],
        "variaciones":   ["permutativa", "modificativa", "variación", "variacion"],
        "patrimonio":    ["activo", "pasivo", "patrimonio neto"],
        "cuentas":       ["cuenta corriente", "debe", "haber", "cuenta t"],
        "ejercicio":     ["ejercicio económico", "ejercicio economico", "período", "periodo"],
        "resultados":    ["resultado positivo", "resultado negativo", "ganancia", "pérdida", "perdida"],
        "capital":       ["capital", "aporte", "socio", "retiro"],
        "asientos":      ["asiento", "registración", "registracion", "libro diario"],
        "balance":       ["balance", "estado contable", "estado financiero"],
    }

    pregunta_lower = pregunta.lower()
    for tema, palabras in TEMAS.items():
        if any(p in pregunta_lower for p in palabras):
            return tema
    return "general"