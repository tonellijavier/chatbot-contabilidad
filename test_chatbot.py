# ==============================================================================
# TEST_CHATBOT.PY — Tests unitarios del chatbot de contabilidad
# ==============================================================================
#
# Testea las funciones del chatbot sin necesitar el LLM ni FAISS.
# Son tests unitarios puros — rápidos, sin costo de tokens, sin conexión.
#
# PARA CORRERLOS:
#   pytest test_chatbot.py -v
#
# PARA CORRER UN TEST ESPECÍFICO:
#   pytest test_chatbot.py::TestMemoria::test_historial_vacio -v
# ==============================================================================

import pytest
from chatbot.retriever import (
    normalizar_pregunta,
    detectar_tipo_pregunta,
    es_fuera_del_dominio,
    obtener_configuracion,
    extraer_contexto_conversacion,
)
from chatbot.feedback import detectar_tema


# ── TESTS DE NORMALIZACIÓN ─────────────────────────────────────────────────────

class TestNormalizacion:
    """
    Verifica que el diccionario de sinónimos normaliza correctamente
    el vocabulario informal de estudiantes universitarios argentinos.
    """

    def test_sinonimo_plata(self):
        resultado = normalizar_pregunta("cuánta plata tengo?")
        assert "patrimonio" in resultado

    def test_sinonimo_guita(self):
        resultado = normalizar_pregunta("qué es la guita en contabilidad?")
        assert "activo" in resultado

    def test_sinonimo_deuda(self):
        resultado = normalizar_pregunta("cómo se registra una deuda?")
        assert "pasivo" in resultado

    def test_sinonimo_ganancias(self):
        resultado = normalizar_pregunta("qué son las ganancias?")
        assert "resultados positivos" in resultado

    def test_sinonimo_abreviacion_pn(self):
        resultado = normalizar_pregunta("cómo calculo el pn?")
        assert "patrimonio neto" in resultado

    def test_no_modifica_terminos_correctos(self):
        pregunta = "qué es el activo corriente?"
        resultado = normalizar_pregunta(pregunta)
        assert "activo" in resultado

    def test_normaliza_en_minusculas(self):
        resultado = normalizar_pregunta("Cuánta PLATA tengo?")
        assert "patrimonio" in resultado


# ── TESTS DE ROUTING ───────────────────────────────────────────────────────────

class TestRouting:
    """
    Verifica que el sistema detecta correctamente el tipo de pregunta
    para elegir la configuración óptima de búsqueda.
    """

    def test_detecta_comparacion_diferencia(self):
        tipo = detectar_tipo_pregunta("¿cuál es la diferencia entre activo y pasivo?")
        assert tipo == "comparacion"

    def test_detecta_comparacion_versus(self):
        tipo = detectar_tipo_pregunta("devengado versus percibido")
        assert tipo == "comparacion"

    def test_detecta_comparacion_distinto(self):
        tipo = detectar_tipo_pregunta("¿en qué son distintas las variaciones permutativas y modificativas?")
        assert tipo == "comparacion"

    def test_detecta_definicion_que_es(self):
        tipo = detectar_tipo_pregunta("¿qué es el patrimonio neto?")
        assert tipo == "definicion"

    def test_detecta_definicion_concepto(self):
        tipo = detectar_tipo_pregunta("¿cuál es el concepto de devengado?")
        assert tipo == "definicion"

    def test_detecta_definicion_significa(self):
        tipo = detectar_tipo_pregunta("¿qué significa hecho sustancial?")
        assert tipo == "definicion"

    def test_detecta_ejemplo(self):
        tipo = detectar_tipo_pregunta("dame un ejemplo de variación permutativa")
        assert tipo == "ejemplo"

    def test_detecta_lista_cuales_son(self):
        tipo = detectar_tipo_pregunta("¿cuáles son los tipos de variaciones patrimoniales?")
        assert tipo == "lista"

    def test_detecta_lista_enumera(self):
        tipo = detectar_tipo_pregunta("enumerá los elementos del patrimonio")
        assert tipo == "lista"

    def test_general_para_preguntas_sin_tipo(self):
        tipo = detectar_tipo_pregunta("el devengado y los resultados")
        assert tipo == "general"


# ── TESTS DE CONFIGURACIÓN POR TIPO ───────────────────────────────────────────

class TestConfiguracion:
    """
    Verifica que cada tipo de pregunta usa la configuración correcta de k y umbral.
    """

    def test_comparacion_usa_k_alto(self):
        config = obtener_configuracion("comparacion")
        assert config["k"] == 12

    def test_definicion_usa_k_bajo(self):
        config = obtener_configuracion("definicion")
        assert config["k"] == 6

    def test_definicion_usa_umbral_bajo(self):
        config = obtener_configuracion("definicion")
        assert config["umbral"] == 1.0

    def test_comparacion_usa_umbral_alto(self):
        config = obtener_configuracion("comparacion")
        assert config["umbral"] == 1.5

    def test_tipo_desconocido_usa_config_general(self):
        config = obtener_configuracion("tipo_que_no_existe")
        assert "k" in config
        assert "umbral" in config


# ── TESTS DE DETECCIÓN FUERA DEL DOMINIO ──────────────────────────────────────

class TestFueraDominio:
    """
    Verifica que el sistema detecta correctamente preguntas que no tienen
    relación con contabilidad.
    """

    def test_pregunta_sobre_futbol(self):
        assert es_fuera_del_dominio("¿quién ganó el mundial 2022?") == True

    def test_pregunta_sobre_cocina(self):
        assert es_fuera_del_dominio("¿cómo se hace una pizza?") == True

    def test_pregunta_sobre_politica(self):
        assert es_fuera_del_dominio("¿quién es el presidente?") == True

    def test_pregunta_contable_no_es_fuera(self):
        assert es_fuera_del_dominio("¿qué es el activo corriente?") == False

    def test_pregunta_patrimonio_no_es_fuera(self):
        assert es_fuera_del_dominio("¿cómo se calcula el patrimonio neto?") == False

    def test_pregunta_devengado_no_es_fuera(self):
        assert es_fuera_del_dominio("¿qué es el principio de devengado?") == False


# ── TESTS DE DETECCIÓN DE TEMA ─────────────────────────────────────────────────

class TestDeteccionTema:
    """
    Verifica que el sistema asigna correctamente el tema contable
    para el análisis de feedback.
    """

    def test_detecta_tema_devengado(self):
        tema = detectar_tema("¿qué es el principio de devengado?")
        assert tema == "devengado"

    def test_detecta_tema_variaciones(self):
        tema = detectar_tema("¿qué es una variación permutativa?")
        assert tema == "variaciones"

    def test_detecta_tema_patrimonio(self):
        tema = detectar_tema("¿cómo se calcula el activo?")
        assert tema == "patrimonio"

    def test_detecta_tema_cuentas(self):
        tema = detectar_tema("¿qué es el debe y el haber?")
        assert tema == "cuentas"

    def test_detecta_tema_ejercicio(self):
        tema = detectar_tema("¿qué es el ejercicio económico?")
        assert tema == "ejercicio"

    def test_tema_general_para_preguntas_sin_tema(self):
        tema = detectar_tema("¿cómo funciona la contabilidad?")
        assert tema == "general"


# ── TESTS DE MEMORIA CONVERSACIONAL ───────────────────────────────────────────

class TestMemoria:
    """
    Verifica que el sistema de memoria conversacional funciona correctamente.

    Testea extraer_contexto_conversacion — la función que enriquece
    la búsqueda en FAISS con palabras clave del historial previo.
    """

    def test_historial_vacio_devuelve_string_vacio(self):
        """Sin historial no hay contexto que extraer."""
        resultado = extraer_contexto_conversacion([])
        assert resultado == ""

    def test_historial_con_una_pregunta(self):
        """Con un turno extrae la pregunta como contexto."""
        historial = [
            {"pregunta": "¿qué es el activo corriente?", "respuesta": "..."}
        ]
        resultado = extraer_contexto_conversacion(historial)
        assert "activo corriente" in resultado

    def test_historial_con_multiples_preguntas_usa_ultimas_dos(self):
        """Con varios turnos solo usa los últimos 2."""
        historial = [
            {"pregunta": "¿qué es el patrimonio?", "respuesta": "..."},
            {"pregunta": "¿qué es el activo?", "respuesta": "..."},
            {"pregunta": "¿qué es el pasivo?", "respuesta": "..."},
        ]
        resultado = extraer_contexto_conversacion(historial)
        # Debe incluir las últimas dos preguntas
        assert "activo" in resultado
        assert "pasivo" in resultado
        # No debe incluir la primera
        assert "patrimonio" not in resultado

    def test_contexto_respeta_limite_de_caracteres(self):
        """El contexto no supera max_chars."""
        historial = [
            {"pregunta": "x" * 200, "respuesta": "..."},
            {"pregunta": "y" * 200, "respuesta": "..."},
        ]
        resultado = extraer_contexto_conversacion(historial, max_chars=150)
        assert len(resultado) <= 150

    def test_contexto_por_defecto_150_caracteres(self):
        """El límite por defecto es 150 caracteres."""
        historial = [
            {"pregunta": "a" * 100, "respuesta": "..."},
            {"pregunta": "b" * 100, "respuesta": "..."},
        ]
        resultado = extraer_contexto_conversacion(historial)
        assert len(resultado) <= 150

    def test_historial_un_turno_devuelve_solo_ese(self):
        """Con un solo turno devuelve solo esa pregunta."""
        historial = [
            {"pregunta": "variaciones patrimoniales", "respuesta": "..."}
        ]
        resultado = extraer_contexto_conversacion(historial)
        assert resultado == "variaciones patrimoniales"