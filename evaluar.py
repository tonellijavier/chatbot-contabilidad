# ==============================================================================
# EVALUAR.PY — Evaluación del chatbot con RAGAS
# ==============================================================================
#
# Mide la calidad de las respuestas del chatbot de contabilidad.
# Usa preguntas y respuestas provistas por los autores del libro.
#
# Importa la lógica de búsqueda desde chatbot/retriever.py y el template
# desde chatbot/prompts.py — garantizando que evaluación y producción
# usen exactamente la misma lógica.
#
# MÉTRICAS:
#   - Faithfulness      → ¿la respuesta está basada en los fragmentos encontrados?
#   - Answer Relevancy  → ¿la respuesta responde realmente la pregunta?
#   - Context Recall    → ¿encontró toda la información necesaria? (requiere ground_truth)
#
# NOTA SOBRE TOKENS:
#   Cada corrida de 10 preguntas consume ~100.000 tokens (límite diario gratuito de Groq).
#   Con 21 preguntas totales, corrés un tema por día.
#
# PARA CORRERLO:
#   python evaluar.py
# ==============================================================================

import os
import json
from datetime import datetime
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Importamos la lógica compartida con app.py
from config import K, UMBRAL_DEFAULT, EMBEDDINGS_MODEL, FAISS_PATH, LLM_MODEL
from chatbot.retriever import buscar_fragmentos, normalizar_pregunta
from chatbot.prompts import TEMPLATE_PRINCIPAL, construir_prompt

load_dotenv()

# ── PREGUNTAS CON GROUND TRUTH ─────────────────────────────────────────────────

PREGUNTAS = [
    # ── TEMA 1: Análisis de las Variaciones Patrimoniales ──────────────────────
    {
        "question": "¿Cuál es la importancia de estudiar las variaciones patrimoniales dentro del sistema contable?",
        "ground_truth": "Es un paso previo fundamental para comprender el funcionamiento del sistema contable. Permite identificar cómo cada operación afecta la estructura y el valor del ente.",
        "tema": "variaciones_patrimoniales"
    },
    {
        "question": "¿Cómo se definen las variaciones patrimoniales permutativas?",
        "ground_truth": "Son cambios que modifican los elementos patrimoniales — activo, pasivo o patrimonio neto — pero no afectan el valor numérico final del patrimonio neto.",
        "tema": "variaciones_patrimoniales"
    },
    {
        "question": "¿Qué distingue fundamentalmente a una variación modificativa de una permutativa?",
        "ground_truth": "La diferencia radica en la alteración de la cuantía del patrimonio neto. Las permutativas mantienen el valor total inalterado, las modificativas alteran los elementos patrimoniales y modifican efectivamente el importe total del patrimonio neto.",
        "tema": "variaciones_patrimoniales"
    },
    {
        "question": "¿Qué hechos originan una variación patrimonial modificativa de capital?",
        "ground_truth": "Se originan por aportes realizados por los socios o por reducciones voluntarias del capital. Son cambios vinculados directamente a la voluntad y acción de los propietarios sobre el capital social.",
        "tema": "variaciones_patrimoniales"
    },
    {
        "question": "¿Cómo se clasifican las variaciones modificativas de resultados acumulados?",
        "ground_truth": "Se dividen en dos categorías: retiros de los propietarios (dividendos) y variaciones originadas por hechos y operaciones, que pueden ser de resultados positivos o negativos.",
        "tema": "variaciones_patrimoniales"
    },
    {
        "question": "¿En qué consiste una variación por retiros de los propietarios?",
        "ground_truth": "Son disminuciones del patrimonio neto originadas en la decisión de los socios de retirar utilidades acumuladas previamente. No deben confundirse con reducciones de capital.",
        "tema": "variaciones_patrimoniales"
    },
    {
        "question": "¿Qué caracteriza a las variaciones modificativas de resultados positivos?",
        "ground_truth": "Son aumentos producidos en el patrimonio neto que no tienen su origen en aportes de los socios. Son incrementos generados por la propia actividad o hechos económicos del ente.",
        "tema": "variaciones_patrimoniales"
    },
    {
        "question": "¿Qué diferencia existe entre un resultado negativo y una reducción voluntaria de capital?",
        "ground_truth": "Ambos disminuyen el patrimonio neto, pero los resultados negativos provienen de la operación o hechos económicos, mientras que las reducciones de capital son actos voluntarios de los socios sobre sus aportes originales.",
        "tema": "variaciones_patrimoniales"
    },
    {
        "question": "¿Qué elementos del patrimonio pueden verse alterados en una variación permutativa?",
        "ground_truth": "Se modifican los elementos del activo o del pasivo, o cambia la composición del patrimonio neto, siempre manteniendo el equilibrio del valor total sin alterar la cuantía del patrimonio neto.",
        "tema": "variaciones_patrimoniales"
    },

    # ── TEMA 2: Principio de Devengado ─────────────────────────────────────────
    {
        "question": "¿Qué problema intenta resolver el principio de devengado dentro del sistema contable?",
        "ground_truth": "Permite determinar en qué período deben reconocerse los resultados, asegurando que la información contable refleje la realidad económica del ente y no únicamente los movimientos de efectivo.",
        "tema": "devengado"
    },
    {
        "question": "¿Cómo se define el principio de devengado?",
        "ground_truth": "Establece que los efectos de las transacciones y hechos económicos deben reconocerse en el período en que ocurren, independientemente del momento en que se cobren o paguen.",
        "tema": "devengado"
    },
    {
        "question": "¿Qué diferencia existe entre el criterio de lo devengado y el criterio de lo percibido?",
        "ground_truth": "El devengado reconoce los hechos económicos cuando se generan, mientras que el percibido los reconoce solo cuando hay cobros o pagos. El percibido no resulta adecuado para medir resultados.",
        "tema": "devengado"
    },
    {
        "question": "¿Qué es un hecho generador de resultados?",
        "ground_truth": "Es el origen económico del resultado, el evento que explica por qué el patrimonio neto aumenta o disminuye en un período.",
        "tema": "devengado"
    },
    {
        "question": "¿Qué se entiende por hecho sustancial?",
        "ground_truth": "Es el hecho que evidencia la realización del resultado y permite determinar el momento en que debe reconocerse contablemente.",
        "tema": "devengado"
    },
    {
        "question": "¿Cuándo se reconoce un resultado positivo originado en la venta de bienes?",
        "ground_truth": "Se reconoce cuando se produce la entrega del bien, es decir, cuando la operación se considera perfeccionada. Según el tipo de bien puede darse con un acto equivalente como la entrega de llaves en una venta de inmueble.",
        "tema": "devengado"
    },
    {
        "question": "¿Qué diferencia existe entre servicios de prestación instantánea y servicios de tracto continuo?",
        "ground_truth": "Los servicios instantáneos se reconocen cuando efectivamente se prestan. Los servicios de tracto continuo se reconocen en función del tiempo transcurrido o al cierre del período.",
        "tema": "devengado"
    },
    {
        "question": "¿Qué criterio se aplica para imputar resultados negativos vinculados con resultados positivos?",
        "ground_truth": "Se imputan en el mismo período que los resultados positivos con los cuales están relacionados, aplicando el criterio de asociación o correspondencia.",
        "tema": "devengado"
    },
    {
        "question": "¿Cómo se imputan los resultados negativos que no tienen vinculación directa con ingresos?",
        "ground_truth": "Se imputan al período correspondiente si están relacionados con la operatoria, o al momento en que se conocen si no tienen relación con ingresos ni con el período.",
        "tema": "devengado"
    },
    {
        "question": "¿Cuál es el criterio para distinguir entre un activo y un resultado negativo?",
        "ground_truth": "La diferencia radica en la capacidad de generar ingresos futuros. Si el hecho tiene capacidad de generar beneficios futuros es un activo; de lo contrario, constituye un resultado negativo.",
        "tema": "devengado"
    },

    # ── TEMA 3: Ejercicio ──────────────────────────────────────────────────────
    {
        "question": "¿Cuál es el concepto de ejercicio económico en contabilidad?",
        "ground_truth": "Se entiende por ejercicio económico la división de la vida del ente entre períodos de igual duración -12 meses- a efectos de suministrar información sobre su situación patrimonial, económica y financiera, y explicar las causas en los cambios en su patrimonio.",
        "tema": "ejercicio"
    },

    # ── TEMA 4: Cuentas Corrientes ─────────────────────────────────────────────
    {
        "question": "¿Qué es una cuenta corriente simple o de gestión?",
        "ground_truth": "Son cuentas abiertas por el vendedor a los clientes con la finalidad de otorgarles cierto crédito o plazo para el pago de las ventas o prestaciones de servicios realizadas, que supone habitualidad en la relación comercial y un necesario grado de confianza. Las operaciones mantienen su individualidad debiendo imputarse, por lo tanto, cada pago al respectivo comprobante que originó el crédito. La finalidad de las cuentas simples o de gestión es la registración de las operaciones para acreditar su existencia y facilitar la organización contable.",
        "tema": "cuentas_corrientes"
    },
]

# ── SELECTOR DE TEMA ───────────────────────────────────────────────────────────
#
# Cambiá TEMA_HOY según el tema a evaluar.
# Usá el slice para limitar la cantidad de preguntas por corrida
# y no superar el límite de tokens de Groq.

TEMA_HOY = "variaciones_patrimoniales"
# TEMA_HOY = "devengado"
# TEMA_HOY = "ejercicio"
# TEMA_HOY = "cuentas_corrientes"

PREGUNTAS_HOY = [p for p in PREGUNTAS if p["tema"] == TEMA_HOY]
# Para correr solo algunas preguntas: [p for p in PREGUNTAS if p["tema"] == TEMA_HOY][:5]


# ── CARGA DE RECURSOS ──────────────────────────────────────────────────────────

def cargar_recursos():
    print("Cargando recursos...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    vector_store = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    llm = ChatGroq(model=LLM_MODEL, temperature=0.2)
    print("   ✓ Recursos cargados\n")
    return embeddings, vector_store, llm


# ── GENERACIÓN DE RESPUESTAS ───────────────────────────────────────────────────

def generar_respuestas(vector_store, llm, preguntas: list) -> list:
    """
    Genera respuestas usando el mismo pipeline que app.py:
    normalización → routing → búsqueda con umbral → LLM
    """
    print(f"Generando respuestas para {len(preguntas)} preguntas...")
    resultados = []

    for i, item in enumerate(preguntas, 1):
        pregunta = item["question"]
        print(f"   {i}/{len(preguntas)} — {pregunta[:60]}...")

        # Mismo pipeline que app.py
        fragmentos, tipo, config = buscar_fragmentos(vector_store, pregunta)
        contexto = "\n\n".join([f.page_content for f in fragmentos])
        prompt_final = construir_prompt(TEMPLATE_PRINCIPAL, contexto, pregunta)

        respuesta = llm.invoke([
            SystemMessage(content=prompt_final),
            HumanMessage(content=pregunta)
        ])

        resultados.append({
            "question": pregunta,
            "answer": respuesta.content,
            "contexts": [f.page_content for f in fragmentos],
            "ground_truth": item["ground_truth"],
            "tema": item["tema"],
            "fragmentos_usados": len(fragmentos),
            "tipo_detectado": tipo,
            "config_usada": config,
        })

    print("   ✓ Respuestas generadas\n")
    return resultados


# ── EVALUACIÓN CON RAGAS ───────────────────────────────────────────────────────

def evaluar(resultados: list, embeddings) -> object:
    print("Evaluando con RAGAS (puede tardar varios minutos)...\n")

    dataset = Dataset.from_list([{
        "question": r["question"],
        "answer": r["answer"],
        "contexts": r["contexts"],
        "ground_truth": r["ground_truth"],
    } for r in resultados])

    llm_evaluador = LangchainLLMWrapper(
        ChatGroq(model=LLM_MODEL, temperature=0)
    )
    embeddings_evaluador = LangchainEmbeddingsWrapper(embeddings)

    return evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextRecall()],
        llm=llm_evaluador,
        embeddings=embeddings_evaluador,
        raise_exceptions=False,
    )


# ── MOSTRAR RESULTADOS ─────────────────────────────────────────────────────────

def mostrar_resultados(evaluacion, resultados: list, tema: str):
    df = evaluacion.to_pandas()

    print("\n" + "=" * 70)
    print(f"RESULTADOS — Tema: {tema.replace('_', ' ').title()}")
    print("=" * 70)

    faith  = df["faithfulness"].mean()
    relev  = df["answer_relevancy"].mean()
    recall = df["context_recall"].mean()

    print(f"\nPROMEDIOS DEL TEMA:")
    print(f"  Faithfulness:     {faith:.2f}   ← ¿la respuesta está basada en los fragmentos?")
    print(f"  Answer Relevancy: {relev:.2f}   ← ¿la respuesta responde la pregunta?")
    print(f"  Context Recall:   {recall:.2f}   ← ¿encontró toda la información necesaria?")

    print(f"\nDETALLE POR PREGUNTA:")
    print("-" * 70)

    for i, (row, res) in enumerate(zip(df.itertuples(), resultados)):
        f_str = f"{row.faithfulness:.2f}" if row.faithfulness == row.faithfulness else "nan"
        r_str = f"{row.answer_relevancy:.2f}" if row.answer_relevancy == row.answer_relevancy else "nan"
        c_str = f"{row.context_recall:.2f}" if row.context_recall == row.context_recall else "nan"

        print(f"\n{i+1}. {res['question'][:70]}")
        print(f"   Tipo detectado: {res['tipo_detectado']} | Config: k={res['config_usada']['k']}, umbral={res['config_usada']['umbral']}")
        print(f"   Faithfulness: {f_str}  Relevancy: {r_str}  Recall: {c_str}")
        print(f"   Fragmentos usados: {res['fragmentos_usados']} de {res['config_usada']['k']} buscados")

        if row.faithfulness == row.faithfulness and row.faithfulness < 0.5:
            print("   ⚠️  Faithfulness baja — posible alucinación")
        if row.context_recall == row.context_recall and row.context_recall < 0.5:
            print("   ⚠️  Context Recall bajo — el retriever no encontró suficiente información")

    print("\nINTERPRETACIÓN:")
    print("  > 0.8 → excelente | 0.6-0.8 → aceptable | < 0.6 → necesita mejoras")
    print("=" * 70 + "\n")


# ── GUARDAR RESULTADOS ─────────────────────────────────────────────────────────

def guardar_resultados(evaluacion, resultados: list, tema: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre = f"evaluacion_{tema}_{timestamp}.json"

    df = evaluacion.to_pandas()

    datos = {
        "tema": tema,
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "promedios": {
            "faithfulness":     float(df["faithfulness"].mean()),
            "answer_relevancy": float(df["answer_relevancy"].mean()),
            "context_recall":   float(df["context_recall"].mean()),
        },
        "detalle": [
            {
                "pregunta":          r["question"],
                "ground_truth":      r["ground_truth"],
                "respuesta_chatbot": r["answer"],
                "tipo_detectado":    r["tipo_detectado"],
                "config_usada":      r["config_usada"],
                "fragmentos_usados": r["fragmentos_usados"],
                "faithfulness":      float(df.iloc[i]["faithfulness"]) if df.iloc[i]["faithfulness"] == df.iloc[i]["faithfulness"] else None,
                "answer_relevancy":  float(df.iloc[i]["answer_relevancy"]) if df.iloc[i]["answer_relevancy"] == df.iloc[i]["answer_relevancy"] else None,
                "context_recall":    float(df.iloc[i]["context_recall"]) if df.iloc[i]["context_recall"] == df.iloc[i]["context_recall"] else None,
            }
            for i, r in enumerate(resultados)
        ]
    }

    with open(nombre, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)

    print(f"   ✓ Resultados guardados en {nombre}")


# ── PUNTO DE ENTRADA ───────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print(f"EVALUACIÓN RAGAS — Tema: {TEMA_HOY.replace('_', ' ').title()}")
    print(f"Preguntas: {len(PREGUNTAS_HOY)}")
    print("=" * 70 + "\n")

    embeddings, vector_store, llm = cargar_recursos()
    resultados = generar_respuestas(vector_store, llm, PREGUNTAS_HOY)
    evaluacion = evaluar(resultados, embeddings)
    mostrar_resultados(evaluacion, resultados, TEMA_HOY)
    guardar_resultados(evaluacion, resultados, TEMA_HOY)