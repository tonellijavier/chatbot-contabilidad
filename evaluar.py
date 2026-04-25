# ==============================================================================
# EVALUAR.PY — Evaluación del chatbot con RAGAS
# ==============================================================================
#
# Mide la calidad de las respuestas del chatbot de contabilidad.
# Usa preguntas y respuestas provistas por los autores del libro.
#
# MÉTRICAS:
#   - Faithfulness      → ¿la respuesta está basada en los fragmentos encontrados?
#   - Answer Relevancy  → ¿la respuesta responde realmente la pregunta?
#   - Context Recall    → ¿encontró toda la información necesaria? (requiere ground_truth)
#
# NOTA SOBRE TOKENS:
#   Cada corrida de 10 preguntas consume ~100.000 tokens (límite diario gratuito de Groq).
#   Con 19 preguntas totales, corrés el Tema 1 un día y el Tema 2 al día siguiente.
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

load_dotenv()

# ── CONFIGURACIÓN ──────────────────────────────────────────────────────────────

K = 12
UMBRAL = 1.0

# ── PREGUNTAS CON GROUND TRUTH ─────────────────────────────────────────────────
#
# Dataset provisto por los autores del libro.
# ground_truth = respuesta esperada correcta — permite medir context_recall.
# tema = agrupa las preguntas para ver resultados por área del libro.

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

# ── TEMA 3: Ejercicio ─────────────────────────────────────────────────────
    },
        {
        "question": "¿Cuál es el concepto de ejercicio económico en contabilidad?",
        "ground_truth": "Se entiende por ejercicio económico la división de la vida del ente entre períodos de igual duracion -12 meses- a efectos de suministrar información sobre su situación patrimonial, económica y financiera, y explicar las causas en los cambios en su patrimonio.",
        "tema": "ejercicio"
    },
    
# ── TEMA 4: Cuenta corriente ──────────────────────────────────────────────
        {
        "question": "¿Qué es una cuenta corriente simple o de gestión?",
        "ground_truth": "Son cuentas abiertas por el vendedor a los clientes con la finalidad de otorgarles cierto crédito o plazo para el pago de las ventas o prestaciones de servicios realizadas, que supone habitualidad en la relación comercial y un necesario grado de confianza. Las operaciones mantienen su individualidad debiendo imputarse, por lo tanto, cada pago al respectivo comprobante que originó el crédito. La finalidad de las cuentas simples o de gestión es la registración de las operaciones para acreditar su existencia y facilitar la organización contable.",
        "tema": "cuentas corrientes"
    },
]

# ── SELECTOR DE TEMA ───────────────────────────────────────────────────────────
#
# Para no gastar todos los tokens en una sola corrida,
# evaluás un tema por día.
#
# Cambiar a "devengado" para el segundo día.

# TEMA_HOY = "variaciones_patrimoniales"
# TEMA_HOY = "devengado"
TEMA_HOY = "ejercicio"
# TEMA_HOY = "cuentas corrientes"


# PREGUNTAS_HOY = [p for p in PREGUNTAS if p["tema"] == TEMA_HOY][5:9]
PREGUNTAS_HOY = [p for p in PREGUNTAS if p["tema"] == TEMA_HOY][-1:]



# ── CARGA DE RECURSOS ──────────────────────────────────────────────────────────

def cargar_recursos():
    print("Cargando recursos...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(
        "./faiss_db",
        embeddings,
        allow_dangerous_deserialization=True
    )
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    print("   ✓ Recursos cargados\n")
    return embeddings, vector_store, llm


# ── BÚSQUEDA CON FILTRO POR UMBRAL ────────────────────────────────────────────

def buscar_con_umbral(vector_store, pregunta: str, k: int, umbral: float) -> list:
    """
    Busca fragmentos relevantes y filtra por umbral de distancia.
    Score bajo = muy similar. Score alto = poco similar.
    Solo pasan los fragmentos con score < umbral.
    """
    docs_con_score = vector_store.similarity_search_with_score(pregunta, k=k)
    fragmentos = [doc for doc, score in docs_con_score if score < umbral]
    if not fragmentos:
        fragmentos = [docs_con_score[0][0]]
    return fragmentos


# ── GENERACIÓN DE RESPUESTAS ───────────────────────────────────────────────────

def generar_respuestas(vector_store, llm, preguntas: list) -> list:
    print(f"Generando respuestas para {len(preguntas)} preguntas...")

    template = """Sos un asistente especializado en contabilidad que responde
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

    resultados = []

    for i, item in enumerate(preguntas, 1):
        pregunta = item["question"]
        print(f"   {i}/{len(preguntas)} — {pregunta[:60]}...")

        fragmentos = buscar_con_umbral(vector_store, pregunta, K, UMBRAL)
        contexto = "\n\n".join([f.page_content for f in fragmentos])

        respuesta = llm.invoke([
            SystemMessage(content=template.replace("{context}", contexto).replace("{question}", "")),
            HumanMessage(content=pregunta)
        ])

        resultados.append({
            "question": pregunta,
            "answer": respuesta.content,
            "contexts": [f.page_content for f in fragmentos],
            "ground_truth": item["ground_truth"],
            "tema": item["tema"],
            "fragmentos_usados": len(fragmentos),
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
        ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
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

    # Promedios globales del tema
    faith  = df["faithfulness"].mean()
    relev  = df["answer_relevancy"].mean()
    recall = df["context_recall"].mean()

    print(f"\nPROMEDIOS DEL TEMA:")
    print(f"  Faithfulness:     {faith:.2f}   ← ¿la respuesta está basada en los fragmentos?")
    print(f"  Answer Relevancy: {relev:.2f}   ← ¿la respuesta responde la pregunta?")
    print(f"  Context Recall:   {recall:.2f}   ← ¿encontró toda la información necesaria?")

    # Detalle por pregunta
    print(f"\nDETALLE POR PREGUNTA:")
    print("-" * 70)

    for i, (row, res) in enumerate(zip(df.itertuples(), resultados)):
        f_str = f"{row.faithfulness:.2f}" if row.faithfulness == row.faithfulness else "nan"
        r_str = f"{row.answer_relevancy:.2f}" if row.answer_relevancy == row.answer_relevancy else "nan"
        c_str = f"{row.context_recall:.2f}" if row.context_recall == row.context_recall else "nan"

        print(f"\n{i+1}. {res['question'][:70]}")
        print(f"   Faithfulness: {f_str}  Relevancy: {r_str}  Recall: {c_str}")
        print(f"   Fragmentos usados: {res['fragmentos_usados']} de {K} buscados")

        # Alertas
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
        "configuracion": {"k": K, "umbral": UMBRAL},
        "promedios": {
            "faithfulness": float(df["faithfulness"].mean()),
            "answer_relevancy": float(df["answer_relevancy"].mean()),
            "context_recall": float(df["context_recall"].mean()),
        },
        "detalle": [
            {
                "pregunta": r["question"],
                "ground_truth": r["ground_truth"],
                "respuesta_chatbot": r["answer"],
                "fragmentos_usados": r["fragmentos_usados"],
                "faithfulness": float(df.iloc[i]["faithfulness"]) if df.iloc[i]["faithfulness"] == df.iloc[i]["faithfulness"] else None,
                "answer_relevancy": float(df.iloc[i]["answer_relevancy"]) if df.iloc[i]["answer_relevancy"] == df.iloc[i]["answer_relevancy"] else None,
                "context_recall": float(df.iloc[i]["context_recall"]) if df.iloc[i]["context_recall"] == df.iloc[i]["context_recall"] else None,
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
    print(f"Preguntas: {len(PREGUNTAS_HOY)} | k={K} | umbral={UMBRAL}")
    print("=" * 70 + "\n")

    embeddings, vector_store, llm = cargar_recursos()
    resultados = generar_respuestas(vector_store, llm, PREGUNTAS_HOY)
    evaluacion = evaluar(resultados, embeddings)
    mostrar_resultados(evaluacion, resultados, TEMA_HOY)
    guardar_resultados(evaluacion, resultados, TEMA_HOY)