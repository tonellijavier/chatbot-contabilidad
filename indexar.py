# ==============================================================================
# INDEXAR.PY — Corre este archivo UNA SOLA VEZ
# ==============================================================================
#
# Lee el PDF, lo divide en fragmentos y guarda la base vectorial en disco.
# Después el chatbot usa esa base vectorial sin necesitar el PDF.
#
# PARA CORRERLO:
#   python indexar.py
# ==============================================================================

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────

RUTA_PDF = r"C:/Users/tonel/Downloads/Lecturas 5a 2023 - Version Autores.pdf"
CARPETA_DB = "./chroma_db"
# La base vectorial se guarda en esta carpeta.
# Esta carpeta SÍ se puede compartir — tiene vectores, no el texto original.
# El PDF NO se comparte — queda solo en tu computadora.


# ── PROCESO ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("INDEXANDO EL LIBRO DE CONTABILIDAD")
    print("=" * 60)

    # Paso 1: verificamos que el PDF existe
    if not Path(RUTA_PDF).exists():
        print(f"\n✗ No se encontró el PDF en:\n  {RUTA_PDF}")
        print("\nVerificá la ruta y volvé a correr este script.")
        exit(1)

    print(f"\n✓ PDF encontrado: {Path(RUTA_PDF).name}")

    # Paso 2: cargamos el modelo de embeddings
    print("\nCargando modelo de embeddings (primera vez ~90MB)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("   ✓ Modelo cargado")

    # Paso 3: cargamos el PDF
    print("\nLeyendo el PDF...")
    loader = PyPDFLoader(RUTA_PDF)
    paginas = loader.load()
    print(f"   ✓ {len(paginas)} páginas cargadas")

    # Paso 4: dividimos en fragmentos
    # chunk_size=1500 — fragmentos más grandes para capturar más contexto
    # chunk_overlap=300 — más overlap para no perder ideas entre fragmentos
    print("\nDividiendo en fragmentos...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
    )
    fragmentos = splitter.split_documents(paginas)
    print(f"   ✓ {len(fragmentos)} fragmentos generados")

    # Paso 5: creamos la base vectorial y la guardamos en disco
    print("\nGenerando vectores y guardando en disco...")
    print("   (esto puede tardar varios minutos con 600+ páginas)")

    vector_store = Chroma.from_documents(
        documents=fragmentos,
        embedding=embeddings,
        persist_directory=CARPETA_DB,
        # persist_directory guarda la base vectorial en disco.
        # Sin esto, los vectores viven solo en memoria y se pierden al cerrar.
    )

    print(f"   ✓ Base vectorial guardada en '{CARPETA_DB}/'")
    print(f"\n{'=' * 60}")
    print("¡Listo! Ya podés correr el chatbot con: streamlit run app.py")
    print(f"{'=' * 60}\n")