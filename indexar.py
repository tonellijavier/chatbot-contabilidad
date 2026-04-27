# ==============================================================================
# INDEXAR.PY — Corre este archivo UNA SOLA VEZ
# ==============================================================================
#
# Lee el PDF, lo divide en fragmentos y guarda la base vectorial en disco.
# Usa FAISS — más liviano que Chroma y compatible con todas las plataformas.
#
# PARA CORRERLO:
#   python indexar.py
# ==============================================================================

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

RUTA_PDF = r"C:/Users/tonel/Downloads/Lecturas 5a 2023 - Version Autores.pdf"
CARPETA_DB = "./faiss_db"

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("INDEXANDO EL LIBRO DE CONTABILIDAD")
    print("=" * 60)

    if not Path(RUTA_PDF).exists():
        print(f"\n✗ No se encontró el PDF en:\n  {RUTA_PDF}")
        exit(1)

    print(f"\n✓ PDF encontrado: {Path(RUTA_PDF).name}")

    print("\nCargando modelo de embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("   ✓ Modelo cargado")

    print("\nLeyendo el PDF...")
    loader = PyPDFLoader(RUTA_PDF)
    paginas = loader.load()
    print(f"   ✓ {len(paginas)} páginas cargadas")

    print("\nDividiendo en fragmentos...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
    )
    fragmentos = splitter.split_documents(paginas)
    print(f"   ✓ {len(fragmentos)} fragmentos generados")

    print("\nGenerando vectores y guardando en disco...")
    print("   (esto puede tardar varios minutos con 600+ páginas)")

    vector_store = FAISS.from_documents(
        documents=fragmentos,
        embedding=embeddings,
    )
    vector_store.save_local(CARPETA_DB)
    # FAISS guarda dos archivos: index.faiss e index.pkl
    # Estos archivos SÍ se pueden subir al repo — no contienen el texto original

    print(f"   ✓ Base vectorial guardada en '{CARPETA_DB}/'")
    print(f"\n{'=' * 60}")
    print("¡Listo! Ya podés correr: streamlit run app.py")
    print(f"{'=' * 60}\n")