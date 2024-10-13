#
# Alfred
# Knowledge Catalogue
#

from argparse import ArgumentParser
from pathlib import Path

from langchain.embeddings.base import Embeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embedding import Embedding, build_embedding


if __name__ == "__main__":
    parser = ArgumentParser(description="Alfred Knowledge Catalogue builder.")
    parser.add_argument(
        "content_dir",
        type=Path,
        help="Path to a directory of course content documents used to build "
        "Alfreds knowledge catalogue. Document formats supported: PDF",
    )
    parser.add_argument(
        "catalog",
        type=Path,
        help="Output path to store the built Knowledge Catalogue.",
    )
    parser.add_argument(
        "-p",
        "--provider",
        type=str,
        default="sentence_transformers",
        choices=[
            "sentence_transformers",
            "voyageai",
        ],
        help="Provider of the embedding model. Ensure that credentials are supplied for the provider (if any).",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="BAAI/bge-en-icl",
        help="Name of the embedding model to use to embed documents into vectors.",
    )
    args = parser.parse_args()
    # load course documents
    # load PDF documents
    docs = []
    for pdf_path in args.content_dir.rglob("*.pdf"):
        docs.extend(PyMuPDFLoader(pdf_path).load())

    # split course documents into document chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # write knowledge catalogue
    args.catalog.mkdir(parents=True, exist_ok=True)
    embedding = Embedding(model=args.model, provider=args.provider)
    # write embedding specification
    with open(str(args.catalog / "embedding.json"), "w") as f:
        f.write(embedding.model_dump_json())
    # write vector db
    vector_store = Milvus.from_documents(
        docs,
        embedding=build_embedding(embedding),
        connection_args={"uri": str(args.catalog / "milvus.db")},
    )
    print("Built Knowledge Catalogue")
