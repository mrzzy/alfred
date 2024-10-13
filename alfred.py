#
# Alfred
# Entrypoint
#


from argparse import ArgumentParser
from pathlib import Path
from typing import List, cast

from langchain.embeddings.base import Embeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter


if __name__ == "__main__":
    # parse arguments
    parser = ArgumentParser(description="Alfred LLM Tutoring assistant")
    parser.add_argument(
        "content_dir",
        type=Path,
        help="Path to a directory of course content documents used by Alfred "
        "to understand the course. Document formats supported: PDF",
    )
    parser.add_argument(
        "-M",
        "--model-provider",
        type=str,
        default="anthropic",
        help="Provider of the LLM model. Ensure that credentials are supplied for the provider.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="claude-3-haiku-20240307",
        help="Name of the LLM model to use to generate responses.",
    )
    parser.add_argument(
        "-E",
        "--embedding-provider",
        type=str,
        default="sentence_transformers",
        help="Provider of the embedding mode. Ensure that credentials are supplied for the provider.",
    )
    parser.add_argument(
        "-e",
        "--embedding",
        type=str,
        default="BAAI/bge-en-icl",
        help="Name of the embedding model to use to embed documents into vectors.",
    )
    args = parser.parse_args()

    # setup LLM model
    if args.model_provider == "anthropic":
        # expects ANTHROPIC_API_KEY env var to pass api key for authenticating with anthropic api
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(
            model_name=args.model,
            temperature=0,
            max_tokens_to_sample=1024,
            timeout=None,
            max_retries=2,
            stop=None,
        )
    else:
        raise ValueError(f"Unsupported model provider: {args.model_provider})")

    # setup embedding model
    if args.embedding_provider == "SentenceTransformer":

        class SentenceTransformerEmbeddings(Embeddings):
            def __init__(self, model_name: str):
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(model_name)

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return self.model.encode(texts).tolist()

            def embed_query(self, text: str) -> list[float]:
                return self.model.encode([text])[0]

        embedding = SentenceTransformerEmbeddings(args.embedding_model)
    else:
        raise ValueError(f"Unsupported embedding provider: {args.embedding_provider})")

    # load course documents
    # load PDF documents
    docs = []
    for pdf_path in args.content_dir.rglob("*.pdf"):
        docs.extend(PyMuPDFLoader(pdf_path).load())

    # split course documents into document chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vector_store = Milvus(
        embedding_function=embedding,
        connection_args={"uri": "./milvus.db"},
    )

    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]
    # ai_msg = model.invoke(messages)
    # print(ai_msg.content)
