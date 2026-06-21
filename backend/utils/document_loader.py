"""
Document ingestion — handles PDF, plain text, and raw strings.
Splits into chunks and returns LangChain Document objects.
"""
import hashlib
import logging
from pathlib import Path
from typing import BinaryIO

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from backend.config import get_settings

logger = logging.getLogger(__name__)


def _make_splitter() -> RecursiveCharacterTextSplitter:
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def load_from_text(text: str, source: str = "manual") -> list[Document]:
    """Chunk raw text into Documents."""
    splitter = _make_splitter()
    doc = Document(page_content=text, metadata={"source": source})
    chunks = splitter.split_documents([doc])
    _annotate_chunks(chunks, source)
    logger.info("Loaded %d chunks from text source '%s'", len(chunks), source)
    return chunks


def load_from_pdf(path: str | Path) -> list[Document]:
    """Load and chunk a PDF file."""
    loader = PyPDFLoader(str(path))
    raw = loader.load()
    splitter = _make_splitter()
    chunks = splitter.split_documents(raw)
    _annotate_chunks(chunks, str(path))
    logger.info("Loaded %d chunks from PDF '%s'", len(chunks), path)
    return chunks


def load_from_upload(file_bytes: bytes, filename: str) -> list[Document]:
    """Handle file upload bytes — dispatches to PDF or text loader."""
    tmp = Path(f"/tmp/{filename}")
    tmp.write_bytes(file_bytes)
    if filename.lower().endswith(".pdf"):
        return load_from_pdf(tmp)
    return load_from_text(tmp.read_text(errors="replace"), source=filename)


def _annotate_chunks(chunks: list[Document], source: str) -> None:
    """Add stable chunk IDs and source metadata."""
    for i, chunk in enumerate(chunks):
        chunk_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:8]
        chunk.metadata.update(
            {
                "chunk_id": f"{chunk_hash}_{i}",
                "source": chunk.metadata.get("source", source),
                "chunk_index": i,
            }
        )
