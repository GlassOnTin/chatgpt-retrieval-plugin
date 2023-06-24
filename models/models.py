from pydantic import BaseModel
from typing import List, Optional


class DocumentMetadata(BaseModel):
    title: Optional[str] = None
    type: Optional[str] = None
    source: Optional[str] = None
    created_at: Optional[str] = None
    status: Optional[str] = None


class DocumentReference(BaseModel):
    document_id: str
    title: str


class DocumentRelationship(BaseModel):
    parents: Optional[List[DocumentReference]] = None
    children: Optional[List[DocumentReference]] = None


class DocumentChunkMetadata(DocumentMetadata):
    document_id: Optional[str] = None
    index: Optional[int] = 0


class DocumentChunk(BaseModel):
    id: Optional[str] = None    
    text: str
    metadata: DocumentChunkMetadata
    relationships: Optional[DocumentRelationship] = None
    embedding: Optional[List[float]] = None


class DocumentChunkWithScore(DocumentChunk):
    score: float


class Document(BaseModel):
    id: Optional[str] = None
    text: str
    metadata: Optional[DocumentMetadata] = None


class DocumentWithChunks(Document):
    chunks: List[DocumentChunk]


class DocumentMetadataFilter(DocumentMetadata):
    document_id: Optional[str] = None
    start_date: Optional[str] = None  # any date string format
    end_date: Optional[str] = None  # any date string format


class Query(BaseModel):
    query: str
    filter: Optional[DocumentMetadataFilter] = None
    top_k: Optional[int] = 3


class QueryWithEmbedding(Query):
    embedding: List[float]


class QueryResult(BaseModel):
    query: str
    results: List[DocumentChunkWithScore]
