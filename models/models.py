from pydantic import BaseModel
from typing import List, Optional


class DocumentMetadata(BaseModel):
    title: Optional[str] = None
    type: Optional[str] = None
    source: Optional[str] = None
    created_at: Optional[str] = None
    status: Optional[str] = None

class DocumentReference(BaseModel):
    document_id: Optional[str] = None
    title: Optional[str] = None
    relationship: Optional[str] = None


class DocumentRelationship(BaseModel):
    from_documents: List[DocumentReference]
    to_documents: List[DocumentReference]

    def __init__(self, **data):
        super().__init__(from_documents=data.get('from_documents', []), to_documents=data.get('to_documents', []), **data)

class DocumentChunkMetadata(DocumentMetadata):
    document_id: Optional[str] = None
    index: Optional[int] = None  # changed from 0 to None


class DocumentChunkMetadataFilter(DocumentChunkMetadata):
    start_date: Optional[str] = None  # any date string format
    end_date: Optional[str] = None  # any date string format

class DocumentChunk(BaseModel):
    text: str
    metadata: DocumentChunkMetadata
    relationships: Optional[DocumentRelationship] = []
    embedding: Optional[List[float]] = None

class DocumentChunkWithScore(DocumentChunk):
    score: float


class Document(BaseModel):
    id: Optional[str] = None
    text: Optional[str] = None
    metadata: Optional[DocumentMetadata] = None


class DocumentChunkMetadata(DocumentMetadata):
    document_id: Optional[str] = None
    index: Optional[int] = None
    
class DocumentWithChunks(Document):
    chunks: Optional[List[DocumentChunk]] = []
    

class Query(BaseModel):
    query: str
    filter: Optional[DocumentChunkMetadataFilter] = None
    top_k: Optional[int] = 3


class QueryWithEmbedding(Query):
    embedding: List[float]

class QueryResult(BaseModel):
    query: str
    results: Optional[List[Optional[DocumentChunkWithScore]]] = []
