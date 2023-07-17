# test_weaviate_datastore.py

import pytest
from weaviate_datastore import WeaviateDataStore, Document

@pytest.fixture(scope="module")
def datastore():
    # Initialize Weaviate
    datastore = WeaviateDataStore()  

    # Define test data 
    datastore.create_document(Document(text="Hello", id="doc1"))
    datastore.create_document(Document(text="World", id="doc2"))
    datastore.add_reference("doc1", "doc2", "related")

    yield datastore  

    # Teardown
    datastore.delete_all()

class DocumentNotFound(Exception):
    pass

def test_create_document(datastore):
    doc = Document(text="Test", id="doc3")
    assert datastore.create_document(doc) == "doc3"

def test_get_document(datastore):
    doc = datastore.get_document("doc1")
    assert doc.text == "Hello"

def test_update_document(datastore):
    datastore.update_document("doc1", {"text": "Hello World"})
    doc = datastore.get_document("doc1")
    assert doc.text == "Hello World"

def test_delete_document(datastore):
    datastore.delete_document("doc1")
    with pytest.raises(DocumentNotFound):
        datastore.get_document("doc1")  

def test_add_reference(datastore):
    datastore.add_reference("doc2", "doc3", "related")
    doc = datastore.get_document("doc2")
    assert len(doc.references) == 1
    assert doc.references[0].document_id == "doc3"

def test_delete_reference(datastore):
    datastore.delete_reference("doc2", "doc3")
    doc = datastore.get_document("doc2")
    assert len(doc.references) == 0

def test_search(datastore):
    results = datastore.search("World")
    assert len(results) == 1
    assert results[0].id == "doc2"