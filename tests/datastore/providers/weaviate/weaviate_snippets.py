import json
import os
import openai
import weaviate
client = weaviate.Client("http://weaviate-uy01:8080")

WEAVIATE_CLASS="OpenAIDocument"

WEAVIATE_RELATIONSHIP_CLASS = "OpenAIRelationship"

#=========== SCHEMAS ===========
#DELETE ALL, restart the server afterwards
#client.schema.delete_all()

# Get all schemas
print(json.dumps(client.schema.get(), indent=4))

# Get OpenAIDocument schema
#openai_class = client.schema.get(WEAVIATE_CLASS)["properties"]
#print(json.dumps(openai_class, indent=4))

#======== Create two documents======
import uuid
doc_id1 = str(uuid.uuid4())
doc_id2 = str(uuid.uuid4())

from_id = client.data_object.create(    
    {
        "document_id": doc_id1,
        "title": "Eiffel Tower",
        "type": "Landmark",
        "source": "User",    
        "status": "Active"
    }, 
    WEAVIATE_CLASS,
    uuid = doc_id1
)

to_id = client.data_object.create(
    {
        "document_id": doc_id2,
        "title": "Gustave Eiffel",
        "type": "Person",
        "source": "User",    
        "status": "Active"
    }, WEAVIATE_CLASS,
    uuid = doc_id2
)

data = client.data_object.get()
print(json.dumps(data, indent=4))


#========= DOCUMENT QUERY =========
QUERY="eiffel"
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_SIZE = 1536
openai.api_key=os.getenv(OPENAI_API_KEY)
response = openai.Embedding.create(input=[QUERY], model=EMBEDDING_MODEL)
# Extract the embedding data from the response
embedding = list(response["data"][0]['embedding'])  # type: ignore

result = (
    client.query.get(
        WEAVIATE_CLASS,
        [
            "chunk_id",
            "document_id",
            "text",
            "source",
            "created_at",
        ],
    )
    .with_hybrid(query=QUERY, alpha=0.5, vector=embedding)
    .do()
)
print(result)
response = result["data"]["Get"][WEAVIATE_CLASS]
print(response)

#=========== RELATIONSHIPS ===========



 # Create a Relationship object for the from_relationship_type
from_relationship_id =  client.data_object.create(
    {
        "from_document": [{
            "beacon": f"weaviate://localhost/{from_id}"
        }],
        "to_document": [{
            "beacon": f"weaviate://localhost/{to_id}"
        }],
        "relationship_type": "Related"
    }, 
    WEAVIATE_RELATIONSHIP_CLASS
)

# Create a Relationship object for the to_relationship_type
to_relationship_id =  client.data_object.create(
    {
        "from_document": [{
            "beacon": f"weaviate://localhost/{from_id}"
        }],
        "to_document": [{
            "beacon": f"weaviate://localhost/{to_id}"
        }],
        "relationship_type": "Related"
    },
    WEAVIATE_RELATIONSHIP_CLASS
)
print(from_relationship_id)
print(to_relationship_id)

# Add a reference from the from_document to the from_relationship_type Relationship object
client.data_object.reference.add(
    from_uuid=from_id,
    from_property_name="relationships",
    to_uuid=from_relationship_id,
    from_class_name=WEAVIATE_CLASS,
    to_class_name=WEAVIATE_RELATIONSHIP_CLASS,
    consistency_level=weaviate.data.replication.ConsistencyLevel.ONE
)

# Add a reference from the to_document to the to_relationship_type Relationship object
client.data_object.reference.add(
    from_uuid=to_id,
    from_property_name="relationships",
    to_uuid=to_relationship_id,
    from_class_name=WEAVIATE_CLASS,
    to_class_name=WEAVIATE_RELATIONSHIP_CLASS,
    consistency_level=weaviate.data.replication.ConsistencyLevel.ONE
)

print(client.data_object.get_by_id(from_id, class_name=WEAVIATE_CLASS))
print(client.data_object.get_by_id(to_id, class_name=WEAVIATE_CLASS))
print(client.data_object.get_by_id(from_relationship_id, class_name=WEAVIATE_CLASS))
print(client.data_object.get_by_id(to_relationship_id, class_name=WEAVIATE_RELATIONSHIP_CLASS))

query = "Document 1"  # replace with your actual query
embedding = [...]  # replace with your actual embedding
filters_ = [...]  # replace with your actual filters
top_k = 10  # replace with your actual limit

result = (
    client.query.get(
        WEAVIATE_CLASS,
        [
            "document_id",
            "index",
            "title",                                
            "text",
            "type",
            "source",
            "created_at",
            "status",
            "relationships { ... on OpenAIRelationship { relationship_type, from_document { ... on OpenAIDocument { document_id, title } }, to_document { ... on OpenAIDocument { document_id, title } } } }"
        ],
    )                        
    .with_additional(["id","score"])
    .do()
)
print(result)

def find_relationship(client, from_id, to_id):
    result = client.query.get(WEAVIATE_RELATIONSHIP_CLASS, ["from_document { ... on " + WEAVIATE_CLASS + " { document_id } }", "to_document { ... on " + WEAVIATE_CLASS + " { document_id } }"]).with_additional(["id"]).do()
    print(f"Query result: {result}")
    if "data" in result and WEAVIATE_RELATIONSHIP_CLASS in result["data"]["Get"]:
        for relationship in result["data"]["Get"][WEAVIATE_RELATIONSHIP_CLASS]:
            from_document_id = relationship["from_document"][0]["document_id"]
            to_document_id = relationship["to_document"][0]["document_id"]
            print(f"Checking relationship from {from_document_id} to {to_document_id}")
            if from_document_id == from_id and to_document_id == to_id:                return relationship["_additional"]["id"]
    return None

relationship_id = find_relationship(client, from_id, to_id)
print(f"Relationship ID: {relationship_id}")