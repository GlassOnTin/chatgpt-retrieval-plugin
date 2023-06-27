import json
import weaviate
client = weaviate.Client("http://weaviate-uy01:8080")
#client.data_object.get()
print(json.dumps(client.schema.get(), indent=4))


openai_class = client.schema.get("OpenAIDocument")["properties"]
print(json.dumps(openai_class, indent=4))

relationships = client.data_object.get("OpenAIRelationship")
print(json.dumps(relationships, indent=4))

#DELETE ALL, restart the server afterwards
#client.schema.delete_all()

QUERY="text"
result = (
    client.query.get(
        "OpenAIDocument",
        [
            "chunk_id",
            "document_id",
            "text",
            "source",
            "created_at",
        ],
    )
    .with_hybrid(query=QUERY, alpha=0.5)
    .do()
)
response = result["data"]["Get"]["OpenAIDocument"]
print(response)
###################################################################

# Create two documents
doc1_id = client.data_object.create({
    "title": "Eiffel Tower",
    "type": "Landmark",
    "source": "User",    
    "status": "Active"
}, "OpenAIDocument")

doc2_id = client.data_object.create({
    "title": "Gustave Eiffel",
    "type": "Person",
    "source": "User",    
    "status": "Active"
}, "OpenAIDocument")

# Create a relationship between the two documents
relationship_id = client.data_object.create({
    "from_document": {
        "beacon": f"weaviate://weaviate-uy01:8080/{doc1_id}"
    },
    "to_document": {
        "beacon": f"weaviate://weaviate-uy01:8080/{doc2_id}"
    },
    "relationship_type": "Designed by"
}, "OpenAIRelationship")

# Add a reference from doc1 to the relationship
client.data_object.reference.add(
    from_class_name="OpenAIDocument",
    from_uuid=doc1_id,
    from_property_name="relationships",
    to_class_name="OpenAIRelationship",
    to_uuid=relationship_id,
)

# Add a reference from doc2 to the relationship
client.data_object.reference.add(
    from_class_name="OpenAIDocument",
    from_uuid=doc2_id,
    from_property_name="relationships",
    to_class_name="OpenAIRelationship",
    to_uuid=relationship_id,
)

#####################################################
print(json.dumps(client.schema.get("OpenAIDocument")), indent=4)
print(json.dumps(client.schema.get("OpenAIRelationship")), indent=4)

doc1_id = client.data_object.create({"title": "Doc1", "text": "This is Doc1"}, "OpenAIDocument")
doc2_id = client.data_object.create({"title": "Doc2", "text": "This is Doc2"}, "OpenAIDocument")
print(client.data_object.get_by_id(doc1_id, class_name="OpenAIDocument"))
print(client.data_object.get_by_id(doc2_id, class_name="OpenAIDocument"))

relationship_id = client.data_object.create({
     "from_document": [{
         "beacon": f"weaviate://localhost/{doc1_id}"
     }],
     "to_document": [{
         "beacon": f"weaviate://localhost/{doc2_id}"
     }],
     "relationship_type": "Related"
}, "OpenAIRelationship")
print(relationship_id)
print(client.data_object.get_by_id(doc1_id, class_name="OpenAIDocument"))
print(client.data_object.get_by_id(doc2_id, class_name="OpenAIDocument"))
