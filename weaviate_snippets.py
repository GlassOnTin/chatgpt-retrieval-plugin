import weaviate
client = weaviate.Client("http://weaviate-uy01:8080")
client.data_object.get()


client.schema.get()

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
            "source_id",
            "url",
            "created_at",
            "author",
            #"refersTo",
            #"referredBy"
        ],
    )
    .with_hybrid(query=QUERY, alpha=0.5)
    .do()
)

response = result["data"]["Get"]["OpenAIDocument"]
print(response)
###################################################################


client.data_object.reference.add(
  "0b56104a-fc38-5556-a4ac-b17294f9383b",
  "refersTo",
  "c15823af-b368-59fb-b748-d5ff2fa593da",
  from_class_name="OpenAIDocument",
  to_class_name="OpenAIDocument",
  consistency_level=weaviate.data.replication.ConsistencyLevel.ALL,  # default QUORUM
)

client.data_object.get()