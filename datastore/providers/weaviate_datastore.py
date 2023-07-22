import asyncio
import datetime
from typing import Dict, List, Tuple, Optional
from loguru import logger
from weaviate import Client
import weaviate
import os
import uuid

from weaviate.util import generate_uuid5

from datastore.datastore import DataStore
from models.models import (
    DocumentReference,
    DocumentRelationship,
    DocumentChunk,
    DocumentChunkMetadata,
    DocumentChunkMetadataFilter,
    QueryResult,
    QueryWithEmbedding,
    DocumentChunkWithScore,
)


WEAVIATE_HOST = os.environ.get("WEAVIATE_HOST", "http://127.0.0.1")
WEAVIATE_PORT = os.environ.get("WEAVIATE_PORT", "8080")
WEAVIATE_USERNAME = os.environ.get("WEAVIATE_USERNAME", None)
WEAVIATE_PASSWORD = os.environ.get("WEAVIATE_PASSWORD", None)
WEAVIATE_SCOPES = os.environ.get("WEAVIATE_SCOPES", "offline_access")
WEAVIATE_CLASS = os.environ.get("WEAVIATE_CLASS", "OpenAIDocument")
WEAVIATE_RELATIONSHIP_CLASS = os.environ.get(
    "WEAVIATE_RELATIONSHIP_CLASS", "OpenAIRelationship"
)

WEAVIATE_BATCH_SIZE = int(os.environ.get("WEAVIATE_BATCH_SIZE", 20))
WEAVIATE_BATCH_DYNAMIC = os.environ.get("WEAVIATE_BATCH_DYNAMIC", False)
WEAVIATE_BATCH_TIMEOUT_RETRIES = int(os.environ.get("WEAVIATE_TIMEOUT_RETRIES", 3))
WEAVIATE_BATCH_NUM_WORKERS = int(os.environ.get("WEAVIATE_BATCH_NUM_WORKERS", 1))

SCHEMA = {
    "class": WEAVIATE_CLASS,
    "description": "The datastore class",
    "properties": [
        {
            "name": "index",
            "dataType": ["int"],
            "description": "The sequential index of the chunk",
        },
        {
            "name": "document_id",
            "dataType": ["string"],
            "description": "The unique id of the document",
        },
        {
            "name": "title",
            "dataType": ["string"],
            "description": "A title for the document",
        },
        {
            "name": "type",
            "dataType": ["string"],
            "description": "The type of artifact required for the work item",
        },
        {
            "name": "text",
            "dataType": ["text"],
            "description": "The text of the chunk",
        },
        {
            "name": "source",
            "dataType": ["string"],
            "description": "The source of the data",
        },
        {
            "name": "created_at",
            "dataType": ["date"],
            "description": "Creation date",
        },
        {
            "name": "status",
            "dataType": ["string"],
            "description": "The current status",
        },
        {
            "name": "priority",
            "dataType": ["string"],
            "description": "The current priority",
        },
        {
            "name": "downcount",
            "dataType": ["string"],
            "description": "The total number of nodes below",
        },
        {
            "name": "upcount",
            "dataType": ["string"],
            "description": "The path length up to home",
        },
    ],
}

relationships_property = {
            "dataType": [WEAVIATE_RELATIONSHIP_CLASS],
            "name": "relationships",
            "description": "The relationships between this document and others",
        }

SCHEMA_RELATIONSHIP = {
    "class": WEAVIATE_RELATIONSHIP_CLASS,
    "description": "The relationship class",
    "properties": [
        {
            "name": "from_document",
            "dataType": [WEAVIATE_CLASS],
            "description": "The document from which this relationship originates",
        },
        {
            "name": "to_document",
            "dataType": [WEAVIATE_CLASS],
            "description": "The document to which this relationship points",
        },
        {
            "name": "relationship_type",
            "dataType": ["string"],
            "description": "The type of this relationship",
        },
    ],
}


class WeaviateDataStore(DataStore):
    def handle_errors(self, results: Optional[List[dict]]) -> List[str]:
        if not self or not results:
            return []

        error_messages = []
        for result in results:
            if (
                "result" not in result
                or "errors" not in result["result"]
                or "error" not in result["result"]["errors"]
            ):
                continue
            for message in result["result"]["errors"]["error"]:
                error_messages.append(message["message"])
                logger.exception(message["message"])

        return error_messages

    def __init__(self):
        auth_credentials = self._build_auth_credentials()

        url = f"{WEAVIATE_HOST}:{WEAVIATE_PORT}"

        logger.debug(
            f"Connecting to weaviate instance at {url} with credential type {type(auth_credentials).__name__}"
        )
        self.client = Client(url, auth_client_secret=auth_credentials)
        self.client.batch.configure(
            batch_size=WEAVIATE_BATCH_SIZE,
            dynamic=WEAVIATE_BATCH_DYNAMIC,  # type: ignore
            callback=self.handle_errors,  # type: ignore
            timeout_retries=WEAVIATE_BATCH_TIMEOUT_RETRIES,
            num_workers=WEAVIATE_BATCH_NUM_WORKERS,
        )
        self._initialize_schema()

    def _initialize_schema(self):
        self._create_or_update_class(WEAVIATE_CLASS, SCHEMA)
        self._create_or_update_class(WEAVIATE_RELATIONSHIP_CLASS, SCHEMA_RELATIONSHIP)
        self._add_relationships_property_to_document_class()

    def _create_or_update_class(self, class_name, schema):
        try:
            existing_schema = self.client.schema.get(class_name)

        except Exception as e:
            existing_schema = None

        if not existing_schema:
            try:
                logger.debug(f"Creating class {class_name} with schema {schema}")
                self.client.schema.create_class(schema)

            except Exception as e:
                logger.error(f"Failed to create weaviate class {class_name}: {e}")
                raise
        else:
            try:
                new_properties = {
                    property["name"]: property for property in schema["properties"]
                }
                existing_properties = {
                    property["name"]: property
                    for property in existing_schema["properties"]
                }
                
                # Don't try to remove the relationships property
                existing_properties.pop(relationships_property["name"], None)

                if len(new_properties) > len(existing_properties):
                    logger.debug(f"Updating class {class_name} with schema {schema}")
                    for property_name, property_schema in new_properties.items():
                        if property_name not in existing_properties:
                            self.client.schema.property.create(
                                class_name, property_schema
                            )

                elif len(new_properties) < len(existing_properties):
                    logger.error(f"Cannot remove properties from class {class_name}")

            except Exception as e:
                logger.error(f"Failed to update weaviate class {class_name}: {e}")
                raise

    def _add_relationships_property_to_document_class(self):

        logger.debug(
            f"Adding relationships property to collection {WEAVIATE_CLASS} with properties {relationships_property}"
        )

        try:
            # Get the schema of the OpenAIDocument class
            schema = self.client.schema.get(WEAVIATE_CLASS)

        except Exception as e:
            logger.debug(f"Failed to get {WEAVIATE_CLASS}")
            raise

        # Check if the 'relationships' property already exists
        try:
            if not any(
                prop for prop in schema["properties"] if prop["name"] == relationships_property["name"]
            ):
                # If the property doesn't exist, add it
                self.client.schema.property.create(
                    WEAVIATE_CLASS, relationships_property
                )

        except Exception as e:
            logger.debug(f"Failed to add relationships property to {WEAVIATE_CLASS}")
            raise

    @staticmethod
    def _build_auth_credentials():
        if WEAVIATE_USERNAME and WEAVIATE_PASSWORD:
            return weaviate.auth.AuthClientPassword(
                WEAVIATE_USERNAME, WEAVIATE_PASSWORD, WEAVIATE_SCOPES
            )
        else:
            return None

    async def _upsert(self, chunks: Dict[str, List[DocumentChunk]]) -> List[str]:
        try:
            doc_ids = []

            with self.client.batch as batch:
                for doc_id, doc_chunks in chunks.items():
                    logger.debug(f"Upserting {doc_id} with {len(doc_chunks)} chunks")
                    for doc_chunk in doc_chunks:
                        if doc_chunk.metadata:
                            if not doc_chunk.metadata.index:
                                j = 0
                            else:
                                j = doc_chunk.metadata.index + 1

                            logger.debug(
                                f"...batching chunk {j+1} of {len(doc_chunks)}"
                            )
                            self._add_chunk_to_batch(batch, doc_chunk)
                        else:
                            logger.error("Metadata missing")

                    doc_ids.append(doc_id)
                batch.flush()
            return doc_ids

        except Exception as e:
            logger.error(f"Error with upsert: {e}", exc=True)
            raise

    def _add_chunk_to_batch(self, batch, doc_chunk: DocumentChunk):
        chunk_uuid = generate_uuid5(doc_chunk, WEAVIATE_CLASS)

        doc_chunk_dict = doc_chunk.dict()

        # Remove the metadata section and store this flat in the schema class
        metadata = doc_chunk_dict.pop("metadata")
        for key, value in metadata.items():
            doc_chunk_dict[key] = value

        # Add the relationships
        doc_chunk_dict["relationships"] = (
            doc_chunk_dict.pop("relationships")
            if doc_chunk_dict["relationships"]
            else None
        )

        # Set the created_at to current time
        doc_chunk_dict["created_at"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()

        # Extract the embedding vector
        embedding = doc_chunk_dict.pop("embedding")

        batch.add_data_object(
            uuid=chunk_uuid,
            data_object=doc_chunk_dict,
            class_name=WEAVIATE_CLASS,
            vector=embedding,
        )

    async def _query_seq(self, queries: List[QueryWithEmbedding]) -> List[QueryResult]:
        try:
            results = []
            for query in queries:
                result = await self._single_query(query)
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error with query: {e}", exc=True)
            raise

    async def _query(self, queries: List[QueryWithEmbedding]) -> List[QueryResult]:
        return await asyncio.gather(*[self._single_query(query) for query in queries])

    async def _single_query(self, query: QueryWithEmbedding) -> QueryResult:
        try:
            result = self._execute_query(query)

            query_results = self._process_response(result)

            #logger.info(f"query_results: {query_results}")

            return QueryResult(query=query.query, results=query_results)

        except Exception as e:
            logger.error(f"Error with _single_query: {e}", exc=True)
            raise

    def _execute_query(self, query: QueryWithEmbedding):
        try:
            filters_ = (
                self.build_filters(query.filter)
                if hasattr(query, "filter") and query.filter
                else None
            )

            query_builder = (
                self.client.query.get(
                    WEAVIATE_CLASS, self._get_fields(SCHEMA, SCHEMA_RELATIONSHIP)
                )
                .with_limit(query.top_k)
                .with_additional(["id", "score"])
            )

            if query.query and query.embedding:
                query_builder = query_builder.with_hybrid(
                    query=query.query, alpha=0.5, vector=query.embedding
                )

            if filters_:
                query_builder = query_builder.with_where(filters_)

            return query_builder.do()

        except Exception as e:
            logger.error(f"Failed to execute_query {query}: {e}", exc_info=True)
            raise

    def _get_fields(self, schema, relationship_schema):
        fields = []
        for property in schema["properties"]:
            fields.append(property["name"])

        relationship_fields = []
        for property in relationship_schema["properties"]:
            if property["dataType"][0] == schema["class"]:
                relationship_fields.append(
                    f"{property['name']} {{ ... on {schema['class']} {{ document_id, title }} }}"
                )
            else:
                relationship_fields.append(property["name"])

        fields.append(
            f"relationships {{ ... on {relationship_schema['class']} {{ {', '.join(relationship_fields)} }} }}"
        )
        return fields

    def _process_response(self, result):
        try:
            #logger.info(f"_process_response{result}")

            if (
                "data" in result
                and "Get" in result["data"]
                and WEAVIATE_CLASS in result["data"]["Get"]
            ):
                response = result["data"]["Get"][WEAVIATE_CLASS]
                if response is None:
                    logger.error(f"Response is None: {result}")
                    return []
                else:
                    return [
                        self._process_document_chunk(resp)
                        for resp in response
                        if resp is not None
                    ]
            else:
                logger.error(f"Expected keys not found in result: {result}")
                return []

        except Exception as e:
            logger.error(f"Failed to process response: {e}", exc_info=True)
            raise

    def _process_document_chunk(self, resp):
        try:
            #logger.info(f"_process_document_chunk{resp}")

            from_documents = []
            to_documents = []
            if resp.get("relationships") is not None:
                for relationship in resp["relationships"]:
                    for direction in ["from_document", "to_document"]:
                        doc = relationship.get(direction)
                        if doc is not None:
                            for ref in doc:
                                if ref is not None:
                                    doc_ref = DocumentReference(
                                        document_id=ref.get("document_id", ""),
                                        title=ref.get("title", ""),
                                        status=ref.get("status", ""),
                                        priority=ref.get("priority", ""),
                                        relationship=relationship["relationship_type"],
                                    )
                                    if direction == "from_document":
                                        from_documents.append(doc_ref)
                                    else:
                                        to_documents.append(doc_ref)

            # Retrieve the fields list from the schema
            fields = [prop["name"] for prop in SCHEMA["properties"]]

            # Define default values for each property
            defaults = {field: "" for field in fields}

            # Use the default value for each property if it's not present in resp
            for key, default in defaults.items():
                resp.setdefault(key, default)

            score = resp["_additional"]["score"]

            # Create metadata by iterating over the attributes of DocumentChunkMetadata
            metadata_dict = {attr: resp.get(attr, "") for attr in resp.keys()}
            metadata = DocumentChunkMetadata(**metadata_dict)

            doc_chunk = DocumentChunkWithScore(
                text=resp["text"],
                score=score,
                metadata=metadata,
                relationships=DocumentRelationship(
                    from_documents=from_documents, to_documents=to_documents
                ),
            )

            #logger.debug(f"doc_chunk={doc_chunk}")

            return doc_chunk

        except Exception as e:
            logger.error(f"Failed to process document chunk: {e}", exc_info=True)
            raise

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[DocumentChunkMetadataFilter] = None,
        delete_all: Optional[bool] = None,
    ) -> bool:
        """
        Removes vectors by ids, filter, or everything in the datastore.
        Returns whether the operation was successful.
        """
        if delete_all:
            logger.debug(f"Deleting all vectors in index {WEAVIATE_CLASS}")
            try:
                self.client.schema.delete_all()
                self._initialize_schema()
            except Exception as e:
                logger.error(f"Failed to delete all vectors: {e}")
                return False
            return True

        if ids is None and filter is None and not delete_all:
            logger.error(
                "No ids or filter provided for deletion and delete_all is not set. Aborting."
            )
            return False

        if ids:
            return await self._delete_by_ids(ids)

        if filter:
            return await self._delete_by_filter(filter)

        return True

    async def _delete_by_ids(self, ids: List[str]) -> bool:
        operands = [
            {"path": ["document_id"], "operator": "Equal", "valueString": id}
            for id in ids
        ]

        where_clause = {"operator": "Or", "operands": operands}

        logger.debug(f"Deleting vectors from index {WEAVIATE_CLASS} with ids {ids}")
        logger.debug(f"Where clause: {where_clause}")
        try:
            result = self.client.batch.delete_objects(
                class_name=WEAVIATE_CLASS, where=where_clause, output="verbose"
            )
            logger.debug(f"Delete result: {result}")

        except Exception as e:
            logger.error(f"Failed to delete vectors with ids {ids}: {e}")
            return False

        if not bool(result["results"]["successful"]):
            logger.debug(
                f"Failed to delete the following objects: {result['results']['objects']}"
            )
            return False

        return True

    async def _delete_by_filter(self, filter: DocumentChunkMetadataFilter) -> bool:
        where_clause = WeaviateDataStore.build_filters(filter)

        logger.debug(
            f"Deleting vectors from index {WEAVIATE_CLASS} with filter {where_clause}"
        )
        logger.debug(f"Where clause: {where_clause}")
        try:
            result = self.client.batch.delete_objects(
                class_name=WEAVIATE_CLASS, where=where_clause
            )
            logger.debug(f"Delete result: {result}")

        except Exception as e:
            logger.error(f"Failed to delete vectors with filter {where_clause}: {e}")
            return False

        if not bool(result["results"]["successful"]):
            logger.debug(
                f"Failed to delete the following objects: {result['results']['objects']}"
            )
            return False

        return True

    @staticmethod
    def build_filters(filter):
        operands = []
        filter_conditions = {
            "start_date": {"operator": "GreaterThanEqual", "value_key": "valueDate"},
            "end_date": {"operator": "LessThanEqual", "value_key": "valueDate"},
            "index": {"operator": "Equal", "value_key": "valueInt"},
            "default": {"operator": "Equal", "value_key": "valueString"},
        }

        #print(f"filter: {filter}")

        for attr, value in filter.__dict__.items():
            if value is not None:
                filter_condition = filter_conditions.get(
                    attr, filter_conditions["default"]
                )
                value_key = filter_condition["value_key"]

                operand = {
                    "path": [
                        attr
                        if not (attr == "start_date" or attr == "end_date")
                        else "created_at"
                    ],
                    "operator": filter_condition["operator"],
                    value_key: value,
                }

                #logger.debug(f"Operand: {operand}")
                
                operands.append(operand)

        # If there's only one operand, return it directly instead of using an 'And' operator
        if len(operands) == 1:
            return operands[0]

        return {"operator": "And", "operands": operands}

    async def add_reference(
        self,
        from_document_id: str,
        to_document_id: str,
        from_relationship_type: str,
        to_relationship_type: str,
        consistency_level: weaviate.data.replication.ConsistencyLevel = weaviate.data.replication.ConsistencyLevel.ALL,
    ) -> bool:
        """
        Adds a two-way cross-reference between two documents properties
        """
        logger.debug(
            f"Adding references between {from_document_id} and {to_document_id}"
        )
        try:
            # Get the chunk IDs for the from_document and to_document
            from_chunk_id = self.get_chunk_id(from_document_id)
            to_chunk_id = self.get_chunk_id(to_document_id)

            # Create a Relationship object for the from_relationship_type
            from_relationship_id = self.create_relationship(
                from_chunk_id, to_chunk_id, from_relationship_type
            )

            # Create a Relationship object for the to_relationship_type
            to_relationship_id = self.create_relationship(
                from_chunk_id, to_chunk_id, to_relationship_type
            )

            # Add a reference from the from_document to the from_relationship_type Relationship object
            self.add_reference_to_relationship(
                from_chunk_id, from_relationship_id, consistency_level
            )

            # Add a reference from the to_document to the to_relationship_type Relationship object
            self.add_reference_to_relationship(
                to_chunk_id, to_relationship_id, consistency_level
            )

            # Update the upcount and downcount of the metadata
            self.update_counts(from_document_id, to_document_id)

            return True
        except Exception as e:
            logger.error(
                f"Failed to add references between {from_document_id} and {to_document_id}: {e}",
                exc_info=True,
            )
            return False

    async def delete_reference(
        self,
        from_document_id: str,
        to_document_id: str,
        consistency_level: weaviate.data.replication.ConsistencyLevel = weaviate.data.replication.ConsistencyLevel.ALL,
    ) -> bool:
        """
        Deletes a two-way cross-reference between two documents properties
        """
        logger.debug(
            f"Deleting references between {from_document_id} and {to_document_id}"
        )
        try:
            # Get the chunk IDs for the from_document and to_document
            from_chunk_id = self.get_chunk_id(from_document_id)
            to_chunk_id = self.get_chunk_id(to_document_id)

            # Delete the reference from the from_document to the to_document
            self.client.data_object.reference.delete(
                from_uuid=from_chunk_id,
                from_property_name="relationships",
                to_uuid=to_chunk_id,
                from_class_name=WEAVIATE_CLASS,
                to_class_name=WEAVIATE_RELATIONSHIP_CLASS,
                consistency_level=consistency_level,
            )

            # Delete the reference from the to_document to the from_document
            self.client.data_object.reference.delete(
                from_uuid=to_chunk_id,
                from_property_name="relationships",
                to_uuid=from_chunk_id,
                from_class_name=WEAVIATE_CLASS,
                to_class_name=WEAVIATE_RELATIONSHIP_CLASS,
                consistency_level=consistency_level,
            )

            return True
        except Exception as e:
            logger.error(
                f"Failed to delete references between {from_document_id} and {to_document_id}: {e}",
                exc_info=True,
            )
            return False

    def get_chunk_id(self, document_id: str) -> str:
        """
        Get the chunk ID for a document
        """
        if not document_id:
            raise ValueError("Document ID cannot be empty")

        try:
            # Build the filter for the document
            document_filter = self.build_filters(
                DocumentChunkMetadataFilter(document_id=document_id, index=0)
            )

            # Get the first chunk for the document
            document_chunk = (
                self.client.query.get(WEAVIATE_CLASS)
                .with_where(document_filter)
                .with_additional(["id"])
                .do()
            )

            # Check if the response is empty or not
            if not document_chunk["data"]["Get"]["OpenAIDocument"]:
                raise ValueError(f"No document found with ID {document_id}")

            # Extract the 'id' from the '_additional' field of the response
            chunk_id = document_chunk["data"]["Get"]["OpenAIDocument"][0][
                "_additional"
            ]["id"]

            return chunk_id
        except (KeyError, IndexError) as e:
            logger.error(
                f"Failed to get chunk ID for {document_id} due to {e.__class__.__name__}: {e}. Document chunk: {document_chunk}",
                exc_info=True,
            )
            raise
        except Exception as e:
            logger.error(
                f"Failed to get chunk ID for {document_id}: {e}", exc_info=True
            )
            raise

    def create_relationship(
        self, from_chunk_id: str, to_chunk_id: str, relationship_type: str
    ) -> str:
        """
        Create a Relationship object
        """
        try:
            relationship_id = self.client.data_object.create(
                {
                    "from_document": [
                        {"beacon": f"weaviate://localhost/{from_chunk_id}"}
                    ],
                    "to_document": [{"beacon": f"weaviate://localhost/{to_chunk_id}"}],
                    "relationship_type": relationship_type,
                },
                WEAVIATE_RELATIONSHIP_CLASS,
            )

            return relationship_id
        except Exception as e:
            logger.error(
                f"Failed to create relationship data object: {e}", exc_info=True
            )
            raise

    def add_reference_to_relationship(
        self,
        chunk_id: str,
        relationship_id: str,
        consistency_level: weaviate.data.replication.ConsistencyLevel,
    ) -> None:
        """
        Add a reference from a document to a Relationship object
        """
        try:
            self.client.data_object.reference.add(
                from_uuid=chunk_id,
                from_property_name="relationships",
                to_uuid=relationship_id,
                from_class_name=WEAVIATE_CLASS,
                to_class_name=WEAVIATE_RELATIONSHIP_CLASS,
                consistency_level=consistency_level,
            )
        except Exception as e:
            logger.error(f"Failed to add reference to relationship: {e}", exc_info=True)
            raise

    def update_counts(self, from_document_id, to_document_id, increment=True):
        # Update the upcount of the 'from' node and all its 'from' descendants
        self.update_count(from_document_id, direction='from', increment=increment)
    
        # Update the downcount of the 'to' node and all its 'to' ancestors
        self.update_count(to_document_id, direction='to', increment=increment)

    def update_count(self, document_id,  direction: str='to', increment=True):
            
        try:
            
            # Get related nodes with new implementation
            related_nodes = self.get_related_nodes(document_id, direction=direction)
            
            logger.info(f"update_count: doc_id={document_id} has {len(related_nodes)} nodes in direction {direction}")
            
            logger.info(related_nodes)
            
            # Determine the count type (upcount or downcount)
            count_type = "downcount" if direction == 'to' else "upcount"
            
            # Update count for each related node
            for related_node_id in related_nodes:
                
                # Get Weaviate ID
                related_node_weaviate_id = self.get_chunk_id(related_node_id)
                
                logger.info(f"related doc id={related_node_id}  uuid={related_node_weaviate_id}")
                
                # Calculate the new count
                if related_nodes:
                    new_count = len(related_nodes)
                else:
                    new_count = 0
                
                    # Get the current count
                    current_count = self.client.data_object.get_by_id(related_node_weaviate_id, class_name=WEAVIATE_CLASS).get('properties', {}).get(count_type)
                    logger.info(f"Current {count_type} for {related_node_id} (uuid={related_node_weaviate_id}): {current_count}")
                    
                    # Update the count in the database
                    self.client.data_object.update( \
                        uuid=related_node_weaviate_id, \
                        class_name=WEAVIATE_CLASS, \
                        data_object={count_type: str(new_count)})
                    
                    # Get the updated count
                    updated_count = self.client.data_object.get_by_id(related_node_weaviate_id, class_name=WEAVIATE_CLASS).get('properties', {}).get(count_type)
                    logger.info(f"Updated {count_type} for {related_node_id} (uuid={related_node_weaviate_id}): {updated_count}")
                    

            
        except Exception as e:
            logger.error(f"Error updating count for {document_id}: {e}")
            raise
            
    def get_related_nodes(self, document_id: str, visited: set = None, direction: str='to') -> List[str]:
                   
        if visited is None:
            visited = set()
            
        if document_id in visited:
            return []
            
        visited.add(document_id)    
        
        try:
            # Get the chunk ID for the document 
            chunk_id = self.get_chunk_id(document_id)
            
            # Fetch the full chunk/doc object using the id
            chunk = self.client.data_object.get_by_id(chunk_id, class_name=WEAVIATE_CLASS)
            
            # Extract relationships
            relationships = chunk.get('properties', {}).get('relationships', [])
            
            related_docs = []
            
            for relationship in relationships:
            
                relationship_id = relationship.get('beacon').split('/')[-1]
                
                # Fetch the relationship object
                relationship_obj = self.client.data_object.get_by_id(relationship_id, class_name='OpenAIRelationship')
                
                # Extract the related document's ID from the 'from_document' or 'to_document' property
                if direction in ['to', 'both']:
                    to_document = relationship_obj.get('properties', {}).get('to_document', [{}])[0]
                    related_chunk_id = to_document.get('beacon').split('/')[-1]
                    
                elif direction == 'from':
                    from_document = relationship_obj.get('properties', {}).get('from_document', [{}])[0]
                    related_chunk_id = from_document.get('beacon').split('/')[-1]
                else:
                    continue
                
                if not related_chunk_id:
                    continue
                
                logger.info(f"related_chunk_id={related_chunk_id}")
                
                chunk = self.client.data_object.get_by_id(related_chunk_id, class_name=WEAVIATE_CLASS)
                
                logger.info(f"related chunk={chunk}")
                
                if not chunk or not chunk.get('properties'):
                    continue
                
                related_doc_id = chunk.get('properties', {}).get('document_id', {})
                
                related_docs.append(related_doc_id)
    
            return related_docs
        
        except Exception as e:
            logger.error(f"Error getting related nodes for {document_id}: {e}")
            raise
    

    @staticmethod
    def _is_valid_weaviate_id(candidate_id: str) -> bool:
        """
        Check if candidate_id is a valid UUID for weaviate's use

        Weaviate supports UUIDs of version 3, 4 and 5. This function checks if the candidate_id is a valid UUID of one of these versions.
        See https://weaviate.io/developers/weaviate/more-resources/faq#q-are-there-restrictions-on-uuid-formatting-do-i-have-to-adhere-to-any-standards
        for more information.
        """
        acceptable_version = [3, 4, 5]

        try:
            result = uuid.UUID(candidate_id)
            if result.version not in acceptable_version:
                return False
            else:
                return True
        except ValueError:
            return False
