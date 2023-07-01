# TODO
import asyncio
import datetime
from typing import Dict, List, Optional
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
    DocumentChunkWithScore
)


WEAVIATE_HOST = os.environ.get("WEAVIATE_HOST", "http://127.0.0.1")
WEAVIATE_PORT = os.environ.get("WEAVIATE_PORT", "8080")
WEAVIATE_USERNAME = os.environ.get("WEAVIATE_USERNAME", None)
WEAVIATE_PASSWORD = os.environ.get("WEAVIATE_PASSWORD", None)
WEAVIATE_SCOPES = os.environ.get("WEAVIATE_SCOPES", "offline_access")
WEAVIATE_CLASS = os.environ.get("WEAVIATE_CLASS", "OpenAIDocument")
WEAVIATE_RELATIONSHIP_CLASS = os.environ.get("WEAVIATE_RELATIONSHIP_CLASS", "OpenAIRelationship")

WEAVIATE_BATCH_SIZE = int(os.environ.get("WEAVIATE_BATCH_SIZE", 20))
WEAVIATE_BATCH_DYNAMIC = os.environ.get("WEAVIATE_BATCH_DYNAMIC", False)
WEAVIATE_BATCH_TIMEOUT_RETRIES = int(os.environ.get("WEAVIATE_TIMEOUT_RETRIES", 3))
WEAVIATE_BATCH_NUM_WORKERS = int(os.environ.get("WEAVIATE_BATCH_NUM_WORKERS", 1))

SCHEMA = {
    "class": WEAVIATE_CLASS,
    "description": "The datastore class",
    "properties": [
        {
            "name": "chunk_id",
            "dataType": ["string"],
            "description": "The unique id of the chunk",
        },
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
            "description": "The current status (To Do, In Progress, Done)",
        }
    ],
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
        }
    ],
}


def extract_schema_properties(schema):
    properties = schema["properties"]

    return {property["name"] for property in properties}


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
        self._create_class_if_not_exists(WEAVIATE_CLASS, SCHEMA)
        self._create_class_if_not_exists(WEAVIATE_RELATIONSHIP_CLASS, SCHEMA_RELATIONSHIP)
        self._add_relationships_property_to_document_class()

    def _create_class_if_not_exists(self, class_name, schema):
        try:            
            existing_schema = self.client.schema.get(class_name)
        except weaviate.exceptions.UnexpectedStatusCodeException:
            existing_schema = None
        if not existing_schema:
            new_schema_properties = extract_schema_properties(schema)
            logger.debug(
                f"Creating collection {class_name} with properties {new_schema_properties}"
            )
            self.client.schema.create_class(schema)

    def _add_relationships_property_to_document_class(self):
        relationships_property = {
            "dataType": [WEAVIATE_RELATIONSHIP_CLASS],
            "name": "relationships",
            "description": "The relationships between this document and others",
        }

        logger.debug(
            f"Adding relationships property to collection {WEAVIATE_CLASS} with properties {relationships_property}"
        )
        
        try: 
            # Get the schema of the OpenAIDocument class
            schema = self.client.schema.get(WEAVIATE_CLASS)
        except weaviate.exceptions.UnexpectedStatusCodeException:
            logger.debug(f"Failed to get {WEAVIATE_CLASS}")   

        # Check if the 'relationships' property already exists
        if not any(prop for prop in schema["properties"] if prop["name"] == "relationships"):
            # If the property doesn't exist, add it
            self.client.schema.property.create(WEAVIATE_CLASS, relationships_property)


    @staticmethod
    def _build_auth_credentials():
        if WEAVIATE_USERNAME and WEAVIATE_PASSWORD:
            return weaviate.auth.AuthClientPassword(
                WEAVIATE_USERNAME, WEAVIATE_PASSWORD, WEAVIATE_SCOPES
            )
        else:
            return None

    async def _upsert(self, chunks: Dict[str, List[DocumentChunk]]) -> List[str]:
        """
        Takes in a dict of list of document chunks and inserts them into the database.
        Return a list of document ids.
        """
        doc_ids = []

        with self.client.batch as batch:
            for doc_id, doc_chunks in chunks.items():
                logger.debug(f"Upserting {doc_id} with {len(doc_chunks)} chunks")
                for doc_chunk in doc_chunks:
                    # we generate a uuid regardless of the format of the document_id because
                    # weaviate needs a uuid to store each document chunk and
                    # a document chunk cannot share the same uuid
                    doc_uuid = generate_uuid5(doc_chunk, WEAVIATE_CLASS)
                    metadata = doc_chunk.metadata
                    doc_chunk_dict = doc_chunk.dict()
                    doc_chunk_dict.pop("metadata")
                    for key, value in metadata.dict().items():
                        doc_chunk_dict[key] = value
                    doc_chunk_dict["relationships"] = (
                        doc_chunk_dict.pop("relationships").value
                        if doc_chunk_dict["relationships"]
                        else None
                    )
                    
                    # Set the 'created_at' field to the current system UTC time
                    doc_chunk_dict['created_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()

                    embedding = doc_chunk_dict.pop("embedding")

                    batch.add_data_object(
                        uuid=doc_uuid,
                        data_object=doc_chunk_dict,
                        class_name=WEAVIATE_CLASS,
                        vector=embedding,
                    )

                doc_ids.append(doc_id)
            batch.flush()
        return doc_ids

    async def _query(
        self,
        queries: List[QueryWithEmbedding],
    ) -> List[QueryResult]:
        """
        Takes in a list of queries with embeddings and filters and returns a list of query results with matching document chunks and scores.
        """

        async def _single_query(query: QueryWithEmbedding) -> QueryResult:
            logger.debug(f"Query: {query.query}")
            if not hasattr(query, "filter") or not query.filter:
                if query.query:
                    logger.debug(f"Querying without filter")
                    result = (
                        self.client.query.get(
                            WEAVIATE_CLASS,
                            [
                                "chunk_id",
                                "document_id",
                                "index",
                                "title",                                
                                "text",
                                "type",
                                "source",
                                "created_at",
                                "status",
                                "relationships { ... on " + WEAVIATE_RELATIONSHIP_CLASS + " { relationship_type, from_document { ... on " + WEAVIATE_CLASS + " { document_id, title } }, to_document { ... on " + WEAVIATE_CLASS + " { document_id, title } } } }"
                            ],
                        )
                        .with_hybrid(query=query.query, alpha=0.5, vector=query.embedding)
                        .with_limit(query.top_k)
                        .with_additional(["id","score","vector"])
                        .do()
                    )
                    
            else:
                filters_ = WeaviateDataStore.build_filters(query.filter)
                
                # Added this check
                if query.query:
                    logger.debug(f"Querying with filters")
                    result = (
                        self.client.query.get(
                            WEAVIATE_CLASS,
                            [
                                "chunk_id",
                                "document_id",
                                "index",
                                "title",                                
                                "text",
                                "type",
                                "source",
                                "created_at",
                                "status",
                                "relationships { ... on " + WEAVIATE_RELATIONSHIP_CLASS + " { relationship_type, from_document { ... on " + WEAVIATE_CLASS + " { document_id, title } }, to_document { ... on " + WEAVIATE_CLASS + " { document_id, title } } } }"
                            ],
                        )                        
                        .with_hybrid(query=query.query, alpha=0.5, vector=query.embedding)
                        .with_where(filters_)
                        .with_limit(query.top_k)
                        .with_additional(["id","score","vector"])
                        .do()
                    )
                
            logger.debug(f"Result: {result}")
            query_results: List[DocumentChunkWithScore] = []
            if "data" not in result:
                logger.error(f"Query result does not contain 'data': {result}")
                return QueryResult(query=query.query, results=[])

            else:
                response = result["data"]["Get"][WEAVIATE_CLASS]

            response = result["data"]["Get"][WEAVIATE_CLASS]

            for resp in response:
                logger.debug(f"Processing document chunk: {resp}")
                from_documents = []
                to_documents = []
                if resp.get("relationships"):
                    for relationship in resp["relationships"]:
                        from_documents.extend([DocumentReference(document_id=ref["document_id"], title=ref["title"], relationship=relationship["relationship_type"]) for ref in relationship.get("from_document", [])])
                        to_documents.extend([DocumentReference(document_id=ref["document_id"], title=ref["title"], relationship=relationship["relationship_type"]) for ref in relationship.get("to_document", [])])
                        
                relationships = DocumentRelationship(from_documents=from_documents, to_documents=to_documents)
                result = DocumentChunkWithScore(
                    text=resp["text"],
                    #embedding=resp["_additional"]["vector"],
                    score=resp["_additional"]["score"],
                    metadata=DocumentChunkMetadata(
                        document_id=resp["document_id"] if resp["document_id"] else "",
                        title=resp["title"] if resp["title"] else "",
                        type=resp["type"] if resp["type"] else "",
                        source=resp["source"] if resp["source"] else "",
                        created_at=resp["created_at"],
                        status=resp["status"] if resp["status"] else ""
                    ),
                    relationships=relationships
                )
                query_results.append(result)
            return QueryResult(query=query.query, results=query_results)

        return await asyncio.gather(*[_single_query(query) for query in queries])

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
            except Exception as e:
                logger.error(f"Failed to delete all vectors: {e}")
                return False
            return True

        if ids is None and filter is None and not delete_all:
            logger.error("No ids or filter provided for deletion and delete_all is not set. Aborting.")
            return False

        if ids:
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

        if filter:
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
            "default": {"operator": "Equal", "value_key": "valueString"}
        }
        
        print(f"filter={filter}")
        
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
                    value_key: value
                }

                logger.debug(f"Operand: {operand}")
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
        logger.debug(f"Adding references between {from_document_id} and {to_document_id}")
        try:
            # Build the filter for the from_document
            from_filter = self.build_filters(DocumentChunkMetadataFilter(document_id=from_document_id, index=0))

            # Get the first chunk for the from_document
            from_chunk = self.client.query.get(WEAVIATE_CLASS).with_where(from_filter).with_additional(["id"]).do()
            
            print(f"from_chunk={from_chunk}")
         
            from_chunk_id = from_chunk['data']['Get']['OpenAIDocument'][0]['_additional']['id']

            # Build the filter for the to_document
            to_filter = self.build_filters(DocumentChunkMetadataFilter(document_id=to_document_id, index=0))

            # Get the first chunk for the to_document
            to_chunk = self.client.query.get(WEAVIATE_CLASS).with_where(to_filter).with_additional(["id"]).do()
            
            print(f"to_chunk={to_chunk}")
            
            # Check if the 'id' is in the '_additional' field of the response
            to_chunk_id = to_chunk['data']['Get']['OpenAIDocument'][0]['_additional']['id']

            # Create a Relationship object for the from_relationship_type
            from_relationship_resp = self.client.data_object.create(
                {
                    "from_document": [{
                        "beacon": f"weaviate://localhost/{from_chunk_id}"
                    }],
                    "to_document": [{
                        "beacon": f"weaviate://localhost/{to_chunk_id}"
                    }],
                    "relationship_type": from_relationship_type
                }, 
                WEAVIATE_RELATIONSHIP_CLASS
            )
            from_relationship_id = from_relationship_resp["_additional"]["id"]

            # Create a Relationship object for the to_relationship_type
            to_relationship_resp = self.client.data_object.create(
                {
                    "from_document": [{
                        "beacon": f"weaviate://localhost/{from_chunk_id}"
                    }],
                    "to_document": [{
                        "beacon": f"weaviate://localhost/{to_chunk_id}"
                    }],
                    "relationship_type": to_relationship_type
                },
                WEAVIATE_RELATIONSHIP_CLASS
            )
            to_relationship_id = to_relationship_resp["_additional"]["id"]

            # Add a reference from the from_document to the from_relationship_type Relationship object
            self.client.data_object.reference.add(
                from_uuid=from_chunk_id,
                from_property_name="relationships",
                to_uuid=from_relationship_id,
                from_class_name=WEAVIATE_CLASS,
                to_class_name=WEAVIATE_RELATIONSHIP_CLASS,
                consistency_level=consistency_level
            )

            # Add a reference from the to_document to the to_relationship_type Relationship object
            self.client.data_object.reference.add(
                from_uuid=to_chunk_id,
                from_property_name="relationships",
                to_uuid=to_relationship_id,
                from_class_name=WEAVIATE_CLASS,
                to_class_name=WEAVIATE_RELATIONSHIP_CLASS,
                consistency_level=consistency_level
            )

            return True
        except Exception as e:
            logger.error(f"Failed to add references between {from_document_id} and {to_document_id}: {e}")
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
        logger.debug(f"Deleting references between {from_document_id} and {to_document_id}")
        try:
            # Build the filter for the from_document
            from_filter = self.build_filters(DocumentChunkMetadataFilter(document_id=from_document_id, index=0))

            # Get the first chunk for the from_document
            from_chunk = self.client.query.get(WEAVIATE_CLASS).with_where(from_filter).with_additional(["id"]).do()

            from_chunk_id = from_chunk['data']['Get']['OpenAIDocument'][0]['_additional']['id']

            # Build the filter for the to_document
            to_filter = self.build_filters(DocumentChunkMetadataFilter(document_id=to_document_id, index=0))

            # Get the first chunk for the to_document
            to_chunk = self.client.query.get(WEAVIATE_CLASS).with_where(to_filter).with_additional(["id"]).do()
            
            print(f"to_chunk={to_chunk}")
            
            # Check if the 'id' is in the '_additional' field of the response
            to_chunk_id = to_chunk['data']['Get']['OpenAIDocument'][0]['_additional']['id']
            
            # Delete the references from the from_document
            for relationship in from_chunk['relationships']:
                self.client.data_object.reference.delete(
                    from_uuid=from_chunk_id,
                    from_property_name="relationships",
                    to_uuid=relationship["_additional"]["id"],
                    from_class_name=WEAVIATE_CLASS,
                    to_class_name=WEAVIATE_RELATIONSHIP_CLASS,
                    consistency_level=consistency_level
                )

            # Delete the references from the to_document
            for relationship in to_chunk['relationships']:
                self.client.data_object.reference.delete(
                    from_uuid=to_chunk_id,
                    from_property_name="relationships",
                    to_uuid=relationship["_additional"]["id"],
                    from_class_name=WEAVIATE_CLASS,
                    to_class_name=WEAVIATE_RELATIONSHIP_CLASS,
                    consistency_level=consistency_level
                )

            return True
        except Exception as e:
            logger.error(f"Failed to delete references between {from_document_id} and {to_document_id}: {e}")
            return False


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
