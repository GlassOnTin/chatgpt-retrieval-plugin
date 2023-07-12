# This is a version of the main.py file found in ../../server/main.py that also gives ChatGPT access to the upsert endpoint
# (allowing it to save information from the chat back to the vector) database.
# Copy and paste this into the main file at ../../server/main.py if you choose to give the model access to the upsert endpoint
# and want to access the openapi.json when you run the app locally at http://0.0.0.0:8000/sub/openapi.json.
import os
from typing import Optional
import json
import yaml
from pydantic import AnyUrl
import uvicorn
from fastapi import FastAPI, APIRouter, File, Form, HTTPException, Depends, Body, Response, Request, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles

from models.api import (
    DeleteRequest,
    DeleteResponse,
    QueryRequest,
    QueryResponse,
    UpsertRequest,
    UpsertResponse,
    AddReferenceRequest, 
    DeleteReferenceRequest, 
    ReferenceResponse
)
from datastore.factory import get_datastore
from services.file import get_document_from_file

from models.models import DocumentMetadata
from loguru import logger

bearer_scheme = HTTPBearer()
BEARER_TOKEN = os.environ.get("BEARER_TOKEN")
assert BEARER_TOKEN is not None


def validate_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return credentials


app = FastAPI()
app.mount("/.well-known", StaticFiles(directory=".well-known"), name="static")

# Create a sub-application, in order to access a subset of the endpoints in the OpenAPI schema, found at http://0.0.0.0:8000/sub/openapi.json when the app is running locally
sub_app = FastAPI(
    title="Graph Memory",
    description= "This plugin provides functionality to insert, update, or delete documents in a vector database. Each document consists of 'text' and 'metadata', with the latter including 'title', 'type', 'source', and 'status'. Documents are stored as chunks in the database, each with a sequential 'index'. All chunks of a document share a common 'document_id'. To update or delete a document, use the 'document_id'. The plugin also supports adding and deleting references between documents, which represent specific types of relationships. When storing new documents, consider the graph structure and memory engrams. A well-structured graph can improve query performance and the quality of results. For example, related documents can be linked via references, and commonly used information can be stored in separate documents and referenced as needed. The 'index' property can be used to retrieve specific chunks of a document.",
    version="1.0.0",
    servers=[{"url": "https://your-app-url.com"}],
    dependencies=[Depends(validate_token)],
)

# Create a router for the endpoints that should be in both the main app and the sub app
common_router = APIRouter()

@app.post(
    "/upsert-file",
    response_model=UpsertResponse,
)
async def upsert_file(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
):
    try:
        metadata_obj = (
            DocumentMetadata.parse_raw(metadata)
            if metadata
            else DocumentMetadata(source=file.filename)
        )
    except:
        metadata_obj = DocumentMetadata(source=file.filename)

    document = await get_document_from_file(file, metadata_obj)

    try:
        ids = await datastore.upsert([document])    # type: ignore
        return UpsertResponse(ids=ids)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"str({e})")

@common_router.post(
    "/upsert",
    response_model=UpsertResponse,
    description="Save chat information. Accepts an array of documents with text and metadata (no ID required as this will be generated).",
)
async def upsert(
    request: UpsertRequest = Body(...),
    token: HTTPAuthorizationCredentials = Depends(validate_token),
):
    try:
        ids = await datastore.upsert(request.documents) # type: ignore
        return UpsertResponse(ids=ids)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"str({e})")
        
@common_router.post(
    "/query",
    response_model=QueryResponse,
    description="Accepts an array of search query objects with a query string and optional filter. The filter refines results based on criteria like creation date, document ID, source, status, title, and type, and can include 'start_date' and 'end_date'. If 'ResponseTooLargeError' occurs, simplify queries or reduce 'top_k'."
)
async def query(
    request: QueryRequest = Body(...),
    token: HTTPAuthorizationCredentials = Depends(validate_token),
):
    try:
        results = await datastore.query(request.queries) # type: ignore
            
        return QueryResponse(results=results)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"str({e})")

@common_router.post(
    "/delete",
    response_model=DeleteResponse,
    description="Accepts IDs array, filter, or delete_all flag. Deletes documents matching IDs or filter criteria. Filter can include 'start_date' and 'end_date'. If delete_all is true, all documents are deleted. Use filter or IDs to avoid performance issues."

)
async def delete(
    request: DeleteRequest = Body(...),
    token: HTTPAuthorizationCredentials = Depends(validate_token),
):
    if not (request.ids or request.filter or request.delete_all):
        raise HTTPException(
            status_code=400,
            detail="One of ids, filter, or delete_all is required",
        )
    try:
        success = await datastore.delete(       # type: ignore
            ids=request.ids,
            filter=request.filter,
            delete_all=request.delete_all,
        )
        return DeleteResponse(success=success)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"str({e})")

@common_router.post(
    "/add_reference",
    response_model=ReferenceResponse,
    description="Adds a reference between two documents.",
)
async def add_reference(
    request: AddReferenceRequest = Body(...),
    token: HTTPAuthorizationCredentials = Depends(validate_token),
):
    try:
        success = await datastore.add_reference(        # type: ignore
            from_document_id=request.from_id,
            to_document_id=request.to_id,
            from_relationship_type=request.from_relationship_type,
            to_relationship_type=request.to_relationship_type
        )
        return ReferenceResponse(success=success)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"str({e})")
    
@common_router.post(
    "/delete_reference",
    response_model=ReferenceResponse,
    description="Deletes a reference between two documents.",
)
async def delete_reference(
    request: DeleteReferenceRequest = Body(...),
    token: HTTPAuthorizationCredentials = Depends(validate_token),
):
    try:
        success = await datastore.delete_reference(     # type: ignore
            from_document_id=request.from_id,
            to_document_id=request.to_id
        )
        return ReferenceResponse(success=success)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"str({e})")

@common_router.get("/openapi.yaml", include_in_schema=False)
async def get_openapi_yaml(request: Request):
    # An endpoint for the openapi.yaml
    openapi_schema = request.app.openapi()
    openapi_yaml = yaml.safe_dump(openapi_schema)
    return Response(content=openapi_yaml, media_type="application/x-yaml")

@sub_app.get("/openapi.yaml", include_in_schema=False)
async def get_sub_openapi_yaml():
    openapi_schema = sub_app.openapi()
    openapi_yaml = yaml.safe_dump(openapi_schema)
    return Response(content=openapi_yaml, media_type="application/x-yaml")

# Add a default representer
def default_representer(dumper, data):
    return dumper.represent_str(str(data))

yaml.representer.SafeRepresenter.add_representer(None, default_representer)

@app.on_event("startup")
async def startup():
    # Generate the OpenAPI schema for the sub app and save it to a file
    openapi_schema = sub_app.openapi()
    openapi_yaml = yaml.safe_dump(openapi_schema)
    with open(".well-known/openapi.yaml", "w") as file:
        file.write(openapi_yaml)

    # Generate the ai-plugin.json file with description from the app
    ai_plugin_info = {
        "schema_version": "v1",
        "name_for_model": "GraphMemory",
        "name_for_human": "Graph Memory",
        "description_for_model": sub_app.description,  # Use the description from the sub_app
        "description_for_human": "Vector Graph Memory with Document Relationships",
        "auth": {
            "type": "user_http",
            "authorization_type": "bearer"
        },
        "api": {
            "type": "openapi",
            "url": "https://your-app-url.com/.well-known/openapi.yaml",
            "has_user_authentication": False
        },
        "logo_url": "https://your-app-url.com/.well-known/logo.png",
        "contact_email": "hello@contact.com", 
        "legal_info_url": "hello@legal.com"
    }
    with open(".well-known/ai-plugin.json", "w") as file:
        json.dump(ai_plugin_info, file, indent=4)

    global datastore
    datastore = await get_datastore()

# Include the common router in the both main and sub app
app.include_router(common_router)
sub_app.include_router(common_router)

app.mount("/sub", sub_app)

def start():
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
