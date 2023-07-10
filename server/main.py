# This is a version of the main.py file found in ../../server/main.py that also gives ChatGPT access to the upsert endpoint
# (allowing it to save information from the chat back to the vector) database.
# Copy and paste this into the main file at ../../server/main.py if you choose to give the model access to the upsert endpoint
# and want to access the openapi.json when you run the app locally at http://0.0.0.0:8000/sub/openapi.json.
import os
from typing import Optional
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

# Create a sub-application, in order to access a subset of the endpoints in the OpenAPI schema, found at http://0.0.0.0:8000/sub/openapi.json when the app is running locally
sub_app = FastAPI(
    title="Retrieval Plugin API",
    description="A retrieval API for querying and filtering documents based on natural language queries and metadata",
    version="1.0.0",
    servers=[{"url": "https://your-app-url.com"}],
    dependencies=[Depends(validate_token)],
)

# Create a router for the endpoints that should be in both the main app and the sub app
common_router = APIRouter()
app_router = APIRouter()
sub_app_router = APIRouter()

@app_router.post(
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
    # NOTE: We are describing the shape of the API endpoint input due to a current limitation in parsing arrays of objects from OpenAPI schemas. This will not be necessary in the future.
    description="Accepts search query objects array each with query and optional filter. Break down complex questions into sub-questions. Refine results by criteria, e.g. time, don't do this often. Split queries if ResponseTooLargeError occurs.",
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
    response_model=DeleteResponse
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

    global datastore
    datastore = await get_datastore()

# Include the common router in the both main and sub app
app.include_router(common_router)
app.include_router(app_router)
sub_app.include_router(common_router)
sub_app.include_router(sub_app_router)

app.mount("/.well-known", StaticFiles(directory=".well-known"), name="static")
app.mount("/sub", sub_app)

def start():
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
