from pathlib import Path

from fastapi import FastAPI, File, UploadFile

import langGraph_agent
from file_utils import create_file_hash
from langGraph_states import ClaimState
from langGraph_states import File as ClaimFile

app = FastAPI(
    title="SuperClaims - Medical Insurance Claim Processing",
    description="API for processing medical insurance claims using LLMs and state machines.",
    version="1.0.0",
)

UPLOAD_DIR = Path("uploads/")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
async def home():
    return {"message": "Welcome to the SuperClaims web-app!"}


@app.post("/process-claim")
async def process_claim(input_files: list[UploadFile] = File(...)):
    file_as_bytes = [await file.read() for file in input_files]
    file_hashes = [create_file_hash(file_bytes) for file_bytes in file_as_bytes]

    if len(set(file_hashes)) != len(file_hashes):
        return {"error": "Duplicate files detected. Please upload unique files."}, 400

    claim_files = [
        ClaimFile(file_hash=file_hash, content_as_bytes=file_bytes)
        for file_hash, file_bytes in zip(file_hashes, file_as_bytes)
    ]
    claim_state = ClaimState(files=claim_files)
    # Invoke the langGraph state machine here with claim_state
    result = await langGraph_agent.invoke(claim_state)

    # TODO: Check if result is ClaimState object
    assert isinstance(result, ClaimState), "Result is not a ClaimState object"
    return {"result": result}  # Modify as per the actual structure of ClaimState
