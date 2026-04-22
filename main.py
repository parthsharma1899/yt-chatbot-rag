from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 
import rag
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#build_index -> a POST request as it builds an index for that video_id not just reads from the server(GET)
class AskRequest(BaseModel):
    video_id : str
    question : str


@app.post("/index/{video_id}")
def build_index(video_id):
    rag.build_index(video_id)
    return "Indexing done"

@app.post("/ask")
def ask_question(request: AskRequest):
    return rag.ask_question(request.video_id, request.question)



