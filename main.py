from fastapi import FastAPI , HTTPException
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
    session_id : str

#adding session id - so need a new request body for build_index as well

class IndexRequest(BaseModel):
    video_id : str
    session_id : str

@app.post("/index")
def build_index(request : IndexRequest):
    try:
        rag.build_index(request.session_id , request.video_id)
        return "Indexing done"
    except Exception as e:
        raise HTTPException (status_code=500 , detail=str(e))

@app.post("/ask")
def ask_question(request: AskRequest):
    try:
        return rag.ask_question(request.video_id, request.question , request.session_id)
    except Exception as e:
        raise HTTPException(status_code=500 , detail=str(e))



