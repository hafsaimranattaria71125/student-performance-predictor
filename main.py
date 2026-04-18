from fastapi import  FastAPI
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from model import load_artifacts,recommend
from schemas import AnalyzeResponse,StudentInput
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()  # runs at startup
    yield            # app runs here
    # optional cleanup code after shutdown

app = FastAPI(lifespan=lifespan)

@app.get("/")
def test():
    return JSONResponse(status_code=200,
        content={"success":True,"message":"Welcome to this app. This is test route."})

@app.post("/analyze",response_model=AnalyzeResponse)
def analyze(features:StudentInput):
    result=recommend(features.model_dump())
    return AnalyzeResponse(predicted_exam_score=result["predicted_exam_score"],
           optimization_steps=result["optimization_steps"],
                           message=result["message"]                 )
