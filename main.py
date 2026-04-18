from fastapi import  FastAPI
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from model import load_artifacts,recommend
from schemas import AnalyzeResponse,StudentInput
from fastapi.middleware.cors import CORSMiddleware
import os

# Initialize FastAPI app
app = FastAPI(
    title="Student Performance Predictor",
    description="Predict exam scores and get optimization strategies",
    version="1.0.0"
)
 
# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for Streamlit Cloud)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Load ML artifacts when app starts"""
    load_artifacts()
    print("✅ Model artifacts loaded successfully")
 
 
@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "message": "Student Performance Predictor API",
        "version": "1.0.0",
        "status": "✅ Running",
        "endpoints": {
            "analyze": "/analyze (POST)",
            "docs": "/docs"
        }
    }

@app.post("/analyze",response_model=AnalyzeResponse)
def analyze(features:StudentInput):
    result=recommend(features.model_dump())
    return AnalyzeResponse(predicted_exam_score=result["predicted_exam_score"],
           optimization_steps=result["optimization_steps"],
                           message=result["message"]                 )


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or default to 7860 (for Hugging Face)
    port = int(os.getenv("PORT", 7860))
    host = "0.0.0.0"  # Required for Docker/Hugging Face
    
    uvicorn.run(app, host=host, port=port)