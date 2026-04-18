from pydantic import BaseModel, Field
from typing import Optional, List, Union

class StudentInput(BaseModel):
    # Numeric features
    hours_studied: float = Field(..., ge=0, le=50, example=20.0, description="Hours studied per week")
    attendance: float = Field(..., ge=0, le=100, example=85.0, description="Attendance percentage")
    sleep_hours: float = Field(..., ge=0, le=14, example=7.0, description="Sleep hours per night")
    previous_scores: float = Field(..., ge=40, le=100, example=75.0, description="Previous exam scores")
    tutoring_sessions: float = Field(..., ge=0, le=10, example=2.0, description="Tutoring sessions per week")
    physical_activity: float = Field(..., ge=0, le=15, example=3.0, description="Physical activity hours per week")
    
    # Categorical features
    parental_involvement: str = Field(..., example="Medium", description="Low / Medium / High")
    access_to_resources: str = Field(..., example="Medium", description="Low / Medium / High")
    extracurricular: str = Field(..., example="No", description="Yes / No")
    motivation_level: str = Field(..., example="Medium", description="Low / Medium / High")
    internet_access: str = Field(..., example="Yes", description="Yes / No")
    family_income: str = Field(..., example="Medium", description="Low / Medium / High")
    teacher_quality: str = Field(..., example="Medium", description="Low / Medium / High")
    school_type: str = Field(..., example="Public", description="Public / Private")
    peer_influence: str = Field(..., example="Neutral", description="Negative / Neutral / Positive")
    learning_disabilities: str = Field(..., example="No", description="Yes / No")
    parental_education: str = Field(..., example="College", description="High School / College / Postgraduate")
    gender: str = Field(..., example="Male", description="Male / Female")
    distance_from_home: str = Field(..., example="Near", description="Near / Moderate / Far")


class PredictResponse(BaseModel):
    predicted_exam_score: float
class OptimizationStep(BaseModel):
    feature: str = Field(..., description="Feature name")
    strategy: str = Field(..., description="Human-readable strategy label")
    from_value: Union[float, str] = Field(..., description="Original value (numeric or categorical)")
    to_value: Union[float, str] = Field(..., description="Recommended value (numeric or categorical)")
    impact: float = Field(..., description="Predicted score improvement")
 
 
class AnalyzeResponse(BaseModel):
    predicted_exam_score: float = Field(..., description="Current predicted exam score")
    optimization_steps: List[OptimizationStep] = Field(default_factory=list, description="List of optimization steps")
    message: str = Field(..., description="Contextual message based on score")