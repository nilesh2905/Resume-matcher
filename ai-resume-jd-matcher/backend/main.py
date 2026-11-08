import os
import json
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai import types

app = FastAPI(title="AI Resume-Job Description Matcher API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

class MatchRequest(BaseModel):
    resume_text: str
    job_description: str

class RewriteSuggestion(BaseModel):
    target_section: str
    rewritten_bullet: str

class LLMAnalysisSchema(BaseModel):
    llm_alignment_score: int
    missing_skills: List[str]
    matched_skills: List[str]
    tailored_rewrite_suggestions: List[RewriteSuggestion]

class MatchResponse(BaseModel):
    match_score: float
    semantic_embedding_score: float
    llm_alignment_score: int
    matched_skills: List[str]
    missing_skills: List[str]
    tailored_rewrite_suggestions: List[RewriteSuggestion]

def compute_embedding_similarity(resume_text: str, jd_text: str) -> float:
    embeddings = embedding_model.encode([resume_text, jd_text], normalize_embeddings=True)
    raw_sim = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
    return round(min(max((raw_sim - 0.20) / 0.65 * 100.0, 0.0), 100.0), 1)

@app.post("/api/match", response_model=MatchResponse)
def match_resume_to_jd(payload: MatchRequest):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set.")
    
    semantic_score = compute_embedding_similarity(payload.resume_text, payload.job_description)
    client = genai.Client(api_key=api_key)
    prompt = f"Analyze resume vs JD and return JSON:\nResume:\n{payload.resume_text}\n\nJD:\n{payload.job_description}"
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=LLMAnalysisSchema,
            temperature=0.2,
        ),
    )
    llm_output = json.loads(response.text)
    combined_score = round((0.40 * semantic_score) + (0.60 * llm_output["llm_alignment_score"]), 1)
    
    return MatchResponse(
        match_score=combined_score,
        semantic_embedding_score=semantic_score,
        llm_alignment_score=llm_output["llm_alignment_score"],
        matched_skills=llm_output["matched_skills"],
        missing_skills=llm_output["missing_skills"],
        tailored_rewrite_suggestions=llm_output["tailored_rewrite_suggestions"]
    )
