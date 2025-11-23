import os
import json
import requests
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict
from playwright.sync_api import sync_playwright
import google.generativeai as genai

# ---------------------------------------------------------------------------
# ENVIRONMENT VARIABLES
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
STUDENT_EMAIL = os.getenv("STUDENT_EMAIL")
STUDENT_SECRET = os.getenv("STUDENT_SECRET")

if not GEMINI_API_KEY or not STUDENT_EMAIL or not STUDENT_SECRET:
    raise RuntimeError("Missing environment variables")

genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel("gemini-pro")

# ---------------------------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------------------------
app = FastAPI()

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str
    model_config = ConfigDict(extra="ignore")

# ---------------------------------------------------------------------------
# FETCH HTML (JS RENDERED)
# ---------------------------------------------------------------------------
def fetch_html(url: str) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=30000)
        page.wait_for_load_state("domcontentloaded")
        html = page.content()
        browser.close()
        return html

# ---------------------------------------------------------------------------
# PARSE QUIZ (LLM)
# ---------------------------------------------------------------------------
def parse_quiz(html: str):
    prompt = f"""
Extract and return ONLY valid JSON:

1. question text
2. submit_url
3. any required data

Format:
{{
  "question":"...",
  "submit_url":"...",
  "data_sources":[]
}}
"""

    response = llm.generate_content(prompt + html)
    text = response.text

    try:
        return json.loads(text[text.index("{"): text.rindex("}")+1])
    except:
        raise RuntimeError("Failed to parse quiz metadata")

# ---------------------------------------------------------------------------
# SOLVE QUIZ (basic)
# ---------------------------------------------------------------------------
def solve_question(question: str):
    response = llm.generate_content(
        f"Answer this clearly and correctly: {question}"
    )
    return response.text.strip()

# ---------------------------------------------------------------------------
# SUBMIT ANSWER
# ---------------------------------------------------------------------------
def submit_answer(submit_url, original_url, answer):
    payload = {
        "email": STUDENT_EMAIL,
        "secret": STUDENT_SECRET,
        "url": original_url,
        "answer": answer
    }

    resp = requests.post(submit_url, json=payload)

    try:
        return resp.json()
    except:
        return {"correct": False, "reason": "Invalid server response"}

# ---------------------------------------------------------------------------
# API ENDPOINT — MUST RETURN FINAL ANSWER
# ---------------------------------------------------------------------------
@app.post("/")
async def root(task: QuizRequest):

    if task.secret != STUDENT_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    html = fetch_html(task.url)
    parsed = parse_quiz(html)

    question = parsed["question"]
    submit_url = parsed["submit_url"]

    answer = solve_question(question)

    result = submit_answer(submit_url, task.url, answer)

    return result  # ✅ full final answer returned immediately

# ---------------------------------------------------------------------------
@app.get("/")
def home():
    return {"status": "running"}
