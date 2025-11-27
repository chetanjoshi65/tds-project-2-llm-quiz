import os
import json
import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict
from playwright.sync_api import sync_playwright
import google.generativeai as genai
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()

# ---------------------------------------------------------------------------
# ENVIRONMENT VARIABLES
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
STUDENT_EMAIL = os.getenv("STUDENT_EMAIL")
STUDENT_SECRET = os.getenv("STUDENT_SECRET")

if not GEMINI_API_KEY or not STUDENT_EMAIL or not STUDENT_SECRET:
    raise RuntimeError("Missing environment variables")

genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel("models/gemini-2.5-flash")

# Create thread pool executor
executor = ThreadPoolExecutor(max_workers=3)

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
# FETCH HTML
# ---------------------------------------------------------------------------
def fetch_html_sync(url: str) -> str:
    """Sync version that runs in thread pool"""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--no-zygote',
                    '--single-process'
                ]
            )
            page = browser.new_page()
            page.goto(url, timeout=30000)
            page.wait_for_load_state("domcontentloaded")
            html = page.content()
            browser.close()
            return html
    except Exception as e:
        print(f"❌ Error fetching HTML: {e}")
        raise


async def fetch_html(url: str) -> str:
    """Async wrapper"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, fetch_html_sync, url)


# ---------------------------------------------------------------------------
# PARSE QUIZ
# ---------------------------------------------------------------------------
def parse_quiz(html: str):
    """Extract question, submit URL, and data from HTML"""
    prompt = f"""
Extract and return ONLY valid JSON with these fields:

1. question: the quiz question text
2. submit_url: the URL to submit the answer
3. data_sources: any data/hints provided (as array)

Return ONLY this JSON format:
{{
  "question": "the question text",
  "submit_url": "the submit URL",
  "data_sources": ["data1", "data2"]
}}

HTML (first 3000 chars):
{html[:3000]}
"""

    response = llm.generate_content(prompt)
    text = response.text

    try:
        # Find JSON in response
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception as e:
        print(f"❌ Failed to parse JSON from LLM: {text[:200]}")
        raise RuntimeError(f"Failed to parse quiz: {e}")


# ---------------------------------------------------------------------------
# SOLVE QUESTION
# ---------------------------------------------------------------------------
def solve_question(question: str, data_sources: list = None):
    """Solve quiz question using provided data"""
    
    # Build context
    context = f"Question: {question}\n"
    
    if data_sources:
        context += f"\nAvailable data: {', '.join(str(d) for d in data_sources)}\n"
    
    prompt = f"""
{context}

Solve this quiz question step by step, then provide ONLY the final answer.

Instructions:
- Use the data provided above
- Think through the problem
- Give a SHORT, DIRECT answer (max 50 characters)
- No explanations in the final answer

Format your response as:
Reasoning: [your thinking process]
Answer: [just the final answer]

Solve now:"""
    
    response = llm.generate_content(prompt)
    text = response.text.strip()
    
    # Extract answer
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    
    # Look for "Answer:" line
    for line in reversed(lines):
        if 'answer:' in line.lower():
            answer = line.split(':', 1)[1].strip()
            # Remove quotes
            answer = answer.strip('"\'')
            return answer
    
    # Fallback: take last line
    answer = lines[-1] if lines else "ERROR"
    
    # Clean up common prefixes
    for prefix in ["Answer:", "The answer is:", "Result:", "Final answer:"]:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()
    
    return answer.strip('"\'')


# ---------------------------------------------------------------------------
# SUBMIT ANSWER
# ---------------------------------------------------------------------------
def submit_answer(submit_url, original_url, answer):
    """Submit answer - handles relative URLs"""
    
    # Fix relative URLs
    if submit_url.startswith('/'):
        parsed = urlparse(original_url)
        submit_url = f"{parsed.scheme}://{parsed.netloc}{submit_url}"
        print(f"🔗 Converted relative URL to: {submit_url}")
    
    payload = {
        "email": STUDENT_EMAIL,
        "secret": STUDENT_SECRET,
        "url": original_url,
        "answer": answer
    }

    print(f"📤 Submitting to: {submit_url}")
    print(f"📦 Answer: {answer[:100]}...")

    try:
        resp = requests.post(submit_url, json=payload, timeout=10)
        return resp.json()
    except Exception as e:
        print(f"❌ Submit error: {e}")
        return {"correct": False, "reason": f"Submission failed: {e}"}


# ---------------------------------------------------------------------------
# MAIN ENDPOINT
# ---------------------------------------------------------------------------
@app.post("/")
async def root(task: QuizRequest):
    """Main endpoint - handles multi-step quiz chains"""
    try:
        print(f"📥 Received quiz request for: {task.url}")
        
        if task.secret != STUDENT_SECRET:
            raise HTTPException(status_code=403, detail="Invalid secret")

        current_url = task.url
        max_iterations = 10
        all_results = []
        
        for iteration in range(max_iterations):
            print(f"\n🔄 Quiz step {iteration + 1}/{max_iterations}")
            print(f"📍 URL: {current_url}")
            
            # Fetch HTML
            print("🌐 Fetching HTML...")
            html = await fetch_html(current_url)
            print(f"✅ HTML fetched: {len(html)} chars")

            # Parse quiz
            print("🤖 Parsing quiz with Gemini...")
            parsed = parse_quiz(html)  # ✅ FIXED - only pass html
            print(f"✅ Parsed: {parsed}")

            question = parsed.get("question", "")
            submit_url = parsed.get("submit_url", "")
            data_sources = parsed.get("data_sources", [])
            
            if not question or not submit_url:
                print("⚠️  No question or submit URL found - ending chain")
                break

            # Solve question
            print(f"💡 Solving: {question}")
            if data_sources:
                print(f"📊 Using data: {data_sources}")
            answer = solve_question(question, data_sources)
            print(f"✅ Answer: {answer}")

            # Submit answer
            print("📤 Submitting answer...")
            result = submit_answer(submit_url, current_url, answer)
            print(f"✅ Result: {result}")
            
            all_results.append({
                "step": iteration + 1,
                "question": question,
                "answer": answer,
                "result": result
            })
            
            # Check if there's a next quiz
            next_url = result.get("url", "")
            
            if not next_url or not next_url.startswith("http"):
                print("✅ Quiz chain complete - no more quizzes")
                break
            
            if result.get("correct") == False:
                print("❌ Answer was incorrect - stopping chain")
                break
                
            # Continue to next quiz
            current_url = next_url
            print(f"➡️  Moving to next quiz: {next_url}")
        
        # Return the final result
        final_result = all_results[-1]["result"] if all_results else {
            "correct": False,
            "reason": "No quizzes were solved",
            "url": "",
            "delay": None
        }
        
        print(f"\n🎉 Completed {len(all_results)} quiz(zes)")
        return final_result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# HEALTH ENDPOINTS
# ---------------------------------------------------------------------------
@app.get("/")
def home():
    return {
        "status": "running",
        "email": STUDENT_EMAIL,
        "gemini_configured": bool(GEMINI_API_KEY)
    }


@app.get("/health")
def health():
    return {"status": "healthy"}



























