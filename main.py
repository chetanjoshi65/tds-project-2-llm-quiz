import os
import json
import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict
from playwright.sync_api import sync_playwright  # ✅ Use sync API
import google.generativeai as genai
from dotenv import load_dotenv

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

# ✅ Create thread pool executor for blocking operations
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
# FETCH HTML (Sync version for thread execution)
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
    """Async wrapper that runs sync playwright in thread"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, fetch_html_sync, url)


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
        print(f"❌ Failed to parse JSON from LLM response: {text[:200]}")
        raise RuntimeError(f"Failed to parse quiz metadata: {e}")


# ---------------------------------------------------------------------------
# SOLVE QUIZ (basic)
# ---------------------------------------------------------------------------
def solve_question(question: str):
    response = llm.generate_content(
        f"Answer this question clearly and correctly: {question}"
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

    print(f"📤 Submitting to: {submit_url}")
    print(f"📦 Payload: {payload}")

    try:
        resp = requests.post(submit_url, json=payload, timeout=10)
        return resp.json()
    except Exception as e:
        print(f"❌ Submit error: {e}")
        return {"correct": False, "reason": f"Submission failed: {e}"}


# ---------------------------------------------------------------------------
# API ENDPOINT
# ---------------------------------------------------------------------------

@app.post("/")
async def root(task: QuizRequest):
    """
    Main endpoint to solve quiz - handles multi-step chains
    """
    try:
        print(f"📥 Received quiz request for: {task.url}")
        
        if task.secret != STUDENT_SECRET:
            raise HTTPException(status_code=403, detail="Invalid secret")

        current_url = task.url
        max_iterations = 10  # Prevent infinite loops
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
            parsed = parse_quiz(html)
            print(f"✅ Parsed: {parsed}")

            question = parsed.get("question", "")
            submit_url = parsed.get("submit_url", "")
            
            if not question or not submit_url:
                print("⚠️  No question or submit URL found - ending chain")
                break

            # Solve question
            print(f"💡 Solving: {question}")
            answer = solve_question(question)
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



'''@app.post("/")
async def root(task: QuizRequest):
    """Main endpoint to solve quiz"""
    try:
        print(f"📥 Received quiz request for: {task.url}")

        if task.secret != STUDENT_SECRET:
            raise HTTPException(status_code=403, detail="Invalid secret")

        # Fetch HTML (runs in thread pool)
        print("🌐 Fetching HTML...")
        html = await fetch_html(task.url)
        print(f"✅ HTML fetched: {len(html)} chars")

        # Parse quiz
        print("🤖 Parsing quiz with Gemini...")
        parsed = parse_quiz(html)
        print(f"✅ Parsed: {parsed}")

        question = parsed.get("question", "")
        submit_url = parsed.get("submit_url", "")

        if not question or not submit_url:
            raise HTTPException(
                status_code=400,
                detail="Could not extract question or submit_url"
            )

        # Solve question
        print(f"💡 Solving: {question}")
        answer = solve_question(question)
        print(f"✅ Answer: {answer}")

        # Submit answer
        print("📤 Submitting answer...")
        result = submit_answer(submit_url, task.url, answer)
        print(f"✅ Result: {result}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))'''


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
















