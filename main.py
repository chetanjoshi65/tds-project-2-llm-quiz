"""
TDS Quiz Solver - Complete Implementation
Handles: Multi-step chains, HTML/API scraping, data extraction, file processing
"""

import os
import json
import requests
import asyncio
import re
import base64
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from playwright.sync_api import sync_playwright
import google.generativeai as genai
from dotenv import load_dotenv
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
STUDENT_EMAIL = os.getenv("STUDENT_EMAIL")
STUDENT_SECRET = os.getenv("STUDENT_SECRET")

if not all([GEMINI_API_KEY, STUDENT_EMAIL, STUDENT_SECRET]):
    raise RuntimeError("❌ Missing environment variables")

genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel("models/gemini-2.5-flash")

executor = ThreadPoolExecutor(max_workers=3)
app = FastAPI(title="TDS Quiz Solver")

# =============================================================================
# MODELS
# =============================================================================
class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str
    model_config = ConfigDict(extra="ignore")

# =============================================================================
# HTML FETCHING (Playwright)
# =============================================================================
def fetch_html_sync(url: str) -> str:
    """Fetch HTML using Playwright - handles JS rendering"""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-setuid-sandbox', 
                      '--disable-dev-shm-usage', '--disable-gpu']
            )
            page = browser.new_page()
            page.goto(url, timeout=30000, wait_until='domcontentloaded')
            html = page.content()
            browser.close()
            return html
    except Exception as e:
        raise RuntimeError(f"Failed to fetch HTML: {e}")

async def fetch_html(url: str) -> str:
    """Async wrapper"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, fetch_html_sync, url)

# =============================================================================
# SMART HTML DATA EXTRACTION
# =============================================================================
def extract_from_html(html: str, instructions: str) -> Optional[str]:
    """
    Extract hidden data from HTML based on instructions
    Handles: class names, IDs, reversed text, hidden elements
    """
    try:
        soup = BeautifulSoup(html, 'html.parser')
        inst_lower = instructions.lower()
        
        # Pattern 1: Extract by class name
        class_match = re.search(r'class[:\s]+["\']?([a-zA-Z0-9\-_]+)', inst_lower)
        if class_match:
            class_name = class_match.group(1)
            elem = soup.find(class_=class_name)
            if elem:
                text = elem.get_text().strip()
                print(f"✓ Found class '{class_name}': {text}")
                
                # Check if reversed
                if 'revers' in inst_lower:
                    text = text[::-1]
                    print(f"✓ Reversed: {text}")
                
                return text
        
        # Pattern 2: Extract by ID
        id_match = re.search(r'id[:\s]+["\']?([a-zA-Z0-9\-_]+)', inst_lower)
        if id_match:
            id_name = id_match.group(1)
            elem = soup.find(id=id_name)
            if elem:
                text = elem.get_text().strip()
                print(f"✓ Found id '{id_name}': {text}")
                
                if 'revers' in inst_lower:
                    text = text[::-1]
                    print(f"✓ Reversed: {text}")
                
                return text
        
        # Pattern 3: Look for specific tag patterns
        if 'hidden' in inst_lower or 'secret' in inst_lower:
            # Try common hidden element patterns
            for elem in soup.find_all(['div', 'span', 'p']):
                if any(word in elem.get('class', []) for word in ['hidden', 'secret', 'password', 'key']):
                    text = elem.get_text().strip()
                    if text:
                        print(f"✓ Found hidden element: {text}")
                        if 'revers' in inst_lower:
                            text = text[::-1]
                        return text
        
        return None
        
    except Exception as e:
        print(f"⚠️ Extraction error: {e}")
        return None

# =============================================================================
# PARSE QUIZ (Extract question, submit_url, data)
# =============================================================================
def parse_quiz(html: str) -> Dict[str, Any]:
    """Extract quiz metadata using LLM"""
    
    prompt = f"""
Extract quiz information from this HTML and return ONLY valid JSON:

{{
  "question": "the question text",
  "submit_url": "the submit URL (absolute or relative)",
  "data_sources": ["any data/hints provided as array"],
  "instructions": "any special instructions for solving"
}}

HTML (first 4000 chars):
{html[:4000]}

Return ONLY the JSON, no other text.
"""
    
    try:
        response = llm.generate_content(prompt)
        text = response.text.strip()
        
        # Extract JSON
        start = text.find("{")
        end = text.rfind("}") + 1
        
        if start == -1 or end == 0:
            raise ValueError("No JSON found in response")
        
        parsed = json.loads(text[start:end])
        
        # Ensure required fields
        if "question" not in parsed or "submit_url" not in parsed:
            raise ValueError("Missing required fields")
        
        if "data_sources" not in parsed:
            parsed["data_sources"] = []
        
        if "instructions" not in parsed:
            parsed["instructions"] = ""
        
        return parsed
        
    except Exception as e:
        print(f"❌ Parse error: {e}")
        print(f"LLM response: {text[:500] if 'text' in locals() else 'N/A'}")
        raise RuntimeError(f"Failed to parse quiz: {e}")

# =============================================================================
# SOLVE QUESTION (Smart extraction + LLM)
# =============================================================================
def solve_question(question: str, data_sources: List[str], instructions: str, html: str) -> str:
    """
    Solve quiz question intelligently:
    1. Try smart HTML extraction if applicable
    2. Use LLM with full context if extraction fails
    """
    
    # Combine instructions
    all_instructions = ' '.join(data_sources) + ' ' + instructions
    
    # Try smart extraction first
    if html and any(word in all_instructions.lower() for word in 
                    ['html', 'div', 'class', 'id', 'hidden', 'revers', 'element']):
        print("🔍 Attempting smart HTML extraction...")
        extracted = extract_from_html(html, all_instructions)
        if extracted:
            print(f"✅ Smart extraction: {extracted}")
            return extracted
    
    # Fallback to LLM
    print("🤖 Using LLM to solve...")
    
    # Build comprehensive context
    context = f"""
Question: {question}

Instructions: {all_instructions}

Data sources: {json.dumps(data_sources)}

HTML content (relevant parts):
{html[:6000] if html else "N/A"}
"""
    
    prompt = f"""
{context}

Solve this quiz question following these rules:

1. READ ALL INSTRUCTIONS CAREFULLY
2. If data is in HTML, EXTRACT it (check for classes, IDs, hidden elements)
3. Apply transformations (reverse text, decode base64, etc.)
4. Calculate or process as needed
5. Return ONLY the final answer (no explanations)

Important:
- Maximum 100 characters
- Direct answer only
- If it's a password/code, give exact value
- If it's a number, give just the number
- If it's a calculation, give the result

Answer:"""
    
    try:
        response = llm.generate_content(prompt)
        answer = response.text.strip()
        
        # Clean up answer
        lines = [l.strip() for l in answer.split('\n') if l.strip()]
        answer = lines[0] if lines else answer
        
        # Remove common prefixes
        for prefix in ['Answer:', 'The answer is:', 'Result:', 'answer:', 'A:']:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
        
        # Remove quotes
        answer = answer.strip('"\'`')
        
        # Limit length
        if len(answer) > 200:
            answer = answer[:200]
        
        return answer
        
    except Exception as e:
        print(f"❌ Solve error: {e}")
        return "ERROR"

# =============================================================================
# SUBMIT ANSWER
# =============================================================================
def submit_answer(submit_url: str, original_url: str, answer: str) -> Dict[str, Any]:
    """Submit answer to quiz endpoint"""
    
    # Handle relative URLs
    if submit_url.startswith('/'):
        parsed = urlparse(original_url)
        submit_url = f"{parsed.scheme}://{parsed.netloc}{submit_url}"
        print(f"🔗 Converted to: {submit_url}")
    
    payload = {
        "email": STUDENT_EMAIL,
        "secret": STUDENT_SECRET,
        "url": original_url,
        "answer": answer
    }
    
    print(f"📤 Submitting to: {submit_url}")
    print(f"📦 Answer: {answer[:100]}")
    
    try:
        resp = requests.post(submit_url, json=payload, timeout=15)
        
        # Try to parse JSON response
        try:
            result = resp.json()
        except:
            result = {
                "correct": False,
                "reason": f"Invalid response: {resp.text[:200]}",
                "url": ""
            }
        
        return result
        
    except requests.exceptions.Timeout:
        return {"correct": False, "reason": "Timeout", "url": ""}
    except Exception as e:
        print(f"❌ Submit error: {e}")
        return {"correct": False, "reason": str(e), "url": ""}

# =============================================================================
# MAIN ENDPOINT - MULTI-STEP QUIZ SOLVER
# =============================================================================
@app.post("/")
async def solve_quiz_chain(task: QuizRequest):
    """
    Main endpoint: Solves multi-step quiz chains
    Handles all question types and data extraction
    """
    try:
        print(f"\n{'='*70}")
        print(f"📥 NEW REQUEST: {task.url}")
        print(f"{'='*70}")
        
        # Verify secret
        if task.secret != STUDENT_SECRET:
            raise HTTPException(status_code=403, detail="Invalid secret")
        
        current_url = task.url
        max_steps = 10
        results = []
        
        # Multi-step quiz chain loop
        for step in range(max_steps):
            print(f"\n🔄 STEP {step + 1}/{max_steps}")
            print(f"📍 URL: {current_url}")
            
            try:
                # Fetch HTML
                print("🌐 Fetching HTML...")
                html = await fetch_html(current_url)
                print(f"✅ Fetched {len(html)} chars")
                
                # Parse quiz
                print("🧩 Parsing quiz...")
                parsed = parse_quiz(html)
                print(f"✅ Parsed: {json.dumps(parsed, indent=2)}")
                
                question = parsed.get("question", "")
                submit_url = parsed.get("submit_url", "")
                data_sources = parsed.get("data_sources", [])
                instructions = parsed.get("instructions", "")
                
                if not question or not submit_url:
                    print("⚠️ Missing question or submit_url - ending chain")
                    break
                
                # Solve question
                print(f"💡 Solving: {question[:100]}...")
                if data_sources:
                    print(f"📊 Data: {data_sources}")
                
                answer = solve_question(question, data_sources, instructions, html)
                print(f"✅ Answer: {answer}")
                
                # Submit answer
                print("📤 Submitting...")
                result = submit_answer(submit_url, current_url, answer)
                print(f"📬 Result: {json.dumps(result, indent=2)}")
                
                # Store result
                results.append({
                    "step": step + 1,
                    "url": current_url,
                    "question": question[:100],
                    "answer": answer,
                    "correct": result.get("correct", False),
                    "reason": result.get("reason", "")
                })
                
                # Check if correct
                if not result.get("correct", False):
                    print(f"❌ Incorrect answer - stopping")
                    print(f"Reason: {result.get('reason', 'Unknown')}")
                    break
                
                # Check for next quiz
                next_url = result.get("url", "")
                
                if not next_url or not next_url.startswith("http"):
                    print("✅ No more quizzes - chain complete!")
                    break
                
                # Move to next quiz
                current_url = next_url
                print(f"➡️ Next: {next_url}")
                
            except Exception as step_error:
                print(f"❌ Step {step + 1} error: {step_error}")
                results.append({
                    "step": step + 1,
                    "error": str(step_error)
                })
                break
        
        # Return final result
        if results and "correct" in results[-1]:
            final = results[-1]
            response = {
                "correct": final.get("correct", False),
                "reason": final.get("reason", ""),
                "url": "",
                "delay": None,
                "steps_completed": len(results)
            }
        else:
            response = {
                "correct": False,
                "reason": "Failed to complete quiz chain",
                "url": "",
                "delay": None
            }
        
        print(f"\n🎉 COMPLETED: {len(results)} step(s)")
        print(f"Final result: {response}")
        print(f"{'='*70}\n")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# HEALTH ENDPOINTS
# =============================================================================
@app.get("/")
def health():
    """Health check"""
    return {
        "status": "running",
        "email": STUDENT_EMAIL,
        "gemini_configured": bool(GEMINI_API_KEY)
    }

@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "playwright": "ready",
        "gemini": "configured",
        "email": STUDENT_EMAIL
    }

# =============================================================================
# STARTUP
# =============================================================================
@app.on_event("startup")
async def startup():
    print("="*70)
    print("🚀 TDS Quiz Solver Started")
    print(f"📧 Email: {STUDENT_EMAIL}")
    print(f"🔑 Gemini: {'Configured' if GEMINI_API_KEY else 'NOT CONFIGURED'}")
    print("="*70)
