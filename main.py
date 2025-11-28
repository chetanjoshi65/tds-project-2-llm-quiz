"""
TDS Quiz Solver - Complete Implementation
Handles: Multi-step chains, HTML/API scraping, data extraction, file processing, pagination, API authentication, data cleaning
"""

import os
import json
import requests
import asyncio
import re
import base64
import time
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
# HEADER EXTRACTION
# =============================================================================
def extract_headers_from_instructions(instructions: str) -> Dict[str, str]:
    """
    Extract required headers from instructions
    Returns dict of headers
    """
    headers = {}
    
    # Pattern 1: "header X-API-Key with value weather-alpha-key"
    # Pattern 1b: "header X-API-Key with the value weather-alpha-key"
    header_pattern = r'header\s+[`"\']?([A-Za-z0-9\-]+)[`"\']?\s+with\s+(?:the\s+)?value\s+[`"\']?([A-Za-z0-9\-]+)[`"\']?'
    matches = re.findall(header_pattern, instructions, re.IGNORECASE)
    
    for header_name, header_value in matches:
        headers[header_name] = header_value
        print(f"🔑 Extracted header: {header_name} = {header_value}")
    
    # Pattern 2: "sending a header X-API-Key with value weather-alpha-key"
    header_pattern2 = r'sending\s+a\s+header\s+[`"\']?([A-Za-z0-9\-]+)[`"\']?\s+with\s+(?:the\s+)?value\s+[`"\']?([A-Za-z0-9\-]+)[`"\']?'
    matches2 = re.findall(header_pattern2, instructions, re.IGNORECASE)
    
    for header_name, header_value in matches2:
        headers[header_name] = header_value
        print(f"🔑 Extracted header (pattern 2): {header_name} = {header_value}")
    
    # Pattern 3: "X-API-Key: weather-alpha-key" or "`X-API-Key` with value `weather-alpha-key`"
    header_pattern3 = r'[`"\']?([A-Za-z0-9\-]+)[`"\']?\s*[:\s]+\s*[`"\']?([A-Za-z0-9\-]+)[`"\']?'
    matches3 = re.findall(header_pattern3, instructions)
    
    for header_name, header_value in matches3:
        # Only extract if it looks like a header (contains "API", "Key", "Auth", etc.)
        if any(word in header_name.upper() for word in ['API', 'KEY', 'AUTH', 'TOKEN', 'X-']):
            if header_name not in headers:  # Don't override already found headers
                headers[header_name] = header_value
                print(f"🔑 Extracted header (pattern 3): {header_name} = {header_value}")
    
    return headers

# =============================================================================
# PRICE CLEANING HELPER
# =============================================================================
def clean_price(price: Any) -> Optional[float]:
    """
    Clean a single price value - COMPREHENSIVE version
    Returns float or None if invalid
    """
    if price is None:
        return None
    
    # If already a number
    if isinstance(price, (int, float)):
        # Check if it's a valid number (not NaN or infinity)
        try:
            if price != price or price == float('inf') or price == float('-inf'):  # NaN check
                return None
            return float(price)
        except:
            return None
    
    # If string, clean it
    if isinstance(price, str):
        # Remove whitespace
        cleaned = price.strip()
        
        # Check for invalid strings (case-insensitive)
        if not cleaned or cleaned.lower() in ['invalid', 'null', 'n/a', 'na', 'none', 'undefined', 'nan', '']:
            return None
        
        # Remove currency symbols and thousands separators
        cleaned = cleaned.replace('$', '').replace('€', '').replace('£', '').replace('₹', '')
        cleaned = cleaned.replace(',', '')  # Remove commas
        cleaned = cleaned.replace(' ', '')  # Remove spaces
        
        # Handle special cases
        if cleaned.lower() == 'free' or cleaned == '0':
            return 0.0
        
        # Try to parse as float
        try:
            value = float(cleaned)
            # Check if it's a valid number (not NaN or infinity)
            if value != value or value == float('inf') or value == float('-inf'):
                return None
            return value
        except ValueError:
            # Not a valid number
            return None
    
    return None

# =============================================================================
# API DATA FETCHING & PAGINATION
# =============================================================================
def fetch_api_data(url: str, headers: Dict[str, str] = None, max_pages: int = 100) -> List[Dict]:
    """
    Fetch data from API with pagination support and custom headers
    Continues until empty list is returned
    Increased max_pages to 100
    """
    all_data = []
    page = 1
    
    if headers is None:
        headers = {}
    
    print(f"🌐 Fetching API data from: {url}")
    if headers:
        print(f"🔑 Using headers: {headers}")
    
    while page <= max_pages:
        try:
            # Build paginated URL
            if 'page=' in url:
                paginated_url = re.sub(r'page=\d+', f'page={page}', url)
            elif '?' in url:
                paginated_url = f"{url}&page={page}"
            else:
                paginated_url = f"{url}?page={page}"
            
            if page % 10 == 1:  # Log every 10th page
                print(f"  📄 Fetching page {page}...")
            
            # Make request with headers
            response = requests.get(paginated_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Check if empty (end of pagination)
            if not data or (isinstance(data, list) and len(data) == 0):
                print(f"  ✅ Reached end at page {page} (empty response)")
                break
            
            # Handle different response formats
            if isinstance(data, list):
                all_data.extend(data)
            elif isinstance(data, dict):
                # Check if this is a single page response (not paginated)
                items = None
                for key in ['cities', 'data', 'weather', 'items', 'results']:
                    if key in data:
                        items = data[key]
                        break
                
                if items is None:
                    items = data.get('items', data.get('results', [data]))
                
                if isinstance(items, list):
                    all_data.extend(items)
                    
                    # If we got data in a non-paginated format, stop after first page
                    if page == 1 and any(key in data for key in ['cities', 'weather', 'data']):
                        print(f"  ✅ Single-page API response detected")
                        break
                else:
                    all_data.append(data)
                    break
            
            page += 1
            time.sleep(0.05)  # Reduced delay
            
        except requests.exceptions.HTTPError as e:
            print(f"  ⚠️ HTTP Error on page {page}: {e}")
            
            # If first page fails with auth error, try without pagination
            if page == 1 and (e.response.status_code == 403 or e.response.status_code == 401):
                print(f"  ℹ️ Trying base URL without pagination...")
                try:
                    response = requests.get(url, headers=headers, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    
                    if isinstance(data, list):
                        all_data.extend(data)
                        print(f"  ✓ Got {len(data)} items without pagination")
                    elif isinstance(data, dict):
                        # Extract array from dict
                        for key in ['cities', 'data', 'weather', 'items', 'results']:
                            if key in data and isinstance(data[key], list):
                                all_data.extend(data[key])
                                print(f"  ✓ Got {len(data[key])} items from '{key}' field")
                                break
                        else:
                            all_data.append(data)
                            print(f"  ✓ Got single object")
                    
                except Exception as e2:
                    print(f"  ❌ Also failed without pagination: {e2}")
            break
        except requests.exceptions.RequestException as e:
            print(f"  ⚠️ Request error on page {page}: {e}")
            break
        except Exception as e:
            print(f"  ⚠️ Parse error on page {page}: {e}")
            break
    
    print(f"✅ Total items fetched: {len(all_data)}")
    return all_data


def search_in_data(data: List[Dict], search_criteria: str) -> Optional[str]:
    """
    Search for specific data in fetched results
    Returns the value based on search criteria
    """
    print(f"🔍 Searching for: {search_criteria[:150]}...")
    
    # Pattern 1: "item with ID 99"
    id_match = re.search(r'id[:\s]+(\d+)', search_criteria.lower())
    if id_match:
        target_id = int(id_match.group(1))
        print(f"  Looking for ID: {target_id}")
        
        for item in data:
            if isinstance(item, dict):
                item_id = item.get('id') or item.get('ID') or item.get('_id')
                
                if item_id == target_id or str(item_id) == str(target_id):
                    name = item.get('name') or item.get('Name') or item.get('title') or item.get('value') or str(item)
                    print(f"  ✅ Found: ID {target_id} → {name}")
                    return name
    
    # Pattern 2: "highest temperature"
    if 'highest' in search_criteria.lower() and 'temperature' in search_criteria.lower():
        print(f"  Looking for highest temperature...")
        
        max_temp = float('-inf')
        max_city = None
        
        for item in data:
            if isinstance(item, dict):
                temp = item.get('temperature') or item.get('temp') or item.get('Temperature') or item.get('Temp')
                city = item.get('city') or item.get('name') or item.get('City') or item.get('location') or item.get('Name')
                
                if temp is not None and city:
                    try:
                        temp_value = float(temp)
                        if temp_value > max_temp:
                            max_temp = temp_value
                            max_city = city
                            print(f"  → New max: {city} at {temp}°")
                    except ValueError:
                        pass
        
        if max_city:
            print(f"  ✅ Highest temperature: {max_city} ({max_temp}°)")
            return max_city
    
    # Pattern 3: "lowest temperature"
    if 'lowest' in search_criteria.lower() and 'temperature' in search_criteria.lower():
        print(f"  Looking for lowest temperature...")
        
        min_temp = float('inf')
        min_city = None
        
        for item in data:
            if isinstance(item, dict):
                temp = item.get('temperature') or item.get('temp') or item.get('Temperature') or item.get('Temp')
                city = item.get('city') or item.get('name') or item.get('City') or item.get('location') or item.get('Name')
                
                if temp is not None and city:
                    try:
                        temp_value = float(temp)
                        if temp_value < min_temp:
                            min_temp = temp_value
                            min_city = city
                            print(f"  → New min: {city} at {temp}°")
                    except ValueError:
                        pass
        
        if min_city:
            print(f"  ✅ Lowest temperature: {min_city} ({min_temp}°)")
            return min_city
    
    # Pattern 4: "sum of prices" or "calculate sum" or "clean data"
    if ('sum' in search_criteria.lower() and 'price' in search_criteria.lower()) or \
       'clean' in search_criteria.lower() or \
       ('calculate' in search_criteria.lower() and 'sum' in search_criteria.lower()):
        print(f"  💰 Calculating sum of prices (NO deduplication - sum ALL items)...")
        
        total = 0.0
        valid_count = 0
        invalid_count = 0
        no_price_field = 0
        
        # Sample first few items for debugging
        print(f"  🔬 Sample data (first 3 items):")
        for i, item in enumerate(data[:3]):
            if isinstance(item, dict):
                price_val = item.get('price', 'NO_FIELD')
                print(f"     Item {i+1}: {item}")
        
        for item in data:
            if isinstance(item, dict):
                # Try different price field names
                price = item.get('price')
                if price is None:
                    price = item.get('Price') or item.get('cost') or item.get('value') or item.get('amount')
                
                if price is None:
                    no_price_field += 1
                    continue
                
                # Clean the price
                cleaned_price = clean_price(price)
                
                if cleaned_price is not None:
                    total += cleaned_price
                    valid_count += 1
                else:
                    invalid_count += 1
        
        print(f"  ✓ Valid prices: {valid_count}")
        print(f"  ✓ Invalid/skipped: {invalid_count}")
        print(f"  ✓ No price field: {no_price_field}")
        print(f"  ✓ Total items checked: {len(data)}")
        print(f"  ✓ Verification: {valid_count} + {invalid_count} + {no_price_field} = {valid_count + invalid_count + no_price_field}")
        print(f"  ✓ Total sum: {total}")
        
        # Return as integer if it's a whole number
        if total == int(total):
            return str(int(total))
        return str(round(total, 2))
    
    return None

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
            for elem in soup.find_all(['div', 'span', 'p']):
                elem_classes = elem.get('class', [])
                if isinstance(elem_classes, list):
                    if any(word in ' '.join(elem_classes).lower() for word in ['hidden', 'secret', 'password', 'key']):
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
  "data_sources": ["any data/hints/URLs provided as array"],
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
# SOLVE QUESTION (Smart extraction + API + LLM)
# =============================================================================
def solve_question(question: str, data_sources: List[str], instructions: str, html: str) -> str:
    """
    Solve quiz question intelligently:
    1. Extract required headers from instructions
    2. Check if API call needed → fetch with headers and search data
    3. Try smart HTML extraction if applicable
    4. Use LLM with full context if extraction fails
    """
    
    # Combine instructions
    all_instructions = ' '.join(data_sources) + ' ' + instructions
    
    # Extract headers from instructions
    headers = extract_headers_from_instructions(all_instructions)
    
    # Check if this requires API fetching
    api_urls = []
    for source in data_sources:
        if isinstance(source, str) and source.startswith('http') and '/api/' in source:
            api_urls.append(source)
    
    context_data = ""
    
    if api_urls:
        print(f"🌐 Detected API data source: {api_urls}")
        
        # Fetch API data with headers
        all_api_data = []
        for api_url in api_urls:
            data = fetch_api_data(api_url, headers=headers)
            all_api_data.extend(data)
        
        print(f"📊 Total API data items: {len(all_api_data)}")
        
        # Search for answer in data
        if all_api_data:
            answer = search_in_data(all_api_data, question + ' ' + all_instructions)
            
            if answer:
                print(f"✅ Found answer in API data: {answer}")
                return answer
            else:
                print(f"⚠️ Could not find answer in API data, trying LLM...")
                context_data = f"\n\nAPI Data (sample):\n{json.dumps(all_api_data[:5], indent=2)}\n"
                context_data += f"\nTotal items: {len(all_api_data)}\n"
        else:
            print(f"⚠️ No API data fetched, trying LLM...")
    
    # Try smart HTML extraction
    if html and any(word in all_instructions.lower() for word in 
                    ['html', 'div', 'class', 'id', 'hidden', 'revers', 'element']):
        print("🔍 Attempting smart HTML extraction...")
        extracted = extract_from_html(html, all_instructions)
        if extracted:
            print(f"✅ Smart extraction: {extracted}")
            return extracted
    
    # Fallback to LLM
    print("🤖 Using LLM to solve...")
    
    context = f"""
Question: {question}

Instructions: {all_instructions}

Data sources: {json.dumps(data_sources)}

{context_data}

HTML content (relevant parts):
{html[:6000] if html else "N/A"}
"""
    
    prompt = f"""
{context}

Solve this quiz question following these rules:

1. READ ALL INSTRUCTIONS CAREFULLY
2. If API data is provided above, ANALYZE it for the answer
3. If data is in HTML, EXTRACT it
4. Apply transformations as needed
5. Return ONLY the final answer (no explanations)

Important:
- Maximum 100 characters
- Direct answer only
- Just the value requested (name, number, city, sum, etc.)

Answer:"""
    
    try:
        response = llm.generate_content(prompt)
        answer = response.text.strip()
        
        lines = [l.strip() for l in answer.split('\n') if l.strip()]
        answer = lines[0] if lines else answer
        
        for prefix in ['Answer:', 'The answer is:', 'Result:', 'answer:', 'A:', 'The name is:', 'Name:', 'City:', 'The city is:', 'Sum:', 'Total:']:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
        
        answer = answer.strip('"\'`')
        
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
