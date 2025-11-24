"""
Test script to verify server is running and can handle requests
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

# Get configuration from environment
TEST_SERVER_URL = os.getenv("TEST_SERVER_URL", "http://localhost:8000")
student_email = os.getenv("STUDENT_EMAIL")
student_secret = os.getenv("STUDENT_SECRET")

# Validate environment variables
if not student_email or not student_secret:
    print("❌ Missing STUDENT_EMAIL or STUDENT_SECRET in .env file")
    exit(1)

print("=" * 70)
print("TDS Quiz Solver - Server Test")
print("=" * 70)
print(f"Server URL: {TEST_SERVER_URL}")
print(f"Student Email: {student_email}")
print()

# Test 1: Health Check
print("Test 1: Health Check (GET /)")
print("-" * 70)
try:
    response = requests.get(TEST_SERVER_URL, timeout=5)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print(f"✅ Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"❌ Unexpected status: {response.text}")
except Exception as e:
    print(f"❌ Health check failed: {e}")
    print("\nMake sure server is running: uvicorn main:app --reload")
    exit(1)

print()

# Test 2: Quiz Submission
print("Test 2: Quiz Submission (POST /)")
print("-" * 70)

payload = {
    "email": student_email,
    "secret": student_secret,
    "url": "https://tds-llm-analysis.s-anand.net/demo"
}

print(f"Payload:")
print(json.dumps(payload, indent=2))
print()

try:
    print("Sending POST request...")
    response = requests.post(
        TEST_SERVER_URL + "/",  # Ensure trailing slash
        json=payload,
        timeout=60  # Quiz solving can take time
    )

    print(f"Response Status Code: {response.status_code}")
    print()

    if response.status_code == 200:
        print("✅ SUCCESS: Server processed the request")
        try:
            result = response.json()
            print("Response:")
            print(json.dumps(result, indent=2))

            # Check if quiz was solved correctly
            if result.get("correct") == True:
                print("\n🎉 Quiz solved correctly!")
            else:
                print("\n⚠️  Quiz answer was incorrect")
                print(f"Reason: {result.get('reason', 'Unknown')}")

        except json.JSONDecodeError:
            print("Response (text):")
            print(response.text)

    elif response.status_code == 403:
        print("❌ FAILED: Invalid credentials (403 Forbidden)")
        print("Check STUDENT_EMAIL and STUDENT_SECRET in .env")

    elif response.status_code == 500:
        print("❌ FAILED: Server error (500 Internal Server Error)")
        print("Check server logs for detailed error message")
        print("\nResponse:")
        print(response.text[:500])

    else:
        print(f"❌ FAILED: Unexpected status code {response.status_code}")
        print("\nResponse:")
        print(response.text[:500])

except requests.exceptions.Timeout:
    print("❌ Request timed out after 60 seconds")
    print("Quiz solving is taking too long - check server logs")

except requests.exceptions.ConnectionError:
    print("❌ Cannot connect to server")
    print(f"Make sure server is running at {TEST_SERVER_URL}")
    print("\nStart server with: uvicorn main:app --reload")

except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback

    traceback.print_exc()

print()
print("=" * 70)
print("Test Complete")
print("=" * 70)



