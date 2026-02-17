import requests
import time
import sys

def test_chat_api():
    base_url = "http://127.0.0.1:8000"
    print(f"Testing Chat API at {base_url}...")
    
    # 1. Health Check
    try:
        r = requests.get(base_url)
        if r.status_code in [200, 307]: # Root might redirect to /app/
            print("✅ Server is online.")
    except Exception as e:
        print(f"❌ Server unreachable: {e}")
        sys.exit(1)

    # 2. Test Initial Narrative Submission (Story)
    payload = {
        "user_message": "I was walking in the dark forest when I saw a tall creature with glowing eyes. It let out a bone-chilling scream."
    }
    try:
        print("Submitting initial narrative...")
        r = requests.post(f"{base_url}/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        
        assert "response" in data
        assert "session_id" in data
        assert data["is_initial"] is True
        print("✅ Initial analysis successful.")
        
        session_id = data["session_id"]
        
        # 3. Test Follow-up Question
        follow_up = {
            "session_id": session_id,
            "user_message": "What specific signals did you detect in this story?"
        }
        print("Submitting follow-up question...")
        r = requests.post(f"{base_url}/chat", json=follow_up)
        r.raise_for_status()
        data = r.json()
        
        assert "response" in data
        assert data["turn_count"] == 1
        print("✅ Follow-up investigation successful.")

    except Exception as e:
        print(f"❌ Chat flow failed: {e}")
        sys.exit(1)

    print("\nAll systems nominal. The investigative pipeline is secure.")

if __name__ == "__main__":
    test_chat_api()
