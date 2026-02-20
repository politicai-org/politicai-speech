import requests
import json
import sys

# Update URL to correct endpoint
url = "http://orbe-api:8003/api/v1/chat"
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "orbe-dev-key-2024"
}
data = {
    "candidate_id": "harvey-colchado",
    "session_id": "test-session",
    "message": "Hola, ¿cómo estás?",
    "stream": True,
    "include_audio": True
}

try:
    print(f"Sending request to {url}...")
    print(f"Payload: {json.dumps(data)}")
    response = requests.post(url, json=data, headers=headers, stream=True)
    print(f"Response status: {response.status_code}")
    
    if response.status_code != 200:
        print(f"Error content: {response.text}")
        sys.exit(1)

    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            # Only print event types to avoid flooding with audio data
            if decoded_line.startswith("event:"):
                print(decoded_line)
            if "error" in decoded_line.lower() and "event: error" in decoded_line:
                print("ERROR DETECTED IN STREAM!")
                print(decoded_line)
            
except Exception as e:
    print(f"Request failed: {e}")
