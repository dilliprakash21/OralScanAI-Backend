import urllib.request
import urllib.error
import urllib.parse
import json

BASE = "http://localhost:8000"

def request(url, method="GET", data=None, headers=None):
    if headers is None: headers = {}
    req_data = None
    if data:
        if isinstance(data, dict):
            req_data = json.dumps(data).encode("utf-8")
            headers["Content-Type"] = "application/json"
        else:
            req_data = data
            
    req = urllib.request.Request(url, data=req_data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")
    except Exception as e:
        return 0, str(e)

def run_tests():
    print("Testing Backend APIs...")
    
    status, body = request(f"{BASE}/health")
    print(f"GET /health -> {status}")
    
    status, body = request(f"{BASE}/auth/signup", method="POST", data={
        "email": "test2@example.com",
        "password": "password123",
        "name": "Test User",
        "role": "health_worker"
    })
    print(f"POST /auth/signup -> {status} {body[:50]}")
    
    status, body = request(f"{BASE}/auth/login", method="POST", data={
        "email": "test2@example.com",
        "password": "password123"
    })
    print(f"POST /auth/login -> {status}")
    
    token = None
    if status == 200:
        try: token = json.loads(body).get("token")
        except: pass
        
    if token:
        status, body = request(f"{BASE}/screenings/stats", headers={"Authorization": f"Bearer {token}"})
        print(f"GET /screenings/stats -> {status}")
    else:
        print("GET /screenings/stats -> FAIL (No Token)")
        
    # multipart form data for analyze
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    payload = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"file\"; filename=\"test.jpg\"\r\n"
        f"Content-Type: image/jpeg\r\n\r\n"
        f"fake image content\r\n"
        f"--{boundary}--\r\n"
    ).encode('utf-8')
    
    status, body = request(f"{BASE}/analyze", method="POST", data=payload, headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}"
    })
    print(f"POST /analyze -> {status} {body[:150]}")

if __name__ == "__main__":
    run_tests()
