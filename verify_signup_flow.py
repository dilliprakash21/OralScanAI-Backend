import urllib.request
import json
import random
import os

BASE_URL = "http://127.0.0.1:8000"

def test_forgot_password(email):
    print(f"\n--- Testing Forgot Password for {email} ---")
    
    # 1. Request Reset
    req = urllib.request.Request(
        f"{BASE_URL}/auth/forgot-password",
        data=json.dumps({"email": email}).encode(),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as res:
            print(f"Step 1 (Forgot) Result: {json.loads(res.read())}")
    except Exception as e:
        print(f"Step 1 Failed: {e}")
        return

    # 2. Get OTP from DB
    import mysql.connector
    cnx = mysql.connector.connect(host=os.getenv('DB_HOST','localhost'), user=os.getenv('DB_USER','root'), password=os.getenv('DB_PASS','1234'), database=os.getenv('DB_NAME','oralscan_db'))
    cur = cnx.cursor()
    cur.execute("SELECT code FROM otp_codes WHERE email=%s AND type='forgot_password' ORDER BY created_at DESC LIMIT 1", (email,))
    row = cur.fetchone()
    if not row:
        print("Step 2 Failed: Reset OTP not found in DB")
        return
    otp_code = row[0]
    cnx.close()
    print(f"Step 2: Found Reset OTP {otp_code}")

    # 3. Reset Password
    reset_payload = {"email": email, "code": otp_code, "password": "new_password_123"}
    req2 = urllib.request.Request(
        f"{BASE_URL}/auth/reset-password",
        data=json.dumps(reset_payload).encode(),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req2) as res:
            print(f"Step 3 (Reset) Result: {json.loads(res.read())}")
            print("✅ Forgot Password flow verified!")
    except Exception as e:
        print(f"Step 3 Failed: {e}")
        if hasattr(e, 'read'): print(e.read().decode())

def test_signup_flow(role="health_worker"):
    email = f"test_{role}_{random.randint(1000, 9999)}@example.com"
    print(f"\n--- Testing Signup for {email} (Role: {role}) ---")
    
    phone = f"9{random.randint(100000000, 999999999)}"
    # 1. Signup
    payload = {
        "name": f"Test {role}",
        "email": email,
        "password": "password123",
        "phone": phone,
        "role": role,
        "license_no": "DENT123" if role == "dentist" else None,
        "clinic": "Test Clinic",
        "location": "Test City"
    }
    
    req = urllib.request.Request(
        f"{BASE_URL}/auth/signup",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req) as res:
            data = json.loads(res.read())
            print(f"Step 1 (Signup) Result: {data}")
    except urllib.error.HTTPError as e:
        print(f"Step 1 Failed with {e.code}")
        print(f"Error Body: {e.read().decode()}")
        return
    except Exception as e:
        print(f"Step 1 Failed: {e}")
        return

    # 2. Check main users table (should be empty for this email)
    # We can't check directly here from urllib, but we can try to signup again 
    # and it should NOT say "email already registered" yet if it's working (REPLACE logic)
    
    # 2. Get OTP from DB
    import mysql.connector
    cnx = mysql.connector.connect(host=os.getenv('DB_HOST','localhost'), user=os.getenv('DB_USER','root'), password=os.getenv('DB_PASS','1234'), database=os.getenv('DB_NAME','oralscan_db'))
    cur = cnx.cursor()
    cur.execute("SELECT otp_code FROM pending_users WHERE email=%s", (email,))
    row = cur.fetchone()
    if not row:
        print("Step 2 Failed: OTP not found in DB")
        return
    otp_code = row[0]
    cnx.close()
    print(f"Step 2: Found OTP {otp_code}")

    # 3. Verify OTP
    verify_payload = {"email": email, "code": otp_code}
    req2 = urllib.request.Request(
        f"{BASE_URL}/auth/verify-signup",
        data=json.dumps(verify_payload).encode(),
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req2) as res:
            data = json.loads(res.read())
            print(f"Step 3 (Verify) Result: {data}")
            if data.get("ok"):
                print("✅ Signup flow 100% verified!")
                test_forgot_password(email)
    except Exception as e:
        print(f"Step 3 Failed: {e}")
        if hasattr(e, 'read'): print(e.read().decode())
        return

if __name__ == "__main__":
    test_signup_flow("health_worker")
    test_signup_flow("dentist")
