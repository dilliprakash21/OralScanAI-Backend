import os
from dotenv import load_dotenv
import mysql.connector

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "1234")
DB_NAME = os.getenv("DB_NAME", "oralscan_db")

def check_structure():
    try:
        cnx = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME
        )
        cur = cnx.cursor()
        cur.execute("SHOW CREATE TABLE otp_codes")
        print(cur.fetchone()[1])
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'cnx' in locals():
            cnx.close()

if __name__ == "__main__":
    check_structure()
