import os
from dotenv import load_dotenv
import mysql.connector

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "1234")
DB_NAME = os.getenv("DB_NAME", "oralscan_db")

def fix_schema():
    try:
        cnx = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME
        )
        cur = cnx.cursor()
        
        print("Dropping old otp_codes table if exists...")
        cur.execute("DROP TABLE IF EXISTS otp_codes")
        
        print("Creating new otp_codes table...")
        cur.execute("""
            CREATE TABLE otp_codes (
                id CHAR(36) PRIMARY KEY,
                email VARCHAR(255) NOT NULL,
                code VARCHAR(10) NOT NULL,
                expires_at DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                INDEX (email)
            )
        """)

        cnx.commit()
        print("✅ Database schema updated successfully (Table recreated).")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'cnx' in locals():
            cnx.close()

if __name__ == "__main__":
    fix_schema()
