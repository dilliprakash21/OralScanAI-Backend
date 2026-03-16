import os
from dotenv import load_dotenv
import mysql.connector

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "1234")
DB_NAME = os.getenv("DB_NAME", "oralscan_db")

def migrate():
    try:
        cnx = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME
        )
        cur = cnx.cursor()
        
        print("Creating pending_users table...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pending_users (
                email VARCHAR(255) PRIMARY KEY,
                payload TEXT NOT NULL,
                otp_code VARCHAR(10) NOT NULL,
                expires_at DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        print("Updating otp_codes table to support type...")
        cur.execute("DESCRIBE otp_codes")
        cols = [row[0] for row in cur.fetchall()]
        if 'type' not in cols:
            cur.execute("ALTER TABLE otp_codes ADD COLUMN type VARCHAR(50) DEFAULT 'signup'")
            
        cnx.commit()
        print("✅ Migration successful.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'cnx' in locals():
            cnx.close()

if __name__ == "__main__":
    migrate()
