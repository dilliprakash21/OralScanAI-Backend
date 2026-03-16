-- Create database and tables for OralScan (MySQL)
CREATE DATABASE IF NOT EXISTS oralscan_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE oralscan_db;

CREATE TABLE IF NOT EXISTS users (
  id CHAR(36) PRIMARY KEY,
  email VARCHAR(255) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  phone VARCHAR(50),
  full_name VARCHAR(255) NOT NULL,
  role VARCHAR(50) NOT NULL,
  email_verified BOOLEAN DEFAULT FALSE,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX (email)
);

CREATE TABLE IF NOT EXISTS profiles (
  id CHAR(36) PRIMARY KEY,
  user_id CHAR(36) NOT NULL UNIQUE,
  name VARCHAR(255),
  email VARCHAR(255),
  phone VARCHAR(50),
  role VARCHAR(50),
  license_no VARCHAR(100),
  clinic VARCHAR(255),
  location VARCHAR(255),
  hospital VARCHAR(255),
  avatar_url TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX (user_id),
  CONSTRAINT fk_profiles_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS screenings (
  id CHAR(36) PRIMARY KEY,
  user_id CHAR(36) NOT NULL,
  patient_id VARCHAR(100) NOT NULL,
  patient_name VARCHAR(255),
  phone VARCHAR(50),
  age INT,
  dob VARCHAR(50),
  gender VARCHAR(50),
  location VARCHAR(255),
  mode VARCHAR(50),
  consent_given BOOLEAN,
  plaque_index DOUBLE,
  gingival_index DOUBLE,
  ai_confidence DOUBLE,
  risk_level VARCHAR(50),
  overridden BOOLEAN,
  override_reason TEXT,
  notes TEXT,
  image_url TEXT,
  heatmap_url TEXT,
  referral_clinic VARCHAR(255),
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX (user_id),
  INDEX (created_at),
  CONSTRAINT fk_screenings_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS notifications (
  id CHAR(36) PRIMARY KEY,
  user_id CHAR(36) NOT NULL,
  text TEXT NOT NULL,
  type VARCHAR(50),
  `read` BOOLEAN DEFAULT FALSE,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX (user_id),
  CONSTRAINT fk_notifications_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS refresh_tokens (
  id CHAR(36) PRIMARY KEY,
  user_id CHAR(36) NOT NULL,
  token VARCHAR(255) NOT NULL UNIQUE,
  expires_at DATETIME NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX (user_id),
  CONSTRAINT fk_refresh_tokens_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS otp_codes (
  id CHAR(36) PRIMARY KEY,
  email VARCHAR(255) NOT NULL,
  code VARCHAR(10) NOT NULL,
  expires_at DATETIME NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX (email)
);

-- ======================================================
-- ADMIN SETUP
-- ======================================================
-- To promote an existing user to Admin, run these:
-- UPDATE users SET role = 'admin' WHERE email = 'oralscanai@gmail.com';
-- UPDATE profiles SET role = 'admin' WHERE email = 'oralscanai@gmail.com';
