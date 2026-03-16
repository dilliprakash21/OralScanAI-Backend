-- ======================================================
-- OralScan AI - Admin Role Setup
-- ======================================================

-- 1. Ensure the role column can handle the 'admin' value (already should be VARCHAR(50))
-- ALTER TABLE users MODIFY role VARCHAR(50) NOT NULL;
-- ALTER TABLE profiles MODIFY role VARCHAR(50);

-- 2. How to upgrade an existing user to Admin manually:
-- Replace 'user@example.com' with the actual email you want to promote.

UPDATE users SET role = 'admin' WHERE email = 'oralscanai@gmail.com';
UPDATE profiles SET role = 'admin' WHERE email = 'oralscanai@gmail.com';

-- 3. Verify the changes:
SELECT id, email, role FROM users WHERE role = 'admin';
