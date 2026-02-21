-- Migration: Add full_content column to file_hash_entries for file retrieval
-- Date: 2026-02-21
-- Description: Enables get_file_by_hash retrieval tool for cached files

ALTER TABLE file_hash_entries 
ADD COLUMN IF NOT EXISTS full_content TEXT NULL;

-- Optional: Add index if retrieval by hash becomes frequent
-- CREATE INDEX IF NOT EXISTS idx_filehash_content_notnull 
-- ON file_hash_entries(content_hash) 
-- WHERE full_content IS NOT NULL;
