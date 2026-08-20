ALTER TABLE content_items
  ADD COLUMN IF NOT EXISTS cover_image_url TEXT,
  ADD COLUMN IF NOT EXISTS cover_image_alt TEXT;
