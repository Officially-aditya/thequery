-- Allow comparison pages alongside existing content kinds.
ALTER TABLE content_items DROP CONSTRAINT IF EXISTS content_items_kind_check;
ALTER TABLE content_items
  ADD CONSTRAINT content_items_kind_check
  CHECK (kind IN ('article', 'guide', 'glossary', 'book', 'chapter', 'comparison'));
