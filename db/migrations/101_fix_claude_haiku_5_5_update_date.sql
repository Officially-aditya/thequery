-- Keep the supplied editorial date when publication happens before UTC midnight.
UPDATE content_items
SET updated_at = TIMESTAMPTZ '2026-10-08 00:00:00+00'
WHERE kind = 'glossary'
  AND slug = 'claude-haiku-5-5'
  AND parent_slug = '';
