-- Remove the duplicated 'gemini-4-argon' entry from the Gemini 4 Argon
-- seoKeywords array written by 096, preserving first-occurrence order.

UPDATE content_items
SET
  metadata = jsonb_set(
    metadata,
    '{seoKeywords}',
    (
      SELECT jsonb_agg(value ORDER BY first_seen)
      FROM (
        SELECT value, min(ord) AS first_seen
        FROM jsonb_array_elements_text(COALESCE(metadata->'seoKeywords', '[]'::jsonb))
        WITH ORDINALITY AS t(value, ord)
        GROUP BY value
      ) AS deduped
    ),
    true
  ),
  updated_at = NOW()
WHERE kind = 'glossary'
  AND slug = 'gemini-4-argon'
  AND parent_slug = '';
