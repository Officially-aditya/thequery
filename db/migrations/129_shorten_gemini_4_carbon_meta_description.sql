-- Keep the model card's search description within Bing's recommended length.
UPDATE content_items
SET
  metadata = jsonb_set(
    COALESCE(metadata, '{}'::jsonb),
    '{seoDescription}',
    to_jsonb('Gemini 4 Carbon model card: benchmarks, $2/$10 introductory input/output prices per million tokens, cyber safeguards and limited Fairwind access.'::text),
    true
  ),
  updated_at = TIMESTAMPTZ '2026-10-30 00:00:00+00'
WHERE kind = 'glossary' AND slug = 'gemini-4-carbon' AND parent_slug = '';
