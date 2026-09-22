-- Fix meta description length for the Jev article.
-- The 049 seed summary was 182 chars (over the ~155-160 truncation limit),
-- so search/GEO tools flag it as too long. Shorten to 151 chars.
UPDATE content_items
SET
  summary = 'Jev claims zero hallucination, 200x speed, and 400x lower cost. Checking the founder''s credit, the self-graded benchmark, and independent test results.',
  updated_at = NOW()
WHERE kind = 'article'
  AND slug = 'jev-zero-hallucination-fine-print-disagrees'
  AND parent_slug = '';
