-- Fix meta description length for the latest comparison page.
-- The 040 seed summary was 182 chars (over the ~155-160 truncation limit),
-- so search/GEO tools flag it as too long. Shorten to 154 chars.
UPDATE content_items
SET
  summary = 'DeepSeek V4.1 Flash vs V4 Flash: what changed on Sept 10, 2026 — 8B/16B CED architecture, benchmark gains, lower pricing, smaller KV cache, native vision.',
  updated_at = NOW()
WHERE kind = 'comparison'
  AND slug = 'deepseek-v4-flash-vs-deepseek-v4-1-flash'
  AND parent_slug = '';
