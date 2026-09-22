-- Update the GPT-6 Sol glossary benchmark table to compare
-- GPT-6 Sol vs GPT-5.6 Sol vs Claude Opus 5.5.
--
-- - Drops the Category column (Benchmark becomes the first column).
-- - Replaces the GPT-6 Luna column with GPT-5.6 Sol values where verified,
--   using "-" where no verified score exists.
-- - Replaces "Not reported in ..." cells with "-".
--
-- Sources already recorded on the row:
-- - GPT-6 Sol figures: OpenAI GPT-6 Sol and Luna launch post, max-effort
--   chart readings (https://openai.com/index/introducing-gpt-6-sol-and-luna/)
-- - GPT-5.6 Sol figures: OpenAI GPT-5.6 launch suite, July 9, 2026
--   (https://openai.com/index/gpt-5-6/) — DeepSWE v1.1 72.7% and
--   OSWorld 2.0 62.6%; no verified GPT-5.6 Sol scores for AutomationBench
--   1.0.6, Agents' Last Exam V1, or FrontierCode 1.1 Main, shown as "-".
-- - Opus 5.5 figures: Anthropic Opus 5.5 launch table
--   (https://www.anthropic.com/claude-opus-5-5).

-- 1. Replace the benchmark table (drop Category, swap Luna for GPT-5.6 Sol,
--    "-" for missing cells).
WITH old_new(old_text, new_text) AS (
  SELECT
    $old$| Category | Benchmark | GPT-6 Sol | GPT-6 Luna | Claude Opus 5.5 |
| --- | --- | ---: | ---: | ---: |
| Professional work | AutomationBench 1.0.6 | 33.2% (xhigh) | 20.7% (max) | 40.0% |
| Professional work | Agents' Last Exam V1 | 56.6% (max) | 50.9% (max) | Not reported in Opus 5.5 launch table |
| Coding | FrontierCode 1.1 Main | 49.3% (max) | 42.4% (max) | 54.4% |
| Coding | DeepSWE v1.1 | 68.8% (max) | 66.6% (max) | Not reported in Opus 5.5 launch table |
| Computer use | OSWorld 2.0 | 64.4% (max) | 52.7% (max) | 81.8% |$old$::text,
    $new$| Benchmark | GPT-6 Sol | GPT-5.6 Sol | Claude Opus 5.5 |
| --- | ---: | ---: | ---: |
| AutomationBench 1.0.6 | 33.2% (xhigh) | - | 40.0% |
| Agents' Last Exam V1 | 56.6% (max) | - | - |
| FrontierCode 1.1 Main | 49.3% (max) | - | 54.4% |
| DeepSWE v1.1 | 68.8% (max) | 72.7% | - |
| OSWorld 2.0 | 64.4% (max) | 62.6% | 81.8% |$new$::text
)
UPDATE content_items
SET
  body = replace(content_items.body, old_new.old_text, old_new.new_text),
  blocks = jsonb_build_array(
    jsonb_build_object(
      'id', 'markdown-1',
      'type', 'markdown',
      'content', replace(content_items.body, old_new.old_text, old_new.new_text)
    )
  ),
  updated_at = NOW()
FROM old_new
WHERE kind = 'glossary'
  AND slug = 'gpt-6-sol'
  AND parent_slug = '';

-- 2. Update the scores sentence to match the new columns.
WITH old_new(old_text, new_text) AS (
  SELECT
    $old$Scores for Sol and Luna are OpenAI-reported chart readings at stated effort levels; Opus 5.5 figures are Anthropic-reported.$old$::text,
    $new$Scores for GPT-6 Sol are OpenAI-reported chart readings at stated effort levels; GPT-5.6 Sol figures are from OpenAI's July 9 GPT-5.6 launch suite; Opus 5.5 figures are Anthropic-reported.$new$::text
)
UPDATE content_items
SET
  body = replace(content_items.body, old_new.old_text, old_new.new_text),
  blocks = jsonb_build_array(
    jsonb_build_object(
      'id', 'markdown-1',
      'type', 'markdown',
      'content', replace(content_items.body, old_new.old_text, old_new.new_text)
    )
  ),
  updated_at = NOW()
FROM old_new
WHERE kind = 'glossary'
  AND slug = 'gpt-6-sol'
  AND parent_slug = '';

-- 3. Keep SEO metadata consistent with the new comparison.
UPDATE content_items
SET
  metadata = jsonb_set(
    jsonb_set(
      COALESCE(metadata, '{}'::jsonb),
      '{seoDescription}',
      '"GPT-6 Sol explained: September 2026 launch pricing ($2/$10), benchmark scorecard vs GPT-5.6 Sol and Opus 5.5, caching, effort levels, and production use cases."'
    ),
    '{seoKeywords}',
    '["GPT-6 Sol", "GPT-6 Sol benchmarks", "GPT-6 Sol pricing", "GPT-6 Sol vs Opus 5.5", "GPT-6 Sol vs GPT-5.6 Sol", "gpt-6-sol", "OpenAI GPT-6 Sol"]'
  ),
  updated_at = NOW()
WHERE kind = 'glossary'
  AND slug = 'gpt-6-sol'
  AND parent_slug = '';
