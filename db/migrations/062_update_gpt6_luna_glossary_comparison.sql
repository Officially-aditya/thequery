-- Update the GPT-6 Luna glossary benchmark table to compare
-- GPT-6 Luna vs GPT-5.6 Luna.
--
-- - Drops the Category column (Benchmark becomes the first column).
-- - Replaces the GPT-6 Sol and Claude Opus 5.5 columns with GPT-5.6 Luna
--   values where verified, using "-" where no verified score exists.
-- - Replaces "Not reported in ..." cells with "-".
--
-- Sources already recorded on the row:
-- - GPT-6 Luna figures: OpenAI GPT-6 Sol and Luna launch post, max-effort
--   chart readings (https://openai.com/index/introducing-gpt-6-sol-and-luna/)
-- - GPT-5.6 Luna figures: OpenAI GPT-5.6 launch suite, July 9, 2026
--   (https://openai.com/index/gpt-5-6/) — DeepSWE v1.1 67.2% and
--   OSWorld 2.0 45.6%; no verified GPT-5.6 Luna scores for AutomationBench
--   1.0.6, Agents' Last Exam V1, or FrontierCode 1.1 Main, shown as "-".

-- 1. Replace the benchmark table (drop Category, swap to Luna vs 5.6 Luna,
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
    $new$| Benchmark | GPT-6 Luna | GPT-5.6 Luna |
| --- | ---: | ---: |
| AutomationBench 1.0.6 | 20.7% (max) | - |
| Agents' Last Exam V1 | 50.9% (max) | - |
| FrontierCode 1.1 Main | 42.4% (max) | - |
| DeepSWE v1.1 | 66.6% (max) | 67.2% |
| OSWorld 2.0 | 52.7% (max) | 45.6% |$new$::text
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
  AND slug = 'gpt-6-luna'
  AND parent_slug = '';

-- 2. Update the scores sentence to match the new columns.
WITH old_new(old_text, new_text) AS (
  SELECT
    $old$Scores for Sol and Luna are OpenAI-reported chart readings at stated effort levels; Opus 5.5 figures are Anthropic-reported. Cross-vendor cells reflect different runs and subsets, so they are not strict head-to-head wins — see the [GPT-6 Sol vs Claude Opus 5.5 comparison](/comparisons/gpt-6-sol-vs-opus-5-5) for the full scorecard with caveats.$old$::text,
    $new$Scores for GPT-6 Luna are OpenAI-reported chart readings at stated effort levels; GPT-5.6 Luna figures are from OpenAI's July 9 GPT-5.6 launch suite.$new$::text
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
  AND slug = 'gpt-6-luna'
  AND parent_slug = '';

-- 3. Keep SEO metadata consistent with the new comparison.
UPDATE content_items
SET
  metadata = jsonb_set(
    jsonb_set(
      COALESCE(metadata, '{}'::jsonb),
      '{seoDescription}',
      '"GPT-6 Luna explained: September 2026 launch pricing ($0.10/$0.50), benchmark scorecard vs GPT-5.6 Luna, caching, and high-volume use cases."'
    ),
    '{seoKeywords}',
    '["GPT-6 Luna", "GPT-6 Luna benchmarks", "GPT-6 Luna pricing", "GPT-6 Luna vs GPT-5.6 Luna", "gpt-6-luna", "OpenAI GPT-6 Luna"]'
  ),
  updated_at = NOW()
WHERE kind = 'glossary'
  AND slug = 'gpt-6-luna'
  AND parent_slug = '';
