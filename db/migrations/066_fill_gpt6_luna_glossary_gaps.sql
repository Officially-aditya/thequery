-- Fill the missing GPT-5.6 Luna cells in the GPT-6 Luna glossary benchmark table.
--
-- - AutomationBench 1.0.6 GPT-5.6 Luna 14.9%: OpenAI GPT-5.6 launch suite
--   Tool use table, July 9, 2026 (https://openai.com/index/gpt-5-6/).
-- - Agents' Last Exam V1 GPT-5.6 Luna 50.3%: OpenAI GPT-5.6 launch suite
--   Professional table, July 9, 2026 (https://openai.com/index/gpt-5-6/).
-- - FrontierCode 1.1 Main GPT-5.6 Luna 39.8%: official FrontierCode
--   leaderboard weighted score (Main), Status Official, Date Jul 9, 2026
--   (https://evals.report/benchmarks/frontiercode?tab=scores, source
--   cognition.com). Same source family as the GPT-5.6 Sol 47.5% and
--   Terra 41.3% Main scores.
-- - Unlike the Sol table, no Zapier-run substitution is needed here: there
--   is no Opus column in this table, so the OpenAI-run AutomationBench
--   figure (14.9%) keeps both Luna columns on the same harness as the
--   GPT-6 Luna 20.7% chart reading.

-- 1. Fill the GPT-5.6 Luna cells.
WITH old_new(old_text, new_text) AS (
  SELECT
    $old$| AutomationBench 1.0.6 | 20.7% | - |
| Agents' Last Exam V1 | 50.9% | - |
| FrontierCode 1.1 Main | 42.4% | - |
| DeepSWE v1.1 | 66.6% | 67.2% |
| OSWorld 2.0 | 52.7% | 45.6% |$old$::text,
    $new$| AutomationBench 1.0.6 | 20.7% | 14.9% |
| Agents' Last Exam V1 | 50.9% | 50.3% |
| FrontierCode 1.1 Main | 42.4% | 39.8% |
| DeepSWE v1.1 | 66.6% | 67.2% |
| OSWorld 2.0 | 52.7% | 45.6% |$new$::text
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

-- 2. Document the FrontierCode source in the scores sentence.
WITH old_new(old_text, new_text) AS (
  SELECT
    $old$Scores for GPT-6 Luna are OpenAI-reported chart readings at max effort; GPT-5.6 Luna figures are from OpenAI's July 9 GPT-5.6 launch suite.$old$::text,
    $new$Scores for GPT-6 Luna are OpenAI-reported chart readings at max effort. GPT-5.6 Luna figures are from OpenAI's July 9 GPT-5.6 launch suite, except FrontierCode 1.1 Main (39.8%), which is the official FrontierCode leaderboard score.$new$::text
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
