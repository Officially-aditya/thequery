-- Fill the missing GPT-5.6 Sol cells in the GPT-6 Sol glossary benchmark table.
--
-- - AutomationBench 1.0.6 GPT-5.6 Sol 28.8%: Zapier-leaderboard run reported
--   in Anthropic's Opus 5.5 launch table
--   (https://www.anthropic.com/claude-opus-5-5) — the same evaluation setup
--   as the Opus 5.5 40.0% cell in the same row. (OpenAI's own July 9 suite
--   separately reports 18.1% on its own AutomationBench run, a different
--   harness, so the Zapier figure is used for comparability.)
-- - Agents' Last Exam V1 GPT-5.6 Sol 52.7%: OpenAI GPT-5.6 launch suite
--   professional table, July 9, 2026 (https://openai.com/index/gpt-5-6/).
-- - FrontierCode 1.1 Main GPT-5.6 Sol 47.5%: OpenAI-reported figure shown in
--   Anthropic's Opus 5.5 launch table, matching the official July 9
--   FrontierCode Main score.
-- - Opus 5.5 Agents' Last Exam V1 and DeepSWE v1.1 remain "-": neither is
--   reported in Anthropic's Opus 5.5 launch table, and the official DeepSWE
--   v1.1 leaderboard (updated September 22, 2026) has no Opus 5.5 row.

-- 1. Fill the GPT-5.6 Sol cells.
WITH old_new(old_text, new_text) AS (
  SELECT
    $old$| AutomationBench 1.0.6 | 33.2% (xhigh) | - | 40.0% |
| Agents' Last Exam V1 | 56.6% (max) | - | - |
| FrontierCode 1.1 Main | 49.3% (max) | - | 54.4% |$old$::text,
    $new$| AutomationBench 1.0.6 | 33.2% (xhigh) | 28.8% | 40.0% |
| Agents' Last Exam V1 | 56.6% (max) | 52.7% | - |
| FrontierCode 1.1 Main | 49.3% (max) | 47.5% | 54.4% |$new$::text
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

-- 2. Update the scores sentence to document the AutomationBench source.
WITH old_new(old_text, new_text) AS (
  SELECT
    $old$Scores for GPT-6 Sol are OpenAI-reported chart readings at stated effort levels; GPT-5.6 Sol figures are from OpenAI's July 9 GPT-5.6 launch suite; Opus 5.5 figures are Anthropic-reported.$old$::text,
    $new$Scores for GPT-6 Sol are OpenAI-reported chart readings at stated effort levels. GPT-5.6 Sol figures are from OpenAI's July 9 GPT-5.6 launch suite, except AutomationBench (28.8%), which is the Zapier-leaderboard run reported in Anthropic's Opus 5.5 launch table — the same setup as the Opus 5.5 40.0% cell. Opus 5.5 figures are Anthropic-reported.$new$::text
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
