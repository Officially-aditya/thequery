-- Remove the common (max) effort labels from the GPT-6 Sol glossary
-- benchmark table. All GPT-6 Sol scores are at max effort except
-- AutomationBench (xhigh), so (max) is redundant — only (xhigh) stays
-- as the anomaly marker. The effort default moves into the scores sentence.

-- 1. Strip " (max)" from the four GPT-6 Sol table cells.
WITH old_new(old_text, new_text) AS (
  SELECT
    $old$| Agents' Last Exam V1 | 56.6% (max) | 52.7% | - |
| FrontierCode 1.1 Main | 49.3% (max) | 47.5% | 54.4% |
| DeepSWE v1.1 | 68.8% (max) | 72.7% | - |
| OSWorld 2.0 | 64.4% (max) | 62.6% | 81.8% |$old$::text,
    $new$| Agents' Last Exam V1 | 56.6% | 52.7% | - |
| FrontierCode 1.1 Main | 49.3% | 47.5% | 54.4% |
| DeepSWE v1.1 | 68.8% | 72.7% | - |
| OSWorld 2.0 | 64.4% | 62.6% | 81.8% |$new$::text
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

-- 2. Move the effort default into the scores sentence.
WITH old_new(old_text, new_text) AS (
  SELECT
    $old$Scores for GPT-6 Sol are OpenAI-reported chart readings at stated effort levels.$old$::text,
    $new$Scores for GPT-6 Sol are OpenAI-reported chart readings at max effort except AutomationBench (xhigh).$new$::text
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
