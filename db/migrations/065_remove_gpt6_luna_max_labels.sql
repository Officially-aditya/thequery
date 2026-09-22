-- Remove the common (max) effort labels from the GPT-6 Luna glossary
-- benchmark table. All GPT-6 Luna scores are at max effort, so (max) is
-- redundant. The effort default moves into the scores sentence, mirroring
-- 064 (Sol).

-- 1. Strip " (max)" from the five GPT-6 Luna table cells.
WITH old_new(old_text, new_text) AS (
  SELECT
    $old$| AutomationBench 1.0.6 | 20.7% (max) | - |
| Agents' Last Exam V1 | 50.9% (max) | - |
| FrontierCode 1.1 Main | 42.4% (max) | - |
| DeepSWE v1.1 | 66.6% (max) | 67.2% |
| OSWorld 2.0 | 52.7% (max) | 45.6% |$old$::text,
    $new$| AutomationBench 1.0.6 | 20.7% | - |
| Agents' Last Exam V1 | 50.9% | - |
| FrontierCode 1.1 Main | 42.4% | - |
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

-- 2. Move the effort default into the scores sentence.
WITH old_new(old_text, new_text) AS (
  SELECT
    $old$Scores for GPT-6 Luna are OpenAI-reported chart readings at stated effort levels;$old$::text,
    $new$Scores for GPT-6 Luna are OpenAI-reported chart readings at max effort;$new$::text
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
