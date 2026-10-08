-- One table entry for Terminal-Bench 4.0, with the evaluator differences
-- retained in an asterisk note. Raw model_benchmarks evidence is untouched.
WITH consolidated AS (
  SELECT id, replace(
    replace(body,
      $rows_before$| Terminal-Bench 4.0 | 28.3% | Mistral launch report |
| Terminal-Bench 4.0 | 26.8% | Artificial Analysis evaluated preview |
| Terminal-Bench 4.0 | 22.73% | Vals model profile |$rows_before$,
      $rows_after$| Terminal-Bench 4.0 | 28.3%\* | Mistral launch report |$rows_after$
    ),
    $note_before$The three Terminal-Bench readings are preserved rather than averaged or treated as the same run. Mistral's headline is the highest, while the two independent profiles report lower values. $note_before$,
    $note_after$\* **Terminal-Bench 4.0:** Mistral reports 28.3%, Artificial Analysis reports 26.8%, and Vals reports 22.73%. The table shows the highest reported result. Evaluation conditions are not fully specified across these sources.

$note_after$
  ) AS body
  FROM content_items
  WHERE kind = 'glossary'
    AND slug = 'mistral-large-4'
    AND parent_slug = ''
)
UPDATE content_items AS content
SET
  body = consolidated.body,
  blocks = jsonb_build_array(jsonb_build_object(
    'id', 'markdown-1', 'type', 'markdown', 'content', consolidated.body
  )),
  updated_at = NOW()
FROM consolidated
WHERE content.id = consolidated.id;
