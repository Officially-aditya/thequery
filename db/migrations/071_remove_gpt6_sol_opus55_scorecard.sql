UPDATE content_items
SET
  blocks = (
    SELECT COALESCE(jsonb_agg(element ORDER BY ordinal), '[]'::jsonb)
    FROM jsonb_array_elements(blocks) WITH ORDINALITY AS elements(element, ordinal)
    WHERE element->>'id' <> 'scorecard-gpt6sol-opus55'
  ),
  updated_at = NOW()
WHERE id = 'comparison:gpt-6-sol-vs-opus-5-5'
  AND blocks @> '[{"id":"scorecard-gpt6sol-opus55","type":"comparison_table"}]'::jsonb;
