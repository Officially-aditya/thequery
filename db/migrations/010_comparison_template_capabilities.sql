UPDATE content_items AS comparison
SET blocks = (
  SELECT jsonb_agg(entry.item ORDER BY entry.position, entry.after_pricing)
  FROM (
    SELECT existing.element AS item, existing.ordinality AS position, 0 AS after_pricing
    FROM jsonb_array_elements(comparison.blocks) WITH ORDINALITY AS existing(element, ordinality)

    UNION ALL

    SELECT jsonb_build_object(
      'id', 'spec-capabilities-access-fable-astra',
      'type', 'spec_table',
      'title', 'Capabilities & access',
      'columns', jsonb_build_array('Claude Fable 5.1', 'GPT-6 Astra'),
      'rows', jsonb_build_array(
        jsonb_build_array('Text input', '**Yes**', '**Yes**'),
        jsonb_build_array('Image / vision input', '**Yes**', '**Yes**'),
        jsonb_build_array('Audio input', 'No native audio input', 'No'),
        jsonb_build_array('Video input', 'No', 'No'),
        jsonb_build_array('Text output', '**Yes**', '**Yes**'),
        jsonb_build_array('Audio output', 'No native audio output', 'No'),
        jsonb_build_array('Tool / function calling', '**Yes**', '**Yes**'),
        jsonb_build_array('Computer use', '**Yes**', '**Yes**'),
        jsonb_build_array('API access', 'Claude API; AWS Bedrock; Google Cloud; Microsoft Foundry', 'OpenAI API; Microsoft Azure; AWS Bedrock'),
        jsonb_build_array('Product access', 'Claude Pro, Max, Team, Enterprise; Claude Code', 'ChatGPT Plus, Pro, Business, Enterprise'),
        jsonb_build_array('Weights / license', 'Proprietary', 'Proprietary')
      )
    ) AS item,
    pricing.ordinality AS position,
    1 AS after_pricing
    FROM jsonb_array_elements(comparison.blocks) WITH ORDINALITY AS pricing(element, ordinality)
    WHERE pricing.element->>'type' = 'spec_table'
      AND lower(pricing.element->>'title') = 'pricing'
  ) AS entry
),
updated_at = NOW()
WHERE comparison.kind = 'comparison'
  AND comparison.title ILIKE '%Fable 5.1%'
  AND comparison.title ILIKE '%Astra%'
  AND EXISTS (
    SELECT 1
    FROM jsonb_array_elements(comparison.blocks) AS block
    WHERE block->>'type' = 'spec_table'
      AND lower(block->>'title') = 'pricing'
  )
  AND NOT EXISTS (
    SELECT 1
    FROM jsonb_array_elements(comparison.blocks) AS block
    WHERE block->>'type' = 'spec_table'
      AND lower(block->>'title') IN ('capabilities & access', 'capabilities and access')
  );
