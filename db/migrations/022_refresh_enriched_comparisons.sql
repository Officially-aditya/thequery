-- Refresh materialized comparison pages after model/benchmark enrichment.
-- Existing non-empty cells win over catalog values so editorial/manual edits are preserved.
-- New canonical benchmark labels are added to every resolved model-vs-model comparison.

WITH labels(section_order, section_title, label_order, label) AS (
  VALUES
    (1, 'Specifications', 1, 'Developer'),
    (1, 'Specifications', 2, 'Release date'),
    (1, 'Specifications', 3, 'API model ID'),
    (1, 'Specifications', 4, 'Context window'),
    (1, 'Specifications', 5, 'Max output'),
    (1, 'Specifications', 6, 'Knowledge cutoff'),
    (1, 'Specifications', 7, 'Reasoning / effort'),
    (2, 'Pricing', 1, 'Input / 1M tokens'),
    (2, 'Pricing', 2, 'Cached input / 1M'),
    (2, 'Pricing', 3, 'Cache write / 1M'),
    (2, 'Pricing', 4, 'Output / 1M tokens'),
    (2, 'Pricing', 5, 'Batch / flex discount'),
    (2, 'Pricing', 6, 'Long-context surcharge'),
    (3, 'Capabilities & access', 1, 'Text input'),
    (3, 'Capabilities & access', 2, 'Image / vision input'),
    (3, 'Capabilities & access', 3, 'Audio input'),
    (3, 'Capabilities & access', 4, 'Video input'),
    (3, 'Capabilities & access', 5, 'Text output'),
    (3, 'Capabilities & access', 6, 'Image output'),
    (3, 'Capabilities & access', 7, 'Audio output'),
    (3, 'Capabilities & access', 8, 'Video output'),
    (3, 'Capabilities & access', 9, 'Tool / function calling'),
    (3, 'Capabilities & access', 10, 'Computer use'),
    (3, 'Capabilities & access', 11, 'API access'),
    (3, 'Capabilities & access', 12, 'Product access'),
    (3, 'Capabilities & access', 13, 'Weights / license'),
    (4, 'Coding', 1, 'SWE-bench Verified'),
    (4, 'Coding', 2, 'SWE-bench Pro'),
    (4, 'Coding', 3, 'FrontierCode 1.1 Main'),
    (4, 'Coding', 4, 'FrontierCode 1.1 Extended'),
    (4, 'Coding', 5, 'DeepSWE v1.1'),
    (4, 'Coding', 6, 'Terminal-Bench 2.1'),
    (4, 'Coding', 7, 'Terminal-Bench 3.0'),
    (4, 'Coding', 8, 'Terminal-Bench 4.0'),
    (4, 'Coding', 9, 'Terminal-Bench Science 0.1'),
    (4, 'Coding', 10, 'Terminal-Bench'),
    (4, 'Coding', 11, 'MLE-Bench'),
    (4, 'Coding', 12, 'LiveCodeBench'),
    (4, 'Coding', 13, 'CursorBench'),
    (5, 'Math & reasoning', 1, 'AIME'),
    (5, 'Math & reasoning', 2, 'HMMT'),
    (5, 'Math & reasoning', 3, 'ARC-AGI'),
    (5, 'Math & reasoning', 4, 'FrontierMath'),
    (5, 'Math & reasoning', 5, 'FrontierMath Tier 4 (v2)'),
    (6, 'Knowledge', 1, 'GPQA Diamond'),
    (6, 'Knowledge', 2, 'Humanity''s Last Exam'),
    (6, 'Knowledge', 3, 'HLE-Verified'),
    (6, 'Knowledge', 4, 'MMLU-Pro'),
    (7, 'Agentic & computer use', 1, 'OSWorld'),
    (7, 'Agentic & computer use', 2, 'OSWorld-Verified'),
    (7, 'Agentic & computer use', 3, 'OSWorld 2.0'),
    (7, 'Agentic & computer use', 4, 'BrowseComp'),
    (7, 'Agentic & computer use', 5, 'GDPval-AA'),
    (7, 'Agentic & computer use', 6, 'GDPval-AA v2'),
    (7, 'Agentic & computer use', 7, 'AutomationBench'),
    (7, 'Agentic & computer use', 8, 'Agents'' Last Exam'),
    (7, 'Agentic & computer use', 9, 'MCP Atlas'),
    (7, 'Agentic & computer use', 10, 'Toolathlon'),
    (7, 'Agentic & computer use', 11, 'MCP / tool-use benchmark')
),
benchmark_rendered AS (
  SELECT
    b.model_slug,
    b.benchmark_name,
    b.evaluation_date,
    b.id,
    b.score_display || CASE
      WHEN concat_ws('; ',
        CASE
          WHEN b.benchmark_version IS NOT NULL
           AND position(lower(b.benchmark_version) in lower(b.benchmark_name)) = 0
          THEN b.benchmark_version
        END,
        CASE WHEN b.tools IS TRUE THEN 'tools' WHEN b.tools IS FALSE THEN 'no tools' END,
        b.reasoning_effort,
        b.harness,
        b.evaluator
      ) <> ''
      THEN ' (' || concat_ws('; ',
        CASE
          WHEN b.benchmark_version IS NOT NULL
           AND position(lower(b.benchmark_version) in lower(b.benchmark_name)) = 0
          THEN b.benchmark_version
        END,
        CASE WHEN b.tools IS TRUE THEN 'tools' WHEN b.tools IS FALSE THEN 'no tools' END,
        b.reasoning_effort,
        b.harness,
        b.evaluator
      ) || ')'
      ELSE ''
    END AS rendered
  FROM model_benchmarks AS b
),
benchmark_grouped AS (
  SELECT
    model_slug,
    benchmark_name,
    string_agg(rendered, ' · ' ORDER BY evaluation_date NULLS LAST, id) AS rendered
  FROM benchmark_rendered
  GROUP BY model_slug, benchmark_name
),
benchmark_objects AS (
  SELECT model_slug, jsonb_object_agg(benchmark_name, rendered) AS comparison_data
  FROM benchmark_grouped
  GROUP BY model_slug
),
model_data AS (
  SELECT
    m.slug,
    m.name,
    COALESCE(bo.comparison_data, '{}'::jsonb) || COALESCE(m.comparison_data, '{}'::jsonb) AS comparison_data
  FROM models AS m
  LEFT JOIN benchmark_objects AS bo ON bo.model_slug = m.slug
),
resolved_pairs AS (
  SELECT
    c.id,
    c.summary,
    c.blocks AS original_blocks,
    c.metadata,
    ma.slug AS model_a_slug,
    ma.name AS model_a_name,
    ma.comparison_data AS model_a_data,
    mb.slug AS model_b_slug,
    mb.name AS model_b_name,
    mb.comparison_data AS model_b_data
  FROM content_items AS c
  LEFT JOIN LATERAL (
    SELECT md.slug, md.name, md.comparison_data
    FROM model_data AS md
    WHERE md.slug = c.metadata->>'modelA'
       OR lower(md.name) = lower(trim(split_part(c.title, ' vs ', 1)))
    ORDER BY CASE WHEN md.slug = c.metadata->>'modelA' THEN 0 ELSE 1 END
    LIMIT 1
  ) AS ma ON TRUE
  LEFT JOIN LATERAL (
    SELECT md.slug, md.name, md.comparison_data
    FROM model_data AS md
    WHERE md.slug = c.metadata->>'modelB'
       OR lower(md.name) = lower(trim(split_part(c.title, ' vs ', 2)))
    ORDER BY CASE WHEN md.slug = c.metadata->>'modelB' THEN 0 ELSE 1 END
    LIMIT 1
  ) AS mb ON TRUE
  WHERE c.kind = 'comparison'
),
flat_existing_rows AS (
  SELECT
    c.id,
    lower(trim(row_value->>0)) AS label,
    COALESCE(row_value->>1, '') AS value_a,
    COALESCE(row_value->>2, '') AS value_b,
    block_ordinality,
    row_ordinality
  FROM content_items AS c
  CROSS JOIN LATERAL jsonb_array_elements(COALESCE(c.blocks, '[]'::jsonb)) WITH ORDINALITY AS block_value(block, block_ordinality)
  CROSS JOIN LATERAL jsonb_array_elements(
    CASE WHEN jsonb_typeof(block_value.block->'rows') = 'array' THEN block_value.block->'rows' ELSE '[]'::jsonb END
  ) WITH ORDINALITY AS row_entry(row_value, row_ordinality)
  WHERE c.kind = 'comparison'
    AND jsonb_typeof(row_value) = 'array'
    AND jsonb_array_length(row_value) >= 3
    AND trim(COALESCE(row_value->>0, '')) <> ''
),
deduped_existing_rows AS (
  SELECT DISTINCT ON (id, label) id, label, value_a, value_b
  FROM flat_existing_rows
  ORDER BY id, label, block_ordinality, row_ordinality
),
existing_rows AS (
  SELECT id, jsonb_object_agg(label, jsonb_build_array(value_a, value_b)) AS values_by_label
  FROM deduped_existing_rows
  GROUP BY id
),
section_rows AS (
  SELECT
    p.id,
    l.section_order,
    l.section_title,
    jsonb_agg(
      jsonb_build_array(
        l.label,
        COALESCE(NULLIF(er.values_by_label->lower(l.label)->>0, ''), p.model_a_data->>l.label, ''),
        COALESCE(NULLIF(er.values_by_label->lower(l.label)->>1, ''), p.model_b_data->>l.label, '')
      ) ORDER BY l.label_order
    ) AS rows
  FROM resolved_pairs AS p
  CROSS JOIN labels AS l
  LEFT JOIN existing_rows AS er ON er.id = p.id
  WHERE p.model_a_slug IS NOT NULL AND p.model_b_slug IS NOT NULL
  GROUP BY p.id, l.section_order, l.section_title, er.values_by_label, p.model_a_data, p.model_b_data
),
sections AS (
  SELECT
    p.id,
    jsonb_agg(
      jsonb_build_object(
        'id', 'spec-enriched-' || sr.section_order,
        'type', 'spec_table',
        'title', sr.section_title,
        'columns', jsonb_build_array(p.model_a_name, p.model_b_name),
        'rows', sr.rows
      ) ORDER BY sr.section_order
    ) AS blocks
  FROM resolved_pairs AS p
  JOIN section_rows AS sr ON sr.id = p.id
  GROUP BY p.id, p.model_a_name, p.model_b_name
),
canonical AS (
  SELECT
    p.id,
    s.blocks || jsonb_build_array(
      jsonb_build_object(
        'id', 'markdown-enriched-bottom-line',
        'type', 'markdown',
        'content', COALESCE(
          (
            SELECT old_block.block->>'content'
            FROM jsonb_array_elements(COALESCE(p.original_blocks, '[]'::jsonb)) WITH ORDINALITY AS old_block(block, ordinality)
            WHERE old_block.block->>'type' = 'markdown'
              AND lower(COALESCE(old_block.block->>'content', '')) LIKE '%bottom line%'
            ORDER BY old_block.ordinality DESC
            LIMIT 1
          ),
          '## Bottom line' || E'\n\n' || COALESCE(NULLIF(trim(p.summary), ''), '')
        )
      )
    ) AS blocks,
    jsonb_set(
      jsonb_set(COALESCE(p.metadata, '{}'::jsonb), '{modelA}', to_jsonb(p.model_a_slug), true),
      '{modelB}', to_jsonb(p.model_b_slug), true
    ) AS metadata
  FROM resolved_pairs AS p
  JOIN sections AS s ON s.id = p.id
  WHERE p.model_a_slug IS NOT NULL AND p.model_b_slug IS NOT NULL
)
UPDATE content_items AS c
SET blocks = canonical.blocks, metadata = canonical.metadata, updated_at = NOW()
FROM canonical
WHERE c.id = canonical.id;