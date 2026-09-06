-- Remove provenance-only benchmark qualifiers from materialized comparison values.
-- Keep benchmark metadata (harness/evaluator/source) normalized in model_benchmarks for auditability.
-- Only auto-generated values that exactly match the previous renderer are rewritten; manual overrides survive.

WITH benchmark_rendered AS (
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
    END AS old_rendered,
    b.score_display || CASE
      WHEN concat_ws('; ',
        CASE
          WHEN b.benchmark_version IS NOT NULL
           AND lower(trim(b.benchmark_version)) <> 'public'
           AND position(lower(b.benchmark_version) in lower(b.benchmark_name)) = 0
          THEN b.benchmark_version
        END,
        CASE WHEN b.tools IS TRUE THEN 'tools' WHEN b.tools IS FALSE THEN 'no tools' END,
        b.reasoning_effort
      ) <> ''
      THEN ' (' || concat_ws('; ',
        CASE
          WHEN b.benchmark_version IS NOT NULL
           AND lower(trim(b.benchmark_version)) <> 'public'
           AND position(lower(b.benchmark_version) in lower(b.benchmark_name)) = 0
          THEN b.benchmark_version
        END,
        CASE WHEN b.tools IS TRUE THEN 'tools' WHEN b.tools IS FALSE THEN 'no tools' END,
        b.reasoning_effort
      ) || ')'
      ELSE ''
    END AS new_rendered
  FROM model_benchmarks AS b
),
benchmark_grouped AS (
  SELECT
    model_slug,
    benchmark_name,
    string_agg(old_rendered, ' · ' ORDER BY evaluation_date NULLS LAST, id) AS old_value,
    string_agg(new_rendered, ' · ' ORDER BY evaluation_date NULLS LAST, id) AS new_value
  FROM benchmark_rendered
  GROUP BY model_slug, benchmark_name
),
rebuilt AS (
  SELECT
    c.id,
    jsonb_agg(
      CASE
        WHEN jsonb_typeof(block_entry.block->'rows') = 'array' THEN
          jsonb_set(
            block_entry.block,
            '{rows}',
            COALESCE(
              (
                SELECT jsonb_agg(
                  CASE
                    WHEN jsonb_typeof(row_entry.row_value) = 'array'
                     AND jsonb_array_length(row_entry.row_value) >= 3 THEN
                      jsonb_set(
                        jsonb_set(
                          row_entry.row_value,
                          '{1}',
                          to_jsonb(
                            CASE
                              WHEN benchmark_a.old_value IS NOT NULL
                               AND COALESCE(row_entry.row_value->>1, '') = benchmark_a.old_value
                              THEN benchmark_a.new_value
                              ELSE COALESCE(row_entry.row_value->>1, '')
                            END
                          ),
                          false
                        ),
                        '{2}',
                        to_jsonb(
                          CASE
                            WHEN benchmark_b.old_value IS NOT NULL
                             AND COALESCE(row_entry.row_value->>2, '') = benchmark_b.old_value
                            THEN benchmark_b.new_value
                            ELSE COALESCE(row_entry.row_value->>2, '')
                          END
                        ),
                        false
                      )
                    ELSE row_entry.row_value
                  END
                  ORDER BY row_entry.row_ordinality
                )
                FROM jsonb_array_elements(block_entry.block->'rows') WITH ORDINALITY AS row_entry(row_value, row_ordinality)
                LEFT JOIN benchmark_grouped AS benchmark_a
                  ON benchmark_a.model_slug = c.metadata->>'modelA'
                 AND lower(benchmark_a.benchmark_name) = lower(COALESCE(row_entry.row_value->>0, ''))
                LEFT JOIN benchmark_grouped AS benchmark_b
                  ON benchmark_b.model_slug = c.metadata->>'modelB'
                 AND lower(benchmark_b.benchmark_name) = lower(COALESCE(row_entry.row_value->>0, ''))
              ),
              '[]'::jsonb
            ),
            true
          )
        ELSE block_entry.block
      END
      ORDER BY block_entry.block_ordinality
    ) AS blocks
  FROM content_items AS c
  CROSS JOIN LATERAL jsonb_array_elements(COALESCE(c.blocks, '[]'::jsonb)) WITH ORDINALITY AS block_entry(block, block_ordinality)
  WHERE c.kind = 'comparison'
  GROUP BY c.id, c.metadata
)
UPDATE content_items AS c
SET blocks = rebuilt.blocks, updated_at = NOW()
FROM rebuilt
WHERE c.id = rebuilt.id
  AND c.blocks IS DISTINCT FROM rebuilt.blocks;
