-- Deepen Muse Spark 1.2/1.3 generated comparison data and benchmark coverage.
-- Primary Meta sources verified 2026-09-07.

WITH patches(slug, comparison_data, sources, notes, metadata) AS (VALUES
  (
    'muse-spark-1-2',
    '{"API model ID":"muse-spark-1.2","Context window":"1M tokens","Reasoning / effort":"xhigh used in Meta''s 1.2 launch evaluations; long-horizon reasoning with compaction and subagents","Input / 1M tokens":"$1.25","Cached input / 1M":"$0.15","Output / 1M tokens":"$4.25","Text input":"Yes","Image / vision input":"Yes","Audio input":"Yes","Video input":"Yes","File / document input":"Yes — multimodal files and PDFs supported in Meta Model API workflows","Text output":"Yes","Tool / function calling":"Yes — native tools, parallel agent workflows and Muse Code integration","Computer use":"Yes — through agentic computer/code harnesses","API access":"Meta Model API","Product access":"Muse Code + Meta Model API + Meta AI","Weights / license":"Proprietary; open weights were announced but not released for 1.2","Primary focus":"Coding-focused update: code generation, complex debugging, codebase understanding and end-to-end developer workflows","Long-horizon work":"Trained on whole-repository generation, large end-to-end projects and auto-research; uses planning, goal conditioning and context compaction to sustain progress","Agent orchestration":"Co-trained with Muse Code; coordinates persistent async subagents that retain task context across a session"}'::jsonb,
    '[{"title":"Muse Spark model card","url":"https://developer.meta.com/ai/models/muse-spark/"},{"title":"Introducing Muse Code and Muse Spark 1.2","url":"https://research.meta.ai/blog/introducing-muse-code-and-muse-spark-1-2"},{"title":"Muse Spark 1.2 evaluation methodology","url":"https://research.meta.ai/static/muse-spark-1-2-methodology"},{"title":"Multimodal intelligence of Muse Spark 1.2","url":"https://research.meta.ai/blog/multimodal-intelligence-of-muse-spark-1-2"}]'::jsonb,
    'Muse Spark 1.2 is a coding-focused Muse release co-trained with Muse Code. Meta documents persistent async subagents, replay-safe long-running workflows, whole-repository work, compaction, multimodal input and expanded Meta Model API access. Pricing/context fields reflect the Meta Model API model card; benchmark provenance remains normalized separately.',
    '{"verification_pass":"muse-comparison-depth-2026-09-07","comparison_depth":"rich","official_model_card":"https://developer.meta.com/ai/models/muse-spark/"}'::jsonb
  ),
  (
    'muse-spark-1-3',
    '{"API model ID":"muse-spark-1.3","Context window":"1M tokens","Reasoning / effort":"xhigh + max; Meta''s launch scorecard uses max for 1.3 while the 1.2 comparison column uses xhigh","Input / 1M tokens":"$1.25","Cached input / 1M":"$0.15","Output / 1M tokens":"$4.25","Text input":"Yes","Image / vision input":"Yes — visual and heterogeneous file workflows","Audio input":"Yes — audio-file editing workflow demonstrated; exact low-level codec matrix not separately disclosed","Video input":"Yes — multimodal file workflows","File / document input":"Yes — documents, spreadsheets, CAD/STEP and other heterogeneous files demonstrated","Text output":"Yes","Tool / function calling":"Yes","Computer use":"Yes — long-horizon agentic workflows","API access":"Meta Model API","Product access":"Muse Code + Meta Model API + Meta AI","Weights / license":"Proprietary; Meta lists open weights as a future release","Primary focus":"Long-horizon agentic and coding work with improved real-world usability","Long-horizon work":"Sustains longer work, generates context across messy/conflicting sources, corrects planning gaps, preserves detailed constraints and tracks multiple workflows in one long thread","Agent orchestration":"Improved multitasking and routing of new prompts to the correct ongoing task; uses tools to build and maintain working context","User collaboration":"Asks clarifying questions for ambiguous prompts, requests help when stuck, adapts update frequency and confirms before consequential actions","Efficiency / generation change":"In Meta engineer comparisons vs 1.2: ~20% fewer tool calls and ~25% fewer tokens for coding work","Safety / approvals":"Improved prompt-injection/adversarial robustness and better calibration around irreversible actions"}'::jsonb,
    '[{"title":"Muse Spark model card","url":"https://developer.meta.com/ai/models/muse-spark/"},{"title":"Introducing Muse Spark 1.3","url":"https://research.meta.ai/blog/introducing-muse-spark-1-3"},{"title":"Muse Spark 1.3 evaluation methodology","url":"https://research.meta.ai/static/muse-spark-1-3-multimodal-evaluation-methodology"}]'::jsonb,
    'Muse Spark 1.3 focuses on long-horizon agentic/coding work, multi-workflow tracking, user collaboration and efficiency. Meta reports about 20% fewer tool calls and 25% fewer tokens than 1.2 in internal engineering comparisons. Launch scorecard results use max reasoning for 1.3 and xhigh for 1.2, so the reasoning-effort difference is preserved in normalized benchmark qualifiers.',
    '{"verification_pass":"muse-comparison-depth-2026-09-07","comparison_depth":"rich","official_model_card":"https://developer.meta.com/ai/models/muse-spark/","generation_baseline":"muse-spark-1-2"}'::jsonb
  )
)
UPDATE models AS m
SET comparison_data = COALESCE(m.comparison_data, '{}'::jsonb) || p.comparison_data,
    sources = p.sources,
    notes = p.notes,
    metadata = COALESCE(m.metadata, '{}'::jsonb) || p.metadata,
    verified_at = NOW(),
    updated_at = NOW()
FROM patches AS p
WHERE m.slug = p.slug;

INSERT INTO model_benchmarks (
  id, model_slug, category, benchmark_name, benchmark_version,
  score_numeric, score_display, score_unit, tools, reasoning_effort,
  harness, evaluator, evaluation_date, source, notes, updated_at
)
SELECT
  b.id, b.model_slug, b.category, b.benchmark_name, b.benchmark_version,
  b.score_numeric, b.score_display, b.score_unit, b.tools, b.reasoning_effort,
  b.harness, b.evaluator, b.evaluation_date::date, b.source, b.notes, NOW()
FROM jsonb_to_recordset($benchmarks$
[
  {"id":"meta-20260902-muse12-jobbench","model_slug":"muse-spark-1-2","category":"professional","benchmark_name":"JobBench","benchmark_version":null,"score_numeric":61.6,"score_display":"61.6%","score_unit":"percent","tools":true,"reasoning_effort":"xhigh","harness":"Muse Spark 1.3 launch scorecard","evaluator":"Meta","evaluation_date":"2026-09-02","source":"https://research.meta.ai/static/muse-spark-1-3-multimodal-evaluation-methodology","notes":"Muse Spark 1.2 comparison value in Meta's 1.3 scorecard; 1.2 is evaluated at xhigh."},
  {"id":"meta-20260902-muse12-osworld20","model_slug":"muse-spark-1-2","category":"agentic_computer_use","benchmark_name":"OSWorld 2.0","benchmark_version":"06.24","score_numeric":47.6,"score_display":"47.6%","score_unit":"percent","tools":true,"reasoning_effort":"xhigh","harness":"Muse Spark 1.3 launch scorecard","evaluator":"Meta","evaluation_date":"2026-09-02","source":"https://research.meta.ai/static/muse-spark-1-3-multimodal-evaluation-methodology","notes":"Meta notes the 1.2 OSWorld 2.0 run uses an older benchmark revision than the 1.3 comparison run."},
  {"id":"meta-20260902-muse12-deepsearchqa","model_slug":"muse-spark-1-2","category":"agentic_computer_use","benchmark_name":"DeepSearchQA","benchmark_version":null,"score_numeric":85.9,"score_display":"85.9%","score_unit":"percent","tools":true,"reasoning_effort":"xhigh","harness":"Muse Spark 1.3 launch scorecard","evaluator":"Meta","evaluation_date":"2026-09-02","source":"https://research.meta.ai/static/muse-spark-1-3-multimodal-evaluation-methodology","notes":"1.2 baseline published in Meta's 1.3 evaluation table."},
  {"id":"meta-20260902-muse12-agentic-if","model_slug":"muse-spark-1-2","category":"agentic_computer_use","benchmark_name":"Agentic IF Index","benchmark_version":"Meta internal","score_numeric":46.2,"score_display":"46.2%","score_unit":"percent","tools":true,"reasoning_effort":"xhigh","harness":"Meta internal agentic instruction-following evaluation","evaluator":"Meta","evaluation_date":"2026-09-02","source":"https://research.meta.ai/static/muse-spark-1-3-multimodal-evaluation-methodology","notes":"Internal benchmark; shown for same-vendor generation comparison, not as an independent leaderboard."},
  {"id":"meta-20260902-muse12-automation","model_slug":"muse-spark-1-2","category":"agentic_computer_use","benchmark_name":"AutomationBench","benchmark_version":null,"score_numeric":38.2,"score_display":"38.2%","score_unit":"percent","tools":true,"reasoning_effort":"xhigh","harness":"Muse Spark 1.3 launch scorecard","evaluator":"Meta","evaluation_date":"2026-09-02","source":"https://research.meta.ai/static/muse-spark-1-3-multimodal-evaluation-methodology","notes":"1.2 baseline published in Meta's 1.3 evaluation table."},
  {"id":"meta-20260902-muse12-mrcr-256-512","model_slug":"muse-spark-1-2","category":"math_reasoning","benchmark_name":"MRCR v2 256K–512K","benchmark_version":"8-needle","score_numeric":66.3,"score_display":"66.3%","score_unit":"percent","tools":false,"reasoning_effort":"xhigh","harness":"MRCR v2 long-context retrieval","evaluator":"Meta","evaluation_date":"2026-09-02","source":"https://research.meta.ai/static/muse-spark-1-3-multimodal-evaluation-methodology","notes":"Long-context retrieval baseline for 1.2."},
  {"id":"meta-20260902-muse12-mrcr-512-1m","model_slug":"muse-spark-1-2","category":"math_reasoning","benchmark_name":"MRCR v2 512K–1M","benchmark_version":"8-needle","score_numeric":55.5,"score_display":"55.5%","score_unit":"percent","tools":false,"reasoning_effort":"xhigh","harness":"MRCR v2 long-context retrieval","evaluator":"Meta","evaluation_date":"2026-09-02","source":"https://research.meta.ai/static/muse-spark-1-3-multimodal-evaluation-methodology","notes":"Long-context retrieval baseline for 1.2."},
  {"id":"meta-20260902-muse12-swe-atlas","model_slug":"muse-spark-1-2","category":"coding","benchmark_name":"SWE-Atlas Codebase QnA","benchmark_version":null,"score_numeric":46.2,"score_display":"46.2%","score_unit":"percent","tools":true,"reasoning_effort":"xhigh","harness":"SWE-Atlas Codebase QnA","evaluator":"Meta","evaluation_date":"2026-09-02","source":"https://research.meta.ai/static/muse-spark-1-3-multimodal-evaluation-methodology","notes":"1.2 baseline published in Meta's 1.3 evaluation table."},

  {"id":"meta-20260902-muse13-jobbench","model_slug":"muse-spark-1-3","category":"professional","benchmark_name":"JobBench","benchmark_version":null,"score_numeric":64.9,"score_display":"64.9%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"Muse Spark 1.3 launch scorecard","evaluator":"Meta","evaluation_date":"2026-09-02","source":"https://research.meta.ai/static/muse-spark-1-3-multimodal-evaluation-methodology","notes":"Professional tool-use evaluation from Meta's launch scorecard."},
  {"id":"meta-20260902-muse13-deepsearchqa","model_slug":"muse-spark-1-3","category":"agentic_computer_use","benchmark_name":"DeepSearchQA","benchmark_version":null,"score_numeric":89.4,"score_display":"89.4%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"Muse Spark 1.3 launch scorecard","evaluator":"Meta","evaluation_date":"2026-09-02","source":"https://research.meta.ai/static/muse-spark-1-3-multimodal-evaluation-methodology","notes":null},
  {"id":"meta-20260902-muse13-agentic-if","model_slug":"muse-spark-1-3","category":"agentic_computer_use","benchmark_name":"Agentic IF Index","benchmark_version":"Meta internal","score_numeric":57.8,"score_display":"57.8%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"Meta internal agentic instruction-following evaluation","evaluator":"Meta","evaluation_date":"2026-09-02","source":"https://research.meta.ai/static/muse-spark-1-3-multimodal-evaluation-methodology","notes":"Internal benchmark; useful for same-vendor generation comparison but not an independent leaderboard."},
  {"id":"meta-20260902-muse13-mrcr-256-512","model_slug":"muse-spark-1-3","category":"math_reasoning","benchmark_name":"MRCR v2 256K–512K","benchmark_version":"8-needle","score_numeric":98.5,"score_display":"98.5%","score_unit":"percent","tools":false,"reasoning_effort":"max","harness":"MRCR v2 long-context retrieval","evaluator":"Meta","evaluation_date":"2026-09-02","source":"https://research.meta.ai/static/muse-spark-1-3-multimodal-evaluation-methodology","notes":null},
  {"id":"meta-20260902-muse13-mrcr-512-1m","model_slug":"muse-spark-1-3","category":"math_reasoning","benchmark_name":"MRCR v2 512K–1M","benchmark_version":"8-needle","score_numeric":98.1,"score_display":"98.1%","score_unit":"percent","tools":false,"reasoning_effort":"max","harness":"MRCR v2 long-context retrieval","evaluator":"Meta","evaluation_date":"2026-09-02","source":"https://research.meta.ai/static/muse-spark-1-3-multimodal-evaluation-methodology","notes":null},
  {"id":"meta-20260902-muse13-swe-atlas","model_slug":"muse-spark-1-3","category":"coding","benchmark_name":"SWE-Atlas Codebase QnA","benchmark_version":null,"score_numeric":59.4,"score_display":"59.4%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"SWE-Atlas Codebase QnA","evaluator":"Meta","evaluation_date":"2026-09-02","source":"https://research.meta.ai/static/muse-spark-1-3-multimodal-evaluation-methodology","notes":null}
]
$benchmarks$::jsonb) AS b(
  id text, model_slug text, category text, benchmark_name text, benchmark_version text,
  score_numeric double precision, score_display text, score_unit text, tools boolean,
  reasoning_effort text, harness text, evaluator text, evaluation_date text, source text, notes text
)
ON CONFLICT (id) DO UPDATE SET
  model_slug=EXCLUDED.model_slug,
  category=EXCLUDED.category,
  benchmark_name=EXCLUDED.benchmark_name,
  benchmark_version=EXCLUDED.benchmark_version,
  score_numeric=EXCLUDED.score_numeric,
  score_display=EXCLUDED.score_display,
  score_unit=EXCLUDED.score_unit,
  tools=EXCLUDED.tools,
  reasoning_effort=EXCLUDED.reasoning_effort,
  harness=EXCLUDED.harness,
  evaluator=EXCLUDED.evaluator,
  evaluation_date=EXCLUDED.evaluation_date,
  source=EXCLUDED.source,
  notes=EXCLUDED.notes,
  updated_at=NOW();

-- Preserve authored comparison cells: only fill cells that already exist and are blank.
WITH r AS (
  SELECT b.model_slug,b.benchmark_name,b.evaluation_date,b.id,
    b.score_display || CASE WHEN concat_ws('; ',
      CASE WHEN b.benchmark_version IS NOT NULL AND lower(trim(b.benchmark_version)) <> 'public' AND position(lower(b.benchmark_version) in lower(b.benchmark_name)) = 0 THEN b.benchmark_version END,
      CASE WHEN b.tools IS TRUE THEN 'tools' WHEN b.tools IS FALSE THEN 'no tools' END,
      b.reasoning_effort
    ) <> '' THEN ' (' || concat_ws('; ',
      CASE WHEN b.benchmark_version IS NOT NULL AND lower(trim(b.benchmark_version)) <> 'public' AND position(lower(b.benchmark_version) in lower(b.benchmark_name)) = 0 THEN b.benchmark_version END,
      CASE WHEN b.tools IS TRUE THEN 'tools' WHEN b.tools IS FALSE THEN 'no tools' END,
      b.reasoning_effort
    ) || ')' ELSE '' END AS rendered
  FROM model_benchmarks b
  WHERE b.model_slug IN ('muse-spark-1-2','muse-spark-1-3')
), g AS (
  SELECT model_slug,benchmark_name,string_agg(rendered,' · ' ORDER BY evaluation_date NULLS LAST,id) AS rendered
  FROM r GROUP BY model_slug,benchmark_name
), x AS (
  SELECT c.id,
    jsonb_agg(
      CASE WHEN jsonb_typeof(be.block->'rows')='array' THEN
        jsonb_set(be.block,'{rows}',COALESCE((
          SELECT jsonb_agg(
            CASE WHEN jsonb_typeof(re.row_value)='array' AND jsonb_array_length(re.row_value)>=3 THEN
              jsonb_set(
                jsonb_set(re.row_value,'{1}',to_jsonb(CASE WHEN COALESCE(re.row_value->>1,'')='' AND a.rendered IS NOT NULL THEN a.rendered ELSE COALESCE(re.row_value->>1,'') END),false),
                '{2}',to_jsonb(CASE WHEN COALESCE(re.row_value->>2,'')='' AND d.rendered IS NOT NULL THEN d.rendered ELSE COALESCE(re.row_value->>2,'') END),false
              )
            ELSE re.row_value END ORDER BY re.ord
          )
          FROM jsonb_array_elements(be.block->'rows') WITH ORDINALITY re(row_value,ord)
          LEFT JOIN g a ON a.model_slug=c.metadata->>'modelA' AND lower(a.benchmark_name)=lower(COALESCE(re.row_value->>0,''))
          LEFT JOIN g d ON d.model_slug=c.metadata->>'modelB' AND lower(d.benchmark_name)=lower(COALESCE(re.row_value->>0,''))
        ),'[]'::jsonb),true)
      ELSE be.block END ORDER BY be.ord
    ) AS blocks
  FROM content_items c
  CROSS JOIN LATERAL jsonb_array_elements(COALESCE(c.blocks,'[]'::jsonb)) WITH ORDINALITY be(block,ord)
  WHERE c.kind='comparison'
    AND (c.metadata->>'modelA' IN ('muse-spark-1-2','muse-spark-1-3') OR c.metadata->>'modelB' IN ('muse-spark-1-2','muse-spark-1-3'))
  GROUP BY c.id,c.metadata
)
UPDATE content_items c
SET blocks=x.blocks,updated_at=NOW()
FROM x
WHERE c.id=x.id AND c.blocks IS DISTINCT FROM x.blocks;
