-- Add current GPT-6 Luna evidence and refresh GPT-6.1 Sol directly from
-- Artificial Analysis. Reuse the existing Sol max-effort row so the older
-- secondary transcription is replaced rather than counted as another run.
INSERT INTO model_benchmarks (
  id, model_slug, category, benchmark_name, benchmark_version,
  score_numeric, score_display, score_unit, tools, reasoning_effort,
  harness, evaluator, evaluation_date, source, notes
)
SELECT
  evidence.id, evidence.model_slug, evidence.category,
  evidence.benchmark_name, evidence.benchmark_version,
  evidence.score_numeric, evidence.score_display, evidence.score_unit,
  evidence.tools, evidence.reasoning_effort, evidence.harness,
  evidence.evaluator, evidence.evaluation_date::date, evidence.source, evidence.notes
FROM jsonb_to_recordset($benchmarks$
[
  {
    "id": "tq-20261008-gpt-6-luna-terminalbench40-aa-high",
    "model_slug": "gpt-6-luna",
    "category": "coding",
    "benchmark_name": "Terminal-Bench 4.0",
    "benchmark_version": null,
    "score_numeric": 4.545454545454549,
    "score_display": "4.5% (AA)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "high",
    "harness": "Artificial Analysis",
    "evaluator": "Artificial Analysis",
    "evaluation_date": null,
    "source": "https://artificialanalysis.ai/models/gpt-6-luna-high",
    "notes": "Current independent Artificial Analysis model-page result at high effort, observed October 8, 2026. Read directly from the public chart data, with the 0 to 1 fraction converted to percent. Tool settings and the exact evaluation date are not published."
  },
  {
    "id": "tq-20261008-gpt-6-luna-terminalbench40-aa-xhigh",
    "model_slug": "gpt-6-luna",
    "category": "coding",
    "benchmark_name": "Terminal-Bench 4.0",
    "benchmark_version": null,
    "score_numeric": 8.08080808080808,
    "score_display": "8.1% (AA)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "xhigh",
    "harness": "Artificial Analysis",
    "evaluator": "Artificial Analysis",
    "evaluation_date": null,
    "source": "https://artificialanalysis.ai/models/gpt-6-luna-xhigh",
    "notes": "Current independent Artificial Analysis model-page result at xhigh effort, observed October 8, 2026. Read directly from the public chart data, with the 0 to 1 fraction converted to percent. Tool settings and the exact evaluation date are not published."
  },
  {
    "id": "tq-20261008-gpt-6-luna-terminalbench40-aa-max",
    "model_slug": "gpt-6-luna",
    "category": "coding",
    "benchmark_name": "Terminal-Bench 4.0",
    "benchmark_version": null,
    "score_numeric": 12.6262626262626,
    "score_display": "12.6% (AA)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Artificial Analysis",
    "evaluator": "Artificial Analysis",
    "evaluation_date": null,
    "source": "https://artificialanalysis.ai/models/gpt-6-luna",
    "notes": "Current independent Artificial Analysis model-page result at max effort, observed October 8, 2026. Read directly from the public chart data, with the 0 to 1 fraction converted to percent. Tool settings and the exact evaluation date are not published."
  },
  {
    "id": "tq-20261008-gpt-6-1-sol-terminalbench40-aa-high",
    "model_slug": "gpt-6-1-sol",
    "category": "coding",
    "benchmark_name": "Terminal-Bench 4.0",
    "benchmark_version": null,
    "score_numeric": 51.5151515151515,
    "score_display": "51.5% (AA)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "high",
    "harness": "Artificial Analysis",
    "evaluator": "Artificial Analysis",
    "evaluation_date": null,
    "source": "https://artificialanalysis.ai/models/gpt-6-1-sol-high",
    "notes": "Current independent Artificial Analysis model-page result at high effort, observed October 8, 2026. Read directly from the public chart data, with the 0 to 1 fraction converted to percent. Tool settings and the exact evaluation date are not published."
  },
  {
    "id": "tq-20261008-gpt-6-1-sol-terminalbench40-aa-xhigh",
    "model_slug": "gpt-6-1-sol",
    "category": "coding",
    "benchmark_name": "Terminal-Bench 4.0",
    "benchmark_version": null,
    "score_numeric": 54.040404040404,
    "score_display": "54.0% (AA)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "xhigh",
    "harness": "Artificial Analysis",
    "evaluator": "Artificial Analysis",
    "evaluation_date": null,
    "source": "https://artificialanalysis.ai/models/gpt-6-1-sol-xhigh",
    "notes": "Current independent Artificial Analysis model-page result at xhigh effort, observed October 8, 2026. Read directly from the public chart data, with the 0 to 1 fraction converted to percent. Tool settings and the exact evaluation date are not published."
  },
  {
    "id": "tq-20260929-gpt61sol-terminalbench40-aa",
    "model_slug": "gpt-6-1-sol",
    "category": "coding",
    "benchmark_name": "Terminal-Bench 4.0",
    "benchmark_version": null,
    "score_numeric": 56.0606060606061,
    "score_display": "56.1% (AA)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Artificial Analysis",
    "evaluator": "Artificial Analysis",
    "evaluation_date": null,
    "source": "https://artificialanalysis.ai/models/gpt-6-1-sol",
    "notes": "Current independent Artificial Analysis model-page result at max effort, observed October 8, 2026. Read directly from the public chart data, with the 0 to 1 fraction converted to percent. Tool settings and the exact evaluation date are not published. Replaces the earlier secondary transcription on the same evidence row; the displayed max-effort score remains 56.1%."
  }
]
$benchmarks$::jsonb) AS evidence(
  id TEXT, model_slug TEXT, category TEXT, benchmark_name TEXT,
  benchmark_version TEXT, score_numeric DOUBLE PRECISION, score_display TEXT,
  score_unit TEXT, tools BOOLEAN, reasoning_effort TEXT, harness TEXT,
  evaluator TEXT, evaluation_date TEXT, source TEXT, notes TEXT
)
ON CONFLICT (id) DO UPDATE SET
  model_slug = EXCLUDED.model_slug,
  category = EXCLUDED.category,
  benchmark_name = EXCLUDED.benchmark_name,
  benchmark_version = EXCLUDED.benchmark_version,
  score_numeric = EXCLUDED.score_numeric,
  score_display = EXCLUDED.score_display,
  score_unit = EXCLUDED.score_unit,
  tools = EXCLUDED.tools,
  reasoning_effort = EXCLUDED.reasoning_effort,
  harness = EXCLUDED.harness,
  evaluator = EXCLUDED.evaluator,
  evaluation_date = EXCLUDED.evaluation_date,
  source = EXCLUDED.source,
  notes = EXCLUDED.notes,
  updated_at = NOW();

UPDATE models
SET verified_at = NOW(), updated_at = NOW()
WHERE slug IN ('gpt-6-luna', 'gpt-6-1-sol');
