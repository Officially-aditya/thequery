-- Add SEC-Bench Pro leaderboard values for the GPT-6 / GPT-5.6 family so
-- database-generated (x vs y) comparisons stop showing blank SEC-Bench Pro
-- cells on the Sol / Luna side.
--
-- Source (fetched and verified):
-- - SEC-Bench Pro leaderboard
--   (https://benchlm.ai/benchmarks/secbenchpro)
--   Last updated September 22 2026, 8 models, cybersecurity benchmark for
--   agentic vulnerability analysis and exploit-oriented security tasks
--   GPT-6 Astra 85.4, GPT-5.6 Sol 71.2, GPT-6 Sol 66.3,
--   DeepSeek V4.1 Flash 62.8, GPT-5.5 45.8, GPT-6 Luna 34.2
--   plus MiMo-V2.6-Pro 66.3 and MiMo-V2-6-Flash 47.5 which have no catalog
--   rows here and are intentionally skipped
-- - About text on the same page states OpenAI reports exact model results in
--   its GPT-5.6 launch table and BenchLM stores the provider-run values as
--   display-only cyber evidence until benchmark-native results are available
--   and the benchmark is excluded from the weighted scoring formula
-- - Category coding matches the existing SEC-Bench Pro rows (040) so the new
--   rows merge into the same Coding row instead of a second section
-- - DeepSeek V4.1 Flash 62.8 already exists (040) and is left untouched
--
-- Fix: insert 5 rows for models present in the catalog with BenchLM as the
-- evaluator and the leaderboard page as the source. Scores and display values
-- are unchanged from the leaderboard. No effort or harness is stated on the
-- leaderboard so those fields stay null.

INSERT INTO model_benchmarks (
  id, model_slug, category, benchmark_name, benchmark_version,
  score_numeric, score_display, score_unit, tools, reasoning_effort,
  harness, evaluator, evaluation_date, source, notes
)
SELECT
  evidence.id,
  evidence.model_slug,
  evidence.category,
  evidence.benchmark_name,
  evidence.benchmark_version,
  evidence.score_numeric,
  evidence.score_display,
  evidence.score_unit,
  evidence.tools,
  evidence.reasoning_effort,
  evidence.harness,
  evidence.evaluator,
  evidence.evaluation_date::date,
  evidence.source,
  evidence.notes
FROM jsonb_to_recordset($benchmarks$
[
  {"id":"tq-20260922-gpt6astra-secbenchpro","model_slug":"gpt-6-astra","category":"coding","benchmark_name":"SEC-Bench Pro","benchmark_version":null,"score_numeric":85.4,"score_display":"85.4%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"BenchLM","evaluation_date":"2026-09-22","source":"https://benchlm.ai/benchmarks/secbenchpro","notes":"BenchLM SEC-Bench Pro leaderboard value, provider-run display-only cyber evidence excluded from weighted scoring"},
  {"id":"tq-20260922-gpt56sol-secbenchpro","model_slug":"gpt-5-6-sol","category":"coding","benchmark_name":"SEC-Bench Pro","benchmark_version":null,"score_numeric":71.2,"score_display":"71.2%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"BenchLM","evaluation_date":"2026-09-22","source":"https://benchlm.ai/benchmarks/secbenchpro","notes":"BenchLM SEC-Bench Pro leaderboard value, provider-run display-only cyber evidence excluded from weighted scoring"},
  {"id":"tq-20260922-gpt6sol-secbenchpro","model_slug":"gpt-6-sol","category":"coding","benchmark_name":"SEC-Bench Pro","benchmark_version":null,"score_numeric":66.3,"score_display":"66.3%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"BenchLM","evaluation_date":"2026-09-22","source":"https://benchlm.ai/benchmarks/secbenchpro","notes":"BenchLM SEC-Bench Pro leaderboard value, provider-run display-only cyber evidence excluded from weighted scoring"},
  {"id":"tq-20260922-gpt55-secbenchpro","model_slug":"gpt-5-5","category":"coding","benchmark_name":"SEC-Bench Pro","benchmark_version":null,"score_numeric":45.8,"score_display":"45.8%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"BenchLM","evaluation_date":"2026-09-22","source":"https://benchlm.ai/benchmarks/secbenchpro","notes":"BenchLM SEC-Bench Pro leaderboard value, provider-run display-only cyber evidence excluded from weighted scoring"},
  {"id":"tq-20260922-gpt6luna-secbenchpro","model_slug":"gpt-6-luna","category":"coding","benchmark_name":"SEC-Bench Pro","benchmark_version":null,"score_numeric":34.2,"score_display":"34.2%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"BenchLM","evaluation_date":"2026-09-22","source":"https://benchlm.ai/benchmarks/secbenchpro","notes":"BenchLM SEC-Bench Pro leaderboard value, provider-run display-only cyber evidence excluded from weighted scoring"}
]$benchmarks$::jsonb) AS evidence(
  id TEXT, model_slug TEXT, category TEXT, benchmark_name TEXT, benchmark_version TEXT,
  score_numeric DOUBLE PRECISION, score_display TEXT, score_unit TEXT, tools BOOLEAN,
  reasoning_effort TEXT, harness TEXT, evaluator TEXT, evaluation_date TEXT,
  source TEXT, notes TEXT
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
