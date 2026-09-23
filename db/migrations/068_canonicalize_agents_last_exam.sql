-- Canonicalize the Agents Last Exam benchmark rows for GPT-6 Sol / Luna so
-- database-generated (x vs y) comparisons render a single row under
-- Agentic & computer use instead of a second Professional section.
--
-- Root cause: the Sol/Luna rows (056/057) were seeded as
-- benchmark_name 'Agents Last Exam' (no apostrophe) with category
-- 'professional', while the rest of the catalog (015, 018, 019, 026, 028,
-- 040) uses benchmark_name 'Agents'' Last Exam' (with apostrophe) with
-- category 'agentic_computer_use'. lib/model-comparison.ts lists
-- "Agents' Last Exam" as a predefined Agentic & computer use label, so the
-- divergent Sol/Luna spelling falls through to the dynamic Professional
-- section while the other side renders under Agentic - two rows for one
-- benchmark. Same class of dedupe as DeepSWE / FrontierCode in 067.
--
-- Fix: rename the two rows to the catalog-wide label and category.
-- Scores, version (V1), tools, effort, harness, evaluator, source and notes
-- are unchanged - the rendered value stays e.g. "56.6% (V1; tools; max)"
-- but now keys to "Agents' Last Exam" and merges with the other model side.
-- The authored GPT-6 Sol vs Opus 5.5 page and glossary tables are untouched.

UPDATE model_benchmarks SET
  benchmark_name = 'Agents'' Last Exam',
  category = 'agentic_computer_use',
  updated_at = NOW()
WHERE id IN ('tq-20260922-gpt6sol-agentslastexamv1', 'tq-20260922-gpt6luna-agentslastexamv1');
