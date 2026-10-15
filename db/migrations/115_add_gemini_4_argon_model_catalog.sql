-- Add Gemini 4 Argon to the model catalog so the generated comparison pages
-- (including gemini-4-argon-vs-opus-5-5) can resolve, following the pattern
-- set by 094 (MiMo V2.6) and 102 (Claude Haiku 5.5). Migration 096 published
-- the Gemini 4 Argon glossary page from the same sources.
--
-- Only the specifications Google and its independent evaluators published at
-- launch are seeded here. Values that were not published (API model ID, price,
-- knowledge cutoff, tool calling, computer use, most benchmark cells) are
-- recorded as not published rather than inferred. Independent benchmark figures
-- are limited to the two normalized scores that map onto existing catalog
-- benchmark labels.

INSERT INTO models (
  slug, name, developer, release_date, ga_date, access, family,
  comparison_data, sources, notes, metadata, verified_at
)
SELECT
  'gemini-4-argon',
  'Gemini 4 Argon',
  'Google DeepMind',
  DATE '2026-09-30',
  NULL,
  'restricted',
  'Gemini 4',
  '{
    "Developer": "Google DeepMind",
    "Release date": "2026-09-30 (announced)",
    "API model ID": "Not published at launch",
    "Context window": "1,000,000 tokens (Artificial Analysis and Vals AI)",
    "Max output": "1,000,000 tokens (Google launch post)",
    "Knowledge cutoff": "Not published at launch",
    "Reasoning / effort": "Highest thinking setting (High)",
    "Input / 1M tokens": "Not published at launch",
    "Cached input / 1M": "Not published at launch",
    "Cache write / 1M": "Not published at launch",
    "Output / 1M tokens": "Not published at launch",
    "Batch / flex discount": "Not published at launch",
    "Long-context surcharge": "Not published at launch",
    "Text input": "Yes",
    "Image / vision input": "Yes",
    "Audio input": "Yes",
    "Video input": "Yes",
    "File / document input": "Yes",
    "Text output": "Yes",
    "Image output": "No",
    "Audio output": "No",
    "Video output": "No",
    "Tool / function calling": "Not published at launch",
    "Computer use": "Not published at launch",
    "API access": "Not publicly available at launch. Rolling out to trusted cyber defenders through Google Fairwind and to internal teams; broader API access not yet available",
    "Product access": "Not generally available. Google says availability for developers, enterprises and consumers will follow, starting with paid API customers and Google AI Ultra subscribers",
    "Weights / license": "Proprietary",
    "Primary focus": "Long-horizon software engineering, enterprise knowledge work (legal, finance, tax), cybersecurity defense, and long-context analysis",
    "Safety / approvals": "Limited release only. No public model card at launch; Google is participating in the US government voluntary pre-release access process"
  }'::jsonb,
  '[
    {"title": "Introducing Gemini 4 Argon", "url": "https://blog.google/technology/ai/gemini-4-argon/"},
    {"title": "TheQuery: Gemini 4 Argon model card", "url": "https://www.thequery.in/glossary/gemini-4-argon"}
  ]'::jsonb,
  'Announced September 30, 2026 by Google DeepMind. Restricted availability at launch: trusted cyber defenders through Google Fairwind and Google internal teams, both without cyber guardrails. No public API model ID, price, knowledge cutoff or full model card was published at launch. Context and output limits are Google-reported or evaluator-reported. Only two normalized benchmark rows are seeded here because Google did not publish values that map onto the catalog''s existing benchmark labels beyond DeepSWE v1.1 and Terminal-Bench 4.0.',
  '{
    "verification_pass": "gemini-4-argon-catalog-2026-09-30",
    "catalog_status": "current",
    "glossary_slug": "gemini-4-argon",
    "availability": "restricted",
    "intelligence_index": 53,
    "intelligence_index_version": "v4.3.2",
    "artificial_analysis_output_tokens_per_second": null
  }'::jsonb,
  NOW()
ON CONFLICT (slug) DO UPDATE SET
  name = EXCLUDED.name,
  developer = EXCLUDED.developer,
  release_date = EXCLUDED.release_date,
  ga_date = EXCLUDED.ga_date,
  access = EXCLUDED.access,
  family = EXCLUDED.family,
  comparison_data = EXCLUDED.comparison_data,
  sources = EXCLUDED.sources,
  notes = EXCLUDED.notes,
  metadata = EXCLUDED.metadata,
  verified_at = EXCLUDED.verified_at;

-- Normalized benchmark rows that map onto existing catalog labels. Values are
-- Google-reported launch figures (pass@1 at the highest thinking setting).
INSERT INTO model_benchmarks (
  id, model_slug, category, benchmark_name, benchmark_version,
  score_numeric, score_display, score_unit, tools, reasoning_effort,
  harness, evaluator, evaluation_date, source, notes
) VALUES
  ('tq-20260930-gemini4argon-deepswe11', 'gemini-4-argon', 'coding', 'DeepSWE v1.1', NULL,
    77.9, '77.9%', 'percent', NULL, 'high', NULL, 'Google', DATE '2026-09-30',
    'https://blog.google/technology/ai/gemini-4-argon/', 'Google-reported pass@1 at the highest thinking setting. Computed by Google with a mini-swe-agent harness'),
  ('tq-20260930-gemini4argon-terminalbench40', 'gemini-4-argon', 'coding', 'Terminal-Bench 4.0', '4.0',
    57.4, '57.4%', 'percent', NULL, 'high', NULL, 'Google', DATE '2026-09-30',
    'https://blog.google/technology/ai/gemini-4-argon/', 'Google-reported result at the highest thinking setting')
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
  notes = EXCLUDED.notes;
