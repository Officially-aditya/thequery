-- Add DeepSeek V4.1 Flash: model catalog entry, benchmark evidence,
-- the authored DeepSeek V4 Flash vs DeepSeek V4.1 Flash comparison page,
-- and the DeepSeek V4.1 Flash glossary entry.

-- 1. Model catalog entry for DeepSeek V4.1 Flash.
INSERT INTO models (
  slug, name, developer, release_date, ga_date, access, family,
  comparison_data, sources, notes, metadata, verified_at
)
SELECT
  'deepseek-v4-1-flash',
  'DeepSeek V4.1 Flash',
  'DeepSeek',
  DATE '2026-09-10',
  DATE '2026-09-10',
  'open_weights',
  'DeepSeek V4',
  '{
    "Developer": "DeepSeek",
    "Release date": "2026-09-10",
    "API model ID": "deepseek-flash",
    "Context window": "1M tokens",
    "Max output": "384K tokens",
    "Reasoning / effort": "Non-thinking + thinking (default); reasoning_effort 1-100",
    "Input / 1M tokens": "$0.15 off-peak / $0.30 peak",
    "Cached input / 1M": "$0.003 off-peak / $0.006 peak",
    "Output / 1M tokens": "$0.60 off-peak / $1.20 peak",
    "Text input": "Yes",
    "Image / vision input": "Yes - native vision encoder",
    "Audio input": "No",
    "Video input": "No",
    "Text output": "Yes",
    "Image output": "No",
    "Audio output": "No",
    "Video output": "No",
    "Tool / function calling": "Yes",
    "Computer use": "Via DeepSeek Harness (developer preview)",
    "API access": "Yes - OpenAI-compatible, Responses API, Anthropic-compatible API",
    "Product access": "DeepSeek app, web, and API + downloadable weights",
    "Weights / license": "Open weights - MIT"
  }'::jsonb,
  '[
    {"title": "DeepSeek-V4.1-Flash: Smarter, Faster, More Efficient", "url": "https://api-docs.deepseek.com/news/news260910"},
    {"title": "DeepSeek-V4.1-Flash Model Card and Weights", "url": "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash"},
    {"title": "DeepSeek API Models & Pricing", "url": "https://api-docs.deepseek.com/quick_start/pricing"},
    {"title": "DeepSeek API Docs - Your First API Call", "url": "https://api-docs.deepseek.com/"}
  ]'::jsonb,
  '552B MoE with CED architecture: 8B active prefill / 16B decode, 890 bytes/token global KV cache (about 1/4 of V4-Flash), FP4 main KV caching, Engram conditional memory, DSpark speculative decoding. Replaces retired V4-Flash and V4-Flash-Vision-Exp; deepseek-v4-pro requests route to V4.1-Flash from Sept 14, 2026.',
  '{
    "verification_pass": "deepseek-v41-flash-2026-09-15",
    "parameters_backbone": "552B",
    "parameters_active_prefill": "8B",
    "parameters_active_decode": "16B",
    "architecture": "Causal Encoder-Decoder MoE with CSA2",
    "kv_cache_bytes_per_token": 890,
    "license": "MIT",
    "concurrency_limit": 2500
  }'::jsonb,
  NOW()
WHERE NOT EXISTS (SELECT 1 FROM models WHERE slug = 'deepseek-v4-1-flash')
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
  metadata = COALESCE(models.metadata, '{}'::jsonb) || EXCLUDED.metadata,
  verified_at = NOW(),
  updated_at = NOW();

-- 2. Mark DeepSeek V4 Flash as retired in the catalog notes.
UPDATE models
SET
  notes = 'Retired September 10, 2026. DeepSeek routes deepseek-v4-flash requests to DeepSeek-V4.1-Flash, billed at V4.1-Flash rates.',
  updated_at = NOW()
WHERE slug = 'deepseek-v4-flash';

-- 3. Benchmark evidence from DeepSeek's V4.1-Flash model card (Sept 10, 2026).
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
  {"id":"deepseek-v4-1-flash-terminal-bench-2-1-90-6-max-deepseek","model_slug":"deepseek-v4-1-flash","category":"coding","benchmark_name":"Terminal-Bench","benchmark_version":"2.1","score_numeric":90.6,"score_display":"90.6","score_unit":"%","tools":null,"reasoning_effort":"max","harness":"DeepSeek Harness minimal mode","evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":null},
  {"id":"deepseek-v4-1-flash-terminal-bench-3-0-30-0-max-deepseek","model_slug":"deepseek-v4-1-flash","category":"coding","benchmark_name":"Terminal-Bench","benchmark_version":"3.0","score_numeric":30.0,"score_display":"30.0","score_unit":"%","tools":null,"reasoning_effort":"max","harness":"DeepSeek Harness minimal mode","evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":null},
  {"id":"deepseek-v4-1-flash-terminal-bench-4-0-31-2-max-deepseek","model_slug":"deepseek-v4-1-flash","category":"coding","benchmark_name":"Terminal-Bench","benchmark_version":"4.0","score_numeric":31.2,"score_display":"31.2","score_unit":"%","tools":null,"reasoning_effort":"max","harness":"DeepSeek Harness minimal mode","evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":null},
  {"id":"deepseek-v4-1-flash-deepswe-v1-1-74-2-max-deepseek","model_slug":"deepseek-v4-1-flash","category":"coding","benchmark_name":"DeepSWE","benchmark_version":"v1.1","score_numeric":74.2,"score_display":"74.2","score_unit":"%","tools":null,"reasoning_effort":"max","harness":"mini-SWE","evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":"DSH Minimal run scores 72.6."},
  {"id":"deepseek-v4-1-flash-nl2repo-bench-64-0-max-deepseek","model_slug":"deepseek-v4-1-flash","category":"coding","benchmark_name":"NL2Repo-Bench","benchmark_version":null,"score_numeric":64.0,"score_display":"64.0","score_unit":"%","tools":null,"reasoning_effort":"max","harness":"DeepSeek Harness minimal mode","evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":null},
  {"id":"deepseek-v4-1-flash-cybergym-88-1-max-deepseek","model_slug":"deepseek-v4-1-flash","category":"coding","benchmark_name":"CyberGym","benchmark_version":null,"score_numeric":88.1,"score_display":"88.1","score_unit":"%","tools":null,"reasoning_effort":"max","harness":"DeepSeek Harness minimal mode","evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":null},
  {"id":"deepseek-v4-1-flash-sec-bench-pro-62-8-max-deepseek","model_slug":"deepseek-v4-1-flash","category":"coding","benchmark_name":"SEC-Bench Pro","benchmark_version":null,"score_numeric":62.8,"score_display":"62.8","score_unit":"%","tools":null,"reasoning_effort":"max","harness":"Claude Code","evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":null},
  {"id":"deepseek-v4-1-flash-codeforces-3471-max-deepseek","model_slug":"deepseek-v4-1-flash","category":"math_reasoning","benchmark_name":"Codeforces","benchmark_version":null,"score_numeric":3471.0,"score_display":"3471","score_unit":"Elo","tools":null,"reasoning_effort":"max","harness":null,"evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":null},
  {"id":"deepseek-v4-1-flash-matharena-apex-65-6-max-deepseek","model_slug":"deepseek-v4-1-flash","category":"math_reasoning","benchmark_name":"MathArena Apex","benchmark_version":null,"score_numeric":65.6,"score_display":"65.6","score_unit":"%","tools":null,"reasoning_effort":"max","harness":null,"evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":null},
  {"id":"deepseek-v4-1-flash-gpqa-diamond-90-9-max-deepseek","model_slug":"deepseek-v4-1-flash","category":"knowledge","benchmark_name":"GPQA Diamond","benchmark_version":null,"score_numeric":90.9,"score_display":"90.9","score_unit":"%","tools":null,"reasoning_effort":"max","harness":null,"evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":null},
  {"id":"deepseek-v4-1-flash-hle-36-8-max-no-tools-deepseek","model_slug":"deepseek-v4-1-flash","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":"text-only subset","score_numeric":36.8,"score_display":"36.8","score_unit":"%","tools":false,"reasoning_effort":"max","harness":null,"evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":null},
  {"id":"deepseek-v4-1-flash-hle-39-1-max-tools-deepseek","model_slug":"deepseek-v4-1-flash","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":"text-only subset","score_numeric":39.1,"score_display":"39.1","score_unit":"%","tools":true,"reasoning_effort":"max","harness":null,"evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":null},
  {"id":"deepseek-v4-1-flash-automationbench-54-8-max-deepseek","model_slug":"deepseek-v4-1-flash","category":"agentic_computer_use","benchmark_name":"AutomationBench","benchmark_version":null,"score_numeric":54.8,"score_display":"54.8","score_unit":"%","tools":true,"reasoning_effort":"max","harness":null,"evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":null},
  {"id":"deepseek-v4-1-flash-agents-last-exam-31-8-max-deepseek","model_slug":"deepseek-v4-1-flash","category":"agentic_computer_use","benchmark_name":"Agents' Last Exam","benchmark_version":null,"score_numeric":31.8,"score_display":"31.8","score_unit":"%","tools":true,"reasoning_effort":"max","harness":null,"evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":null},
  {"id":"deepseek-v4-1-flash-chartography-78-9-max-tools-deepseek","model_slug":"deepseek-v4-1-flash","category":"multimodal","benchmark_name":"Chartography","benchmark_version":null,"score_numeric":78.9,"score_display":"78.9","score_unit":"%","tools":true,"reasoning_effort":"max","harness":"Claude Code","evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":"Visual agent benchmark with tools."},
  {"id":"deepseek-v4-1-flash-babyvision-89-6-max-tools-deepseek","model_slug":"deepseek-v4-1-flash","category":"multimodal","benchmark_name":"BabyVision","benchmark_version":null,"score_numeric":89.6,"score_display":"89.6","score_unit":"%","tools":true,"reasoning_effort":"max","harness":"Claude Code","evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":"Visual agent benchmark with tools."},
  {"id":"deepseek-v4-1-flash-zerobench-49-0-max-tools-deepseek","model_slug":"deepseek-v4-1-flash","category":"multimodal","benchmark_name":"ZeroBench","benchmark_version":"main Pass@5","score_numeric":49.0,"score_display":"49.0","score_unit":"%","tools":true,"reasoning_effort":"max","harness":"Claude Code","evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":"Visual agent benchmark with tools."},
  {"id":"deepseek-v4-flash-gpqa-diamond-89-9-v41-card-deepseek","model_slug":"deepseek-v4-flash","category":"knowledge","benchmark_name":"GPQA Diamond","benchmark_version":null,"score_numeric":89.9,"score_display":"89.9","score_unit":"%","tools":null,"reasoning_effort":"max","harness":null,"evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":"DS-V4-Flash comparison value in DeepSeek's V4.1-Flash model card."},
  {"id":"deepseek-v4-flash-codeforces-3289-v41-card-deepseek","model_slug":"deepseek-v4-flash","category":"math_reasoning","benchmark_name":"Codeforces","benchmark_version":null,"score_numeric":3289.0,"score_display":"3289","score_unit":"Elo","tools":null,"reasoning_effort":"max","harness":null,"evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":"DS-V4-Flash comparison value in DeepSeek's V4.1-Flash model card."},
  {"id":"deepseek-v4-flash-matharena-apex-58-6-v41-card-deepseek","model_slug":"deepseek-v4-flash","category":"math_reasoning","benchmark_name":"MathArena Apex","benchmark_version":null,"score_numeric":58.6,"score_display":"58.6","score_unit":"%","tools":null,"reasoning_effort":"max","harness":null,"evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":"DS-V4-Flash comparison value in DeepSeek's V4.1-Flash model card."},
  {"id":"deepseek-v4-flash-terminal-bench-3-0-7-6-v41-card-deepseek","model_slug":"deepseek-v4-flash","category":"coding","benchmark_name":"Terminal-Bench","benchmark_version":"3.0","score_numeric":7.6,"score_display":"7.6","score_unit":"%","tools":null,"reasoning_effort":"max","harness":"DeepSeek Harness minimal mode","evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":"DS-V4-Flash comparison value in DeepSeek's V4.1-Flash model card."},
  {"id":"deepseek-v4-flash-terminal-bench-4-0-7-0-v41-card-deepseek","model_slug":"deepseek-v4-flash","category":"coding","benchmark_name":"Terminal-Bench","benchmark_version":"4.0","score_numeric":7.0,"score_display":"7.0","score_unit":"%","tools":null,"reasoning_effort":"max","harness":"DeepSeek Harness minimal mode","evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":"DS-V4-Flash comparison value in DeepSeek's V4.1-Flash model card."},
  {"id":"deepseek-v4-flash-deepswe-v1-1-54-4-v41-card-deepseek","model_slug":"deepseek-v4-flash","category":"coding","benchmark_name":"DeepSWE","benchmark_version":"v1.1","score_numeric":54.4,"score_display":"54.4","score_unit":"%","tools":null,"reasoning_effort":"max","harness":"mini-SWE","evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":"DS-V4-Flash comparison value in DeepSeek's V4.1-Flash model card."},
  {"id":"deepseek-v4-flash-cybergym-76-7-v41-card-deepseek","model_slug":"deepseek-v4-flash","category":"coding","benchmark_name":"CyberGym","benchmark_version":null,"score_numeric":76.7,"score_display":"76.7","score_unit":"%","tools":null,"reasoning_effort":"max","harness":"DeepSeek Harness minimal mode","evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":"DS-V4-Flash comparison value in DeepSeek's V4.1-Flash model card."},
  {"id":"deepseek-v4-flash-sec-bench-pro-30-9-v41-card-deepseek","model_slug":"deepseek-v4-flash","category":"coding","benchmark_name":"SEC-Bench Pro","benchmark_version":null,"score_numeric":30.9,"score_display":"30.9","score_unit":"%","tools":null,"reasoning_effort":"max","harness":"Claude Code","evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":"DS-V4-Flash comparison value in DeepSeek's V4.1-Flash model card."},
  {"id":"deepseek-v4-flash-nl2repo-bench-54-2-v41-card-deepseek","model_slug":"deepseek-v4-flash","category":"coding","benchmark_name":"NL2Repo-Bench","benchmark_version":null,"score_numeric":54.2,"score_display":"54.2","score_unit":"%","tools":null,"reasoning_effort":"max","harness":"DeepSeek Harness minimal mode","evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":"DS-V4-Flash comparison value in DeepSeek's V4.1-Flash model card."},
  {"id":"deepseek-v4-flash-hle-37-8-max-no-tools-v41-card-deepseek","model_slug":"deepseek-v4-flash","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":"text-only subset","score_numeric":37.8,"score_display":"37.8","score_unit":"%","tools":false,"reasoning_effort":"max","harness":null,"evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":"DS-V4-Flash comparison value in DeepSeek's V4.1-Flash model card."},
  {"id":"deepseek-v4-flash-automationbench-37-7-v41-card-deepseek","model_slug":"deepseek-v4-flash","category":"agentic_computer_use","benchmark_name":"AutomationBench","benchmark_version":null,"score_numeric":37.7,"score_display":"37.7","score_unit":"%","tools":true,"reasoning_effort":"max","harness":null,"evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":"DS-V4-Flash comparison value in DeepSeek's V4.1-Flash model card."},
  {"id":"deepseek-v4-flash-agents-last-exam-25-2-v41-card-deepseek","model_slug":"deepseek-v4-flash","category":"agentic_computer_use","benchmark_name":"Agents' Last Exam","benchmark_version":null,"score_numeric":25.2,"score_display":"25.2","score_unit":"%","tools":true,"reasoning_effort":"max","harness":null,"evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":"DS-V4-Flash comparison value in DeepSeek's V4.1-Flash model card."},
  {"id":"deepseek-v4-pro-gpqa-diamond-92-4-v41-card-deepseek","model_slug":"deepseek-v4-pro","category":"knowledge","benchmark_name":"GPQA Diamond","benchmark_version":null,"score_numeric":92.4,"score_display":"92.4","score_unit":"%","tools":null,"reasoning_effort":"max","harness":null,"evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":"DS-V4-Pro comparison value in DeepSeek's V4.1-Flash model card."},
  {"id":"deepseek-v4-pro-codeforces-3348-v41-card-deepseek","model_slug":"deepseek-v4-pro","category":"math_reasoning","benchmark_name":"Codeforces","benchmark_version":null,"score_numeric":3348.0,"score_display":"3348","score_unit":"Elo","tools":null,"reasoning_effort":"max","harness":null,"evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":"DS-V4-Pro comparison value in DeepSeek's V4.1-Flash model card."},
  {"id":"deepseek-v4-pro-matharena-apex-65-3-v41-card-deepseek","model_slug":"deepseek-v4-pro","category":"math_reasoning","benchmark_name":"MathArena Apex","benchmark_version":null,"score_numeric":65.3,"score_display":"65.3","score_unit":"%","tools":null,"reasoning_effort":"max","harness":null,"evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":"DS-V4-Pro comparison value in DeepSeek's V4.1-Flash model card."},
  {"id":"deepseek-v4-pro-terminal-bench-3-0-11-8-v41-card-deepseek","model_slug":"deepseek-v4-pro","category":"coding","benchmark_name":"Terminal-Bench","benchmark_version":"3.0","score_numeric":11.8,"score_display":"11.8","score_unit":"%","tools":null,"reasoning_effort":"max","harness":"DeepSeek Harness minimal mode","evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":"DS-V4-Pro comparison value in DeepSeek's V4.1-Flash model card."},
  {"id":"deepseek-v4-pro-terminal-bench-4-0-12-4-v41-card-deepseek","model_slug":"deepseek-v4-pro","category":"coding","benchmark_name":"Terminal-Bench","benchmark_version":"4.0","score_numeric":12.4,"score_display":"12.4","score_unit":"%","tools":null,"reasoning_effort":"max","harness":"DeepSeek Harness minimal mode","evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":"DS-V4-Pro comparison value in DeepSeek's V4.1-Flash model card."},
  {"id":"deepseek-v4-pro-deepswe-v1-1-62-7-v41-card-deepseek","model_slug":"deepseek-v4-pro","category":"coding","benchmark_name":"DeepSWE","benchmark_version":"v1.1","score_numeric":62.7,"score_display":"62.7","score_unit":"%","tools":null,"reasoning_effort":"max","harness":"mini-SWE","evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":"DS-V4-Pro comparison value in DeepSeek's V4.1-Flash model card."},
  {"id":"deepseek-v4-pro-cybergym-83-3-v41-card-deepseek","model_slug":"deepseek-v4-pro","category":"coding","benchmark_name":"CyberGym","benchmark_version":null,"score_numeric":83.3,"score_display":"83.3","score_unit":"%","tools":null,"reasoning_effort":"max","harness":"DeepSeek Harness minimal mode","evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":"DS-V4-Pro comparison value in DeepSeek's V4.1-Flash model card."},
  {"id":"deepseek-v4-pro-sec-bench-pro-56-4-v41-card-deepseek","model_slug":"deepseek-v4-pro","category":"coding","benchmark_name":"SEC-Bench Pro","benchmark_version":null,"score_numeric":56.4,"score_display":"56.4","score_unit":"%","tools":null,"reasoning_effort":"max","harness":"Claude Code","evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":"DS-V4-Pro comparison value in DeepSeek's V4.1-Flash model card."},
  {"id":"deepseek-v4-pro-nl2repo-bench-61-5-v41-card-deepseek","model_slug":"deepseek-v4-pro","category":"coding","benchmark_name":"NL2Repo-Bench","benchmark_version":null,"score_numeric":61.5,"score_display":"61.5","score_unit":"%","tools":null,"reasoning_effort":"max","harness":"DeepSeek Harness minimal mode","evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":"DS-V4-Pro comparison value in DeepSeek's V4.1-Flash model card."},
  {"id":"deepseek-v4-pro-automationbench-43-2-v41-card-deepseek","model_slug":"deepseek-v4-pro","category":"agentic_computer_use","benchmark_name":"AutomationBench","benchmark_version":null,"score_numeric":43.2,"score_display":"43.2","score_unit":"%","tools":true,"reasoning_effort":"max","harness":null,"evaluator":"DeepSeek","evaluation_date":"2026-09-10","source":"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash","notes":"DS-V4-Pro comparison value in DeepSeek's V4.1-Flash model card. The earlier V4 preview value was 31.8 under a different run."}
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

-- 4. Authored comparison page: DeepSeek V4 Flash vs DeepSeek V4.1 Flash.
WITH comparison_notes AS (
  SELECT $notes$
<small>*Scores are vendor-reported from DeepSeek's V4.1-Flash model card (September 10, 2026) at maximum reasoning effort (reasoning_effort=100). Agentic rows use DeepSeek Harness minimal mode, mini-SWE for DeepSWE v1.1, and the Claude Code harness for SEC-Bench Pro. Prices are list rates; off-peak is 50% of peak (peak hours 01:00-04:00 and 06:00-10:00 UTC, Monday-Friday).*</small>

Retired September 10, 2026 - legacy `deepseek-v4-flash` requests now route to V4.1-Flash at V4.1-Flash rates, so the left column describes what shipped until that date.

Same 1M / 384K windows and API feature set - the structural changes are active parameters (8B prefill / 16B decode vs 13B), KV cache size, and price.

Cache-hit reads drop to $0.003 and cache-miss input to $0.15 off-peak - roughly half the final V4-Flash rates - and cache economics dominate long agent sessions.

The clearest same-scorecard sweep: DeepSWE v1.1 74.2 vs 54.4, Terminal-Bench 2.1 90.6 vs 82.7, Codeforces 3471 vs 3289, AutomationBench 54.8 vs 37.7.

The one regression: no-tools text-only HLE slips from 37.8% to 36.8%, and Terminal-Bench 3.0/4.0 stay far behind Opus 5.0.

Vision moves from a separate experimental model to native image input on the same endpoint.
$notes$::text AS notes
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'comparison:deepseek-v4-flash-vs-deepseek-v4-1-flash',
  'comparison',
  'deepseek-v4-flash-vs-deepseek-v4-1-flash',
  '',
  'comparisons/deepseek-v4-flash-vs-deepseek-v4-1-flash',
  'DeepSeek V4 Flash vs DeepSeek V4.1 Flash',
  'V4.1-Flash replaced and retired V4-Flash on September 10, 2026: what the new 8B/16B CED architecture gains on benchmarks, price, and KV cache, and where the old model still holds up.',
  comparison_notes.notes,
  '[
    {"id": "spec-deepseek-flash-v41-1", "type": "spec_table", "title": "Specifications",
     "columns": ["DeepSeek V4 Flash", "DeepSeek V4.1 Flash"],
     "rows": [
       ["Developer", "DeepSeek", "DeepSeek"],
       ["Release date", "2026-07-31", "2026-09-10"],
       ["API model ID", "deepseek-v4-flash (retired; routes to V4.1-Flash)", "deepseek-flash"],
       ["Context window", "1M tokens", "1M tokens"],
       ["Max output", "384K tokens", "384K tokens"],
       ["Reasoning / effort", "Non-think / Think High / Think Max", "Non-thinking + thinking (default); reasoning_effort 1-100"]
     ]},
    {"id": "spec-deepseek-flash-v41-2", "type": "spec_table", "title": "Pricing",
     "columns": ["DeepSeek V4 Flash", "DeepSeek V4.1 Flash"],
     "rows": [
       ["Input / 1M tokens", "$0.22 off-peak / $0.44 peak", "$0.15 off-peak / $0.30 peak"],
       ["Cached input / 1M", "$0.007 off-peak / $0.014 peak", "$0.003 off-peak / $0.006 peak"],
       ["Output / 1M tokens", "$0.66 off-peak / $1.32 peak", "$0.60 off-peak / $1.20 peak"]
     ]},
    {"id": "spec-deepseek-flash-v41-3", "type": "spec_table", "title": "Capabilities & access",
     "columns": ["DeepSeek V4 Flash", "DeepSeek V4.1 Flash"],
     "rows": [
       ["Text input", "Yes", "Yes"],
       ["Image / vision input", "Separate experimental model (V4-Flash-Vision-Exp)", "Yes - native vision encoder"],
       ["Audio input", "No", "No"],
       ["Video input", "No", "No"],
       ["Text output", "Yes", "Yes"],
       ["Image output", "No", "No"],
       ["Audio output", "No", "No"],
       ["Video output", "No", "No"],
       ["Tool / function calling", "Yes", "Yes"],
       ["Computer use", "Via DeepSeek Harness", "Via DeepSeek Harness (developer preview)"],
       ["API access", "OpenAI-compatible + Anthropic-compatible + Responses API", "OpenAI-compatible + Anthropic-compatible + Responses API"],
       ["Product access", "DeepSeek app, web, API + downloadable weights", "DeepSeek app, web, API + downloadable weights"],
       ["Weights / license", "Open weights - MIT (retired)", "Open weights - MIT"]
     ]},
    {"id": "spec-deepseek-flash-v41-4", "type": "spec_table", "title": "Coding",
     "columns": ["DeepSeek V4 Flash", "DeepSeek V4.1 Flash"],
     "rows": [
       ["DeepSWE v1.1", "54.4%", "**74.2%**"],
       ["Terminal-Bench 2.1", "82.7%", "**90.6%**"],
       ["Terminal-Bench 3.0", "7.6%", "**30.0%**"],
       ["Terminal-Bench 4.0", "7.0%", "**31.2%**"],
       ["NL2Repo-Bench", "54.2%", "**64.0%**"],
       ["CyberGym", "76.7%", "**88.1%**"],
       ["SEC-Bench Pro", "30.9%", "**62.8%**"]
     ]},
    {"id": "spec-deepseek-flash-v41-5", "type": "spec_table", "title": "Math & reasoning",
     "columns": ["DeepSeek V4 Flash", "DeepSeek V4.1 Flash"],
     "rows": [
       ["Codeforces rating", "3289", "**3471**"],
       ["MathArena Apex", "58.6%", "**65.6%**"]
     ]},
    {"id": "spec-deepseek-flash-v41-6", "type": "spec_table", "title": "Knowledge",
     "columns": ["DeepSeek V4 Flash", "DeepSeek V4.1 Flash"],
     "rows": [
       ["GPQA Diamond", "89.9%", "**90.9%**"],
       ["Humanity''s Last Exam", "**37.8%** (text-only, no tools)", "36.8% no tools / 39.1% with tools (text-only)"]
     ]},
    {"id": "spec-deepseek-flash-v41-7", "type": "spec_table", "title": "Agentic & computer use",
     "columns": ["DeepSeek V4 Flash", "DeepSeek V4.1 Flash"],
     "rows": [
       ["AutomationBench", "37.7%", "**54.8%**"],
       ["Agents'' Last Exam", "25.2%", "**31.8%**"],
       ["Toolathlon", "70.3%", ""]
     ]},
    {"id": "markdown-deepseek-flash-v41-bottom-line", "type": "markdown",
     "content": "## Bottom line\n\nV4.1-Flash replaces V4-Flash outright: cheaper at both cache hit and miss, roughly a quarter of the KV cache, native vision, and same-scorecard wins on most agentic benchmarks. V4-Flash remains the record of what DeepSeek retired on September 10, 2026."}
  ]'::jsonb,
  jsonb_build_array(
    jsonb_build_object('title', 'DeepSeek-V4.1-Flash: Smarter, Faster, More Efficient', 'url', 'https://api-docs.deepseek.com/news/news260910'),
    jsonb_build_object('title', 'DeepSeek-V4.1-Flash Model Card and Weights', 'url', 'https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash'),
    jsonb_build_object('title', 'DeepSeek API Models & Pricing', 'url', 'https://api-docs.deepseek.com/quick_start/pricing'),
    jsonb_build_object('title', 'DeepSeek-V4-Flash-Vision-Exp Release 2026/08/21', 'url', 'https://api-docs.deepseek.com/news/news260821'),
    jsonb_build_object('title', 'DeepSeek API Change Log', 'url', 'https://api-docs.deepseek.com/updates/')
  ),
  jsonb_build_object(
    'modelA', 'deepseek-v4-flash',
    'modelB', 'deepseek-v4-1-flash',
    'verification_pass', 'deepseek-v41-flash-2026-09-15'
  ),
  NULL,
  NULL,
  'published',
  DATE '2026-09-10',
  0
FROM comparison_notes
WHERE true
ON CONFLICT (kind, slug, parent_slug) DO UPDATE SET
  path = EXCLUDED.path,
  title = EXCLUDED.title,
  summary = EXCLUDED.summary,
  body = EXCLUDED.body,
  blocks = EXCLUDED.blocks,
  sources = EXCLUDED.sources,
  metadata = EXCLUDED.metadata,
  cover_image_url = EXCLUDED.cover_image_url,
  cover_image_alt = EXCLUDED.cover_image_alt,
  status = EXCLUDED.status,
  published_at = EXCLUDED.published_at,
  sort_order = EXCLUDED.sort_order,
  updated_at = NOW();

-- 5. Glossary entry for DeepSeek V4.1 Flash.
WITH ds41_entry AS (
  SELECT $ds41$
DeepSeek V4.1 Flash is DeepSeek's 552B-parameter multimodal Mixture-of-Experts model and the smallest member of its new Causal Encoder-Decoder architecture family. DeepSeek released it on September 10, 2026, as the API model `deepseek-flash`, replacing the retired [DeepSeek V4 Flash](/glossary/deepseek-v4-flash) and V4 Flash Vision Exp builds while keeping a 1 million token [context window](/glossary/context-window) and up to 384K output tokens.

For a normal user, V4.1 Flash is now the only Flash endpoint on the DeepSeek API: legacy names such as `deepseek-v4-flash` still work, but their requests are served by V4.1-Flash and billed at V4.1-Flash rates. For a developer, it is an open-weight, MIT-licensed model with native image understanding, tool calling, JSON output, OpenAI-compatible Responses API support, an Anthropic-compatible endpoint, and a continuously controllable reasoning-effort setting from 1 to 100.

## Key specifications

| Specification | DeepSeek V4.1 Flash |
| --- | --- |
| Model version | DeepSeek-V4.1-Flash |
| API model ID | `deepseek-flash` |
| Release date | September 10, 2026 |
| Backbone parameters | 552B (MoE) |
| Active parameters | 8B per token at prefill / 16B at decode |
| Architecture | Causal Encoder-Decoder with CSA2 sparse attention |
| Context / maximum output | 1M / 384K tokens |
| Reasoning effort | Non-thinking + thinking (default); continuous 1-100 |
| Inputs | Text and images (native vision encoder) |
| API features | JSON, tools, Responses API, Anthropic API, prefix completion, FIM (non-thinking only) |
| KV cache | 890 bytes per token, about 1/4 of V4-Flash |
| Concurrency limit | 2500 |
| License | MIT |

The 552B backbone includes a 196B-parameter Engram conditional-memory component that is sparsely accessed through token-based lookup, so the active-parameter figure is the better clue to per-token compute. Open weights do not mean lightweight deployment: the full checkpoint still requires serious multi-GPU infrastructure or a hosted inference provider.

## Architecture

The signature change is the **Causal Encoder-Decoder (CED)**: a 40-layer Transformer organized as a 20-layer causal encoder followed by a 20-layer decoder. The decoder's global KV cache is projected from the final encoder hidden states rather than derived from each decoder layer's own hidden states. That is what lets the model activate only 8B parameters per token during prefill and 16B during decode, which matters most for input-heavy agentic workloads where documents and repositories dominate the prompt.

The model uses one shared expert and 384 routed experts per MoE layer, activating 6 routed experts per token. Attention uses **CSA2**, which assigns each layer one of three static modes - Full, Reindex, or Reuse - to share main KV and indexer keys across layers. A Hierarchical Sparse Indexer bounds deeper indexer cost independently of context length, and **SWA Bounded Replay** reconstructs missing sliding-window KV states by replaying only the most recent tokens instead of persisting them to SSD.

Additional components include Single-Pass mHC for residual-stream mixing, Engram conditional memory, and DSpark [speculative decoding](/glossary/speculative-decoding) with confidence-scheduled verification. A DeepSeek-ViT vision encoder and a two-layer MLP projector convert images into visual embeddings that are processed jointly with text from the start of pre-training, which is why image understanding is native rather than bolted on.

The model was trained from scratch on a 45T-token multimodal corpus, with sparse attention trained at 64K sequence length and context extended to 1M at 34T tokens. Post-training follows the standard SFT to RL to on-policy distillation paradigm; the substantive changes lie in large-scale automated synthesis of agent tasks and environments.

## KV cache economics

Cache-hit charges often dominate agent costs because long sessions repeatedly reread a growing prefix. V4.1-Flash's KV cache needs about 1/4 the HBM and 1/8 the SSD storage of the previous generation: roughly 890 bytes per token against about 4x that for [DeepSeek V4 Flash](/glossary/deepseek-v4-flash). DeepSeek attributes the reduction to FP4 main KV caching, CSA2 layer sharing, and SWA Bounded Replay.

For production planning, the practical consequence is that self-hosted deployments fit the same context window on much less memory, and API users pay materially less for cached input.

## Official benchmark comparison

These are DeepSeek's reported instruct-model results at maximum reasoning effort (reasoning_effort=100). Agentic benchmarks use DeepSeek Harness minimal mode, mini-SWE for DeepSWE v1.1, and the Claude Code harness for SEC-Bench Pro, so cross-vendor rows should be read as vendor-reported snapshots rather than identical configurations.

| Benchmark | V4.1-Flash | V4-Flash | V4-Pro | Opus 5.0 |
| --- | ---: | ---: | ---: | ---: |
| GPQA Diamond | 90.9 | 89.9 | 92.4 | 93.4 |
| HLE, text-only, no tools | 36.8 | 37.8 | 42.7 | 56.3 |
| Codeforces rating | **3471** | 3289 | 3348 | - |
| MathArena Apex | **65.6** | 58.6 | 65.3 | - |
| Terminal-Bench 2.1 | **90.6** | 82.7 | 87.9 | 89.1 |
| Terminal-Bench 3.0 | 30.0 | 7.6 | 11.8 | **43.3** |
| Terminal-Bench 4.0 | 31.2 | 7.0 | 12.4 | **51.8** |
| DeepSWE v1.1 | **74.2** | 54.4 | 62.7 | 74.0 |
| AutomationBench | **54.8** | 37.7 | 43.2 | 50.3 |
| Agent's Last Exam | **31.8** | 25.2 | 25.7 | 28.6 |

The useful pattern is not a sweep. V4.1-Flash beats both [DeepSeek V4](/glossary/deepseek-v4) predecessors and Opus 5.0 on Terminal-Bench 2.1, DeepSWE v1.1, Codeforces, and AutomationBench, but stays well behind Opus 5.0 on the harder Terminal-Bench 3.0/4.0 releases, and text-only HLE slips slightly below V4-Flash. The base model also beats V4-Pro-Base on MMLU-Pro (74.1 vs 73.5) while activating a fraction of the parameters.

## Pricing

V4.1-Flash pricing took effect at 04:00 UTC on September 10, 2026. Off-peak rates are 50% of peak rates; peak hours are 01:00-04:00 and 06:00-10:00 UTC, Monday through Friday.

| Token type | Off-peak | Peak |
| --- | ---: | ---: |
| 1M input, cache hit | $0.003 | $0.006 |
| 1M input, cache miss | $0.15 | $0.30 |
| 1M output | $0.60 | $1.20 |

That is roughly half the final V4-Flash cache-miss rates ($0.22 input, $0.66 output off-peak) with cache-hit reads down to a third of a cent. The concurrency limit is 2500, against 500 for DeepSeek V4 Pro.

## V4.1 Flash versus V4 Flash

V4.1-Flash replaces V4-Flash outright rather than supplementing it. V4-Flash and V4-Flash-Vision-Exp were retired on September 10, 2026; their API names temporarily route to V4.1-Flash at V4.1-Flash rates. The same-scorecard benchmark table shows V4.1-Flash ahead on nearly every agentic benchmark, with the largest gaps on DeepSWE v1.1 (74.2 vs 54.4) and Terminal-Bench 3.0/4.0, where V4-Flash effectively failed the newer task releases.

Vision also changes status: V4-Flash was text-only and served images through a separate experimental model, while V4.1-Flash accepts images natively through one endpoint.

DeepSeek has also said all `deepseek-v4-pro` requests route to V4.1-Flash at V4.1-Flash rates starting 04:00 UTC on September 14, 2026, until V4.1-Pro launches. V4.1-Flash therefore functions as the default DeepSeek endpoint for both Flash and Pro API names.

## When to use DeepSeek V4.1 Flash

V4.1-Flash fits high-volume agent workloads, repository-scale coding, extraction, summarization, document and image understanding in one endpoint, and any workload that benefits from the 2500-concurrency limit and off-peak scheduling. The 8B-active prefill makes input-heavy requests - large prompts, long [context windows](/glossary/context-window) - disproportionately cheaper than on models that activate parameters symmetrically.

It is not the right tool for the hardest terminal and exploit-development tasks, where Opus 5.0 retains a clear lead on DeepSeek's own table, and text-only HLE suggests no factual-knowledge advantage over its predecessor. Self-hosting is possible under the MIT license, but the 552B backbone keeps local deployment in multi-GPU territory.

## Bottom line

DeepSeek V4.1 Flash is not an incremental Flash update. It is a new Causal Encoder-Decoder architecture that cuts active parameters, KV cache, and API prices at once, adds native vision, and takes over both retired Flash builds and, temporarily, V4-Pro API traffic. The vendor-reported table shows real same-scorecard wins on agent benchmarks, with the usual caveats about harness-dependent results.
$ds41$::text AS body
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'glossary:deepseek-v4-1-flash',
  'glossary',
  'deepseek-v4-1-flash',
  '',
  'glossary/deepseek-v4-1-flash',
  'DeepSeek V4.1 Flash',
  'DeepSeek V4.1 Flash is DeepSeek''s 552B-parameter multimodal MoE model with an 8B-active prefill and 16B-active decode, released September 10, 2026 as the API model deepseek-flash.',
  ds41_entry.body,
  jsonb_build_array(jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', ds41_entry.body)),
  jsonb_build_array(
    jsonb_build_object('title', 'DeepSeek-V4.1-Flash: Smarter, Faster, More Efficient', 'url', 'https://api-docs.deepseek.com/news/news260910'),
    jsonb_build_object('title', 'DeepSeek-V4.1-Flash Model Card and Weights', 'url', 'https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash'),
    jsonb_build_object('title', 'DeepSeek API Models & Pricing', 'url', 'https://api-docs.deepseek.com/quick_start/pricing'),
    jsonb_build_object('title', 'DeepSeek API Docs - Your First API Call', 'url', 'https://api-docs.deepseek.com/')
  ),
  jsonb_build_object(
    'category', 'Models & Architectures',
    'relatedTerms', jsonb_build_array('deepseek-v4', 'deepseek-v4-flash', 'deepseek-v4-pro', 'large-language-model', 'mixture-of-experts', 'context-window', 'speculative-decoding', 'kv-cache', 'inference', 'benchmark', 'open-weight-model', 'reasoning', 'api'),
    'analogy', 'V4.1 Flash is a new building on the same street: the old Flash office closed, its address now forwards there, and the rent is half while the staff reading your files cost a quarter of the previous team.',
    'seoDescription', 'DeepSeek V4.1 Flash explained: 552B MoE with 8B/16B CED architecture, 1M context, native vision, September 2026 pricing, benchmarks, and V4-Flash retirement.',
    'seoKeywords', jsonb_build_array('what is DeepSeek V4.1 Flash', 'DeepSeek V4.1 Flash', 'DeepSeek V4.1 Flash benchmarks', 'DeepSeek V4.1 Flash pricing', 'DeepSeek V4.1 Flash vs V4 Flash', 'deepseek-flash API', 'DeepSeek V4 Flash retired', 'Causal Encoder-Decoder architecture', '890 bytes per token KV cache', '552B MoE model')
  ),
  NULL,
  NULL,
  'published',
  DATE '2026-09-10',
  0
FROM ds41_entry
WHERE true
ON CONFLICT (kind, slug, parent_slug) DO UPDATE SET
  path = EXCLUDED.path,
  title = EXCLUDED.title,
  summary = EXCLUDED.summary,
  body = EXCLUDED.body,
  blocks = EXCLUDED.blocks,
  sources = EXCLUDED.sources,
  metadata = EXCLUDED.metadata,
  cover_image_url = EXCLUDED.cover_image_url,
  cover_image_alt = EXCLUDED.cover_image_alt,
  status = EXCLUDED.status,
  published_at = EXCLUDED.published_at,
  sort_order = EXCLUDED.sort_order,
  updated_at = NOW();

-- 6. Cross-link the new glossary slug from the existing DeepSeek entries.
UPDATE content_items
SET
  metadata = CASE
    WHEN COALESCE(metadata->'relatedTerms', '[]'::jsonb) @> '["deepseek-v4-1-flash"]'::jsonb
      THEN metadata
    ELSE jsonb_set(
      COALESCE(metadata, '{}'::jsonb),
      '{relatedTerms}',
      COALESCE(metadata->'relatedTerms', '[]'::jsonb) || jsonb_build_array('deepseek-v4-1-flash'),
      true
    )
  END,
  updated_at = NOW()
WHERE kind = 'glossary'
  AND slug IN ('deepseek-v4', 'deepseek-v4-flash', 'deepseek-v4-pro')
  AND parent_slug = '';
