-- Enrich the GPT-6 Sol / GPT-6 Luna model catalog rows with the official
-- OpenAI model-docs specs so database-generated (x vs y) comparisons stop
-- showing blank Sol/Luna cells, and dedupe the DeepSWE / FrontierCode
-- benchmark names so each renders as one row instead of two.
--
-- Sources (fetched and verified):
-- - GPT-6 Sol model page
--   (https://developers.openai.com/api/docs/models/gpt-6-sol)
--   Model ID gpt-6-sol, 1,050,000-token context window, 128,000 max output,
--   Apr 20, 2026 knowledge cutoff, effort none/low/medium(default)/high/
--   xhigh/max, $2 input, $0.20 cached input, $2.50 cache writes, $10 output,
--   Batch and Flex 50 percent off Standard, >272K prompts at 2x input/cache
--   and 1.5x output, text+image in, text out, Responses tools incl. computer
--   use and MCP.
-- - GPT-6 Luna model page
--   (https://developers.openai.com/api/docs/models/gpt-6-luna)
--   Model ID gpt-6-luna, 1,050,000-token context window, 128,000 max output,
--   May 18, 2026 knowledge cutoff, same effort ladder, $0.10 input, $0.01
--   cached input, $0.125 cache writes, $0.50 output, same Batch/Flex and
--   >272K terms, text+image in, text out, same Responses tool set.
-- - Benchmark scores themselves are unchanged (migrations 056-057, 063, 066).
--   Only the two divergent benchmark_name spellings are canonicalized:
--   DeepSWE 1.1 -> DeepSWE v1.1 and FrontierCode 1.1 Main -> the shared
--   FrontierCode 1.1 Main label used by the rest of the catalog (018, 052),
--   so generated comparisons render one merged row per benchmark.
-- - The authored GPT-6 Sol vs Claude Opus 5.5 page (059, 067) is untouched.

-- 1. GPT-6 Sol catalog specs from the official model page.
UPDATE models SET
  comparison_data = (comparison_data - 'Cached input') || '{
    "Developer": "OpenAI",
    "Release date": "2026-09-22",
    "API model ID": "gpt-6-sol",
    "Context window": "1,050,000 tokens",
    "Max output": "128,000 tokens",
    "Knowledge cutoff": "Apr 20, 2026",
    "Reasoning / effort": "none / low / medium (default) / high / xhigh / max",
    "Input / 1M tokens": "$2",
    "Cached input / 1M": "$0.20",
    "Cache write / 1M": "$2.50",
    "Output / 1M tokens": "$10",
    "Batch / flex discount": "Batch and Flex 50 percent off Standard rates",
    "Long-context surcharge": "Prompts over 272K input tokens: 2x input and cache rates plus 1.5x output",
    "Text input": "Yes",
    "Image / vision input": "Yes",
    "Audio input": "No",
    "Video input": "No",
    "Text output": "Yes",
    "Image output": "No",
    "Audio output": "No",
    "Video output": "No",
    "Tool / function calling": "Yes",
    "Computer use": "Yes",
    "API access": "Yes",
    "Product access": "ChatGPT Work and Codex (Plus, Pro, Business, Enterprise, Edu)",
    "Weights / license": "Proprietary",
    "Primary focus": "Built to power complex coding and agentic workflows"
  }'::jsonb,
  sources = sources || '[
    {"title": "GPT-6 Sol - OpenAI Developers", "url": "https://developers.openai.com/api/docs/models/gpt-6-sol"},
    {"title": "Compare models - GPT-6 Sol - OpenAI Developers", "url": "https://developers.openai.com/api/docs/models/compare?model=gpt-6-sol"}
  ]'::jsonb,
  metadata = COALESCE(metadata, '{}'::jsonb) || '{"verification_pass": "gpt6-sol-luna-catalog-dedupe-2026-09-23"}'::jsonb,
  verified_at = NOW(),
  updated_at = NOW()
WHERE slug = 'gpt-6-sol';

-- 2. GPT-6 Luna catalog specs from the official model page.
UPDATE models SET
  comparison_data = (comparison_data - 'Cached input') || '{
    "Developer": "OpenAI",
    "Release date": "2026-09-22",
    "API model ID": "gpt-6-luna",
    "Context window": "1,050,000 tokens",
    "Max output": "128,000 tokens",
    "Knowledge cutoff": "May 18, 2026",
    "Reasoning / effort": "none / low / medium (default) / high / xhigh / max",
    "Input / 1M tokens": "$0.10",
    "Cached input / 1M": "$0.01",
    "Cache write / 1M": "$0.125",
    "Output / 1M tokens": "$0.50",
    "Batch / flex discount": "Batch and Flex 50 percent off Standard rates",
    "Long-context surcharge": "Prompts over 272K input tokens: 2x input and cache rates plus 1.5x output",
    "Text input": "Yes",
    "Image / vision input": "Yes",
    "Audio input": "No",
    "Video input": "No",
    "Text output": "Yes",
    "Image output": "No",
    "Audio output": "No",
    "Video output": "No",
    "Tool / function calling": "Yes",
    "Computer use": "Yes",
    "API access": "Yes",
    "Product access": "ChatGPT Work and Codex plus Luna on desktop for Free and Go users",
    "Weights / license": "Proprietary",
    "Primary focus": "Most efficient model for focused, high-volume tasks"
  }'::jsonb,
  sources = sources || '[
    {"title": "GPT-6 Luna - OpenAI Developers", "url": "https://developers.openai.com/api/docs/models/gpt-6-luna"}
  ]'::jsonb,
  metadata = COALESCE(metadata, '{}'::jsonb) || '{"verification_pass": "gpt6-sol-luna-catalog-dedupe-2026-09-23"}'::jsonb,
  verified_at = NOW(),
  updated_at = NOW()
WHERE slug = 'gpt-6-luna';

-- 3. Canonicalize the divergent DeepSWE spelling so Sol/Luna share the
-- catalog-wide DeepSWE v1.1 label (018, 052) instead of a second row.
UPDATE model_benchmarks SET
  benchmark_name = 'DeepSWE v1.1',
  benchmark_version = NULL,
  updated_at = NOW()
WHERE id IN ('tq-20260922-gpt6sol-deepswe11', 'tq-20260922-gpt6luna-deepswe11');

-- 4. Canonicalize the divergent FrontierCode spelling so Sol/Luna share the
-- catalog-wide FrontierCode 1.1 Main label (018, 052) instead of a second row.
UPDATE model_benchmarks SET
  benchmark_name = 'FrontierCode 1.1 Main',
  benchmark_version = NULL,
  updated_at = NOW()
WHERE id IN ('tq-20260922-gpt6sol-frontiercode11main', 'tq-20260922-gpt6luna-frontiercode11main');
