-- Publish the DeepSeek V4.1 Flash 1017 model card from DeepSeek's launch
-- announcement, official model repository, and current API pricing page.
WITH deepseek_v41_flash_1017_entry AS (
  SELECT $body$DeepSeek V4.1 Flash 1017 is the October 17, 2026 update to [DeepSeek V4.1 Flash](/glossary/deepseek-v4-1-flash), DeepSeek's model for reasoning, coding, agent tasks and image understanding. DeepSeek describes it as a minor update and reports stronger benchmark results, shorter outputs and faster task completion. It is available on DeepSeek's website, app and API, with downloadable model weights under the MIT license.

For a normal user, the update is available through the existing DeepSeek products. For a developer, the API model name remains `deepseek-flash`, which now serves `DeepSeek-V4.1-Flash-1017`. The API lists a 1M-token [context window](/glossary/context-window), up to 384K output tokens, image input and tool calling. The benchmark scores below come from DeepSeek's launch materials and are vendor-reported.

## Key specifications

| Specification | DeepSeek V4.1 Flash 1017 |
| --- | --- |
| Developer | DeepSeek |
| Release date | October 17, 2026 |
| Model version | `DeepSeek-V4.1-Flash-1017` |
| API model ID | `deepseek-flash` |
| Inputs / output | Text and images / text |
| Context window | 1M tokens |
| Maximum API output | 384K tokens |
| Thinking modes | Thinking, enabled by default, and non-thinking |
| API features | JSON output, tool calls, Responses API and Anthropic-compatible API |
| Completion features | Chat prefix completion; fill-in-the-middle in non-thinking mode only |
| Flash API concurrency limit | 2500 |
| Availability | DeepSeek website, app, API, Hugging Face and ModelScope |
| Model weights and repository license | MIT |
| Recommended sampling | Temperature 1.0; top-p 1.0 |

The context window is the model's working desk: prompts, conversation history and generated text have to fit within its token budget. A maximum output limit is a ceiling for a response, rather than a target length. DeepSeek's downloadable-model instructions recommend setting `max_tokens` to at least 256K; the hosted API pricing page separately lists 384K as the maximum output.

## What changed from V4.1 Flash

DeepSeek positions 1017 as an update to the September 10, 2026 V4.1 Flash release. Its launch post emphasizes improvements across reasoning, coding, agent tasks and vision, alongside reduced task execution time and output length. The official model card calls the release a minor update.

The launch also highlights end-to-end project development, including games and 3D scenes. DeepSeek says reinforcement learning with self-feedback contributes to those improvements. In that process, a model practices tasks and uses feedback on its own attempts to improve, like a programmer running a project and using the result to revise the next attempt. The demonstrations show selected outcomes; they do not establish a general success rate for building complete projects.

DeepSeek recommends DeepSeek Harness v0.2.1-rc.1 or later and says the model is optimized for its harness, Tencent CodeBuddy, WorkBuddy and other agent environments. An agent harness is the surrounding workbench that gives the model tools, runs its commands and returns the results. Its configuration affects both performance and cost.

## Official benchmark results

The following figures are from the [official 1017 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017). The predecessor and Opus 5 columns reproduce the comparisons in that scorecard. Opus 5 is a reference model from DeepSeek's table, rather than a comparison with every model available at release.

| Benchmark | V4.1 Flash 1017 | V4.1 Flash | Opus 5 reference |
| --- | ---: | ---: | ---: |
| Humanity's Last Exam, without tools | 42.6 | 36.8 | 56.3 |
| Terminal-Bench 4.0 | 42.3 | 31.2 | 51.8 |
| DeepSWE v1.1 | 77.0 | 74.2 | 74.0 |
| ProgramBench | 29.1 | 20.3 | 37.0 |
| NL2Repo-Bench | 68.2 | 65.4 | 75.3 |
| Automation-Bench v1.0.6 | 60.5 | 54.8 | 50.3 |
| Agents' Last Exam | 35.4 | 31.8 | 28.6 |
| Apex Bench v1.1 | 72.1 | 64.6 | 65.8 |
| Chartography, with tools | 83.0 | 78.9 | 84.0 |
| BabyVision, with tools | 92.5 | 89.6 | 94.1 |
| ZeroBench-main, with tools | 51.0 | 49.0 | 52.0 |

DeepSeek separately reports 44.0 on the text-only subset of Humanity's Last Exam, against 39.1 for the predecessor. Those subset scores should not be substituted for the full HLE results in the table. The three vision rows explicitly use tools, so they measure a model-and-tools workflow.

The 1017 model improves on the predecessor in all 11 listed rows. Terminal-Bench 4.0 rises by 11.1 points, ProgramBench by 8.8 and DeepSWE v1.1 by 2.8. Against the Opus 5 reference, it leads on DeepSWE, Automation-Bench, Agents' Last Exam and Apex Bench, and trails on the other seven rows.

DeepSeek specifies Claude Code as the harness for Terminal-Bench 4.0 and Mini-SWE-V2 for DeepSWE v1.1. These settings matter: TheQuery's September V4.1 Flash card records different harnesses for those evaluations. The predecessor figures here come from the new scorecard; the table is not an independently controlled test isolating model changes from evaluation changes.

## Task cost and output length

DeepSeek says 1017 completes tasks faster while producing shorter outputs. Its launch chart plots DeepSWE v1.1 pass@1 against cost per task using the mini-swe-v2 harness. Pass@1 asks whether a task succeeds on the first attempt, like checking whether the first submitted repair actually fixes a bug.

Shorter outputs can reduce output-token charges, and faster completion can reduce waiting time. Neither establishes a fixed saving for another workload. Total task cost also depends on input length, cache reuse, tool calls, retries and reasoning settings. DeepSeek's claim of a better performance-cost tradeoff is vendor-reported; the launch material does not establish a universal percentage cost reduction.

## API pricing

The [DeepSeek pricing page](https://api-docs.deepseek.com/quick_start/pricing) lists the following Flash rates, checked on October 17, 2026. Prices are in US dollars per million tokens and apply to the current `deepseek-flash` endpoint.

| Token type | Off-peak | Peak |
| --- | ---: | ---: |
| Input, cache hit | $0.003 | $0.006 |
| Input, cache miss | $0.15 | $0.30 |
| Output | $0.60 | $1.20 |

A cache hit reuses a previously processed prompt prefix, like retrieving an already-read file from a desk instead of reading it again. It has a separate input rate and does not discount output tokens.

Peak hours are 01:00-04:00 and 06:00-10:00 UTC, Monday through Friday, excluding Chinese public holidays. All other hours, including weekends and Chinese public holidays, are off-peak. The API still accepts the retired `deepseek-v4-flash` and `deepseek-v4-flash-vision-exp` model names, but serves those requests with V4.1 Flash 1017 at Flash rates.

## Search and Fetch APIs

The same launch opens DeepSeek's standalone Search and Fetch APIs to everyone. Each endpoint costs $0.75 per 1,000 successful requests. These request-based charges are separate from the model's token prices.

DeepSeek says server-side Search and Fetch calls through DeepSeek Harness, Claude Code and Codex are temporarily free for DeepSeek API users. Billing for those calls begins on October 31, 2026 at 16:00 UTC, equivalent to November 1 at 00:00 Beijing time. That temporary offer does not make all standalone Search and Fetch requests free.

## Running the downloadable model

The [official repository](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) provides MIT-licensed weights and local inference instructions. It does not include a Jinja-format chat template. Instead, its encoding directory contains Python reference code for multi-turn conversations, tool calling, thinking mode, numeric reasoning effort and image content. The prompt format is the envelope around a conversation that tells the model which text belongs to the user, assistant or a tool.

DeepSeek also provides deepseek-recipe, Rust libraries with Python bindings for converting Messages, Chat Completions and Responses requests into its conversation format and parsing model outputs. Inference, tool execution and HTTP transport remain the caller's responsibility. MIT licensing permits self-hosting, while the checkpoint still requires suitable inference infrastructure.

## Limits and workflow fit

The launch evidence supports evaluating 1017 for coding agents, project creation, tool-driven work and image-understanding workflows. It retains the existing Flash endpoint and published token rates while DeepSeek reports better task results and lower output length.

The release materials reviewed here do not state a new knowledge cutoff or provide an independent evaluation of the cost and speed claims. Check the model on representative tasks with the intended harness, tool permissions and spending limits. The relevant measure is how often the whole task succeeds at an acceptable total cost.
$body$::text AS body
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'glossary:deepseek-v4-1-flash-1017',
  'glossary',
  'deepseek-v4-1-flash-1017',
  '',
  'glossary/deepseek-v4-1-flash-1017',
  'DeepSeek V4.1 Flash 1017',
  'DeepSeek''s October 17, 2026 V4.1 Flash update, with a 1M context window, text and image input, MIT-licensed weights and vendor-reported gains in reasoning, coding and agent tasks. Available as deepseek-flash.',
  deepseek_v41_flash_1017_entry.body,
  jsonb_build_array(jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', deepseek_v41_flash_1017_entry.body)),
  jsonb_build_array(
    jsonb_build_object('title', 'DeepSeek V4.1 Flash 1017 launch announcement', 'url', 'https://x.com/deepseek_ai/status/2111375640717033777'),
    jsonb_build_object('title', 'DeepSeek V4.1 Flash 1017 official model card and weights', 'url', 'https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017'),
    jsonb_build_object('title', 'DeepSeek V4.1 Flash 1017 release notes', 'url', 'https://api-docs.deepseek.com/news/news261017'),
    jsonb_build_object('title', 'DeepSeek API models and pricing', 'url', 'https://api-docs.deepseek.com/quick_start/pricing')
  ),
  jsonb_build_object(
    'category', 'Models & Architectures',
    'relatedTerms', jsonb_build_array('deepseek-v4-1-flash', 'context-window', 'tool-calling', 'benchmark', 'open-weight-model', 'reasoning'),
    'seoDescription', 'DeepSeek V4.1 Flash 1017 model card: October 17 release, 1M context, 384K output, MIT weights, official benchmarks, API pricing and Search/Fetch fees.',
    'seoKeywords', jsonb_build_array('DeepSeek V4.1 Flash 1017', 'DeepSeek-V4.1-Flash-1017', 'DeepSeek V4.1 Flash 1017 benchmarks', 'DeepSeek V4.1 Flash 1017 pricing', 'deepseek-flash'),
    'verification_pass', 'deepseek-v41-flash-1017-2026-10-17'
  ),
  NULL,
  NULL,
  'published',
  DATE '2026-10-17',
  0
FROM deepseek_v41_flash_1017_entry
WHERE true
ON CONFLICT (kind, slug, parent_slug) DO UPDATE SET
  path = EXCLUDED.path,
  title = EXCLUDED.title,
  summary = EXCLUDED.summary,
  body = EXCLUDED.body,
  blocks = EXCLUDED.blocks,
  sources = EXCLUDED.sources,
  metadata = EXCLUDED.metadata,
  status = EXCLUDED.status,
  published_at = EXCLUDED.published_at,
  sort_order = EXCLUDED.sort_order,
  updated_at = NOW();
