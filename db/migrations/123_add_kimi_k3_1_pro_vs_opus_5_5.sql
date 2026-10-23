-- Authored comparison with explicit tables so vendor-specific benchmark
-- settings and the pending weight release remain visible beside the figures.
WITH copy AS (
  SELECT
  $intro$[Kimi K3.1 Pro](/glossary/kimi-k3-1-pro) and [Claude Opus 5.5](/glossary/claude-opus-5-5) both target long-running coding and knowledge work with roughly one-million-token context windows. Kimi's standard API input and output rates are half Opus's. Their cache-read rates are identical, and their public Terminal-Bench 4.0 scores are separated by just 0.7 percentage points.

That makes Kimi a credible lower-price model to test, but the launch evidence does not establish that it completes the same tasks at half the cost. The comparison below uses Moonshot's October 23, 2026 launch, its API documentation and Anthropic's Opus sources, checked October 24. Benchmark results are vendor-reported or reproduced in a vendor's chart; they are not a new independent head-to-head evaluation.
$intro$::text AS intro,
  $context$## Context and reasoning

The context window is the model's working desk: both can keep a large amount of code or documentation within reach. Kimi's API lists 1,048,576 tokens; Anthropic advertises 1M. That capacity alone says little about how accurately either model uses details buried inside a long request.

Kimi's quickstart allows `max_completion_tokens` up to 1,048,576, with a 131,072 default. Opus documents 128K synchronous output and a separate 300K Batch API beta. Kimi's request setting should not be read as a guarantee of a million-token answer in addition to a full context window.

Both models always reason, but their defaults differ. Kimi's quickstart defaults to `max`, while Opus defaults to `medium`. Reasoning effort is like the review time given to an analyst: comparing default settings can compare different work budgets. Kimi supports forced tool calling through `tool_choice="required"`; Opus 5.5 rejects forced tool use. Existing agent integrations need to account for that difference.
$context$::text AS context,
  $cost$## What the price difference buys

At standard rates, a request using 100,000 uncached input tokens and 10,000 output tokens costs $0.30 on Kimi and $0.60 on Opus. If that input is already cached, the token-read and output subtotal becomes $0.12 versus $0.22. Both examples exclude cache writes, external tools, taxes and retries, and assume identical token counts.

Caching is like retaining a frequently used reference file on the desk instead of fetching it again. Both models charge $0.20 per million cached input tokens, so repeated reads do not deliver the same 50% saving as uncached input. Different tokenizers, reasoning lengths and numbers of tool calls can also change the cost of a completed task.

Kimi Highspeed costs $4 input and $20 output per million tokens, matching standard Opus's base rates. Moonshot advertises two to three times the speed of standard Kimi; that is not a measured speed comparison with Opus. Opus offers a 50% Batch API discount for eligible asynchronous input and output. Kimi's current batch guide does not list K3.1 Pro as a supported model, so its older-model batch discount is excluded here.
$cost$::text AS cost,
  $bench$## How to read the benchmark gap

Terminal-Bench 4.0 is a test course for agents working in a terminal. Kimi uses the Claude Code harness at maximum reasoning effort. Anthropic reports Opus at `xhigh`, its best reported setting for this test. The 66.4% versus 65.7% gap is smaller than Anthropic's own 2.6-percentage-point standard error for Opus. Without a matched evaluation and uncertainty estimate for both models, those numbers do not establish a reliable winner.

BenchCAD measures how closely generated geometry overlaps the target shape, rather than the proportion of completely correct models. Moonshot evaluates open models on a 1,000-case subset and imports the Opus results from Anthropic's system card. The with-tools and without-tools rows describe different setups and should be read separately.

Kimi's factuality advantage in the table comes from Moonshot's own challenging-prompt benchmark. The metric counts unsupported claims, not incorrect responses, and lower is better. These are internal stress-test results, not everyday error rates or independently reproduced evidence that one model is generally more truthful.
$bench$::text AS bench,
  $choice$## Which model fits the workflow

**Kimi K3.1 Pro is worth testing when token cost, visual coding or 3D tool workflows matter.** It accepts text, images and video, supports structured output and forced tool calls, and is available in Kimi's own coding and desktop products. Moonshot also reports 64.4 on OfficeQA Pro v2 and 58.6 on Frontier Finance. Those are evidence for targeted document and finance-workflow trials, not a substitute for checking the calculations and sources in the resulting work.

**Opus 5.5 fits teams that already depend on Claude's APIs, cloud distribution or batch processing.** Its published interface supports text and image input, text output and computer-use tooling, with access through the Claude API and major cloud partners. The API and workflow differences can outweigh a lower token rate if a migration changes the tool loop, output handling or review workload.

**Self-hosting remains a future option for Kimi at this launch.** Moonshot promises full K3.1 Pro weights by November 4, 2026. The reviewed sources do not establish a downloadable checkpoint and final license on October 24. Opus remains proprietary. Kimi's 3D demonstrations show software operating through tools, not native 3D or video generation from the base model endpoint.

The useful trial is the same repository task or document workflow, with the same tool permissions, time limit and acceptance checks. Measure completion rate, total billed tokens, elapsed time and the human corrections required. Kimi has the lower standard token price. The cheaper completed job still has to be measured.
$choice$::text AS choice
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks,
  sources, metadata, status, published_at, sort_order
)
SELECT
  'comparison:kimi-k3-1-pro-vs-opus-5-5', 'comparison',
  'kimi-k3-1-pro-vs-opus-5-5', '', 'comparisons/kimi-k3-1-pro-vs-opus-5-5',
  'Kimi K3.1 Pro vs Claude Opus 5.5',
  'Kimi K3.1 Pro vs Claude Opus 5.5: API pricing, 1M context, coding and CAD benchmarks, reasoning, tool support and the planned Kimi weight release.',
  concat_ws(E'\n\n', copy.intro, copy.context, copy.cost, copy.bench, copy.choice),
  jsonb_build_array(
    jsonb_build_object('id', 'kimi-opus-intro', 'type', 'markdown', 'content', copy.intro),
    jsonb_build_object('id', 'kimi-opus-specs', 'type', 'spec_table', 'title', 'Specifications',
      'columns', jsonb_build_array('Kimi K3.1 Pro', 'Claude Opus 5.5'),
      'rows', $rows$[
        ["Developer", "Moonshot AI", "Anthropic"],
        ["Release date", "October 23, 2026", "September 22, 2026"],
        ["API model ID", "kimi-k3.1-pro", "claude-opus-5-5"],
        ["Context window", "1,048,576 tokens", "1M tokens"],
        ["Output limit", "Completion setting: 131,072 default; up to 1,048,576, subject to context budget", "128K synchronous; 300K Batch API beta"],
        ["Reasoning", "Always on; low / high / max; max default in quickstart", "Adaptive, always on; medium default"],
        ["Input", "Text, images and video", "Text and images"],
        ["Output", "Text and tool calls", "Text and tool calls"],
        ["Forced tool use", "tool_choice=required supported", "Not supported; returns an error"],
        ["Knowledge cutoff", "Not specified in reviewed sources", "June 2026"],
        ["Weights", "Promised by November 4, 2026; final license not verified", "Proprietary"]
      ]$rows$::jsonb,
      'sourceNote', 'Official model documentation, checked October 24, 2026. Kimi completion settings and Opus output caps are documented differently.'),
    jsonb_build_object('id', 'kimi-opus-context', 'type', 'markdown', 'content', copy.context),
    jsonb_build_object('id', 'kimi-opus-pricing', 'type', 'spec_table', 'title', 'Pricing',
      'columns', jsonb_build_array('Kimi K3.1 Pro', 'Claude Opus 5.5'),
      'rows', $rows$[
        ["Uncached input / 1M tokens", "$2.00", "$4.00"],
        ["Output / 1M tokens", "$10.00", "$20.00"],
        ["Cached input read / 1M", "$0.20", "$0.20"],
        ["Five-minute cache write / 1M", "$2.00", "$5.00"],
        ["One-hour cache write / 1M", "$4.00", "$8.00"],
        ["Batch discount", "K3.1 Pro not listed in the current supported-model guide", "50% off input and output"],
        ["100K uncached input + 10K output", "$0.30", "$0.60"],
        ["100K cached reads + 10K output", "$0.12", "$0.22"]
      ]$rows$::jsonb,
      'sourceNote', 'USD per million tokens at standard rates. Examples exclude cache writes, tool fees, taxes and retries; they assume equal token counts.'),
    jsonb_build_object('id', 'kimi-opus-cost', 'type', 'markdown', 'content', copy.cost),
    jsonb_build_object('id', 'kimi-opus-benchmarks', 'type', 'spec_table', 'title', 'Reported benchmarks',
      'columns', jsonb_build_array('Kimi K3.1 Pro', 'Claude Opus 5.5'),
      'rows', $rows$[
        ["Terminal-Bench 4.0", "65.7% (max; Moonshot)", "66.4% (xhigh; Anthropic)"],
        ["BenchCAD, with tools", "92.7% voxel IoU (Moonshot, 1,000-case sample)", "96.2% (Anthropic figure cited by Moonshot)"],
        ["BenchCAD, without tools", "72.8% voxel IoU (Moonshot, 1,000-case sample)", "73.0% (Anthropic figure cited by Moonshot)"],
        ["Kimi Hallu Bench, no tools; lower is better", "5.4% claim-level hallucination rate", "6.8% claim-level hallucination rate"],
        ["Kimi Hallu Bench Hard, with tools; lower is better", "8.3% claim-level hallucination rate", "15.0% claim-level hallucination rate"]
      ]$rows$::jsonb,
      'sourceNote', 'Moonshot launch charts and footnotes, plus Anthropic launch methodology for Terminal-Bench. BenchCAD is geometry overlap; hallucination rows are Moonshot internal stress tests. These are not matched independent evaluations.'),
    jsonb_build_object('id', 'kimi-opus-bench-notes', 'type', 'markdown', 'content', copy.bench),
    jsonb_build_object('id', 'kimi-opus-choice', 'type', 'markdown', 'content', copy.choice)
  ),
  $sources$[
    {"title":"Introducing Kimi K3.1 Pro","url":"https://www.kimi.ai/blog/kimi-k3-1-pro"},
    {"title":"Kimi K3.1 Pro API quickstart","url":"https://platform.kimi.ai/docs/guide/kimi-k3-1-pro-quickstart"},
    {"title":"Kimi API model pricing","url":"https://platform.kimi.ai/docs/pricing/chat"},
    {"title":"Kimi Batch API supported models","url":"https://platform.kimi.ai/docs/guide/use-batch-api"},
    {"title":"Introducing Claude Opus 5.5","url":"https://www.anthropic.com/claude-opus-5-5"},
    {"title":"Claude Opus 5.5 API overview","url":"https://platform.claude.com/docs/en/models/opus-5-5/overview"}
  ]$sources$::jsonb,
  '{"modelA":"kimi-k3-1-pro","modelB":"claude-opus-5-5","verification_pass":"kimi-k31-pro-opus55-2026-10-24"}'::jsonb,
  'published', NOW(), 0
FROM copy
WHERE true
ON CONFLICT (kind, slug, parent_slug) DO UPDATE SET
  title = EXCLUDED.title, summary = EXCLUDED.summary, body = EXCLUDED.body,
  blocks = EXCLUDED.blocks, sources = EXCLUDED.sources, metadata = EXCLUDED.metadata,
  status = EXCLUDED.status, updated_at = NOW();
