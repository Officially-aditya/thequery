-- Enrich Claude Opus 4.6/4.7/4.8/5 and Sonnet 4.6/5 with current verified specs,
-- pricing, availability and normalized benchmark evidence.
-- All six models already exist in the canonical catalog, so this migration updates them in place.

INSERT INTO models (
  slug, name, developer, release_date, ga_date, access, family,
  comparison_data, sources, notes, metadata, verified_at
)
SELECT
  seed.slug,
  seed.name,
  seed.developer,
  seed.release_date::date,
  NULLIF(seed.ga_date, '')::date,
  seed.access,
  seed.family,
  seed.comparison_data,
  seed.sources,
  seed.notes,
  seed.metadata,
  NOW()
FROM jsonb_to_recordset($models$
[
  {
    "slug":"claude-opus-5",
    "name":"Claude Opus 5",
    "developer":"Anthropic",
    "release_date":"2026-07-24",
    "ga_date":"2026-07-24",
    "access":"proprietary",
    "family":"Claude Opus 5",
    "comparison_data":{
      "Developer":"Anthropic",
      "Release date":"2026-07-24",
      "API model ID":"claude-opus-5",
      "Context window":"1M tokens",
      "Max output":"128K tokens",
      "Knowledge cutoff":"May 2026",
      "Reasoning / effort":"Adaptive thinking; high default",
      "Input / 1M tokens":"$5",
      "Cached input / 1M":"$0.50",
      "Cache write / 1M":"$6.25 (5 min) / $10 (1 hr)",
      "Output / 1M tokens":"$25",
      "Batch / flex discount":"Batch API: 50% discount",
      "Long-context surcharge":"None — standard pricing through 1M context",
      "Text input":"Yes",
      "Image / vision input":"Yes",
      "Audio input":"No native audio input documented",
      "Video input":"No native video input documented",
      "Text output":"Yes",
      "Image output":"No",
      "Audio output":"No",
      "Video output":"No",
      "Tool / function calling":"Yes",
      "Computer use":"Yes",
      "API access":"Yes",
      "Product access":"Claude + Claude Code + Claude API + cloud partners",
      "Weights / license":"Proprietary"
    },
    "sources":[
      {"title":"Introducing Claude Opus 5","url":"https://www.anthropic.com/news/claude-opus-5"},
      {"title":"Claude Opus 5 platform docs","url":"https://platform.claude.com/docs/en/models/opus-5/whats-new-opus-5"},
      {"title":"Claude pricing","url":"https://platform.claude.com/docs/en/about-claude/pricing"}
    ],
    "notes":"Current Opus model. Fast mode is also available at a higher per-token rate; standard catalog pricing records the default API tier.",
    "metadata":{"verification_pass":"anthropic-frontier-2026-09-07","catalog_status":"current","default_effort":"high"}
  },
  {
    "slug":"claude-opus-4-8",
    "name":"Claude Opus 4.8",
    "developer":"Anthropic",
    "release_date":"2026-05-28",
    "ga_date":"2026-05-28",
    "access":"proprietary",
    "family":"Claude Opus 4",
    "comparison_data":{
      "Developer":"Anthropic",
      "Release date":"2026-05-28",
      "API model ID":"claude-opus-4-8",
      "Context window":"1M tokens",
      "Max output":"128K tokens",
      "Knowledge cutoff":"Jan 2026",
      "Reasoning / effort":"Adaptive thinking; high default",
      "Input / 1M tokens":"$5",
      "Cached input / 1M":"$0.50",
      "Cache write / 1M":"$6.25 (5 min) / $10 (1 hr)",
      "Output / 1M tokens":"$25",
      "Batch / flex discount":"Batch API: 50% discount",
      "Long-context surcharge":"None — standard pricing through 1M context",
      "Text input":"Yes",
      "Image / vision input":"Yes",
      "Audio input":"No native audio input documented",
      "Video input":"No native video input documented",
      "Text output":"Yes",
      "Image output":"No",
      "Audio output":"No",
      "Video output":"No",
      "Tool / function calling":"Yes",
      "Computer use":"Yes",
      "API access":"Yes (legacy model)",
      "Product access":"Claude + Claude Code + Claude API + cloud partners",
      "Weights / license":"Proprietary"
    },
    "sources":[
      {"title":"Introducing Claude Opus 4.8","url":"https://www.anthropic.com/news/claude-opus-4-8"},
      {"title":"Claude Opus 4.8 platform docs","url":"https://platform.claude.com/docs/en/models/opus-4-8/overview"},
      {"title":"Claude pricing","url":"https://platform.claude.com/docs/en/about-claude/pricing"}
    ],
    "notes":"Still available as a legacy model. Fast mode is supported; standard catalog pricing records the default API tier.",
    "metadata":{"verification_pass":"anthropic-frontier-2026-09-07","catalog_status":"legacy_active","default_effort":"high"}
  },
  {
    "slug":"claude-opus-4-7",
    "name":"Claude Opus 4.7",
    "developer":"Anthropic",
    "release_date":"2026-04-16",
    "ga_date":"2026-04-16",
    "access":"proprietary",
    "family":"Claude Opus 4",
    "comparison_data":{
      "Developer":"Anthropic",
      "Release date":"2026-04-16",
      "API model ID":"claude-opus-4-7",
      "Context window":"1M tokens",
      "Max output":"128K tokens",
      "Knowledge cutoff":"Jan 2026",
      "Reasoning / effort":"Adaptive thinking; high default; xhigh supported",
      "Input / 1M tokens":"$5",
      "Cached input / 1M":"$0.50",
      "Cache write / 1M":"$6.25 (5 min) / $10 (1 hr)",
      "Output / 1M tokens":"$25",
      "Batch / flex discount":"Batch API: 50% discount",
      "Long-context surcharge":"None — standard pricing through 1M context",
      "Text input":"Yes",
      "Image / vision input":"Yes",
      "Audio input":"No native audio input documented",
      "Video input":"No native video input documented",
      "Text output":"Yes",
      "Image output":"No",
      "Audio output":"No",
      "Video output":"No",
      "Tool / function calling":"Yes",
      "Computer use":"Yes",
      "API access":"Yes (legacy model)",
      "Product access":"Claude + Claude Code + Claude API + cloud partners",
      "Weights / license":"Proprietary"
    },
    "sources":[
      {"title":"Introducing Claude Opus 4.7","url":"https://www.anthropic.com/news/claude-opus-4-7"},
      {"title":"Claude Opus 4.7 platform docs","url":"https://platform.claude.com/docs/en/models/opus-4-7/overview"},
      {"title":"Claude pricing","url":"https://platform.claude.com/docs/en/about-claude/pricing"}
    ],
    "notes":"Still available as a legacy model. Uses Anthropic's newer tokenizer generation introduced with Claude 4.7.",
    "metadata":{"verification_pass":"anthropic-frontier-2026-09-07","catalog_status":"legacy_active","default_effort":"high"}
  },
  {
    "slug":"claude-opus-4-6",
    "name":"Claude Opus 4.6",
    "developer":"Anthropic",
    "release_date":"2026-02-05",
    "ga_date":"2026-02-05",
    "access":"proprietary",
    "family":"Claude Opus 4",
    "comparison_data":{
      "Developer":"Anthropic",
      "Release date":"2026-02-05",
      "API model ID":"claude-opus-4-6",
      "Context window":"1M tokens",
      "Max output":"128K tokens",
      "Knowledge cutoff":"May 2025",
      "Reasoning / effort":"Adaptive thinking; extended thinking deprecated; high default",
      "Input / 1M tokens":"$5",
      "Cached input / 1M":"$0.50",
      "Cache write / 1M":"$6.25 (5 min) / $10 (1 hr)",
      "Output / 1M tokens":"$25",
      "Batch / flex discount":"Batch API: 50% discount",
      "Long-context surcharge":"None — standard pricing through 1M context",
      "Text input":"Yes",
      "Image / vision input":"Yes",
      "Audio input":"No native audio input documented",
      "Video input":"No native video input documented",
      "Text output":"Yes",
      "Image output":"No",
      "Audio output":"No",
      "Video output":"No",
      "Tool / function calling":"Yes",
      "Computer use":"Yes",
      "API access":"Yes (legacy model)",
      "Product access":"Claude + Claude Code + Claude API + cloud partners",
      "Weights / license":"Proprietary"
    },
    "sources":[
      {"title":"Introducing Claude Opus 4.6","url":"https://www.anthropic.com/news/claude-opus-4-6"},
      {"title":"Claude Opus 4.6 platform docs","url":"https://platform.claude.com/docs/en/models/opus-4-6/overview"},
      {"title":"Claude pricing","url":"https://platform.claude.com/docs/en/about-claude/pricing"}
    ],
    "notes":"Still available as a legacy model. This is the final Opus release in the older tokenizer generation.",
    "metadata":{"verification_pass":"anthropic-frontier-2026-09-07","catalog_status":"legacy_active","default_effort":"high"}
  },
  {
    "slug":"claude-sonnet-5",
    "name":"Claude Sonnet 5",
    "developer":"Anthropic",
    "release_date":"2026-06-30",
    "ga_date":"2026-06-30",
    "access":"proprietary",
    "family":"Claude Sonnet 5",
    "comparison_data":{
      "Developer":"Anthropic",
      "Release date":"2026-06-30",
      "API model ID":"claude-sonnet-5",
      "Context window":"1M tokens",
      "Max output":"128K tokens",
      "Knowledge cutoff":"Jan 2026",
      "Reasoning / effort":"Adaptive thinking on by default; high default",
      "Input / 1M tokens":"$2",
      "Cached input / 1M":"$0.20",
      "Cache write / 1M":"$2.50 (5 min) / $4 (1 hr)",
      "Output / 1M tokens":"$10",
      "Batch / flex discount":"Batch API: 50% discount",
      "Long-context surcharge":"None — standard pricing through 1M context",
      "Text input":"Yes",
      "Image / vision input":"Yes",
      "Audio input":"No native audio input documented",
      "Video input":"No native video input documented",
      "Text output":"Yes",
      "Image output":"No",
      "Audio output":"No",
      "Video output":"No",
      "Tool / function calling":"Yes",
      "Computer use":"Yes; stable computer toolset and browser use",
      "API access":"Yes",
      "Product access":"Claude + Claude Code + Claude API + cloud partners",
      "Weights / license":"Proprietary"
    },
    "sources":[
      {"title":"Introducing Claude Sonnet 5","url":"https://www.anthropic.com/news/claude-sonnet-5"},
      {"title":"What's new in Claude Sonnet 5","url":"https://platform.claude.com/docs/en/docs/about-claude/models/whats-new-sonnet-5"},
      {"title":"Claude pricing","url":"https://platform.claude.com/docs/en/about-claude/pricing"}
    ],
    "notes":"The launch $2/$10 input/output rate was made permanent in August 2026. Sonnet 5 uses Anthropic's newer tokenizer and keeps adaptive thinking enabled by default.",
    "metadata":{"verification_pass":"anthropic-frontier-2026-09-07","catalog_status":"current","default_effort":"high"}
  },
  {
    "slug":"claude-sonnet-4-6",
    "name":"Claude Sonnet 4.6",
    "developer":"Anthropic",
    "release_date":"2026-02-17",
    "ga_date":"2026-02-17",
    "access":"proprietary",
    "family":"Claude Sonnet 4",
    "comparison_data":{
      "Developer":"Anthropic",
      "Release date":"2026-02-17",
      "API model ID":"claude-sonnet-4-6",
      "Context window":"1M tokens",
      "Max output":"128K tokens",
      "Knowledge cutoff":"Aug 2025",
      "Reasoning / effort":"Adaptive thinking; extended thinking deprecated; high default",
      "Input / 1M tokens":"$3",
      "Cached input / 1M":"$0.30",
      "Cache write / 1M":"$3.75 (5 min) / $6 (1 hr)",
      "Output / 1M tokens":"$15",
      "Batch / flex discount":"Batch API: 50% discount",
      "Long-context surcharge":"None — standard pricing through 1M context",
      "Text input":"Yes",
      "Image / vision input":"Yes",
      "Audio input":"No native audio input documented",
      "Video input":"No native video input documented",
      "Text output":"Yes",
      "Image output":"No",
      "Audio output":"No",
      "Video output":"No",
      "Tool / function calling":"Yes",
      "Computer use":"Yes; computer_20251124 tool generation",
      "API access":"Yes (legacy model)",
      "Product access":"Claude + Claude Code + Claude API + cloud partners",
      "Weights / license":"Proprietary"
    },
    "sources":[
      {"title":"Introducing Claude Sonnet 4.6","url":"https://www.anthropic.com/news/claude-sonnet-4-6"},
      {"title":"Claude Sonnet 4.6 platform docs","url":"https://platform.claude.com/docs/en/models/sonnet-4-6/overview"},
      {"title":"Claude pricing","url":"https://platform.claude.com/docs/en/about-claude/pricing"}
    ],
    "notes":"Still available as a legacy model. Uses the pre-4.7 tokenizer generation; later Anthropic comparison tables revised several agentic benchmark values for methodology consistency.",
    "metadata":{"verification_pass":"anthropic-frontier-2026-09-07","catalog_status":"legacy_active","default_effort":"high"}
  }
]
$models$::jsonb) AS seed(
  slug text, name text, developer text, release_date text, ga_date text,
  access text, family text, comparison_data jsonb, sources jsonb, notes text, metadata jsonb
)
ON CONFLICT (slug) DO UPDATE SET
  name = EXCLUDED.name,
  developer = EXCLUDED.developer,
  release_date = EXCLUDED.release_date,
  ga_date = EXCLUDED.ga_date,
  access = EXCLUDED.access,
  family = EXCLUDED.family,
  comparison_data = COALESCE(models.comparison_data, '{}'::jsonb) || EXCLUDED.comparison_data,
  sources = EXCLUDED.sources,
  notes = EXCLUDED.notes,
  metadata = COALESCE(models.metadata, '{}'::jsonb) || EXCLUDED.metadata,
  verified_at = NOW(),
  updated_at = NOW();

INSERT INTO model_benchmarks (
  id, model_slug, category, benchmark_name, benchmark_version, score_numeric, score_display,
  score_unit, tools, reasoning_effort, harness, evaluator, evaluation_date, source, notes, updated_at
)
SELECT
  b.id, b.model_slug, b.category, b.benchmark_name, b.benchmark_version, b.score_numeric, b.score_display,
  b.score_unit, b.tools, b.reasoning_effort, b.harness, b.evaluator,
  NULLIF(b.evaluation_date, '')::date, b.source, b.notes, NOW()
FROM jsonb_to_recordset($benchmarks$
[
  {"id":"anthropic-20260724-opus5-swe-verified","model_slug":"claude-opus-5","category":"coding","benchmark_name":"SWE-bench Verified","benchmark_version":"Verified","score_numeric":96.0,"score_display":"96.0%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":"Anthropic launch evaluation","evaluator":"Anthropic","evaluation_date":"2026-07-24","source":"https://www.anthropic.com/news/claude-opus-5","notes":null},
  {"id":"anthropic-20260724-opus5-swe-pro","model_slug":"claude-opus-5","category":"coding","benchmark_name":"SWE-bench Pro","benchmark_version":null,"score_numeric":79.2,"score_display":"79.2%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":"Anthropic launch evaluation","evaluator":"Anthropic","evaluation_date":"2026-07-24","source":"https://www.anthropic.com/news/claude-opus-5","notes":null},
  {"id":"anthropic-20260724-opus5-browsecomp","model_slug":"claude-opus-5","category":"agentic_computer_use","benchmark_name":"BrowseComp","benchmark_version":null,"score_numeric":90.8,"score_display":"90.8%","score_unit":"percent","tools":true,"reasoning_effort":"high","harness":"Anthropic launch evaluation","evaluator":"Anthropic","evaluation_date":"2026-07-24","source":"https://www.anthropic.com/news/claude-opus-5","notes":null},
  {"id":"anthropic-20260724-opus5-mcp-atlas","model_slug":"claude-opus-5","category":"agentic_computer_use","benchmark_name":"MCP Atlas","benchmark_version":null,"score_numeric":85.8,"score_display":"85.8%","score_unit":"percent","tools":true,"reasoning_effort":"high","harness":"Anthropic launch evaluation","evaluator":"Anthropic","evaluation_date":"2026-07-24","source":"https://www.anthropic.com/news/claude-opus-5","notes":null},
  {"id":"anthropic-20260724-opus5-automation","model_slug":"claude-opus-5","category":"agentic_computer_use","benchmark_name":"AutomationBench","benchmark_version":null,"score_numeric":26.0,"score_display":"26.0%","score_unit":"percent","tools":true,"reasoning_effort":"high","harness":"Anthropic launch evaluation","evaluator":"Anthropic","evaluation_date":"2026-07-24","source":"https://www.anthropic.com/news/claude-opus-5","notes":null},
  {"id":"anthropic-20260724-opus5-hle-no-tools","model_slug":"claude-opus-5","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":null,"score_numeric":56.3,"score_display":"56.3%","score_unit":"percent","tools":false,"reasoning_effort":"high","harness":"Anthropic launch evaluation","evaluator":"Anthropic","evaluation_date":"2026-07-24","source":"https://www.anthropic.com/news/claude-opus-5","notes":null},
  {"id":"anthropic-20260724-opus5-hle-tools","model_slug":"claude-opus-5","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":null,"score_numeric":64.7,"score_display":"64.7%","score_unit":"percent","tools":true,"reasoning_effort":"high","harness":"Search + code tools","evaluator":"Anthropic","evaluation_date":"2026-07-24","source":"https://www.anthropic.com/news/claude-opus-5","notes":null},
  {"id":"aa-20260724-opus5-gdpval-v2","model_slug":"claude-opus-5","category":"agentic_computer_use","benchmark_name":"GDPval-AA v2","benchmark_version":null,"score_numeric":1861,"score_display":"1861 Elo","score_unit":"Elo","tools":true,"reasoning_effort":"high","harness":"Artificial Analysis","evaluator":"Artificial Analysis","evaluation_date":"2026-07-24","source":"https://www.anthropic.com/news/claude-opus-5","notes":"GDPval-AA v2; do not conflate with the original GDPval-AA benchmark."},
  {"id":"anthropic-20260724-opus5-arcagi3","model_slug":"claude-opus-5","category":"math_reasoning","benchmark_name":"ARC-AGI","benchmark_version":"3","score_numeric":30.2,"score_display":"30.2%","score_unit":"percent","tools":false,"reasoning_effort":"high","harness":"ARC-AGI-3 evaluation","evaluator":"Anthropic","evaluation_date":"2026-07-24","source":"https://www.anthropic.com/news/claude-opus-5","notes":null},
  {"id":"anthropic-20260724-opus5-osworld20-first","model_slug":"claude-opus-5","category":"agentic_computer_use","benchmark_name":"OSWorld 2.0","benchmark_version":"First-attempt success","score_numeric":70.57,"score_display":"70.57%","score_unit":"percent","tools":true,"reasoning_effort":"high","harness":"Anthropic system-card evaluation","evaluator":"Anthropic","evaluation_date":"2026-07-24","source":"https://www.anthropic.com/news/claude-opus-5","notes":"First-attempt success rate; keep distinct from partial-credit and other OSWorld 2.0 methodologies."},

  {"id":"anthropic-20260528-opus48-swe-verified","model_slug":"claude-opus-4-8","category":"coding","benchmark_name":"SWE-bench Verified","benchmark_version":"Verified","score_numeric":88.6,"score_display":"88.6%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":"Anthropic comparison evaluation","evaluator":"Anthropic","evaluation_date":"2026-05-28","source":"https://www.anthropic.com/news/claude-opus-4-8","notes":null},
  {"id":"anthropic-20260528-opus48-swe-pro","model_slug":"claude-opus-4-8","category":"coding","benchmark_name":"SWE-bench Pro","benchmark_version":null,"score_numeric":69.2,"score_display":"69.2%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":"Anthropic comparison evaluation","evaluator":"Anthropic","evaluation_date":"2026-05-28","source":"https://www.anthropic.com/news/claude-opus-4-8","notes":null},
  {"id":"anthropic-20260630-opus48-terminal21","model_slug":"claude-opus-4-8","category":"coding","benchmark_name":"Terminal-Bench 2.1","benchmark_version":null,"score_numeric":82.7,"score_display":"82.7%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":"Terminus-2 public harness","evaluator":"Anthropic","evaluation_date":"2026-06-30","source":"https://www.anthropic.com/news/claude-sonnet-5","notes":null},
  {"id":"anthropic-20260630-opus48-hle-no-tools","model_slug":"claude-opus-4-8","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":null,"score_numeric":49.8,"score_display":"49.8%","score_unit":"percent","tools":false,"reasoning_effort":"high","harness":"Anthropic Sonnet 5 comparison table","evaluator":"Anthropic","evaluation_date":"2026-06-30","source":"https://www.anthropic.com/news/claude-sonnet-5","notes":null},
  {"id":"anthropic-20260630-opus48-hle-tools","model_slug":"claude-opus-4-8","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":null,"score_numeric":57.9,"score_display":"57.9%","score_unit":"percent","tools":true,"reasoning_effort":"high","harness":"Anthropic Sonnet 5 comparison table","evaluator":"Anthropic","evaluation_date":"2026-06-30","source":"https://www.anthropic.com/news/claude-sonnet-5","notes":null},
  {"id":"anthropic-20260630-opus48-osworld-verified","model_slug":"claude-opus-4-8","category":"agentic_computer_use","benchmark_name":"OSWorld-Verified","benchmark_version":null,"score_numeric":83.4,"score_display":"83.4%","score_unit":"percent","tools":true,"reasoning_effort":"high","harness":"Anthropic Sonnet 5 comparison table","evaluator":"Anthropic","evaluation_date":"2026-06-30","source":"https://www.anthropic.com/news/claude-sonnet-5","notes":null},
  {"id":"aa-20260630-opus48-gdpval-v2","model_slug":"claude-opus-4-8","category":"agentic_computer_use","benchmark_name":"GDPval-AA v2","benchmark_version":null,"score_numeric":1615,"score_display":"1615 Elo","score_unit":"Elo","tools":true,"reasoning_effort":"high","harness":"Artificial Analysis","evaluator":"Artificial Analysis","evaluation_date":"2026-06-30","source":"https://www.anthropic.com/news/claude-sonnet-5","notes":"GDPval-AA v2."},
  {"id":"anthropic-20260630-opus48-gpqa","model_slug":"claude-opus-4-8","category":"knowledge","benchmark_name":"GPQA Diamond","benchmark_version":null,"score_numeric":93.6,"score_display":"93.6%","score_unit":"percent","tools":false,"reasoning_effort":"high","harness":"Anthropic Sonnet 5 comparison table","evaluator":"Anthropic","evaluation_date":"2026-06-30","source":"https://www.anthropic.com/news/claude-sonnet-5","notes":null},

  {"id":"anthropic-20260416-opus47-swe-verified","model_slug":"claude-opus-4-7","category":"coding","benchmark_name":"SWE-bench Verified","benchmark_version":"Verified","score_numeric":87.6,"score_display":"87.6%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":"Anthropic launch evaluation","evaluator":"Anthropic","evaluation_date":"2026-04-16","source":"https://www.anthropic.com/news/claude-opus-4-7","notes":null},
  {"id":"anthropic-20260416-opus47-swe-pro","model_slug":"claude-opus-4-7","category":"coding","benchmark_name":"SWE-bench Pro","benchmark_version":null,"score_numeric":64.3,"score_display":"64.3%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":"Anthropic launch evaluation","evaluator":"Anthropic","evaluation_date":"2026-04-16","source":"https://www.anthropic.com/news/claude-opus-4-7","notes":null},
  {"id":"anthropic-20260416-opus47-terminal20","model_slug":"claude-opus-4-7","category":"coding","benchmark_name":"Terminal-Bench 2.0","benchmark_version":null,"score_numeric":69.4,"score_display":"69.4%","score_unit":"percent","tools":null,"reasoning_effort":"thinking disabled","harness":"Terminus-2","evaluator":"Anthropic","evaluation_date":"2026-04-16","source":"https://www.anthropic.com/news/claude-opus-4-7","notes":"Anthropic reports this run with thinking disabled."},
  {"id":"anthropic-20260416-opus47-hle-no-tools","model_slug":"claude-opus-4-7","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":null,"score_numeric":46.9,"score_display":"46.9%","score_unit":"percent","tools":false,"reasoning_effort":"high","harness":"Anthropic launch evaluation","evaluator":"Anthropic","evaluation_date":"2026-04-16","source":"https://www.anthropic.com/news/claude-opus-4-7","notes":null},
  {"id":"anthropic-20260416-opus47-hle-tools","model_slug":"claude-opus-4-7","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":null,"score_numeric":54.7,"score_display":"54.7%","score_unit":"percent","tools":true,"reasoning_effort":"high","harness":"Anthropic launch evaluation","evaluator":"Anthropic","evaluation_date":"2026-04-16","source":"https://www.anthropic.com/news/claude-opus-4-7","notes":null},
  {"id":"anthropic-20260416-opus47-browsecomp","model_slug":"claude-opus-4-7","category":"agentic_computer_use","benchmark_name":"BrowseComp","benchmark_version":null,"score_numeric":79.3,"score_display":"79.3%","score_unit":"percent","tools":true,"reasoning_effort":"high","harness":"Anthropic launch evaluation","evaluator":"Anthropic","evaluation_date":"2026-04-16","source":"https://www.anthropic.com/news/claude-opus-4-7","notes":null},
  {"id":"anthropic-20260416-opus47-mcp-atlas","model_slug":"claude-opus-4-7","category":"agentic_computer_use","benchmark_name":"MCP Atlas","benchmark_version":null,"score_numeric":77.3,"score_display":"77.3%","score_unit":"percent","tools":true,"reasoning_effort":"high","harness":"Updated Anthropic grading","evaluator":"Anthropic","evaluation_date":"2026-04-16","source":"https://www.anthropic.com/news/claude-opus-4-7","notes":null},
  {"id":"anthropic-20260528-opus47-osworld-verified","model_slug":"claude-opus-4-7","category":"agentic_computer_use","benchmark_name":"OSWorld-Verified","benchmark_version":null,"score_numeric":82.3,"score_display":"82.3%","score_unit":"percent","tools":true,"reasoning_effort":"high","harness":"Revised Anthropic methodology","evaluator":"Anthropic","evaluation_date":"2026-05-28","source":"https://www.anthropic.com/news/claude-opus-4-8","notes":"Updated from the original Opus 4.7 launch value using Anthropic's revised OSWorld methodology."},
  {"id":"anthropic-20260416-opus47-gpqa","model_slug":"claude-opus-4-7","category":"knowledge","benchmark_name":"GPQA Diamond","benchmark_version":null,"score_numeric":94.2,"score_display":"94.2%","score_unit":"percent","tools":false,"reasoning_effort":"high","harness":"Anthropic launch evaluation","evaluator":"Anthropic","evaluation_date":"2026-04-16","source":"https://www.anthropic.com/news/claude-opus-4-7","notes":null},

  {"id":"anthropic-20260416-opus46-swe-verified","model_slug":"claude-opus-4-6","category":"coding","benchmark_name":"SWE-bench Verified","benchmark_version":"Verified","score_numeric":80.8,"score_display":"80.8%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":"Anthropic comparison evaluation","evaluator":"Anthropic","evaluation_date":"2026-04-16","source":"https://www.anthropic.com/news/claude-opus-4-7","notes":null},
  {"id":"anthropic-20260416-opus46-swe-pro","model_slug":"claude-opus-4-6","category":"coding","benchmark_name":"SWE-bench Pro","benchmark_version":null,"score_numeric":53.4,"score_display":"53.4%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":"Anthropic comparison evaluation","evaluator":"Anthropic","evaluation_date":"2026-04-16","source":"https://www.anthropic.com/news/claude-opus-4-7","notes":null},
  {"id":"anthropic-20260416-opus46-terminal20","model_slug":"claude-opus-4-6","category":"coding","benchmark_name":"Terminal-Bench 2.0","benchmark_version":null,"score_numeric":65.4,"score_display":"65.4%","score_unit":"percent","tools":null,"reasoning_effort":"thinking disabled","harness":"Terminus-2","evaluator":"Anthropic","evaluation_date":"2026-04-16","source":"https://www.anthropic.com/news/claude-opus-4-7","notes":null},
  {"id":"anthropic-20260416-opus46-hle-no-tools","model_slug":"claude-opus-4-6","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":null,"score_numeric":40.0,"score_display":"40.0%","score_unit":"percent","tools":false,"reasoning_effort":"high","harness":"Anthropic comparison evaluation","evaluator":"Anthropic","evaluation_date":"2026-04-16","source":"https://www.anthropic.com/news/claude-opus-4-7","notes":null},
  {"id":"anthropic-20260416-opus46-hle-tools","model_slug":"claude-opus-4-6","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":null,"score_numeric":53.3,"score_display":"53.3%","score_unit":"percent","tools":true,"reasoning_effort":"high","harness":"Search + fetch + code + compaction","evaluator":"Anthropic","evaluation_date":"2026-04-16","source":"https://www.anthropic.com/news/claude-opus-4-7","notes":null},
  {"id":"anthropic-20260416-opus46-browsecomp","model_slug":"claude-opus-4-6","category":"agentic_computer_use","benchmark_name":"BrowseComp","benchmark_version":null,"score_numeric":83.7,"score_display":"83.7%","score_unit":"percent","tools":true,"reasoning_effort":"high","harness":"Anthropic comparison evaluation","evaluator":"Anthropic","evaluation_date":"2026-04-16","source":"https://www.anthropic.com/news/claude-opus-4-7","notes":null},
  {"id":"anthropic-20260416-opus46-mcp-atlas","model_slug":"claude-opus-4-6","category":"agentic_computer_use","benchmark_name":"MCP Atlas","benchmark_version":null,"score_numeric":75.8,"score_display":"75.8%","score_unit":"percent","tools":true,"reasoning_effort":"high","harness":"Updated Anthropic grading","evaluator":"Anthropic","evaluation_date":"2026-04-16","source":"https://www.anthropic.com/news/claude-opus-4-7","notes":"Uses the updated grading reported with the Opus 4.7 comparison table."},
  {"id":"anthropic-20260416-opus46-osworld-verified","model_slug":"claude-opus-4-6","category":"agentic_computer_use","benchmark_name":"OSWorld-Verified","benchmark_version":null,"score_numeric":72.7,"score_display":"72.7%","score_unit":"percent","tools":true,"reasoning_effort":"high","harness":"Anthropic comparison evaluation","evaluator":"Anthropic","evaluation_date":"2026-04-16","source":"https://www.anthropic.com/news/claude-opus-4-7","notes":null},
  {"id":"anthropic-20260416-opus46-gpqa","model_slug":"claude-opus-4-6","category":"knowledge","benchmark_name":"GPQA Diamond","benchmark_version":null,"score_numeric":91.3,"score_display":"91.3%","score_unit":"percent","tools":false,"reasoning_effort":"high","harness":"Anthropic comparison evaluation","evaluator":"Anthropic","evaluation_date":"2026-04-16","source":"https://www.anthropic.com/news/claude-opus-4-7","notes":null},
  {"id":"aa-20260205-opus46-gdpval","model_slug":"claude-opus-4-6","category":"agentic_computer_use","benchmark_name":"GDPval-AA","benchmark_version":null,"score_numeric":1606,"score_display":"1606 Elo","score_unit":"Elo","tools":true,"reasoning_effort":"high","harness":"Artificial Analysis","evaluator":"Artificial Analysis","evaluation_date":"2026-02-05","source":"https://www.anthropic.com/news/claude-opus-4-6","notes":"Original GDPval-AA benchmark; do not compare as if it were GDPval-AA v2."},

  {"id":"anthropic-20260630-sonnet5-swe-verified","model_slug":"claude-sonnet-5","category":"coding","benchmark_name":"SWE-bench Verified","benchmark_version":"Verified","score_numeric":85.2,"score_display":"85.2%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":"Anthropic launch evaluation","evaluator":"Anthropic","evaluation_date":"2026-06-30","source":"https://www.anthropic.com/news/claude-sonnet-5","notes":null},
  {"id":"anthropic-20260630-sonnet5-swe-pro","model_slug":"claude-sonnet-5","category":"coding","benchmark_name":"SWE-bench Pro","benchmark_version":null,"score_numeric":63.2,"score_display":"63.2%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":"Anthropic launch evaluation","evaluator":"Anthropic","evaluation_date":"2026-06-30","source":"https://www.anthropic.com/news/claude-sonnet-5","notes":null},
  {"id":"anthropic-20260630-sonnet5-frontiercode-main","model_slug":"claude-sonnet-5","category":"coding","benchmark_name":"FrontierCode 1.1 Main","benchmark_version":null,"score_numeric":38.8,"score_display":"38.8%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":"Anthropic launch evaluation","evaluator":"Anthropic","evaluation_date":"2026-06-30","source":"https://www.anthropic.com/news/claude-sonnet-5","notes":null},
  {"id":"anthropic-20260630-sonnet5-terminal21","model_slug":"claude-sonnet-5","category":"coding","benchmark_name":"Terminal-Bench 2.1","benchmark_version":null,"score_numeric":80.4,"score_display":"80.4%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":"Terminus-2 public harness","evaluator":"Anthropic","evaluation_date":"2026-06-30","source":"https://www.anthropic.com/news/claude-sonnet-5","notes":null},
  {"id":"anthropic-20260630-sonnet5-hle-no-tools","model_slug":"claude-sonnet-5","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":null,"score_numeric":43.2,"score_display":"43.2%","score_unit":"percent","tools":false,"reasoning_effort":"high","harness":"Anthropic launch evaluation","evaluator":"Anthropic","evaluation_date":"2026-06-30","source":"https://www.anthropic.com/news/claude-sonnet-5","notes":null},
  {"id":"anthropic-20260630-sonnet5-hle-tools","model_slug":"claude-sonnet-5","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":null,"score_numeric":57.4,"score_display":"57.4%","score_unit":"percent","tools":true,"reasoning_effort":"high","harness":"Anthropic launch evaluation","evaluator":"Anthropic","evaluation_date":"2026-06-30","source":"https://www.anthropic.com/news/claude-sonnet-5","notes":null},
  {"id":"anthropic-20260630-sonnet5-browsecomp","model_slug":"claude-sonnet-5","category":"agentic_computer_use","benchmark_name":"BrowseComp","benchmark_version":"Single-agent","score_numeric":84.7,"score_display":"84.7%","score_unit":"percent","tools":true,"reasoning_effort":"high","harness":"Anthropic launch evaluation","evaluator":"Anthropic","evaluation_date":"2026-06-30","source":"https://www.anthropic.com/news/claude-sonnet-5","notes":null},
  {"id":"anthropic-20260630-sonnet5-osworld-verified","model_slug":"claude-sonnet-5","category":"agentic_computer_use","benchmark_name":"OSWorld-Verified","benchmark_version":null,"score_numeric":81.2,"score_display":"81.2%","score_unit":"percent","tools":true,"reasoning_effort":"high","harness":"Anthropic launch evaluation","evaluator":"Anthropic","evaluation_date":"2026-06-30","source":"https://www.anthropic.com/news/claude-sonnet-5","notes":null},
  {"id":"aa-20260630-sonnet5-gdpval-v2","model_slug":"claude-sonnet-5","category":"agentic_computer_use","benchmark_name":"GDPval-AA v2","benchmark_version":null,"score_numeric":1618,"score_display":"1618 Elo","score_unit":"Elo","tools":true,"reasoning_effort":"high","harness":"Artificial Analysis","evaluator":"Artificial Analysis","evaluation_date":"2026-06-30","source":"https://www.anthropic.com/news/claude-sonnet-5","notes":"GDPval-AA v2."},
  {"id":"anthropic-20260630-sonnet5-toolathlon","model_slug":"claude-sonnet-5","category":"agentic_computer_use","benchmark_name":"Toolathlon","benchmark_version":null,"score_numeric":54.3,"score_display":"54.3%","score_unit":"percent","tools":true,"reasoning_effort":"high","harness":"Anthropic launch evaluation","evaluator":"Anthropic","evaluation_date":"2026-06-30","source":"https://www.anthropic.com/news/claude-sonnet-5","notes":null},
  {"id":"anthropic-20260630-sonnet5-automation","model_slug":"claude-sonnet-5","category":"agentic_computer_use","benchmark_name":"AutomationBench","benchmark_version":null,"score_numeric":13.5,"score_display":"13.5%","score_unit":"percent","tools":true,"reasoning_effort":"high","harness":"Anthropic launch evaluation","evaluator":"Anthropic","evaluation_date":"2026-06-30","source":"https://www.anthropic.com/news/claude-sonnet-5","notes":null},

  {"id":"anthropic-20260217-sonnet46-swe-verified","model_slug":"claude-sonnet-4-6","category":"coding","benchmark_name":"SWE-bench Verified","benchmark_version":"Verified","score_numeric":79.6,"score_display":"79.6%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":"Anthropic launch evaluation","evaluator":"Anthropic","evaluation_date":"2026-02-17","source":"https://www.anthropic.com/news/claude-sonnet-4-6","notes":null},
  {"id":"anthropic-20260630-sonnet46-swe-pro","model_slug":"claude-sonnet-4-6","category":"coding","benchmark_name":"SWE-bench Pro","benchmark_version":null,"score_numeric":58.1,"score_display":"58.1%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":"Anthropic Sonnet 5 comparison table","evaluator":"Anthropic","evaluation_date":"2026-06-30","source":"https://www.anthropic.com/news/claude-sonnet-5","notes":null},
  {"id":"anthropic-20260217-sonnet46-terminal20","model_slug":"claude-sonnet-4-6","category":"coding","benchmark_name":"Terminal-Bench 2.0","benchmark_version":null,"score_numeric":59.1,"score_display":"59.1%","score_unit":"percent","tools":null,"reasoning_effort":"thinking disabled","harness":"Terminus-2","evaluator":"Anthropic","evaluation_date":"2026-02-17","source":"https://www.anthropic.com/news/claude-sonnet-4-6","notes":"Anthropic reports this run with thinking disabled."},
  {"id":"anthropic-20260630-sonnet46-terminal21","model_slug":"claude-sonnet-4-6","category":"coding","benchmark_name":"Terminal-Bench 2.1","benchmark_version":null,"score_numeric":67.0,"score_display":"67.0%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":"Terminus-2 public harness","evaluator":"Anthropic","evaluation_date":"2026-06-30","source":"https://www.anthropic.com/news/claude-sonnet-5","notes":"Later Terminal-Bench 2.1 run; kept distinct from the original Terminal-Bench 2.0 result."},
  {"id":"anthropic-20260630-sonnet46-hle-no-tools","model_slug":"claude-sonnet-4-6","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":null,"score_numeric":34.6,"score_display":"34.6%","score_unit":"percent","tools":false,"reasoning_effort":"high","harness":"Updated Anthropic grader","evaluator":"Anthropic","evaluation_date":"2026-06-30","source":"https://www.anthropic.com/news/claude-sonnet-5","notes":"Updated grader value from Anthropic's Sonnet 5 comparison table."},
  {"id":"anthropic-20260630-sonnet46-hle-tools","model_slug":"claude-sonnet-4-6","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":null,"score_numeric":46.8,"score_display":"46.8%","score_unit":"percent","tools":true,"reasoning_effort":"high","harness":"Updated Anthropic grader","evaluator":"Anthropic","evaluation_date":"2026-06-30","source":"https://www.anthropic.com/news/claude-sonnet-5","notes":"Updated grader value from Anthropic's Sonnet 5 comparison table."},
  {"id":"anthropic-20260630-sonnet46-browsecomp","model_slug":"claude-sonnet-4-6","category":"agentic_computer_use","benchmark_name":"BrowseComp","benchmark_version":"Single-agent","score_numeric":74.01,"score_display":"74.01%","score_unit":"percent","tools":true,"reasoning_effort":"high","harness":"Anthropic system-card revised value","evaluator":"Anthropic","evaluation_date":"2026-06-30","source":"https://www.anthropic.com/news/claude-sonnet-5","notes":"Uses the revised single-agent score rather than the earlier launch-table value."},
  {"id":"anthropic-20260630-sonnet46-osworld-verified","model_slug":"claude-sonnet-4-6","category":"agentic_computer_use","benchmark_name":"OSWorld-Verified","benchmark_version":null,"score_numeric":78.5,"score_display":"78.5%","score_unit":"percent","tools":true,"reasoning_effort":"high","harness":"Updated Anthropic methodology","evaluator":"Anthropic","evaluation_date":"2026-06-30","source":"https://www.anthropic.com/news/claude-sonnet-5","notes":null},
  {"id":"aa-20260630-sonnet46-gdpval-v2","model_slug":"claude-sonnet-4-6","category":"agentic_computer_use","benchmark_name":"GDPval-AA v2","benchmark_version":null,"score_numeric":1395,"score_display":"1395 Elo","score_unit":"Elo","tools":true,"reasoning_effort":"high","harness":"Artificial Analysis","evaluator":"Artificial Analysis","evaluation_date":"2026-06-30","source":"https://www.anthropic.com/news/claude-sonnet-5","notes":"GDPval-AA v2."},
  {"id":"anthropic-20260217-sonnet46-arcagi2","model_slug":"claude-sonnet-4-6","category":"math_reasoning","benchmark_name":"ARC-AGI","benchmark_version":"2","score_numeric":58.3,"score_display":"58.3%","score_unit":"percent","tools":false,"reasoning_effort":"high","harness":"ARC-AGI-2 evaluation","evaluator":"Anthropic","evaluation_date":"2026-02-17","source":"https://www.anthropic.com/news/claude-sonnet-4-6","notes":null}
]
$benchmarks$::jsonb) AS b(
  id text, model_slug text, category text, benchmark_name text, benchmark_version text,
  score_numeric double precision, score_display text, score_unit text, tools boolean,
  reasoning_effort text, harness text, evaluator text, evaluation_date text, source text, notes text
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

-- Fill only empty benchmark cells in already-authored comparison pages.
-- Normalized benchmark data remains authoritative for database-generated comparisons.
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
    END AS rendered
  FROM model_benchmarks AS b
  WHERE b.model_slug IN (
    'claude-opus-5','claude-opus-4-8','claude-opus-4-7','claude-opus-4-6',
    'claude-sonnet-5','claude-sonnet-4-6'
  )
),
benchmark_grouped AS (
  SELECT
    model_slug,
    benchmark_name,
    string_agg(rendered, ' · ' ORDER BY evaluation_date NULLS LAST, id) AS rendered
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
            COALESCE((
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
                            WHEN COALESCE(row_entry.row_value->>1, '') = ''
                             AND bench_a.rendered IS NOT NULL
                            THEN bench_a.rendered
                            ELSE COALESCE(row_entry.row_value->>1, '')
                          END
                        ),
                        false
                      ),
                      '{2}',
                      to_jsonb(
                        CASE
                          WHEN COALESCE(row_entry.row_value->>2, '') = ''
                           AND bench_b.rendered IS NOT NULL
                          THEN bench_b.rendered
                          ELSE COALESCE(row_entry.row_value->>2, '')
                        END
                      ),
                      false
                    )
                  ELSE row_entry.row_value
                END
                ORDER BY row_entry.row_ordinality
              )
              FROM jsonb_array_elements(block_entry.block->'rows') WITH ORDINALITY
                AS row_entry(row_value, row_ordinality)
              LEFT JOIN benchmark_grouped AS bench_a
                ON bench_a.model_slug = c.metadata->>'modelA'
               AND lower(bench_a.benchmark_name) = lower(COALESCE(row_entry.row_value->>0, ''))
              LEFT JOIN benchmark_grouped AS bench_b
                ON bench_b.model_slug = c.metadata->>'modelB'
               AND lower(bench_b.benchmark_name) = lower(COALESCE(row_entry.row_value->>0, ''))
            ), '[]'::jsonb),
            true
          )
        ELSE block_entry.block
      END
      ORDER BY block_entry.block_ordinality
    ) AS blocks
  FROM content_items AS c
  CROSS JOIN LATERAL jsonb_array_elements(COALESCE(c.blocks, '[]'::jsonb)) WITH ORDINALITY
    AS block_entry(block, block_ordinality)
  WHERE c.kind = 'comparison'
    AND (
      c.metadata->>'modelA' IN (
        'claude-opus-5','claude-opus-4-8','claude-opus-4-7','claude-opus-4-6',
        'claude-sonnet-5','claude-sonnet-4-6'
      )
      OR c.metadata->>'modelB' IN (
        'claude-opus-5','claude-opus-4-8','claude-opus-4-7','claude-opus-4-6',
        'claude-sonnet-5','claude-sonnet-4-6'
      )
    )
  GROUP BY c.id, c.metadata
)
UPDATE content_items AS c
SET blocks = rebuilt.blocks, updated_at = NOW()
FROM rebuilt
WHERE c.id = rebuilt.id
  AND c.blocks IS DISTINCT FROM rebuilt.blocks;
