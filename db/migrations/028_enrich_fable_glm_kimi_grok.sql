-- Enrich Claude Fable 5, GLM-5.2, Kimi K2.6, Grok 4.6 and Grok 4.5.
-- Existing canonical rows are updated in place. Benchmark provenance remains normalized;
-- materialized comparisons are only filled where the authored benchmark cell is blank.

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
    "slug":"claude-fable-5",
    "name":"Claude Fable 5",
    "developer":"Anthropic",
    "release_date":"2026-06-09",
    "ga_date":"2026-06-09",
    "access":"proprietary",
    "family":"Claude Fable 5",
    "comparison_data":{
      "Developer":"Anthropic",
      "Release date":"2026-06-09",
      "API model ID":"claude-fable-5",
      "Context window":"1M tokens",
      "Max output":"128K tokens",
      "Knowledge cutoff":"Jan 2026",
      "Reasoning / effort":"Adaptive thinking (always on); high default",
      "Input / 1M tokens":"$10",
      "Cached input / 1M":"$1",
      "Cache write / 1M":"$12.50 (5 min) / $20 (1 hr)",
      "Output / 1M tokens":"$50",
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
      {"title":"Claude Fable 5 launch","url":"https://www.anthropic.com/news/claude-fable-5-mythos-5"},
      {"title":"Claude Fable 5 platform docs","url":"https://platform.claude.com/docs/en/models/fable-5/introducing-claude-fable-5-and-claude-mythos-5"},
      {"title":"Claude pricing","url":"https://platform.claude.com/docs/en/about-claude/pricing"}
    ],
    "notes":"Released June 9, temporarily suspended June 12 under an export-control directive, and restored globally July 1. Current API status is active.",
    "metadata":{"verification_pass":"frontier-followup-2026-09-07","catalog_status":"active","default_effort":"high"}
  },
  {
    "slug":"glm-5-2",
    "name":"GLM-5.2",
    "developer":"Z.ai",
    "release_date":"2026-06-24",
    "ga_date":"2026-06-24",
    "access":"open_source",
    "family":"GLM-5",
    "comparison_data":{
      "Developer":"Z.ai",
      "Release date":"2026-06-24",
      "API model ID":"glm-5.2",
      "Context window":"1M tokens",
      "Max output":"128K tokens",
      "Reasoning / effort":"Multiple thinking modes; low / high / max supported",
      "Input / 1M tokens":"$1.40",
      "Cached input / 1M":"$0.26",
      "Output / 1M tokens":"$4.40",
      "Text input":"Yes",
      "Image / vision input":"No",
      "Audio input":"No",
      "Video input":"No",
      "Text output":"Yes",
      "Image output":"No",
      "Audio output":"No",
      "Video output":"No",
      "Tool / function calling":"Yes; function calling, MCP and structured output",
      "Computer use":"Via agent harnesses; no native computer-use API documented",
      "API access":"Yes",
      "Product access":"Z.ai API + open weights",
      "Weights / license":"Open source — MIT"
    },
    "sources":[
      {"title":"GLM-5.2 developer docs","url":"https://docs.z.ai/guides/llm/glm-5.2"},
      {"title":"GLM-5.2 official weights","url":"https://huggingface.co/zai-org/GLM-5.2"},
      {"title":"Z.ai pricing","url":"https://docs.z.ai/guides/overview/pricing"},
      {"title":"GLM-5.3-Flash comparison","url":"https://autoclaw.z.ai/blog/model/glm-5.3-flash/"}
    ],
    "notes":"753B-parameter open-source flagship with a solid 1M context window. Pricing is Z.ai's current API rate.",
    "metadata":{"verification_pass":"frontier-followup-2026-09-07","parameters_total":"753B","license":"MIT"}
  },
  {
    "slug":"kimi-k2-6",
    "name":"Kimi K2.6",
    "developer":"Moonshot AI",
    "release_date":"2026-04-20",
    "ga_date":"2026-04-20",
    "access":"open_weights",
    "family":"Kimi K2",
    "comparison_data":{
      "Developer":"Moonshot AI",
      "Release date":"2026-04-20",
      "API model ID":"kimi-k2.6",
      "Context window":"256K tokens",
      "Reasoning / effort":"Thinking and non-thinking modes",
      "Text input":"Yes",
      "Image / vision input":"Yes",
      "Audio input":"No native audio input documented",
      "Video input":"No native video input documented",
      "Text output":"Yes",
      "Image output":"No",
      "Audio output":"No",
      "Video output":"No",
      "Tool / function calling":"Yes",
      "Computer use":"Via Kimi Agent / Agent Swarm",
      "API access":"Yes",
      "Product access":"Kimi + Kimi Agent + Kimi API + open weights",
      "Weights / license":"Open weights — Modified MIT"
    },
    "sources":[
      {"title":"Kimi K2.6 official model card","url":"https://huggingface.co/moonshotai/Kimi-K2.6"},
      {"title":"Kimi Agent overview","url":"https://www.kimi.com/en/help/agent/agent-overview"},
      {"title":"Kimi Agent Swarm","url":"https://www.kimi.com/en/help/agent/agent-swarm"}
    ],
    "notes":"1T-parameter MoE with 32B active parameters, native vision, 256K context and upgraded Agent Swarm. Hosted pricing is intentionally omitted because a directly attributable first-party K2.6 rate was not available in the verified source set.",
    "metadata":{"verification_pass":"frontier-followup-2026-09-07","parameters_total":"1T","parameters_active":"32B","license":"Modified MIT"}
  },
  {
    "slug":"grok-4-6",
    "name":"Grok 4.6",
    "developer":"xAI",
    "release_date":"2026-08-12",
    "ga_date":"2026-08-12",
    "access":"proprietary",
    "family":"Grok 4",
    "comparison_data":{
      "Developer":"xAI",
      "Release date":"2026-08-12",
      "API model ID":"grok-4.6",
      "Context window":"500K tokens",
      "Max output":"No text output limit",
      "Knowledge cutoff":"Jan 2026",
      "Reasoning / effort":"low / medium / high / xhigh; high default",
      "Input / 1M tokens":"$2 <=200K / $4 >200K",
      "Cached input / 1M":"$0.50 <=200K / $1 >200K",
      "Output / 1M tokens":"$6 <=200K / $12 >200K",
      "Long-context surcharge":">=200K prompt tokens use the higher tier",
      "Text input":"Yes",
      "Image / vision input":"Yes",
      "Audio input":"No native audio input documented",
      "Video input":"No native video input documented",
      "Text output":"Yes",
      "Image output":"No",
      "Audio output":"No",
      "Video output":"No",
      "Tool / function calling":"Yes; function calling, web search, X search, code execution",
      "Computer use":"Via Grok Build / agent environments",
      "API access":"Yes",
      "Product access":"Grok Build + Cursor + xAI API + partner gateways",
      "Weights / license":"Proprietary"
    },
    "sources":[
      {"title":"Introducing Grok 4.6","url":"https://x.ai/news/grok-4-6"},
      {"title":"Grok 4.6 model docs","url":"https://docs.x.ai/developers/models/grok-4.6"},
      {"title":"xAI pricing","url":"https://docs.x.ai/developers/pricing"}
    ],
    "notes":"Current frontier coding/agent model. Fast variant is available at twice the standard token price.",
    "metadata":{"verification_pass":"frontier-followup-2026-09-07","default_effort":"high","catalog_status":"current"}
  },
  {
    "slug":"grok-4-5",
    "name":"Grok 4.5",
    "developer":"xAI",
    "release_date":"2026-07-16",
    "ga_date":"2026-07-16",
    "access":"proprietary",
    "family":"Grok 4",
    "comparison_data":{
      "Developer":"xAI",
      "Release date":"2026-07-16",
      "API model ID":"grok-4.5",
      "Context window":"500K tokens",
      "Reasoning / effort":"low / medium / high; high default",
      "Input / 1M tokens":"$2 <=200K / $4 >200K",
      "Cached input / 1M":"$0.30 <=200K / $0.60 >200K",
      "Output / 1M tokens":"$6 <=200K / $12 >200K",
      "Long-context surcharge":">=200K prompt tokens use the higher tier",
      "Text input":"Yes",
      "Image / vision input":"Yes",
      "Audio input":"No native audio input documented",
      "Video input":"No native video input documented",
      "Text output":"Yes",
      "Image output":"No",
      "Audio output":"No",
      "Video output":"No",
      "Tool / function calling":"Yes",
      "Computer use":"Via Grok Build / agent environments",
      "API access":"Yes",
      "Product access":"Grok Build + Cursor + GitHub Copilot + xAI API",
      "Weights / license":"Proprietary"
    },
    "sources":[
      {"title":"Introducing Grok 4.5","url":"https://x.ai/news/grok-4-5"},
      {"title":"Grok 4.5 model docs","url":"https://docs.x.ai/developers/models/grok-4.5"},
      {"title":"xAI pricing","url":"https://docs.x.ai/developers/pricing"}
    ],
    "notes":"Coding- and agent-focused model. xAI's API release notes show API availability before the July 16 announcement; the catalog release date tracks the public model announcement.",
    "metadata":{"verification_pass":"frontier-followup-2026-09-07","default_effort":"high","catalog_status":"active"}
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
  id, model_slug, category, benchmark_name, benchmark_version,
  score_numeric, score_display, score_unit, tools, reasoning_effort,
  harness, evaluator, evaluation_date, source, notes, updated_at
)
SELECT
  b.id, b.model_slug, b.category, b.benchmark_name, b.benchmark_version,
  b.score_numeric, b.score_display, b.score_unit, b.tools, b.reasoning_effort,
  b.harness, b.evaluator, NULLIF(b.evaluation_date, '')::date, b.source, b.notes, NOW()
FROM jsonb_to_recordset($benchmarks$
[
  {"id":"tq-20260907-fable5-deepswe11","model_slug":"claude-fable-5","category":"coding","benchmark_name":"DeepSWE v1.1","benchmark_version":null,"score_numeric":70.0,"score_display":"70.0%","score_unit":"percent","tools":null,"reasoning_effort":"max","harness":"mini-swe-agent","evaluator":"SpaceXAI","evaluation_date":"2026-08-12","source":"https://x.ai/news/grok-4-6","notes":"Cross-vendor comparison reported by SpaceXAI."},
  {"id":"tq-20260907-fable5-terminal21","model_slug":"claude-fable-5","category":"coding","benchmark_name":"Terminal-Bench 2.1","benchmark_version":null,"score_numeric":84.3,"score_display":"84.3%","score_unit":"percent","tools":null,"reasoning_effort":"max","harness":"Terminus-2","evaluator":"SpaceXAI","evaluation_date":"2026-07-16","source":"https://x.ai/news/grok-4-5","notes":"Cross-vendor comparison reported by SpaceXAI."},
  {"id":"tq-20260907-fable5-swepro","model_slug":"claude-fable-5","category":"coding","benchmark_name":"SWE-bench Pro","benchmark_version":"Public","score_numeric":80.4,"score_display":"80.4%","score_unit":"percent","tools":null,"reasoning_effort":"max","harness":null,"evaluator":"SpaceXAI","evaluation_date":"2026-07-16","source":"https://x.ai/news/grok-4-5","notes":"Cross-vendor comparison reported by SpaceXAI."},
  {"id":"tq-20260907-fable5-frontiercode-ext","model_slug":"claude-fable-5","category":"coding","benchmark_name":"FrontierCode 1.1 Extended","benchmark_version":null,"score_numeric":63.6,"score_display":"63.6%","score_unit":"percent","tools":null,"reasoning_effort":"max","harness":null,"evaluator":"SpaceXAI","evaluation_date":"2026-08-12","source":"https://x.ai/news/grok-4-6","notes":"Cross-vendor comparison reported by SpaceXAI."},
  {"id":"tq-20260907-fable5-terminal30","model_slug":"claude-fable-5","category":"coding","benchmark_name":"Terminal-Bench 3.0","benchmark_version":null,"score_numeric":34.1,"score_display":"34.1%","score_unit":"percent","tools":null,"reasoning_effort":"max","harness":null,"evaluator":"SpaceXAI","evaluation_date":"2026-08-12","source":"https://x.ai/news/grok-4-6","notes":"Cross-vendor comparison reported by SpaceXAI."},
  {"id":"tq-20260907-fable5-terminal-science","model_slug":"claude-fable-5","category":"coding","benchmark_name":"Terminal-Bench Science 0.1","benchmark_version":null,"score_numeric":24.7,"score_display":"24.7%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":"Anthropic reproduction; Claude Code","evaluator":"Anthropic","evaluation_date":"2026-09-01","source":"https://www.anthropic.com/claude/fable","notes":"Anthropic reproduction of the public leaderboard setup; public leaderboard score cited as 21.4%."},
  {"id":"tq-20260907-fable5-cursor32","model_slug":"claude-fable-5","category":"coding","benchmark_name":"CursorBench","benchmark_version":"3.2","score_numeric":70.5,"score_display":"70.5%","score_unit":"percent","tools":null,"reasoning_effort":"max","harness":"CursorBench 3.2","evaluator":"SpaceXAI","evaluation_date":"2026-08-12","source":"https://x.ai/news/grok-4-6","notes":"Cross-vendor comparison reported by SpaceXAI."},
  {"id":"tq-20260907-fable5-gdpv2","model_slug":"claude-fable-5","category":"agentic_computer_use","benchmark_name":"GDPval-AA v2","benchmark_version":null,"score_numeric":1741,"score_display":"1741 Elo","score_unit":"Elo","tools":true,"reasoning_effort":"max","harness":"Artificial Analysis","evaluator":"SpaceXAI","evaluation_date":"2026-08-12","source":"https://x.ai/news/grok-4-6","notes":"Cross-vendor comparison; underlying evaluation is Artificial Analysis."},

  {"id":"tq-20260907-glm52-terminal21","model_slug":"glm-5-2","category":"coding","benchmark_name":"Terminal-Bench 2.1","benchmark_version":null,"score_numeric":81.0,"score_display":"81.0%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":"Claude Code","evaluator":"Z.ai","evaluation_date":"2026-06-24","source":"https://docs.z.ai/guides/llm/glm-5.2","notes":null},
  {"id":"tq-20260907-glm52-swepro","model_slug":"glm-5-2","category":"coding","benchmark_name":"SWE-bench Pro","benchmark_version":"Public","score_numeric":62.1,"score_display":"62.1%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"Z.ai","evaluation_date":"2026-06-24","source":"https://docs.z.ai/guides/llm/glm-5.2","notes":null},
  {"id":"tq-20260907-glm52-deepswe","model_slug":"glm-5-2","category":"coding","benchmark_name":"DeepSWE v1.1","benchmark_version":null,"score_numeric":46.2,"score_display":"46.2%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"Z.ai","evaluation_date":"2026-09-06","source":"https://autoclaw.z.ai/blog/model/glm-5.3-flash/","notes":"GLM-5.2 comparison value in Z.ai's GLM-5.3-Flash evaluation table."},
  {"id":"tq-20260907-glm52-toolathlon","model_slug":"glm-5-2","category":"agentic_computer_use","benchmark_name":"Toolathlon","benchmark_version":"Verified","score_numeric":59.9,"score_display":"59.9%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":null,"evaluator":"Z.ai","evaluation_date":"2026-09-06","source":"https://autoclaw.z.ai/blog/model/glm-5.3-flash/","notes":null},
  {"id":"tq-20260907-glm52-automation","model_slug":"glm-5-2","category":"agentic_computer_use","benchmark_name":"AutomationBench","benchmark_version":"v1.0.6","score_numeric":26.2,"score_display":"26.2%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":null,"evaluator":"Z.ai","evaluation_date":"2026-09-06","source":"https://autoclaw.z.ai/blog/model/glm-5.3-flash/","notes":null},
  {"id":"tq-20260907-glm52-agents-last-exam","model_slug":"glm-5-2","category":"agentic_computer_use","benchmark_name":"Agents' Last Exam","benchmark_version":null,"score_numeric":20.4,"score_display":"20.4%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":null,"evaluator":"Z.ai","evaluation_date":"2026-09-06","source":"https://autoclaw.z.ai/blog/model/glm-5.3-flash/","notes":null},
  {"id":"tq-20260907-glm52-hle-tools","model_slug":"glm-5-2","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":null,"score_numeric":54.7,"score_display":"54.7%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":null,"evaluator":"Z.ai","evaluation_date":"2026-09-06","source":"https://autoclaw.z.ai/blog/model/glm-5.3-flash/","notes":null},
  {"id":"tq-20260907-glm52-gdpv2","model_slug":"glm-5-2","category":"agentic_computer_use","benchmark_name":"GDPval-AA v2","benchmark_version":null,"score_numeric":1504,"score_display":"1504 Elo","score_unit":"Elo","tools":true,"reasoning_effort":null,"harness":"Artificial Analysis","evaluator":"Z.ai","evaluation_date":"2026-09-06","source":"https://autoclaw.z.ai/blog/model/glm-5.3-flash/","notes":null},

  {"id":"tq-20260907-k26-sweverified","model_slug":"kimi-k2-6","category":"coding","benchmark_name":"SWE-bench Verified","benchmark_version":null,"score_numeric":80.2,"score_display":"80.2%","score_unit":"percent","tools":null,"reasoning_effort":"thinking","harness":"Moonshot in-house SWE-agent adaptation","evaluator":"Moonshot AI","evaluation_date":"2026-04-20","source":"https://huggingface.co/moonshotai/Kimi-K2.6","notes":"Average over 10 independent runs."},
  {"id":"tq-20260907-k26-swepro","model_slug":"kimi-k2-6","category":"coding","benchmark_name":"SWE-bench Pro","benchmark_version":"Public","score_numeric":58.6,"score_display":"58.6%","score_unit":"percent","tools":null,"reasoning_effort":"thinking","harness":"Moonshot in-house SWE-agent adaptation","evaluator":"Moonshot AI","evaluation_date":"2026-04-20","source":"https://huggingface.co/moonshotai/Kimi-K2.6","notes":"Average over 10 independent runs."},
  {"id":"tq-20260907-k26-terminal20","model_slug":"kimi-k2-6","category":"coding","benchmark_name":"Terminal-Bench 2.0","benchmark_version":"Terminus-2","score_numeric":66.7,"score_display":"66.7%","score_unit":"percent","tools":null,"reasoning_effort":"thinking","harness":"Terminus-2; preserve thinking","evaluator":"Moonshot AI","evaluation_date":"2026-04-20","source":"https://huggingface.co/moonshotai/Kimi-K2.6","notes":null},
  {"id":"tq-20260907-k26-livecode","model_slug":"kimi-k2-6","category":"coding","benchmark_name":"LiveCodeBench","benchmark_version":"v6","score_numeric":89.6,"score_display":"89.6%","score_unit":"percent","tools":null,"reasoning_effort":"thinking","harness":null,"evaluator":"Moonshot AI","evaluation_date":"2026-04-20","source":"https://huggingface.co/moonshotai/Kimi-K2.6","notes":null},
  {"id":"tq-20260907-k26-aime","model_slug":"kimi-k2-6","category":"math_reasoning","benchmark_name":"AIME","benchmark_version":"2026","score_numeric":96.4,"score_display":"96.4%","score_unit":"percent","tools":false,"reasoning_effort":"thinking","harness":null,"evaluator":"Moonshot AI","evaluation_date":"2026-04-20","source":"https://huggingface.co/moonshotai/Kimi-K2.6","notes":null},
  {"id":"tq-20260907-k26-hmmt","model_slug":"kimi-k2-6","category":"math_reasoning","benchmark_name":"HMMT","benchmark_version":"2026 Feb","score_numeric":92.7,"score_display":"92.7%","score_unit":"percent","tools":false,"reasoning_effort":"thinking","harness":null,"evaluator":"Moonshot AI","evaluation_date":"2026-04-20","source":"https://huggingface.co/moonshotai/Kimi-K2.6","notes":null},
  {"id":"tq-20260907-k26-gpqa","model_slug":"kimi-k2-6","category":"knowledge","benchmark_name":"GPQA Diamond","benchmark_version":null,"score_numeric":90.5,"score_display":"90.5%","score_unit":"percent","tools":false,"reasoning_effort":"thinking","harness":null,"evaluator":"Moonshot AI","evaluation_date":"2026-04-20","source":"https://huggingface.co/moonshotai/Kimi-K2.6","notes":null},
  {"id":"tq-20260907-k26-hle-no-tools","model_slug":"kimi-k2-6","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":"Full","score_numeric":34.7,"score_display":"34.7%","score_unit":"percent","tools":false,"reasoning_effort":"thinking","harness":null,"evaluator":"Moonshot AI","evaluation_date":"2026-04-20","source":"https://huggingface.co/moonshotai/Kimi-K2.6","notes":null},
  {"id":"tq-20260907-k26-hle-tools","model_slug":"kimi-k2-6","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":"Full","score_numeric":54.0,"score_display":"54.0%","score_unit":"percent","tools":true,"reasoning_effort":"thinking","harness":null,"evaluator":"Moonshot AI","evaluation_date":"2026-04-20","source":"https://huggingface.co/moonshotai/Kimi-K2.6","notes":null},
  {"id":"tq-20260907-k26-browsecomp","model_slug":"kimi-k2-6","category":"agentic_computer_use","benchmark_name":"BrowseComp","benchmark_version":null,"score_numeric":83.2,"score_display":"83.2%","score_unit":"percent","tools":true,"reasoning_effort":"thinking","harness":"Single-agent","evaluator":"Moonshot AI","evaluation_date":"2026-04-20","source":"https://huggingface.co/moonshotai/Kimi-K2.6","notes":"Agent Swarm result is intentionally not added as a second display value to avoid collapsing distinct harnesses in the UI."},
  {"id":"tq-20260907-k26-toolathlon","model_slug":"kimi-k2-6","category":"agentic_computer_use","benchmark_name":"Toolathlon","benchmark_version":null,"score_numeric":50.0,"score_display":"50.0%","score_unit":"percent","tools":true,"reasoning_effort":"thinking","harness":null,"evaluator":"Moonshot AI","evaluation_date":"2026-04-20","source":"https://huggingface.co/moonshotai/Kimi-K2.6","notes":null},
  {"id":"tq-20260907-k26-osworld","model_slug":"kimi-k2-6","category":"agentic_computer_use","benchmark_name":"OSWorld-Verified","benchmark_version":null,"score_numeric":73.1,"score_display":"73.1%","score_unit":"percent","tools":true,"reasoning_effort":"thinking","harness":null,"evaluator":"Moonshot AI","evaluation_date":"2026-04-20","source":"https://huggingface.co/moonshotai/Kimi-K2.6","notes":null},

  {"id":"tq-20260907-grok46-deepswe","model_slug":"grok-4-6","category":"coding","benchmark_name":"DeepSWE v1.1","benchmark_version":null,"score_numeric":65.9,"score_display":"65.9%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":null,"evaluator":"SpaceXAI","evaluation_date":"2026-08-12","source":"https://x.ai/news/grok-4-6","notes":null},
  {"id":"tq-20260907-grok46-cursor","model_slug":"grok-4-6","category":"coding","benchmark_name":"CursorBench","benchmark_version":"3.2","score_numeric":69.9,"score_display":"69.9%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":"CursorBench 3.2","evaluator":"SpaceXAI","evaluation_date":"2026-08-12","source":"https://x.ai/news/grok-4-6","notes":null},
  {"id":"tq-20260907-grok46-frontiercode","model_slug":"grok-4-6","category":"coding","benchmark_name":"FrontierCode 1.1 Extended","benchmark_version":null,"score_numeric":61.3,"score_display":"61.3%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":null,"evaluator":"SpaceXAI","evaluation_date":"2026-08-12","source":"https://x.ai/news/grok-4-6","notes":null},
  {"id":"tq-20260907-grok46-terminal30","model_slug":"grok-4-6","category":"coding","benchmark_name":"Terminal-Bench 3.0","benchmark_version":null,"score_numeric":26.0,"score_display":"26.0%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":null,"evaluator":"SpaceXAI","evaluation_date":"2026-08-12","source":"https://x.ai/news/grok-4-6","notes":null},
  {"id":"tq-20260907-grok46-gdpv2","model_slug":"grok-4-6","category":"agentic_computer_use","benchmark_name":"GDPval-AA v2","benchmark_version":null,"score_numeric":1753,"score_display":"1753 Elo","score_unit":"Elo","tools":true,"reasoning_effort":"high","harness":"Artificial Analysis","evaluator":"SpaceXAI","evaluation_date":"2026-08-12","source":"https://x.ai/news/grok-4-6","notes":null},

  {"id":"tq-20260907-grok45-terminal21","model_slug":"grok-4-5","category":"coding","benchmark_name":"Terminal-Bench 2.1","benchmark_version":null,"score_numeric":83.3,"score_display":"83.3%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":null,"evaluator":"SpaceXAI","evaluation_date":"2026-07-16","source":"https://x.ai/news/grok-4-5","notes":null},
  {"id":"tq-20260907-grok45-swepro","model_slug":"grok-4-5","category":"coding","benchmark_name":"SWE-bench Pro","benchmark_version":"Public","score_numeric":64.7,"score_display":"64.7%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":null,"evaluator":"SpaceXAI","evaluation_date":"2026-07-16","source":"https://x.ai/news/grok-4-5","notes":null},
  {"id":"tq-20260907-grok45-deepswe","model_slug":"grok-4-5","category":"coding","benchmark_name":"DeepSWE v1.1","benchmark_version":null,"score_numeric":54.0,"score_display":"54.0%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":null,"evaluator":"SpaceXAI","evaluation_date":"2026-08-12","source":"https://x.ai/news/grok-4-6","notes":"Later SpaceXAI comparison value; retained instead of the earlier 53% launch-table value."},
  {"id":"tq-20260907-grok45-cursor","model_slug":"grok-4-5","category":"coding","benchmark_name":"CursorBench","benchmark_version":"3.2","score_numeric":66.7,"score_display":"66.7%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":"CursorBench 3.2","evaluator":"SpaceXAI","evaluation_date":"2026-08-12","source":"https://x.ai/news/grok-4-6","notes":null},
  {"id":"tq-20260907-grok45-frontiercode","model_slug":"grok-4-5","category":"coding","benchmark_name":"FrontierCode 1.1 Extended","benchmark_version":null,"score_numeric":56.6,"score_display":"56.6%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":null,"evaluator":"SpaceXAI","evaluation_date":"2026-08-12","source":"https://x.ai/news/grok-4-6","notes":null},
  {"id":"tq-20260907-grok45-terminal30","model_slug":"grok-4-5","category":"coding","benchmark_name":"Terminal-Bench 3.0","benchmark_version":null,"score_numeric":15.7,"score_display":"15.7%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":null,"evaluator":"SpaceXAI","evaluation_date":"2026-08-12","source":"https://x.ai/news/grok-4-6","notes":null},
  {"id":"tq-20260907-grok45-gdpv2","model_slug":"grok-4-5","category":"agentic_computer_use","benchmark_name":"GDPval-AA v2","benchmark_version":null,"score_numeric":1526,"score_display":"1526 Elo","score_unit":"Elo","tools":true,"reasoning_effort":"high","harness":"Artificial Analysis","evaluator":"SpaceXAI","evaluation_date":"2026-08-12","source":"https://x.ai/news/grok-4-6","notes":null}
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
  WHERE b.model_slug IN ('claude-fable-5', 'glm-5-2', 'kimi-k2-6', 'grok-4-6', 'grok-4-5')
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
                        to_jsonb(CASE
                          WHEN COALESCE(row_entry.row_value->>1, '') = '' AND bench_a.rendered IS NOT NULL
                          THEN bench_a.rendered
                          ELSE COALESCE(row_entry.row_value->>1, '')
                        END),
                        false
                      ),
                      '{2}',
                      to_jsonb(CASE
                        WHEN COALESCE(row_entry.row_value->>2, '') = '' AND bench_b.rendered IS NOT NULL
                        THEN bench_b.rendered
                        ELSE COALESCE(row_entry.row_value->>2, '')
                      END),
                      false
                    )
                  ELSE row_entry.row_value
                END
                ORDER BY row_entry.row_ordinality
              )
              FROM jsonb_array_elements(block_entry.block->'rows') WITH ORDINALITY AS row_entry(row_value, row_ordinality)
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
  CROSS JOIN LATERAL jsonb_array_elements(COALESCE(c.blocks, '[]'::jsonb)) WITH ORDINALITY AS block_entry(block, block_ordinality)
  WHERE c.kind = 'comparison'
    AND (c.metadata->>'modelA' IN ('claude-fable-5', 'glm-5-2', 'kimi-k2-6', 'grok-4-6', 'grok-4-5')
      OR c.metadata->>'modelB' IN ('claude-fable-5', 'glm-5-2', 'kimi-k2-6', 'grok-4-6', 'grok-4-5'))
  GROUP BY c.id, c.metadata
)
UPDATE content_items AS c
SET blocks = rebuilt.blocks, updated_at = NOW()
FROM rebuilt
WHERE c.id = rebuilt.id
  AND c.blocks IS DISTINCT FROM rebuilt.blocks;
