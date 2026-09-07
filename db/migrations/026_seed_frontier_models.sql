-- Enrich the late-2026 frontier-model catalog with verified model specs and normalized benchmarks.
-- Six slugs already exist in earlier catalog seeds and are updated in place. GLM-5.3 is inserted here.
-- Benchmark provenance stays normalized in model_benchmarks; comparison rendering omits provenance-only qualifiers.

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
    "slug":"kimi-k3",
    "name":"Kimi K3",
    "developer":"Moonshot AI",
    "release_date":"2026-07-16",
    "ga_date":"2026-07-16",
    "access":"open_weights",
    "family":"Kimi K3",
    "comparison_data":{
      "Developer":"Moonshot AI",
      "Release date":"2026-07-16",
      "API model ID":"kimi-k3",
      "Context window":"1M tokens",
      "Reasoning / effort":"low / high / max",
      "Input / 1M tokens":"$3.00 cache miss",
      "Cached input / 1M":"$0.30 cache hit",
      "Output / 1M tokens":"$15.00",
      "Text input":"Yes",
      "Image / vision input":"Yes",
      "Audio input":"No native audio input documented",
      "Video input":"Yes",
      "Text output":"Yes",
      "Image output":"No",
      "Audio output":"No",
      "Video output":"No",
      "Tool / function calling":"Yes",
      "Computer use":"Via Kimi agent and coding harnesses",
      "API access":"Yes",
      "Product access":"Kimi + Kimi Work + Kimi Code + Kimi API + weights",
      "Weights / license":"Open weights — Kimi K3 License"
    },
    "sources":[
      {"title":"Kimi K3 research release","url":"https://www.kimi.com/blog/kimi-k3"},
      {"title":"Kimi K3 official model card","url":"https://huggingface.co/moonshotai/Kimi-K3"},
      {"title":"Kimi API pricing","url":"https://platform.moonshot.ai/docs/pricing"}
    ],
    "notes":"2.8T-parameter MoE model with native vision and a 1M context window. Benchmark rows below use the direct Kimi evaluation where available rather than later cross-vendor reruns.",
    "metadata":{"verification_pass":"frontier-models-2026-09-07","parameters_total":"2.8T","architecture":"MoE with Kimi Delta Attention"}
  },
  {
    "slug":"minimax-m3",
    "name":"MiniMax M3",
    "developer":"MiniMax",
    "release_date":"2026-06-01",
    "ga_date":"2026-06-01",
    "access":"open_weights",
    "family":"MiniMax M3",
    "comparison_data":{
      "Developer":"MiniMax",
      "Release date":"2026-06-01",
      "API model ID":"MiniMax-M3",
      "Context window":"1M tokens",
      "Reasoning / effort":"Thinking can be enabled or disabled",
      "Input / 1M tokens":"$0.30 <=512K / $0.60 >512K–1M",
      "Cached input / 1M":"$0.06 <=512K / $0.12 >512K–1M",
      "Output / 1M tokens":"$1.20 <=512K / $2.40 >512K–1M",
      "Long-context surcharge":">512K uses the long-context tier",
      "Text input":"Yes",
      "Image / vision input":"Yes",
      "Audio input":"No native audio input documented",
      "Video input":"Yes",
      "Text output":"Yes",
      "Image output":"No",
      "Audio output":"No",
      "Video output":"No",
      "Tool / function calling":"Yes",
      "Computer use":"Yes",
      "API access":"Yes",
      "Product access":"MiniMax Code + MiniMax API + weights",
      "Weights / license":"Open weights — MiniMax Community License"
    },
    "sources":[
      {"title":"MiniMax M3","url":"https://www.minimax.io/news/minimax-m3"},
      {"title":"MiniMax M3 official weights","url":"https://huggingface.co/MiniMaxAI/MiniMax-M3"},
      {"title":"MiniMax Token Plan pricing","url":"https://www.minimax.io/platform/pricing"}
    ],
    "notes":"Native multimodal agent model with roughly 428B total and 23B active parameters. Current hosted prices reflect MiniMax's permanent 50% Token Plan rate.",
    "metadata":{"verification_pass":"frontier-models-2026-09-07","parameters_total":"~428B","parameters_active":"~23B"}
  },
  {
    "slug":"glm-5-3",
    "name":"GLM-5.3",
    "developer":"Z.ai",
    "release_date":"2026-08-28",
    "ga_date":"2026-08-28",
    "access":"open_weights",
    "family":"GLM-5",
    "comparison_data":{
      "Developer":"Z.ai",
      "Release date":"2026-08-28",
      "API model ID":"glm-5.3; ZHIPU/GLM-5.3 on Alibaba Model Studio",
      "Context window":"1M tokens",
      "Max output":"128K tokens",
      "Reasoning / effort":"low / high / max; max default",
      "Input / 1M tokens":"$1.40 hosted via Alibaba Model Studio Singapore",
      "Cached input / 1M":"$0.26 implicit cache via Alibaba Model Studio Singapore",
      "Output / 1M tokens":"$4.40 hosted via Alibaba Model Studio Singapore",
      "Text input":"Yes",
      "Image / vision input":"No",
      "Audio input":"No",
      "Video input":"No",
      "Text output":"Yes",
      "Image output":"No",
      "Audio output":"No",
      "Video output":"No",
      "Tool / function calling":"Yes",
      "Computer use":"Via agent harnesses; no native computer-use API documented",
      "API access":"Yes",
      "Product access":"Z.ai API + open weights + third-party hosting",
      "Weights / license":"Open weights — GLM-5.3 License"
    },
    "sources":[
      {"title":"GLM-5.3 official model card","url":"https://huggingface.co/zai-org/GLM-5.3"},
      {"title":"GLM-5.3 on Alibaba Model Studio","url":"https://www.alibabacloud.com/help/en/model-studio/models"}
    ],
    "notes":"Text-only GLM-5.2-base derivative whose gains come primarily from post-training. Hosted price fields are explicitly the Alibaba Model Studio Singapore rates, not a claimed universal Z.ai API price.",
    "metadata":{"verification_pass":"frontier-models-2026-09-07","launch_stage":"ga_open_weights"}
  },
  {
    "slug":"deepseek-v4-flash",
    "name":"DeepSeek V4 Flash",
    "developer":"DeepSeek",
    "release_date":"2026-04-24",
    "ga_date":"2026-07-31",
    "access":"open_weights",
    "family":"DeepSeek V4",
    "comparison_data":{
      "Developer":"DeepSeek",
      "Release date":"2026-04-24",
      "API model ID":"deepseek-v4-flash",
      "Context window":"1M tokens",
      "Max output":"384K tokens",
      "Reasoning / effort":"Thinking + non-thinking; low / high / max, high default",
      "Input / 1M tokens":"$0.22 off-peak / $0.44 peak cache miss",
      "Cached input / 1M":"$0.007 off-peak / $0.014 peak cache hit",
      "Output / 1M tokens":"$0.66 off-peak / $1.32 peak",
      "Text input":"Yes",
      "Image / vision input":"No on the main endpoint",
      "Audio input":"No",
      "Video input":"No",
      "Text output":"Yes",
      "Image output":"No",
      "Audio output":"No",
      "Video output":"No",
      "Tool / function calling":"Yes",
      "Computer use":"Via agent harnesses; no native computer-use tool documented",
      "API access":"Yes",
      "Product access":"DeepSeek API + weights",
      "Weights / license":"Open weights — MIT"
    },
    "sources":[
      {"title":"DeepSeek V4 Preview","url":"https://www.deepseek.com/en/news/v4-preview/"},
      {"title":"DeepSeek API updates","url":"https://api-docs.deepseek.com/updates"},
      {"title":"DeepSeek API pricing","url":"https://api-docs.deepseek.com/quick_start/pricing"},
      {"title":"DeepSeek V4 Flash weights","url":"https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash"}
    ],
    "notes":"Current hosted endpoint is the 0731 Flash revision. Vision is a separate experimental V4-Flash-Vision endpoint and is not attributed to this text endpoint.",
    "metadata":{"verification_pass":"frontier-models-2026-09-07","current_revision":"DeepSeek-V4-Flash-0731","parameters_total":"284B","parameters_active":"13B"}
  },
  {
    "slug":"deepseek-v4-pro",
    "name":"DeepSeek V4 Pro",
    "developer":"DeepSeek",
    "release_date":"2026-04-24",
    "ga_date":"2026-08-13",
    "access":"open_weights",
    "family":"DeepSeek V4",
    "comparison_data":{
      "Developer":"DeepSeek",
      "Release date":"2026-04-24",
      "API model ID":"deepseek-v4-pro",
      "Context window":"1M tokens",
      "Max output":"384K tokens",
      "Reasoning / effort":"Thinking + non-thinking; low / high / max, high default",
      "Input / 1M tokens":"$0.66 off-peak / $1.32 peak cache miss",
      "Cached input / 1M":"$0.022 off-peak / $0.044 peak cache hit",
      "Output / 1M tokens":"$1.98 off-peak / $3.96 peak",
      "Text input":"Yes",
      "Image / vision input":"No on the main endpoint",
      "Audio input":"No",
      "Video input":"No",
      "Text output":"Yes",
      "Image output":"No",
      "Audio output":"No",
      "Video output":"No",
      "Tool / function calling":"Yes",
      "Computer use":"Via agent harnesses; no native computer-use tool documented",
      "API access":"Yes",
      "Product access":"DeepSeek API + weights",
      "Weights / license":"Open weights — MIT"
    },
    "sources":[
      {"title":"DeepSeek V4 Preview","url":"https://www.deepseek.com/en/news/v4-preview/"},
      {"title":"DeepSeek API updates","url":"https://api-docs.deepseek.com/updates"},
      {"title":"DeepSeek API pricing","url":"https://api-docs.deepseek.com/quick_start/pricing"},
      {"title":"DeepSeek V4 Pro weights","url":"https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro"}
    ],
    "notes":"Current hosted endpoint is the 0813 Pro revision. Benchmark evidence remains tied to the documented DeepSeek harness rather than being flattened with third-party runs.",
    "metadata":{"verification_pass":"frontier-models-2026-09-07","current_revision":"DeepSeek-V4-Pro-0813","parameters_total":"1.6T","parameters_active":"49B"}
  },
  {
    "slug":"qwen3-8-max",
    "name":"Qwen3.8-Max",
    "developer":"Alibaba / Qwen",
    "release_date":"2026-08-02",
    "ga_date":"2026-08-02",
    "access":"open_weights",
    "family":"Qwen3.8",
    "comparison_data":{
      "Developer":"Alibaba / Qwen",
      "Release date":"2026-08-02",
      "API model ID":"qwen3.8-max",
      "Context window":"1M tokens",
      "Max output":"128K tokens",
      "Reasoning / effort":"Hybrid thinking / non-thinking; thinking enabled by default",
      "Input / 1M tokens":"$2.00 Alibaba Model Studio Singapore",
      "Cached input / 1M":"$0.25 implicit / $0.17 explicit cache read",
      "Cache write / 1M":"$2.50 explicit cache creation",
      "Output / 1M tokens":"$6.00 Alibaba Model Studio Singapore",
      "Text input":"Yes",
      "Image / vision input":"Yes",
      "Audio input":"No native audio input documented",
      "Video input":"Yes",
      "Text output":"Yes",
      "Image output":"No",
      "Audio output":"No",
      "Video output":"No",
      "Tool / function calling":"Yes",
      "Computer use":"Via multimodal agent stacks",
      "API access":"Yes",
      "Product access":"Alibaba Model Studio + Qwen weights",
      "Weights / license":"Open weights — Apache 2.0 counterpart Qwen3.8-2.4T-A95B"
    },
    "sources":[
      {"title":"Qwen3.8-Max release","url":"https://www.alibabacloud.com/help/en/model-studio/newly-released-models"},
      {"title":"Qwen3.8-Max model docs","url":"https://www.alibabacloud.com/help/en/model-studio/models"},
      {"title":"Qwen3.8 open model card","url":"https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B"}
    ],
    "notes":"Hosted flagship model with a 2.4T-parameter open-weight counterpart. Current hosted snapshot is qwen3.8-max-0902; pricing fields use Alibaba Model Studio Singapore international rates.",
    "metadata":{"verification_pass":"frontier-models-2026-09-07","current_snapshot":"qwen3.8-max-0902","parameters_total":"2.4T","parameters_active":"~95B"}
  },
  {
    "slug":"qwen3-8-27b",
    "name":"Qwen3.8-27B",
    "developer":"Alibaba / Qwen",
    "release_date":"2026-08-17",
    "ga_date":"2026-08-17",
    "access":"open_weights",
    "family":"Qwen3.8",
    "comparison_data":{
      "Developer":"Alibaba / Qwen",
      "Release date":"2026-08-17",
      "API model ID":"qwen3.8-27b",
      "Context window":"1M tokens",
      "Max output":"128K tokens",
      "Reasoning / effort":"Hybrid thinking / non-thinking; thinking enabled by default",
      "Input / 1M tokens":"$0.50 Alibaba Model Studio Singapore",
      "Cached input / 1M":"$0.10 implicit / $0.05 explicit cache read",
      "Cache write / 1M":"$0.625 explicit cache creation",
      "Output / 1M tokens":"$3.00 Alibaba Model Studio Singapore",
      "Text input":"Yes",
      "Image / vision input":"Yes",
      "Audio input":"No native audio input documented",
      "Video input":"Yes",
      "Text output":"Yes",
      "Image output":"No",
      "Audio output":"No",
      "Video output":"No",
      "Tool / function calling":"Yes",
      "Computer use":"Via multimodal agent stacks",
      "API access":"Yes",
      "Product access":"Alibaba Model Studio + weights",
      "Weights / license":"Open weights — Apache 2.0"
    },
    "sources":[
      {"title":"Qwen3.8-27B model docs","url":"https://www.alibabacloud.com/help/en/model-studio/models"},
      {"title":"Qwen3.8-27B official weights","url":"https://huggingface.co/Qwen/Qwen3.8-27B"}
    ],
    "notes":"27.3B dense native multimodal model. Hosted API supports a 1M context and 128K maximum output; pricing fields use Alibaba Model Studio Singapore international rates.",
    "metadata":{"verification_pass":"frontier-models-2026-09-07","parameters_total":"27.3B"}
  }
]
$models$::jsonb) AS seed(
  slug text, name text, developer text, release_date text, ga_date text, access text, family text,
  comparison_data jsonb, sources jsonb, notes text, metadata jsonb
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
  {"id":"tq-20260907-kimi-k3-gpqa","model_slug":"kimi-k3","category":"knowledge","benchmark_name":"GPQA Diamond","benchmark_version":null,"score_numeric":93.5,"score_display":"93.5%","score_unit":"percent","tools":false,"reasoning_effort":"max","harness":"Kimi K3 direct evaluation","evaluator":"Moonshot AI","evaluation_date":"2026-07-16","source":"https://huggingface.co/moonshotai/Kimi-K3","notes":null},
  {"id":"tq-20260907-kimi-k3-hle-no-tools","model_slug":"kimi-k3","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":null,"score_numeric":43.5,"score_display":"43.5%","score_unit":"percent","tools":false,"reasoning_effort":"max","harness":"Kimi K3 direct evaluation","evaluator":"Moonshot AI","evaluation_date":"2026-07-16","source":"https://huggingface.co/moonshotai/Kimi-K3","notes":null},
  {"id":"tq-20260907-kimi-k3-hle-tools","model_slug":"kimi-k3","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":null,"score_numeric":56.0,"score_display":"56.0%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"Kimi K3 direct evaluation","evaluator":"Moonshot AI","evaluation_date":"2026-07-16","source":"https://huggingface.co/moonshotai/Kimi-K3","notes":null},
  {"id":"tq-20260907-kimi-k3-deepswe","model_slug":"kimi-k3","category":"coding","benchmark_name":"DeepSWE v1.1","benchmark_version":null,"score_numeric":67.5,"score_display":"67.5%","score_unit":"percent","tools":null,"reasoning_effort":"max","harness":"Kimi Code","evaluator":"Moonshot AI","evaluation_date":"2026-07-16","source":"https://huggingface.co/moonshotai/Kimi-K3","notes":null},
  {"id":"tq-20260907-kimi-k3-terminal21","model_slug":"kimi-k3","category":"coding","benchmark_name":"Terminal-Bench 2.1","benchmark_version":null,"score_numeric":88.3,"score_display":"88.3%","score_unit":"percent","tools":null,"reasoning_effort":"max","harness":"Kimi Code","evaluator":"Moonshot AI","evaluation_date":"2026-07-16","source":"https://huggingface.co/moonshotai/Kimi-K3","notes":null},
  {"id":"tq-20260907-kimi-k3-browsecomp","model_slug":"kimi-k3","category":"agentic_computer_use","benchmark_name":"BrowseComp","benchmark_version":null,"score_numeric":91.2,"score_display":"91.2%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"Kimi agent with context compaction","evaluator":"Moonshot AI","evaluation_date":"2026-07-16","source":"https://huggingface.co/moonshotai/Kimi-K3","notes":"Kimi reports context compaction in the BrowseComp setup."},
  {"id":"tq-20260907-kimi-k3-gdpv2","model_slug":"kimi-k3","category":"agentic_computer_use","benchmark_name":"GDPval-AA v2","benchmark_version":null,"score_numeric":1686,"score_display":"1686 Elo","score_unit":"Elo","tools":true,"reasoning_effort":"max","harness":"Artificial Analysis","evaluator":"Artificial Analysis","evaluation_date":"2026-07-16","source":"https://huggingface.co/moonshotai/Kimi-K3","notes":"Value reproduced in Kimi's official comparison table."},
  {"id":"tq-20260907-kimi-k3-toolathlon","model_slug":"kimi-k3","category":"agentic_computer_use","benchmark_name":"Toolathlon","benchmark_version":"Verified","score_numeric":76.5,"score_display":"76.5%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"Kimi K3 direct evaluation","evaluator":"Moonshot AI","evaluation_date":"2026-07-16","source":"https://huggingface.co/moonshotai/Kimi-K3","notes":null},
  {"id":"tq-20260907-kimi-k3-mcp-atlas","model_slug":"kimi-k3","category":"agentic_computer_use","benchmark_name":"MCP Atlas","benchmark_version":null,"score_numeric":84.2,"score_display":"84.2%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"Kimi K3 direct evaluation","evaluator":"Moonshot AI","evaluation_date":"2026-07-16","source":"https://huggingface.co/moonshotai/Kimi-K3","notes":null},
  {"id":"tq-20260907-kimi-k3-automation","model_slug":"kimi-k3","category":"agentic_computer_use","benchmark_name":"AutomationBench","benchmark_version":"Public 600-task subset","score_numeric":30.8,"score_display":"30.8%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"Kimi K3 direct evaluation","evaluator":"Moonshot AI","evaluation_date":"2026-07-16","source":"https://huggingface.co/moonshotai/Kimi-K3","notes":null},
  {"id":"tq-20260907-kimi-k3-ale","model_slug":"kimi-k3","category":"agentic_computer_use","benchmark_name":"Agents' Last Exam","benchmark_version":null,"score_numeric":28.3,"score_display":"28.3%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"Kimi K3 direct evaluation","evaluator":"Moonshot AI","evaluation_date":"2026-07-16","source":"https://huggingface.co/moonshotai/Kimi-K3","notes":null},
  {"id":"tq-20260907-kimi-k3-osworld-verified","model_slug":"kimi-k3","category":"agentic_computer_use","benchmark_name":"OSWorld-Verified","benchmark_version":null,"score_numeric":84.8,"score_display":"84.8%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"Kimi K3 direct evaluation","evaluator":"Moonshot AI","evaluation_date":"2026-07-16","source":"https://huggingface.co/moonshotai/Kimi-K3","notes":null},
  {"id":"tq-20260907-kimi-k3-osworld20","model_slug":"kimi-k3","category":"agentic_computer_use","benchmark_name":"OSWorld 2.0","benchmark_version":null,"score_numeric":58.3,"score_display":"58.3%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"Kimi K3 direct evaluation","evaluator":"Moonshot AI","evaluation_date":"2026-07-16","source":"https://huggingface.co/moonshotai/Kimi-K3","notes":null},

  {"id":"tq-20260907-minimax-m3-swe-pro","model_slug":"minimax-m3","category":"coding","benchmark_name":"SWE-bench Pro","benchmark_version":null,"score_numeric":59.0,"score_display":"59.0%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":"Claude Code scaffold on MiniMax internal infrastructure","evaluator":"MiniMax","evaluation_date":"2026-06-01","source":"https://www.minimax.io/news/minimax-m3","notes":null},
  {"id":"tq-20260907-minimax-m3-terminal21","model_slug":"minimax-m3","category":"coding","benchmark_name":"Terminal-Bench 2.1","benchmark_version":null,"score_numeric":66.0,"score_display":"66.0%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":"Terminus 2; 8C16G; 2h","evaluator":"MiniMax","evaluation_date":"2026-06-01","source":"https://www.minimax.io/news/minimax-m3","notes":"MiniMax reports a 128K max output setting for this evaluation harness; that is not promoted here as a universal model output limit."},
  {"id":"tq-20260907-minimax-m3-mcp-atlas","model_slug":"minimax-m3","category":"agentic_computer_use","benchmark_name":"MCP Atlas","benchmark_version":"Public set","score_numeric":74.2,"score_display":"74.2%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":"Official MCP Atlas codebase","evaluator":"MiniMax","evaluation_date":"2026-06-01","source":"https://www.minimax.io/news/minimax-m3","notes":null},
  {"id":"tq-20260907-minimax-m3-browsecomp","model_slug":"minimax-m3","category":"agentic_computer_use","benchmark_name":"BrowseComp","benchmark_version":null,"score_numeric":83.5,"score_display":"83.5%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":"WebExplorer agent framework","evaluator":"MiniMax","evaluation_date":"2026-06-01","source":"https://www.minimax.io/news/minimax-m3","notes":"Evaluation discards history beyond 64K tokens."},

  {"id":"tq-20260907-glm53-terminal21","model_slug":"glm-5-3","category":"coding","benchmark_name":"Terminal-Bench 2.1","benchmark_version":null,"score_numeric":88.2,"score_display":"88.2%","score_unit":"percent","tools":null,"reasoning_effort":"max","harness":"GLM-5.3 official reproduction","evaluator":"Z.ai","evaluation_date":"2026-08-28","source":"https://huggingface.co/zai-org/GLM-5.3","notes":null},
  {"id":"tq-20260907-glm53-terminal30","model_slug":"glm-5-3","category":"coding","benchmark_name":"Terminal-Bench 3.0","benchmark_version":null,"score_numeric":28.3,"score_display":"28.3%","score_unit":"percent","tools":null,"reasoning_effort":"max","harness":"GLM-5.3 official reproduction","evaluator":"Z.ai","evaluation_date":"2026-08-28","source":"https://huggingface.co/zai-org/GLM-5.3","notes":null},
  {"id":"tq-20260907-glm53-deepswe","model_slug":"glm-5-3","category":"coding","benchmark_name":"DeepSWE v1.1","benchmark_version":null,"score_numeric":66.9,"score_display":"66.9%","score_unit":"percent","tools":null,"reasoning_effort":"max","harness":"mini-swe-agent; 400K context","evaluator":"Z.ai","evaluation_date":"2026-08-28","source":"https://huggingface.co/zai-org/GLM-5.3","notes":null},
  {"id":"tq-20260907-glm53-toolathlon","model_slug":"glm-5-3","category":"agentic_computer_use","benchmark_name":"Toolathlon","benchmark_version":"Verified","score_numeric":73.0,"score_display":"73.0%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"Official evaluation service; pass@1 average of 3 runs","evaluator":"Z.ai","evaluation_date":"2026-08-28","source":"https://huggingface.co/zai-org/GLM-5.3","notes":null},
  {"id":"tq-20260907-glm53-automation","model_slug":"glm-5-3","category":"agentic_computer_use","benchmark_name":"AutomationBench","benchmark_version":"v1.0.6","score_numeric":48.2,"score_display":"48.2%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"GLM-5.3 official reproduction","evaluator":"Z.ai","evaluation_date":"2026-08-28","source":"https://huggingface.co/zai-org/GLM-5.3","notes":null},
  {"id":"tq-20260907-glm53-ale","model_slug":"glm-5-3","category":"agentic_computer_use","benchmark_name":"Agents' Last Exam","benchmark_version":"ALE-CLI","score_numeric":28.5,"score_display":"28.5%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"Claude Code; 1M context; 64K output","evaluator":"Z.ai","evaluation_date":"2026-08-28","source":"https://huggingface.co/zai-org/GLM-5.3","notes":null},
  {"id":"tq-20260907-glm53-hle-tools","model_slug":"glm-5-3","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":null,"score_numeric":62.5,"score_display":"62.5%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"300K context with context management","evaluator":"Z.ai","evaluation_date":"2026-08-28","source":"https://huggingface.co/zai-org/GLM-5.3","notes":"Official footnote uses GPT-5.6 Luna medium as judge."},
  {"id":"tq-20260907-glm53-gdpv2","model_slug":"glm-5-3","category":"agentic_computer_use","benchmark_name":"GDPval-AA v2","benchmark_version":null,"score_numeric":1769,"score_display":"1769 Elo","score_unit":"Elo","tools":true,"reasoning_effort":"max","harness":"Artificial Analysis","evaluator":"Artificial Analysis","evaluation_date":"2026-08-28","source":"https://huggingface.co/zai-org/GLM-5.3","notes":"Reported in Z.ai's official model-card comparison table."},

  {"id":"tq-20260907-dsv4flash-terminal21","model_slug":"deepseek-v4-flash","category":"coding","benchmark_name":"Terminal-Bench 2.1","benchmark_version":null,"score_numeric":82.7,"score_display":"82.7%","score_unit":"percent","tools":null,"reasoning_effort":"max","harness":"DeepSeek Harness minimal mode","evaluator":"DeepSeek","evaluation_date":"2026-07-31","source":"https://api-docs.deepseek.com/updates","notes":"Public Code Agent tasks; top_p 0.95 and temperature 1.0 in the documented setup."},
  {"id":"tq-20260907-dsv4flash-deepswe","model_slug":"deepseek-v4-flash","category":"coding","benchmark_name":"DeepSWE v1.1","benchmark_version":null,"score_numeric":54.4,"score_display":"54.4%","score_unit":"percent","tools":null,"reasoning_effort":"max","harness":"DeepSeek Harness minimal mode","evaluator":"DeepSeek","evaluation_date":"2026-07-31","source":"https://api-docs.deepseek.com/updates","notes":null},
  {"id":"tq-20260907-dsv4flash-toolathlon","model_slug":"deepseek-v4-flash","category":"agentic_computer_use","benchmark_name":"Toolathlon","benchmark_version":"Verified","score_numeric":70.3,"score_display":"70.3%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"DeepSeek Harness minimal mode","evaluator":"DeepSeek","evaluation_date":"2026-07-31","source":"https://api-docs.deepseek.com/updates","notes":null},
  {"id":"tq-20260907-dsv4flash-ale","model_slug":"deepseek-v4-flash","category":"agentic_computer_use","benchmark_name":"Agents' Last Exam","benchmark_version":null,"score_numeric":25.2,"score_display":"25.2%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"DeepSeek Harness minimal mode","evaluator":"DeepSeek","evaluation_date":"2026-07-31","source":"https://api-docs.deepseek.com/updates","notes":null},
  {"id":"tq-20260907-dsv4flash-automation","model_slug":"deepseek-v4-flash","category":"agentic_computer_use","benchmark_name":"AutomationBench","benchmark_version":"Public","score_numeric":25.1,"score_display":"25.1%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"DeepSeek Harness minimal mode","evaluator":"DeepSeek","evaluation_date":"2026-07-31","source":"https://api-docs.deepseek.com/updates","notes":null},
  {"id":"tq-20260907-dsv4flash-hle-no-tools","model_slug":"deepseek-v4-flash","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":null,"score_numeric":37.8,"score_display":"37.8%","score_unit":"percent","tools":false,"reasoning_effort":"max","harness":"DeepSeek V4 Pro comparison evaluation","evaluator":"DeepSeek","evaluation_date":"2026-08-13","source":"https://api-docs.deepseek.com/updates","notes":null},
  {"id":"tq-20260907-dsv4flash-hle-tools","model_slug":"deepseek-v4-flash","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":null,"score_numeric":51.5,"score_display":"51.5%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"DeepSeek V4 Pro comparison evaluation","evaluator":"DeepSeek","evaluation_date":"2026-08-13","source":"https://api-docs.deepseek.com/updates","notes":null},

  {"id":"tq-20260907-dsv4pro-hle-no-tools","model_slug":"deepseek-v4-pro","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":null,"score_numeric":42.7,"score_display":"42.7%","score_unit":"percent","tools":false,"reasoning_effort":"max","harness":"DeepSeek V4 Pro direct evaluation","evaluator":"DeepSeek","evaluation_date":"2026-08-13","source":"https://api-docs.deepseek.com/updates","notes":null},
  {"id":"tq-20260907-dsv4pro-hle-tools","model_slug":"deepseek-v4-pro","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":null,"score_numeric":60.0,"score_display":"60.0%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"DeepSeek V4 Pro direct evaluation","evaluator":"DeepSeek","evaluation_date":"2026-08-13","source":"https://api-docs.deepseek.com/updates","notes":null},
  {"id":"tq-20260907-dsv4pro-terminal21","model_slug":"deepseek-v4-pro","category":"coding","benchmark_name":"Terminal-Bench 2.1","benchmark_version":null,"score_numeric":87.9,"score_display":"87.9%","score_unit":"percent","tools":null,"reasoning_effort":"max","harness":"DeepSeek Harness","evaluator":"DeepSeek","evaluation_date":"2026-08-13","source":"https://api-docs.deepseek.com/updates","notes":null},
  {"id":"tq-20260907-dsv4pro-deepswe","model_slug":"deepseek-v4-pro","category":"coding","benchmark_name":"DeepSWE v1.1","benchmark_version":null,"score_numeric":62.7,"score_display":"62.7%","score_unit":"percent","tools":null,"reasoning_effort":"max","harness":"DeepSeek Harness","evaluator":"DeepSeek","evaluation_date":"2026-08-13","source":"https://api-docs.deepseek.com/updates","notes":null},
  {"id":"tq-20260907-dsv4pro-toolathlon","model_slug":"deepseek-v4-pro","category":"agentic_computer_use","benchmark_name":"Toolathlon","benchmark_version":"Verified","score_numeric":74.1,"score_display":"74.1%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"DeepSeek Harness","evaluator":"DeepSeek","evaluation_date":"2026-08-13","source":"https://api-docs.deepseek.com/updates","notes":null},
  {"id":"tq-20260907-dsv4pro-ale","model_slug":"deepseek-v4-pro","category":"agentic_computer_use","benchmark_name":"Agents' Last Exam","benchmark_version":null,"score_numeric":25.7,"score_display":"25.7%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"DeepSeek Harness","evaluator":"DeepSeek","evaluation_date":"2026-08-13","source":"https://api-docs.deepseek.com/updates","notes":null},
  {"id":"tq-20260907-dsv4pro-automation","model_slug":"deepseek-v4-pro","category":"agentic_computer_use","benchmark_name":"AutomationBench","benchmark_version":"Public","score_numeric":31.8,"score_display":"31.8%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"DeepSeek Harness","evaluator":"DeepSeek","evaluation_date":"2026-08-13","source":"https://api-docs.deepseek.com/updates","notes":null},

  {"id":"tq-20260907-qwen38max-terminal21","model_slug":"qwen3-8-max","category":"coding","benchmark_name":"Terminal-Bench 2.1","benchmark_version":null,"score_numeric":86.6,"score_display":"86.6%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":"Qwen3.8 official evaluation","evaluator":"Alibaba / Qwen","evaluation_date":"2026-08-02","source":"https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B","notes":null},
  {"id":"tq-20260907-qwen38max-swe-pro","model_slug":"qwen3-8-max","category":"coding","benchmark_name":"SWE-bench Pro","benchmark_version":null,"score_numeric":67.7,"score_display":"67.7%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":"Qwen3.8 official evaluation","evaluator":"Alibaba / Qwen","evaluation_date":"2026-08-02","source":"https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B","notes":null},
  {"id":"tq-20260907-qwen38max-deepswe","model_slug":"qwen3-8-max","category":"coding","benchmark_name":"DeepSWE v1.1","benchmark_version":null,"score_numeric":56.6,"score_display":"56.6%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":"Qwen3.8 official evaluation","evaluator":"Alibaba / Qwen","evaluation_date":"2026-08-02","source":"https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B","notes":null},
  {"id":"tq-20260907-qwen38max-ale","model_slug":"qwen3-8-max","category":"agentic_computer_use","benchmark_name":"Agents' Last Exam","benchmark_version":"Pass","score_numeric":27.0,"score_display":"27.0%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":"Qwen3.8 official evaluation","evaluator":"Alibaba / Qwen","evaluation_date":"2026-08-02","source":"https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B","notes":"Official table also reports overall ALE score 52.4; canonical comparison stores the pass rate."},
  {"id":"tq-20260907-qwen38max-automation","model_slug":"qwen3-8-max","category":"agentic_computer_use","benchmark_name":"AutomationBench","benchmark_version":null,"score_numeric":27.3,"score_display":"27.3%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":"Qwen3.8 official evaluation","evaluator":"Alibaba / Qwen","evaluation_date":"2026-08-02","source":"https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B","notes":null},
  {"id":"tq-20260907-qwen38max-toolathlon","model_slug":"qwen3-8-max","category":"agentic_computer_use","benchmark_name":"Toolathlon","benchmark_version":"Verified","score_numeric":72.5,"score_display":"72.5%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":"Qwen3.8 official evaluation","evaluator":"Alibaba / Qwen","evaluation_date":"2026-08-02","source":"https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B","notes":null},
  {"id":"tq-20260907-qwen38max-hle-tools","model_slug":"qwen3-8-max","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":null,"score_numeric":56.2,"score_display":"56.2%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":"Qwen3.8 official evaluation","evaluator":"Alibaba / Qwen","evaluation_date":"2026-08-02","source":"https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B","notes":null},
  {"id":"tq-20260907-qwen38max-gpqa","model_slug":"qwen3-8-max","category":"knowledge","benchmark_name":"GPQA Diamond","benchmark_version":null,"score_numeric":92.6,"score_display":"92.6%","score_unit":"percent","tools":false,"reasoning_effort":null,"harness":"Qwen3.8 official evaluation","evaluator":"Alibaba / Qwen","evaluation_date":"2026-08-02","source":"https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B","notes":null},
  {"id":"tq-20260907-qwen38max-hle-no-tools","model_slug":"qwen3-8-max","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":null,"score_numeric":43.6,"score_display":"43.6%","score_unit":"percent","tools":false,"reasoning_effort":null,"harness":"Qwen3.8 official evaluation","evaluator":"Alibaba / Qwen","evaluation_date":"2026-08-02","source":"https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B","notes":null},
  {"id":"tq-20260907-qwen38max-gdpv2","model_slug":"qwen3-8-max","category":"agentic_computer_use","benchmark_name":"GDPval-AA v2","benchmark_version":null,"score_numeric":1739,"score_display":"1739 Elo","score_unit":"Elo","tools":true,"reasoning_effort":null,"harness":"Artificial Analysis","evaluator":"Artificial Analysis","evaluation_date":"2026-08-28","source":"https://huggingface.co/zai-org/GLM-5.3","notes":"Cross-vendor value reported in Z.ai's official GLM-5.3 comparison table; kept distinct from Qwen's direct benchmark rows."},

  {"id":"tq-20260907-qwen3827b-terminal21","model_slug":"qwen3-8-27b","category":"coding","benchmark_name":"Terminal-Bench 2.1","benchmark_version":null,"score_numeric":73.0,"score_display":"73.0%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":"Terminus","evaluator":"Alibaba / Qwen","evaluation_date":"2026-08-17","source":"https://huggingface.co/Qwen/Qwen3.8-27B","notes":null},
  {"id":"tq-20260907-qwen3827b-swe-pro","model_slug":"qwen3-8-27b","category":"coding","benchmark_name":"SWE-bench Pro","benchmark_version":null,"score_numeric":61.7,"score_display":"61.7%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":"Claude Code; 256K context","evaluator":"Alibaba / Qwen","evaluation_date":"2026-08-17","source":"https://huggingface.co/Qwen/Qwen3.8-27B","notes":null},
  {"id":"tq-20260907-qwen3827b-deepswe","model_slug":"qwen3-8-27b","category":"coding","benchmark_name":"DeepSWE v1.1","benchmark_version":null,"score_numeric":42.2,"score_display":"42.2%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":"Claude Code; 256K context","evaluator":"Alibaba / Qwen","evaluation_date":"2026-08-17","source":"https://huggingface.co/Qwen/Qwen3.8-27B","notes":null},
  {"id":"tq-20260907-qwen3827b-ale","model_slug":"qwen3-8-27b","category":"agentic_computer_use","benchmark_name":"Agents' Last Exam","benchmark_version":"Pass@1","score_numeric":20.4,"score_display":"20.4%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":"Qwen3.8-27B official evaluation","evaluator":"Alibaba / Qwen","evaluation_date":"2026-08-17","source":"https://huggingface.co/Qwen/Qwen3.8-27B","notes":"Official table also reports overall ALE score 42.9; canonical comparison stores pass@1."},
  {"id":"tq-20260907-qwen3827b-gpqa","model_slug":"qwen3-8-27b","category":"knowledge","benchmark_name":"GPQA Diamond","benchmark_version":null,"score_numeric":89.2,"score_display":"89.2%","score_unit":"percent","tools":false,"reasoning_effort":null,"harness":"Qwen3.8-27B official evaluation","evaluator":"Alibaba / Qwen","evaluation_date":"2026-08-17","source":"https://huggingface.co/Qwen/Qwen3.8-27B","notes":null},
  {"id":"tq-20260907-qwen3827b-hle","model_slug":"qwen3-8-27b","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":null,"score_numeric":30.8,"score_display":"30.8%","score_unit":"percent","tools":false,"reasoning_effort":null,"harness":"Qwen3.8-27B official evaluation","evaluator":"Alibaba / Qwen","evaluation_date":"2026-08-17","source":"https://huggingface.co/Qwen/Qwen3.8-27B","notes":null},
  {"id":"tq-20260907-qwen3827b-livecode","model_slug":"qwen3-8-27b","category":"coding","benchmark_name":"LiveCodeBench","benchmark_version":"v6","score_numeric":90.3,"score_display":"90.3%","score_unit":"percent","tools":false,"reasoning_effort":null,"harness":"Qwen3.8-27B official evaluation","evaluator":"Alibaba / Qwen","evaluation_date":"2026-08-17","source":"https://huggingface.co/Qwen/Qwen3.8-27B","notes":null},
  {"id":"tq-20260907-qwen3827b-toolathlon","model_slug":"qwen3-8-27b","category":"agentic_computer_use","benchmark_name":"Toolathlon","benchmark_version":"Verified","score_numeric":67.1,"score_display":"67.1%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":"Qwen3.8-27B official evaluation","evaluator":"Alibaba / Qwen","evaluation_date":"2026-08-17","source":"https://huggingface.co/Qwen/Qwen3.8-27B","notes":null},
  {"id":"tq-20260907-qwen3827b-osworld-verified","model_slug":"qwen3-8-27b","category":"agentic_computer_use","benchmark_name":"OSWorld-Verified","benchmark_version":null,"score_numeric":84.3,"score_display":"84.3%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":"Qwen3.8-27B multimodal agent evaluation","evaluator":"Alibaba / Qwen","evaluation_date":"2026-08-17","source":"https://huggingface.co/Qwen/Qwen3.8-27B","notes":null},
  {"id":"tq-20260907-qwen3827b-osworld20-binary","model_slug":"qwen3-8-27b","category":"agentic_computer_use","benchmark_name":"OSWorld 2.0","benchmark_version":"Binary","score_numeric":19.4,"score_display":"19.4%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":"Qwen3.8-27B multimodal agent evaluation","evaluator":"Alibaba / Qwen","evaluation_date":"2026-08-17","source":"https://huggingface.co/Qwen/Qwen3.8-27B","notes":null},
  {"id":"tq-20260907-qwen3827b-osworld20-partial","model_slug":"qwen3-8-27b","category":"agentic_computer_use","benchmark_name":"OSWorld 2.0","benchmark_version":"Partial","score_numeric":48.0,"score_display":"48.0%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":"Qwen3.8-27B multimodal agent evaluation","evaluator":"Alibaba / Qwen","evaluation_date":"2026-08-17","source":"https://huggingface.co/Qwen/Qwen3.8-27B","notes":null}
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

-- Populate empty benchmark cells on authored/materialized comparisons without replacing editorial values.
WITH target_models(slug) AS (
  VALUES
    ('kimi-k3'), ('minimax-m3'), ('glm-5-3'), ('deepseek-v4-flash'),
    ('deepseek-v4-pro'), ('qwen3-8-max'), ('qwen3-8-27b')
),
benchmark_rendered AS (
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
  JOIN target_models AS tm ON tm.slug = b.model_slug
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
                          WHEN COALESCE(row_entry.row_value->>1, '') = '' AND bench_a.rendered IS NOT NULL THEN bench_a.rendered
                          ELSE COALESCE(row_entry.row_value->>1, '')
                        END),
                        false
                      ),
                      '{2}',
                      to_jsonb(CASE
                        WHEN COALESCE(row_entry.row_value->>2, '') = '' AND bench_b.rendered IS NOT NULL THEN bench_b.rendered
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
    AND (
      c.metadata->>'modelA' IN (SELECT slug FROM target_models)
      OR c.metadata->>'modelB' IN (SELECT slug FROM target_models)
    )
  GROUP BY c.id, c.metadata
)
UPDATE content_items AS c
SET blocks = rebuilt.blocks, updated_at = NOW()
FROM rebuilt
WHERE c.id = rebuilt.id
  AND c.blocks IS DISTINCT FROM rebuilt.blocks;
