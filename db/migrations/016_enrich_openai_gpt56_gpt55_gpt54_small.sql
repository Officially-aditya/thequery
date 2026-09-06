-- First-party verified enrichment for OpenAI GPT-5.6 Sol/Terra/Luna,
-- GPT-5.5, GPT-5.4 mini, and GPT-5.4 nano.
-- Sources: OpenAI launch posts and current OpenAI API model pages, checked 2026-09-06.

WITH verified_models AS (
  SELECT *
  FROM jsonb_to_recordset($models$
[
  {
    "slug":"gpt-5-6-sol",
    "family":"GPT-5.6",
    "ga_date":"2026-07-09",
    "comparison_data":{
      "Developer":"OpenAI",
      "Release date":"2026-07-09",
      "API model ID":"gpt-5.6-sol (gpt-5.6 alias routes to Sol)",
      "Context window":"1,050,000 tokens",
      "Max output":"128,000 tokens",
      "Knowledge cutoff":"2026-02-16",
      "Reasoning / effort":"none / low / medium / high / xhigh / max",
      "Input / 1M tokens":"$4.00 (current promotional API price)",
      "Cached input / 1M":"$0.40",
      "Cache write / 1M":"$5.00 (1.25× uncached input)",
      "Output / 1M tokens":"$20.00 (current promotional API price)",
      "Long-context surcharge":">272K input: 2× input and 1.5× output for the full request",
      "Text input":"Yes",
      "Image / vision input":"Yes",
      "Audio input":"No",
      "Video input":"No",
      "Text output":"Yes",
      "Image output":"No native image output; image-generation tool supported",
      "Audio output":"No",
      "Video output":"No",
      "Tool / function calling":"Yes",
      "Computer use":"Yes",
      "API access":"Yes",
      "Product access":"ChatGPT / ChatGPT Work + Codex + OpenAI API",
      "Weights / license":"Proprietary"
    },
    "sources":[
      {"title":"GPT-5.6 launch","url":"https://openai.com/index/gpt-5-6/"},
      {"title":"GPT-5.6 Sol API model page","url":"https://developers.openai.com/api/docs/models/gpt-5.6-sol"},
      {"title":"GPT-5.6 current price-performance update","url":"https://openai.com/index/advancing-the-price-performance-frontier-with-gpt-5-6/"}
    ],
    "notes":"GPT-5.6 Sol launched GA on July 9 after a June 26 limited preview. Current $4/$20 API pricing is promotional and OpenAI says it is available at least through November 21, 2026."
  },
  {
    "slug":"gpt-5-6-terra",
    "family":"GPT-5.6",
    "ga_date":"2026-07-09",
    "comparison_data":{
      "Developer":"OpenAI",
      "Release date":"2026-07-09",
      "API model ID":"gpt-5.6-terra",
      "Context window":"1,050,000 tokens",
      "Max output":"128,000 tokens",
      "Knowledge cutoff":"2026-02-16",
      "Reasoning / effort":"none / low / medium / high / xhigh / max",
      "Input / 1M tokens":"$2.00",
      "Cached input / 1M":"$0.20",
      "Cache write / 1M":"$2.50 (1.25× uncached input)",
      "Output / 1M tokens":"$12.00",
      "Long-context surcharge":">272K input: 2× input and 1.5× output for the full request",
      "Text input":"Yes",
      "Image / vision input":"Yes",
      "Audio input":"No",
      "Video input":"No",
      "Text output":"Yes",
      "Image output":"No native image output; image-generation tool supported",
      "Audio output":"No",
      "Video output":"No",
      "Tool / function calling":"Yes",
      "Computer use":"Yes",
      "API access":"Yes",
      "Product access":"ChatGPT / ChatGPT Work + Codex + OpenAI API",
      "Weights / license":"Proprietary"
    },
    "sources":[
      {"title":"GPT-5.6 launch","url":"https://openai.com/index/gpt-5-6/"},
      {"title":"GPT-5.6 Terra API model page","url":"https://developers.openai.com/api/docs/models/gpt-5.6-terra"},
      {"title":"GPT-5.6 current price-performance update","url":"https://openai.com/index/advancing-the-price-performance-frontier-with-gpt-5-6/"}
    ],
    "notes":"Terra is the balanced GPT-5.6 tier. OpenAI reduced its API price on July 30, 2026 to $2 input / $12 output per 1M tokens."
  },
  {
    "slug":"gpt-5-6-luna",
    "family":"GPT-5.6",
    "ga_date":"2026-07-09",
    "comparison_data":{
      "Developer":"OpenAI",
      "Release date":"2026-07-09",
      "API model ID":"gpt-5.6-luna",
      "Context window":"1,050,000 tokens",
      "Max output":"128,000 tokens",
      "Knowledge cutoff":"2026-02-16",
      "Reasoning / effort":"none / low / medium / high / xhigh / max",
      "Input / 1M tokens":"$0.20",
      "Cached input / 1M":"$0.02",
      "Cache write / 1M":"$0.25 (1.25× uncached input)",
      "Output / 1M tokens":"$1.20",
      "Long-context surcharge":">272K input: 2× input and 1.5× output for the full request",
      "Text input":"Yes",
      "Image / vision input":"Yes",
      "Audio input":"No",
      "Video input":"No",
      "Text output":"Yes",
      "Image output":"No native image output; image-generation tool supported",
      "Audio output":"No",
      "Video output":"No",
      "Tool / function calling":"Yes",
      "Computer use":"Yes",
      "API access":"Yes",
      "Product access":"ChatGPT / ChatGPT Work + Codex + OpenAI API",
      "Weights / license":"Proprietary"
    },
    "sources":[
      {"title":"GPT-5.6 launch","url":"https://openai.com/index/gpt-5-6/"},
      {"title":"GPT-5.6 Luna API model page","url":"https://developers.openai.com/api/docs/models/gpt-5.6-luna"},
      {"title":"GPT-5.6 current price-performance update","url":"https://openai.com/index/advancing-the-price-performance-frontier-with-gpt-5-6/"}
    ],
    "notes":"Luna is the fastest and most affordable GPT-5.6 tier. OpenAI reduced its API price on July 30, 2026 to $0.20 input / $1.20 output per 1M tokens."
  },
  {
    "slug":"gpt-5-5",
    "family":"GPT-5.5",
    "ga_date":"2026-04-23",
    "comparison_data":{
      "Developer":"OpenAI",
      "Release date":"2026-04-23",
      "API model ID":"gpt-5.5",
      "Context window":"1,050,000 tokens",
      "Max output":"128,000 tokens",
      "Knowledge cutoff":"2025-12-01",
      "Reasoning / effort":"none / low / medium / high / xhigh",
      "Input / 1M tokens":"$5.00",
      "Cached input / 1M":"$0.50",
      "Output / 1M tokens":"$30.00",
      "Batch / flex discount":"50% off standard token rates",
      "Long-context surcharge":">272K input: 2× input and 1.5× output for the full session",
      "Text input":"Yes",
      "Image / vision input":"Yes",
      "Audio input":"No",
      "Video input":"No",
      "Text output":"Yes",
      "Image output":"No native image output; image-generation tool supported",
      "Audio output":"No",
      "Video output":"No",
      "Tool / function calling":"Yes",
      "Computer use":"Yes",
      "API access":"Yes",
      "Product access":"ChatGPT + Codex + OpenAI API",
      "Weights / license":"Proprietary"
    },
    "sources":[
      {"title":"Introducing GPT-5.5","url":"https://openai.com/index/introducing-gpt-5-5/"},
      {"title":"GPT-5.5 API model page","url":"https://developers.openai.com/api/docs/models/gpt-5.5"}
    ],
    "notes":"GPT-5.5 launched in ChatGPT and Codex on April 23; API availability followed on April 24, 2026."
  },
  {
    "slug":"gpt-5-4-mini",
    "family":"GPT-5.4",
    "ga_date":"2026-03-17",
    "comparison_data":{
      "Developer":"OpenAI",
      "Release date":"2026-03-17",
      "API model ID":"gpt-5.4-mini",
      "Context window":"400,000 tokens",
      "Max output":"128,000 tokens",
      "Knowledge cutoff":"2025-08-31",
      "Reasoning / effort":"none (default) / low / medium / high / xhigh",
      "Input / 1M tokens":"$0.75",
      "Cached input / 1M":"$0.075",
      "Output / 1M tokens":"$4.50",
      "Text input":"Yes",
      "Image / vision input":"Yes",
      "Audio input":"No",
      "Video input":"No",
      "Text output":"Yes",
      "Image output":"No native image output; image-generation tool supported",
      "Audio output":"No",
      "Video output":"No",
      "Tool / function calling":"Yes",
      "Computer use":"Yes",
      "API access":"Yes",
      "Product access":"OpenAI API + Codex + ChatGPT",
      "Weights / license":"Proprietary"
    },
    "sources":[
      {"title":"Introducing GPT-5.4 mini and nano","url":"https://openai.com/index/introducing-gpt-5-4-mini-and-nano/"},
      {"title":"GPT-5.4 mini API model page","url":"https://developers.openai.com/api/docs/models/gpt-5.4-mini"}
    ],
    "notes":"OpenAI describes GPT-5.4 mini as optimized for coding, computer use, high-volume workloads, and subagents."
  },
  {
    "slug":"gpt-5-4-nano",
    "family":"GPT-5.4",
    "ga_date":"2026-03-17",
    "comparison_data":{
      "Developer":"OpenAI",
      "Release date":"2026-03-17",
      "API model ID":"gpt-5.4-nano",
      "Context window":"400,000 tokens",
      "Max output":"128,000 tokens",
      "Knowledge cutoff":"2025-08-31",
      "Reasoning / effort":"none (default) / low / medium / high / xhigh",
      "Input / 1M tokens":"$0.20",
      "Cached input / 1M":"$0.02",
      "Output / 1M tokens":"$1.25",
      "Text input":"Yes",
      "Image / vision input":"Yes",
      "Audio input":"No",
      "Video input":"No",
      "Text output":"Yes",
      "Image output":"No native image output; image-generation tool supported",
      "Audio output":"No",
      "Video output":"No",
      "Tool / function calling":"Yes",
      "Computer use":"No",
      "API access":"Yes",
      "Product access":"OpenAI API only",
      "Weights / license":"Proprietary"
    },
    "sources":[
      {"title":"Introducing GPT-5.4 mini and nano","url":"https://openai.com/index/introducing-gpt-5-4-mini-and-nano/"},
      {"title":"GPT-5.4 nano API model page","url":"https://developers.openai.com/api/docs/models/gpt-5.4-nano"}
    ],
    "notes":"OpenAI positions GPT-5.4 nano for classification, data extraction, ranking, and simpler coding subagents. Its API model page explicitly lists computer use as unsupported."
  }
]
$models$::jsonb) AS v(
    slug text,
    family text,
    ga_date text,
    comparison_data jsonb,
    sources jsonb,
    notes text
  )
)
UPDATE models AS m
SET
  family = v.family,
  ga_date = v.ga_date::date,
  comparison_data = COALESCE(m.comparison_data, '{}'::jsonb) || v.comparison_data,
  sources = v.sources,
  notes = v.notes,
  metadata = COALESCE(m.metadata, '{}'::jsonb) || jsonb_build_object('verification_pass', 'openai-2026-09-06'),
  verified_at = NOW(),
  updated_at = NOW()
FROM verified_models AS v
WHERE m.slug = v.slug;

-- Replace first-party rows from these source pages for the six enriched models so
-- repeatable migrations do not leave stale duplicate launch-eval records.
DELETE FROM model_benchmarks
WHERE model_slug IN (
  'gpt-5-6-sol', 'gpt-5-6-terra', 'gpt-5-6-luna',
  'gpt-5-5', 'gpt-5-4-mini', 'gpt-5-4-nano'
)
AND source IN (
  'https://openai.com/index/gpt-5-6/',
  'https://openai.com/index/introducing-gpt-5-5/',
  'https://openai.com/index/introducing-gpt-5-4-mini-and-nano/'
);

INSERT INTO model_benchmarks (
  id, model_slug, category, benchmark_name, benchmark_version,
  score_numeric, score_display, score_unit, tools, reasoning_effort,
  harness, evaluator, evaluation_date, source, notes, updated_at
)
VALUES
  -- GPT-5.6 family + GPT-5.5: coherent comparison suite from the GPT-5.6 launch post.
  ('openai-20260709-gpt56-sol-swe-pro','gpt-5-6-sol','coding','SWE-bench Pro',NULL,64.6,'64.6%','percent',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/','Public SWE-bench Pro result reported by OpenAI',NOW()),
  ('openai-20260709-gpt56-terra-swe-pro','gpt-5-6-terra','coding','SWE-bench Pro',NULL,63.4,'63.4%','percent',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/','Public SWE-bench Pro result reported by OpenAI',NOW()),
  ('openai-20260709-gpt56-luna-swe-pro','gpt-5-6-luna','coding','SWE-bench Pro',NULL,62.7,'62.7%','percent',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/','Public SWE-bench Pro result reported by OpenAI',NOW()),
  ('openai-20260709-gpt55-swe-pro','gpt-5-5','coding','SWE-bench Pro',NULL,59.4,'59.4%','percent',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/','Later OpenAI comparison run; differs from the April GPT-5.5 launch score',NOW()),

  ('openai-20260709-gpt56-sol-deepswe','gpt-5-6-sol','coding','DeepSWE v1.1','v1.1',72.7,'72.7%','percent',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/',NULL,NOW()),
  ('openai-20260709-gpt56-terra-deepswe','gpt-5-6-terra','coding','DeepSWE v1.1','v1.1',69.6,'69.6%','percent',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/',NULL,NOW()),
  ('openai-20260709-gpt56-luna-deepswe','gpt-5-6-luna','coding','DeepSWE v1.1','v1.1',67.2,'67.2%','percent',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/',NULL,NOW()),
  ('openai-20260709-gpt55-deepswe','gpt-5-5','coding','DeepSWE v1.1','v1.1',67.0,'67.0%','percent',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/',NULL,NOW()),

  ('openai-20260709-gpt56-sol-terminal21','gpt-5-6-sol','coding','Terminal-Bench 2.1','2.1',88.8,'88.8%','percent',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/',NULL,NOW()),
  ('openai-20260709-gpt56-terra-terminal21','gpt-5-6-terra','coding','Terminal-Bench 2.1','2.1',87.4,'87.4%','percent',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/',NULL,NOW()),
  ('openai-20260709-gpt56-luna-terminal21','gpt-5-6-luna','coding','Terminal-Bench 2.1','2.1',84.7,'84.7%','percent',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/',NULL,NOW()),
  ('openai-20260709-gpt55-terminal21','gpt-5-5','coding','Terminal-Bench 2.1','2.1',85.6,'85.6%','percent',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/',NULL,NOW()),

  ('openai-20260709-gpt56-sol-frontiermath13','gpt-5-6-sol','math_reasoning','FrontierMath','v2 Tier 1-3',89.0,'89.0%','percent',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/',NULL,NOW()),
  ('openai-20260709-gpt56-terra-frontiermath13','gpt-5-6-terra','math_reasoning','FrontierMath','v2 Tier 1-3',84.9,'84.9%','percent',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/',NULL,NOW()),
  ('openai-20260709-gpt56-luna-frontiermath13','gpt-5-6-luna','math_reasoning','FrontierMath','v2 Tier 1-3',78.6,'78.6%','percent',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/',NULL,NOW()),
  ('openai-20260709-gpt55-frontiermath13','gpt-5-5','math_reasoning','FrontierMath','v2 Tier 1-3',85.3,'85.3%','percent',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/',NULL,NOW()),
  ('openai-20260709-gpt56-sol-frontiermath4','gpt-5-6-sol','math_reasoning','FrontierMath','v2 Tier 4',83.0,'83.0%','percent',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/',NULL,NOW()),
  ('openai-20260709-gpt56-terra-frontiermath4','gpt-5-6-terra','math_reasoning','FrontierMath','v2 Tier 4',68.3,'68.3%','percent',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/',NULL,NOW()),
  ('openai-20260709-gpt56-luna-frontiermath4','gpt-5-6-luna','math_reasoning','FrontierMath','v2 Tier 4',58.5,'58.5%','percent',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/',NULL,NOW()),
  ('openai-20260709-gpt55-frontiermath4','gpt-5-5','math_reasoning','FrontierMath','v2 Tier 4',72.5,'72.5%','percent',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/',NULL,NOW()),

  ('openai-20260709-gpt56-sol-gpqa','gpt-5-6-sol','knowledge','GPQA Diamond',NULL,94.6,'94.6%','percent',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/',NULL,NOW()),
  ('openai-20260709-gpt56-terra-gpqa','gpt-5-6-terra','knowledge','GPQA Diamond',NULL,92.9,'92.9%','percent',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/',NULL,NOW()),
  ('openai-20260709-gpt56-luna-gpqa','gpt-5-6-luna','knowledge','GPQA Diamond',NULL,92.3,'92.3%','percent',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/',NULL,NOW()),
  ('openai-20260709-gpt55-gpqa','gpt-5-5','knowledge','GPQA Diamond',NULL,93.6,'93.6%','percent',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/',NULL,NOW()),

  ('openai-20260423-gpt55-hle-no-tools','gpt-5-5','knowledge','Humanity''s Last Exam',NULL,41.4,'41.4%','percent',FALSE,'xhigh','OpenAI GPT-5.5 launch evaluation','OpenAI','2026-04-23','https://openai.com/index/introducing-gpt-5-5/','No tools',NOW()),
  ('openai-20260423-gpt55-hle-tools','gpt-5-5','knowledge','Humanity''s Last Exam',NULL,52.2,'52.2%','percent',TRUE,'xhigh','OpenAI GPT-5.5 launch evaluation','OpenAI','2026-04-23','https://openai.com/index/introducing-gpt-5-5/','With tools',NOW()),

  ('openai-20260709-gpt56-sol-osworld20','gpt-5-6-sol','agentic_computer_use','OSWorld 2.0','2.0',62.6,'62.6%','percent',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/',NULL,NOW()),
  ('openai-20260709-gpt56-terra-osworld20','gpt-5-6-terra','agentic_computer_use','OSWorld 2.0','2.0',50.2,'50.2%','percent',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/',NULL,NOW()),
  ('openai-20260709-gpt56-luna-osworld20','gpt-5-6-luna','agentic_computer_use','OSWorld 2.0','2.0',45.6,'45.6%','percent',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/',NULL,NOW()),
  ('openai-20260709-gpt55-osworld20','gpt-5-5','agentic_computer_use','OSWorld 2.0','2.0',47.5,'47.5%','percent',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/',NULL,NOW()),
  ('openai-20260709-gpt56-sol-browsecomp','gpt-5-6-sol','agentic_computer_use','BrowseComp',NULL,90.4,'90.4%','percent',NULL,NULL,'Single-model GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/','Do not confuse with Sol Ultra / multi-agent 92.2%',NOW()),
  ('openai-20260709-gpt56-terra-browsecomp','gpt-5-6-terra','agentic_computer_use','BrowseComp',NULL,87.5,'87.5%','percent',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/',NULL,NOW()),
  ('openai-20260709-gpt56-luna-browsecomp','gpt-5-6-luna','agentic_computer_use','BrowseComp',NULL,83.3,'83.3%','percent',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/',NULL,NOW()),
  ('openai-20260709-gpt55-browsecomp','gpt-5-5','agentic_computer_use','BrowseComp',NULL,84.4,'84.4%','percent',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/',NULL,NOW()),
  ('openai-20260709-gpt56-sol-gdpvalaa2','gpt-5-6-sol','agentic_computer_use','GDPval-AA v2','v2',1747.8,'1,747.8 Elo','Elo',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/',NULL,NOW()),
  ('openai-20260709-gpt56-terra-gdpvalaa2','gpt-5-6-terra','agentic_computer_use','GDPval-AA v2','v2',1593.0,'1,593 Elo','Elo',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/',NULL,NOW()),
  ('openai-20260709-gpt56-luna-gdpvalaa2','gpt-5-6-luna','agentic_computer_use','GDPval-AA v2','v2',1591.8,'1,591.8 Elo','Elo',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/',NULL,NOW()),
  ('openai-20260709-gpt55-gdpvalaa2','gpt-5-5','agentic_computer_use','GDPval-AA v2','v2',1493.7,'1,493.7 Elo','Elo',NULL,NULL,'OpenAI GPT-5.6 launch comparison suite','OpenAI','2026-07-09','https://openai.com/index/gpt-5-6/',NULL,NOW()),

  -- GPT-5.4 mini / nano launch evals. OpenAI's table explicitly reports xhigh.
  ('openai-20260317-gpt54mini-swe-pro','gpt-5-4-mini','coding','SWE-bench Pro','Public',54.4,'54.4%','percent',NULL,'xhigh','OpenAI GPT-5.4 mini/nano launch evaluation','OpenAI','2026-03-17','https://openai.com/index/introducing-gpt-5-4-mini-and-nano/',NULL,NOW()),
  ('openai-20260317-gpt54nano-swe-pro','gpt-5-4-nano','coding','SWE-bench Pro','Public',52.4,'52.4%','percent',NULL,'xhigh','OpenAI GPT-5.4 mini/nano launch evaluation','OpenAI','2026-03-17','https://openai.com/index/introducing-gpt-5-4-mini-and-nano/',NULL,NOW()),
  ('openai-20260317-gpt54mini-terminal20','gpt-5-4-mini','coding','Terminal-Bench','2.0',60.0,'60.0%','percent',NULL,'xhigh','OpenAI GPT-5.4 mini/nano launch evaluation','OpenAI','2026-03-17','https://openai.com/index/introducing-gpt-5-4-mini-and-nano/',NULL,NOW()),
  ('openai-20260317-gpt54nano-terminal20','gpt-5-4-nano','coding','Terminal-Bench','2.0',46.3,'46.3%','percent',NULL,'xhigh','OpenAI GPT-5.4 mini/nano launch evaluation','OpenAI','2026-03-17','https://openai.com/index/introducing-gpt-5-4-mini-and-nano/',NULL,NOW()),
  ('openai-20260317-gpt54mini-gpqa','gpt-5-4-mini','knowledge','GPQA Diamond',NULL,88.0,'88.0%','percent',NULL,'xhigh','OpenAI GPT-5.4 mini/nano launch evaluation','OpenAI','2026-03-17','https://openai.com/index/introducing-gpt-5-4-mini-and-nano/',NULL,NOW()),
  ('openai-20260317-gpt54nano-gpqa','gpt-5-4-nano','knowledge','GPQA Diamond',NULL,82.8,'82.8%','percent',NULL,'xhigh','OpenAI GPT-5.4 mini/nano launch evaluation','OpenAI','2026-03-17','https://openai.com/index/introducing-gpt-5-4-mini-and-nano/',NULL,NOW()),
  ('openai-20260317-gpt54mini-osworld','gpt-5-4-mini','agentic_computer_use','OSWorld','Verified',72.1,'72.1%','percent',NULL,'xhigh','OpenAI GPT-5.4 mini/nano launch evaluation','OpenAI','2026-03-17','https://openai.com/index/introducing-gpt-5-4-mini-and-nano/',NULL,NOW()),
  ('openai-20260317-gpt54nano-osworld','gpt-5-4-nano','agentic_computer_use','OSWorld','Verified',39.0,'39.0%','percent',NULL,'xhigh','OpenAI GPT-5.4 mini/nano launch evaluation','OpenAI','2026-03-17','https://openai.com/index/introducing-gpt-5-4-mini-and-nano/',NULL,NOW()),
  ('openai-20260317-gpt54mini-toolathlon','gpt-5-4-mini','agentic_computer_use','Toolathlon',NULL,42.9,'42.9%','percent',NULL,'xhigh','OpenAI GPT-5.4 mini/nano launch evaluation','OpenAI','2026-03-17','https://openai.com/index/introducing-gpt-5-4-mini-and-nano/',NULL,NOW()),
  ('openai-20260317-gpt54nano-toolathlon','gpt-5-4-nano','agentic_computer_use','Toolathlon',NULL,35.5,'35.5%','percent',NULL,'xhigh','OpenAI GPT-5.4 mini/nano launch evaluation','OpenAI','2026-03-17','https://openai.com/index/introducing-gpt-5-4-mini-and-nano/',NULL,NOW())
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