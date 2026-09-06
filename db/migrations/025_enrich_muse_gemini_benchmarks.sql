-- Backfill normalized benchmark evidence for sparse Gemini and Muse comparison models.
-- Sources verified 2026-09-06. Vendor, harness, evaluator and effort remain normalized
-- in model_benchmarks even though provenance-only details are omitted from cell rendering.

INSERT INTO model_benchmarks (
  id,model_slug,category,benchmark_name,benchmark_version,score_numeric,score_display,
  score_unit,tools,reasoning_effort,harness,evaluator,evaluation_date,source,notes,updated_at
)
SELECT
  b.id,b.model_slug,b.category,b.benchmark_name,b.benchmark_version,b.score_numeric,b.score_display,
  b.score_unit,b.tools,b.reasoning_effort,b.harness,b.evaluator,
  NULLIF(b.evaluation_date,'')::date,b.source,b.notes,NOW()
FROM jsonb_to_recordset($benchmarks$
[
 {"id":"tq-20260906-benchfill-g31pro-swe-verified","model_slug":"gemini-3-1-pro","category":"coding","benchmark_name":"SWE-bench Verified","benchmark_version":null,"score_numeric":80.6,"score_display":"80.6%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":"Single attempt","evaluator":"Google DeepMind","evaluation_date":"2026-02-19","source":"https://deepmind.google/models/model-cards/gemini-3-1-pro/","notes":null},
 {"id":"tq-20260906-benchfill-g31pro-gpqa","model_slug":"gemini-3-1-pro","category":"knowledge","benchmark_name":"GPQA Diamond","benchmark_version":null,"score_numeric":94.3,"score_display":"94.3%","score_unit":"percent","tools":false,"reasoning_effort":"high","harness":null,"evaluator":"Google DeepMind","evaluation_date":"2026-02-19","source":"https://deepmind.google/models/model-cards/gemini-3-1-pro/","notes":null},
 {"id":"tq-20260906-benchfill-g31pro-hle-no-tools","model_slug":"gemini-3-1-pro","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":null,"score_numeric":44.4,"score_display":"44.4%","score_unit":"percent","tools":false,"reasoning_effort":"high","harness":null,"evaluator":"Google DeepMind","evaluation_date":"2026-02-19","source":"https://deepmind.google/models/model-cards/gemini-3-1-pro/","notes":null},
 {"id":"tq-20260906-benchfill-g31pro-hle-tools","model_slug":"gemini-3-1-pro","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":null,"score_numeric":51.4,"score_display":"51.4%","score_unit":"percent","tools":true,"reasoning_effort":"high","harness":"Search (blocklist) + Code","evaluator":"Google DeepMind","evaluation_date":"2026-02-19","source":"https://deepmind.google/models/model-cards/gemini-3-1-pro/","notes":null},
 {"id":"tq-20260906-benchfill-g31pro-arcagi2","model_slug":"gemini-3-1-pro","category":"math_reasoning","benchmark_name":"ARC-AGI","benchmark_version":"2","score_numeric":77.1,"score_display":"77.1%","score_unit":"percent","tools":false,"reasoning_effort":"high","harness":"ARC Prize Verified","evaluator":"Google DeepMind","evaluation_date":"2026-02-19","source":"https://deepmind.google/models/model-cards/gemini-3-1-pro/","notes":null},
 {"id":"tq-20260906-benchfill-g31pro-gdpval-aa","model_slug":"gemini-3-1-pro","category":"agentic_computer_use","benchmark_name":"GDPval-AA","benchmark_version":null,"score_numeric":1314,"score_display":"1314 Elo","score_unit":"Elo","tools":true,"reasoning_effort":"high","harness":null,"evaluator":"Google DeepMind","evaluation_date":"2026-05-19","source":"https://deepmind.google/models/model-cards/gemini-3-5-flash/","notes":"Later Google comparison value; distinct from GDPval-AA v2."},
 {"id":"tq-20260906-benchfill-g31pro-mcp-atlas","model_slug":"gemini-3-1-pro","category":"agentic_computer_use","benchmark_name":"MCP Atlas","benchmark_version":null,"score_numeric":78.2,"score_display":"78.2%","score_unit":"percent","tools":true,"reasoning_effort":"high","harness":null,"evaluator":"Google DeepMind","evaluation_date":"2026-05-19","source":"https://deepmind.google/models/model-cards/gemini-3-5-flash/","notes":null},
 {"id":"tq-20260906-benchfill-g31pro-browsecomp","model_slug":"gemini-3-1-pro","category":"agentic_computer_use","benchmark_name":"BrowseComp","benchmark_version":null,"score_numeric":85.9,"score_display":"85.9%","score_unit":"percent","tools":true,"reasoning_effort":"high","harness":"Search + Python + Browse","evaluator":"Google DeepMind","evaluation_date":"2026-02-19","source":"https://deepmind.google/models/model-cards/gemini-3-1-pro/","notes":null},
 {"id":"tq-20260906-benchfill-g31lite-swe-pro","model_slug":"gemini-3-1-flash-lite","category":"coding","benchmark_name":"SWE-bench Pro","benchmark_version":"Public","score_numeric":38.3,"score_display":"38.3%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"Google DeepMind","evaluation_date":"2026-07-21","source":"https://deepmind.google/models/model-cards/gemini-3-5-flash-lite/","notes":null},
 {"id":"tq-20260906-benchfill-g31lite-terminal21","model_slug":"gemini-3-1-flash-lite","category":"coding","benchmark_name":"Terminal-Bench 2.1","benchmark_version":"Terminus-2","score_numeric":31.0,"score_display":"31.0%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":"Terminus-2","evaluator":"Google DeepMind","evaluation_date":"2026-07-21","source":"https://deepmind.google/models/model-cards/gemini-3-5-flash-lite/","notes":null},
 {"id":"tq-20260906-benchfill-g31lite-mle","model_slug":"gemini-3-1-flash-lite","category":"coding","benchmark_name":"MLE-Bench","benchmark_version":null,"score_numeric":22.0,"score_display":"22.0%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"Google DeepMind","evaluation_date":"2026-07-21","source":"https://deepmind.google/models/model-cards/gemini-3-5-flash-lite/","notes":null},
 {"id":"tq-20260906-benchfill-g31lite-gdpv2","model_slug":"gemini-3-1-flash-lite","category":"agentic_computer_use","benchmark_name":"GDPval-AA v2","benchmark_version":null,"score_numeric":642,"score_display":"642 Elo","score_unit":"Elo","tools":true,"reasoning_effort":null,"harness":"Artificial Analysis","evaluator":"Google DeepMind","evaluation_date":"2026-07-21","source":"https://deepmind.google/models/model-cards/gemini-3-5-flash-lite/","notes":null},
 {"id":"tq-20260906-benchfill-g31lite-osworld-verified","model_slug":"gemini-3-1-flash-lite","category":"agentic_computer_use","benchmark_name":"OSWorld-Verified","benchmark_version":null,"score_numeric":54.3,"score_display":"54.3%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":null,"evaluator":"Google DeepMind","evaluation_date":"2026-07-21","source":"https://deepmind.google/models/model-cards/gemini-3-5-flash-lite/","notes":null},
 {"id":"tq-20260906-benchfill-g35-toolathlon","model_slug":"gemini-3-5-flash","category":"agentic_computer_use","benchmark_name":"Toolathlon","benchmark_version":null,"score_numeric":56.5,"score_display":"56.5%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":null,"evaluator":"Google DeepMind","evaluation_date":"2026-05-19","source":"https://deepmind.google/models/model-cards/gemini-3-5-flash/","notes":null},
 {"id":"tq-20260906-benchfill-g35-gdpval-aa","model_slug":"gemini-3-5-flash","category":"agentic_computer_use","benchmark_name":"GDPval-AA","benchmark_version":null,"score_numeric":1656,"score_display":"1656 Elo","score_unit":"Elo","tools":true,"reasoning_effort":null,"harness":null,"evaluator":"Google DeepMind","evaluation_date":"2026-05-19","source":"https://deepmind.google/models/model-cards/gemini-3-5-flash/","notes":"Original GDPval-AA benchmark; distinct from GDPval-AA v2."},
 {"id":"tq-20260906-benchfill-g35-hle","model_slug":"gemini-3-5-flash","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":null,"score_numeric":40.2,"score_display":"40.2%","score_unit":"percent","tools":false,"reasoning_effort":null,"harness":null,"evaluator":"Google DeepMind","evaluation_date":"2026-05-19","source":"https://deepmind.google/models/model-cards/gemini-3-5-flash/","notes":null},
 {"id":"tq-20260906-benchfill-g35-arcagi2","model_slug":"gemini-3-5-flash","category":"math_reasoning","benchmark_name":"ARC-AGI","benchmark_version":"2","score_numeric":72.1,"score_display":"72.1%","score_unit":"percent","tools":false,"reasoning_effort":null,"harness":null,"evaluator":"Google DeepMind","evaluation_date":"2026-05-19","source":"https://deepmind.google/models/model-cards/gemini-3-5-flash/","notes":null},
 {"id":"tq-20260906-benchfill-g35lite-swe-pro","model_slug":"gemini-3-5-flash-lite","category":"coding","benchmark_name":"SWE-bench Pro","benchmark_version":"Public","score_numeric":54.2,"score_display":"54.2%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"Google DeepMind","evaluation_date":"2026-07-21","source":"https://deepmind.google/models/model-cards/gemini-3-5-flash-lite/","notes":null},
 {"id":"tq-20260906-benchfill-g35lite-terminal21","model_slug":"gemini-3-5-flash-lite","category":"coding","benchmark_name":"Terminal-Bench 2.1","benchmark_version":"Terminus-2","score_numeric":54.0,"score_display":"54.0%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":"Terminus-2","evaluator":"Google DeepMind","evaluation_date":"2026-07-21","source":"https://deepmind.google/models/model-cards/gemini-3-5-flash-lite/","notes":null},
 {"id":"tq-20260906-benchfill-g35lite-mle","model_slug":"gemini-3-5-flash-lite","category":"coding","benchmark_name":"MLE-Bench","benchmark_version":null,"score_numeric":39.2,"score_display":"39.2%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"Google DeepMind","evaluation_date":"2026-07-21","source":"https://deepmind.google/models/model-cards/gemini-3-5-flash-lite/","notes":null},
 {"id":"tq-20260906-benchfill-g35lite-gdpv2","model_slug":"gemini-3-5-flash-lite","category":"agentic_computer_use","benchmark_name":"GDPval-AA v2","benchmark_version":null,"score_numeric":1140,"score_display":"1140 Elo","score_unit":"Elo","tools":true,"reasoning_effort":null,"harness":"Artificial Analysis","evaluator":"Google DeepMind","evaluation_date":"2026-07-21","source":"https://deepmind.google/models/model-cards/gemini-3-5-flash-lite/","notes":null},
 {"id":"tq-20260906-benchfill-g35lite-osworld-verified","model_slug":"gemini-3-5-flash-lite","category":"agentic_computer_use","benchmark_name":"OSWorld-Verified","benchmark_version":null,"score_numeric":74.0,"score_display":"74.0%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":null,"evaluator":"Google DeepMind","evaluation_date":"2026-07-21","source":"https://deepmind.google/models/model-cards/gemini-3-5-flash-lite/","notes":null},
 {"id":"tq-20260906-benchfill-g37-terminal40","model_slug":"gemini-3-7-flash","category":"coding","benchmark_name":"Terminal-Bench 4.0","benchmark_version":null,"score_numeric":11.2,"score_display":"11.2%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"Google DeepMind","evaluation_date":"2026-09-02","source":"https://deepmind.google/models/model-cards/gemini-3-8-flash/","notes":"Reported in the Gemini 3.8 Flash comparison table."},
 {"id":"tq-20260906-benchfill-g37-cursorbench32","model_slug":"gemini-3-7-flash","category":"coding","benchmark_name":"CursorBench","benchmark_version":"3.2","score_numeric":61.6,"score_display":"61.6%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":"CursorBench 3.2","evaluator":"Cursor","evaluation_date":"2026-09-02","source":"https://cursor.com/evals","notes":null},
 {"id":"tq-20260906-benchfill-g37-livecodebench","model_slug":"gemini-3-7-flash","category":"coding","benchmark_name":"LiveCodeBench","benchmark_version":null,"score_numeric":88.7,"score_display":"88.7%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":null,"evaluator":"Vals AI","evaluation_date":"2026-09-05","source":"https://www.vals.ai/benchmarks/lcb","notes":null},
 {"id":"tq-20260906-benchfill-g38-deepswe","model_slug":"gemini-3-8-flash","category":"coding","benchmark_name":"DeepSWE v1.1","benchmark_version":null,"score_numeric":73.7,"score_display":"73.7%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"Google DeepMind","evaluation_date":"2026-09-02","source":"https://deepmind.google/models/model-cards/gemini-3-8-flash/","notes":null},
 {"id":"tq-20260906-benchfill-g38-terminal21","model_slug":"gemini-3-8-flash","category":"coding","benchmark_name":"Terminal-Bench 2.1","benchmark_version":null,"score_numeric":89.4,"score_display":"89.4%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"Google DeepMind","evaluation_date":"2026-09-02","source":"https://deepmind.google/models/model-cards/gemini-3-8-flash/","notes":null},
 {"id":"tq-20260906-benchfill-g38-terminal40","model_slug":"gemini-3-8-flash","category":"coding","benchmark_name":"Terminal-Bench 4.0","benchmark_version":null,"score_numeric":19.1,"score_display":"19.1%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"Google DeepMind","evaluation_date":"2026-09-02","source":"https://deepmind.google/models/model-cards/gemini-3-8-flash/","notes":null},
 {"id":"tq-20260906-benchfill-g38-gdpv2","model_slug":"gemini-3-8-flash","category":"agentic_computer_use","benchmark_name":"GDPval-AA v2","benchmark_version":null,"score_numeric":1545,"score_display":"1545 Elo","score_unit":"Elo","tools":true,"reasoning_effort":null,"harness":null,"evaluator":"Google DeepMind","evaluation_date":"2026-09-02","source":"https://deepmind.google/models/model-cards/gemini-3-8-flash/","notes":null},
 {"id":"tq-20260906-benchfill-g38-osworld20","model_slug":"gemini-3-8-flash","category":"agentic_computer_use","benchmark_name":"OSWorld 2.0","benchmark_version":null,"score_numeric":59.0,"score_display":"59.0%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":null,"evaluator":"Google DeepMind","evaluation_date":"2026-09-02","source":"https://deepmind.google/models/model-cards/gemini-3-8-flash/","notes":null},
 {"id":"tq-20260906-benchfill-g38-cursorbench32","model_slug":"gemini-3-8-flash","category":"coding","benchmark_name":"CursorBench","benchmark_version":"3.2","score_numeric":69.2,"score_display":"69.2%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":"CursorBench 3.2","evaluator":"Cursor","evaluation_date":"2026-09-02","source":"https://cursor.com/evals","notes":null},
 {"id":"tq-20260906-benchfill-g38-livecodebench","model_slug":"gemini-3-8-flash","category":"coding","benchmark_name":"LiveCodeBench","benchmark_version":null,"score_numeric":89.5,"score_display":"89.5%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":null,"evaluator":"Vals AI","evaluation_date":"2026-09-05","source":"https://www.vals.ai/benchmarks/lcb","notes":null},
 {"id":"tq-20260906-benchfill-muse1-mcp-atlas","model_slug":"muse-spark","category":"agentic_computer_use","benchmark_name":"MCP Atlas","benchmark_version":null,"score_numeric":82.2,"score_display":"82.2%","score_unit":"percent","tools":true,"reasoning_effort":"Contemplating","harness":"Muse Spark 1.1 launch comparison","evaluator":"Meta","evaluation_date":"2026-07-09","source":"https://research.meta.ai/blog/introducing-muse-spark-meta-model-api","notes":"Comparator result reported by Meta in the Muse Spark 1.1 evaluation."},
 {"id":"tq-20260906-benchfill-muse1-toolathlon","model_slug":"muse-spark","category":"agentic_computer_use","benchmark_name":"Toolathlon","benchmark_version":"Verified","score_numeric":49.4,"score_display":"49.4%","score_unit":"percent","tools":true,"reasoning_effort":"Contemplating","harness":"Muse Spark 1.1 launch comparison","evaluator":"Meta","evaluation_date":"2026-07-09","source":"https://research.meta.ai/blog/introducing-muse-spark-meta-model-api","notes":"Comparator result reported by Meta in the Muse Spark 1.1 evaluation."},
 {"id":"tq-20260906-benchfill-muse1-osworld-verified","model_slug":"muse-spark","category":"agentic_computer_use","benchmark_name":"OSWorld-Verified","benchmark_version":null,"score_numeric":53.3,"score_display":"53.3%","score_unit":"percent","tools":true,"reasoning_effort":"Contemplating","harness":"Muse Spark 1.1 launch comparison","evaluator":"Meta","evaluation_date":"2026-07-09","source":"https://research.meta.ai/blog/introducing-muse-spark-meta-model-api","notes":"Comparator result reported by Meta in the Muse Spark 1.1 evaluation."},
 {"id":"tq-20260906-benchfill-muse1-hle-tools","model_slug":"muse-spark","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":null,"score_numeric":50.4,"score_display":"50.4%","score_unit":"percent","tools":true,"reasoning_effort":"Contemplating","harness":"Muse Spark 1.1 launch comparison","evaluator":"Meta","evaluation_date":"2026-07-09","source":"https://research.meta.ai/blog/introducing-muse-spark-meta-model-api","notes":"Comparator result reported by Meta in the Muse Spark 1.1 evaluation."},
 {"id":"tq-20260906-benchfill-muse1-terminal21","model_slug":"muse-spark","category":"coding","benchmark_name":"Terminal-Bench 2.1","benchmark_version":null,"score_numeric":67.3,"score_display":"67.3%","score_unit":"percent","tools":true,"reasoning_effort":"Contemplating","harness":"Muse Spark 1.1 launch comparison","evaluator":"Meta","evaluation_date":"2026-07-09","source":"https://research.meta.ai/blog/introducing-muse-spark-meta-model-api","notes":"Comparator result reported by Meta in the Muse Spark 1.1 evaluation."},
 {"id":"tq-20260906-benchfill-muse1-swe-pro","model_slug":"muse-spark","category":"coding","benchmark_name":"SWE-bench Pro","benchmark_version":"Public","score_numeric":55.0,"score_display":"55.0%","score_unit":"percent","tools":true,"reasoning_effort":"Contemplating","harness":"Muse Spark 1.1 launch comparison","evaluator":"Meta","evaluation_date":"2026-07-09","source":"https://research.meta.ai/blog/introducing-muse-spark-meta-model-api","notes":"Comparator result reported by Meta in the Muse Spark 1.1 evaluation."},
 {"id":"tq-20260906-benchfill-muse1-deepswe","model_slug":"muse-spark","category":"coding","benchmark_name":"DeepSWE v1.1","benchmark_version":null,"score_numeric":10.0,"score_display":"10.0%","score_unit":"percent","tools":true,"reasoning_effort":"Contemplating","harness":"Muse Spark 1.1 launch comparison","evaluator":"Meta","evaluation_date":"2026-07-09","source":"https://research.meta.ai/blog/introducing-muse-spark-meta-model-api","notes":"Comparator result reported by Meta in the Muse Spark 1.1 evaluation."},
 {"id":"tq-20260906-benchfill-muse11-mcp-atlas","model_slug":"muse-spark-1-1","category":"agentic_computer_use","benchmark_name":"MCP Atlas","benchmark_version":null,"score_numeric":88.1,"score_display":"88.1%","score_unit":"percent","tools":true,"reasoning_effort":"xhigh","harness":"Muse Spark 1.1 evaluation","evaluator":"Meta","evaluation_date":"2026-07-09","source":"https://research.meta.ai/blog/introducing-muse-spark-meta-model-api","notes":null},
 {"id":"tq-20260906-benchfill-muse11-toolathlon","model_slug":"muse-spark-1-1","category":"agentic_computer_use","benchmark_name":"Toolathlon","benchmark_version":"Verified","score_numeric":75.6,"score_display":"75.6%","score_unit":"percent","tools":true,"reasoning_effort":"xhigh","harness":"Toolathlon-Verified","evaluator":"Meta","evaluation_date":"2026-07-09","source":"https://research.meta.ai/blog/introducing-muse-spark-meta-model-api","notes":null},
 {"id":"tq-20260906-benchfill-muse11-osworld-verified","model_slug":"muse-spark-1-1","category":"agentic_computer_use","benchmark_name":"OSWorld-Verified","benchmark_version":null,"score_numeric":80.8,"score_display":"80.8%","score_unit":"percent","tools":true,"reasoning_effort":"xhigh","harness":"Muse Spark 1.1 evaluation","evaluator":"Meta","evaluation_date":"2026-07-09","source":"https://research.meta.ai/blog/introducing-muse-spark-meta-model-api","notes":null},
 {"id":"tq-20260906-benchfill-muse11-hle-tools","model_slug":"muse-spark-1-1","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":null,"score_numeric":62.1,"score_display":"62.1%","score_unit":"percent","tools":true,"reasoning_effort":"xhigh","harness":"Browser + bash tools","evaluator":"Meta","evaluation_date":"2026-07-09","source":"https://research.meta.ai/blog/introducing-muse-spark-meta-model-api","notes":null},
 {"id":"tq-20260906-benchfill-muse11-terminal21","model_slug":"muse-spark-1-1","category":"coding","benchmark_name":"Terminal-Bench 2.1","benchmark_version":null,"score_numeric":80.0,"score_display":"80.0%","score_unit":"percent","tools":true,"reasoning_effort":"xhigh","harness":"Muse Spark 1.1 agent harness","evaluator":"Meta","evaluation_date":"2026-07-09","source":"https://research.meta.ai/blog/introducing-muse-spark-meta-model-api","notes":null},
 {"id":"tq-20260906-benchfill-muse11-swe-pro","model_slug":"muse-spark-1-1","category":"coding","benchmark_name":"SWE-bench Pro","benchmark_version":"Public","score_numeric":61.5,"score_display":"61.5%","score_unit":"percent","tools":true,"reasoning_effort":"xhigh","harness":"mini-swe-agent","evaluator":"Scale AI","evaluation_date":"2026-07-09","source":"https://research.meta.ai/blog/introducing-muse-spark-meta-model-api","notes":null},
 {"id":"tq-20260906-benchfill-muse11-deepswe","model_slug":"muse-spark-1-1","category":"coding","benchmark_name":"DeepSWE v1.1","benchmark_version":null,"score_numeric":53.3,"score_display":"53.3%","score_unit":"percent","tools":true,"reasoning_effort":"xhigh","harness":"mini-swe-agent fork","evaluator":"Meta","evaluation_date":"2026-07-09","source":"https://research.meta.ai/blog/introducing-muse-spark-meta-model-api","notes":null},
 {"id":"tq-20260906-benchfill-muse11-gdpv2","model_slug":"muse-spark-1-1","category":"agentic_computer_use","benchmark_name":"GDPval-AA v2","benchmark_version":null,"score_numeric":1381,"score_display":"1381 Elo","score_unit":"Elo","tools":true,"reasoning_effort":"xhigh","harness":"Stirrup","evaluator":"Artificial Analysis","evaluation_date":"2026-07-09","source":"https://research.meta.ai/blog/introducing-muse-spark-meta-model-api","notes":null},
 {"id":"tq-20260906-benchfill-muse12-mcp-atlas","model_slug":"muse-spark-1-2","category":"agentic_computer_use","benchmark_name":"MCP Atlas","benchmark_version":null,"score_numeric":90.3,"score_display":"90.3%","score_unit":"percent","tools":true,"reasoning_effort":"xhigh","harness":"Scale AI MCP Atlas","evaluator":"Scale AI","evaluation_date":"2026-08-05","source":"https://research.meta.ai/blog/introducing-muse-code-and-muse-spark-1-2","notes":null},
 {"id":"tq-20260906-benchfill-muse12-deepswe-meta","model_slug":"muse-spark-1-2","category":"coding","benchmark_name":"DeepSWE v1.1","benchmark_version":null,"score_numeric":59.3,"score_display":"59.3%","score_unit":"percent","tools":true,"reasoning_effort":"xhigh","harness":"Muse Code","evaluator":"Meta","evaluation_date":"2026-08-05","source":"https://research.meta.ai/blog/introducing-muse-code-and-muse-spark-1-2","notes":"Meta first-party evaluation; retained separately from Google's cross-vendor comparison row."},
 {"id":"tq-20260906-benchfill-muse12-gdpv2-aa","model_slug":"muse-spark-1-2","category":"agentic_computer_use","benchmark_name":"GDPval-AA v2","benchmark_version":null,"score_numeric":1631,"score_display":"1631 Elo","score_unit":"Elo","tools":true,"reasoning_effort":"xhigh","harness":"Stirrup","evaluator":"Artificial Analysis","evaluation_date":"2026-08-05","source":"https://research.meta.ai/blog/introducing-muse-code-and-muse-spark-1-2","notes":"Provider benchmark result reported in Meta methodology; retained separately from Google's cross-vendor comparison row."},
 {"id":"tq-20260906-benchfill-muse13-deepswe","model_slug":"muse-spark-1-3","category":"coding","benchmark_name":"DeepSWE v1.1","benchmark_version":null,"score_numeric":75.4,"score_display":"75.4%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"Muse Spark 1.3 launch evaluation","evaluator":"Meta","evaluation_date":"2026-09-02","source":"https://research.meta.ai/blog/introducing-muse-spark-1-3","notes":null},
 {"id":"tq-20260906-benchfill-muse13-terminal21","model_slug":"muse-spark-1-3","category":"coding","benchmark_name":"Terminal-Bench 2.1","benchmark_version":null,"score_numeric":88.8,"score_display":"88.8%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"Muse Spark 1.3 launch evaluation","evaluator":"Meta","evaluation_date":"2026-09-02","source":"https://research.meta.ai/blog/introducing-muse-spark-1-3","notes":null},
 {"id":"tq-20260906-benchfill-muse13-gdpv2","model_slug":"muse-spark-1-3","category":"agentic_computer_use","benchmark_name":"GDPval-AA v2","benchmark_version":null,"score_numeric":1754,"score_display":"1754 Elo","score_unit":"Elo","tools":true,"reasoning_effort":"max","harness":"Stirrup","evaluator":"Artificial Analysis","evaluation_date":"2026-09-02","source":"https://research.meta.ai/blog/introducing-muse-spark-1-3","notes":null},
 {"id":"tq-20260906-benchfill-muse13-osworld20","model_slug":"muse-spark-1-3","category":"agentic_computer_use","benchmark_name":"OSWorld 2.0","benchmark_version":null,"score_numeric":66.9,"score_display":"66.9%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"Muse Spark 1.3 launch evaluation","evaluator":"Meta","evaluation_date":"2026-09-02","source":"https://research.meta.ai/blog/introducing-muse-spark-1-3","notes":null},
 {"id":"tq-20260906-benchfill-muse13-automation","model_slug":"muse-spark-1-3","category":"agentic_computer_use","benchmark_name":"AutomationBench","benchmark_version":null,"score_numeric":49.4,"score_display":"49.4%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"Muse Spark 1.3 launch evaluation","evaluator":"Meta","evaluation_date":"2026-09-02","source":"https://research.meta.ai/blog/introducing-muse-spark-1-3","notes":null},
 {"id":"tq-20260906-benchfill-glimmer-mcp-atlas","model_slug":"muse-glimmer","category":"agentic_computer_use","benchmark_name":"MCP Atlas","benchmark_version":"Public","score_numeric":75.5,"score_display":"75.5%","score_unit":"percent","tools":true,"reasoning_effort":"High","harness":"Muse Glimmer high-reasoning evaluation","evaluator":"Meta","evaluation_date":"2026-08-10","source":"https://huggingface.co/meta-models/Muse-Glimmer-30B","notes":null},
 {"id":"tq-20260906-benchfill-glimmer-gdpv2","model_slug":"muse-glimmer","category":"agentic_computer_use","benchmark_name":"GDPval-AA v2","benchmark_version":null,"score_numeric":953,"score_display":"953 Elo","score_unit":"Elo","tools":true,"reasoning_effort":"High","harness":"Stirrup","evaluator":"Artificial Analysis","evaluation_date":"2026-08-10","source":"https://huggingface.co/meta-models/Muse-Glimmer-30B","notes":null},
 {"id":"tq-20260906-benchfill-glimmer-osworld-verified","model_slug":"muse-glimmer","category":"agentic_computer_use","benchmark_name":"OSWorld-Verified","benchmark_version":null,"score_numeric":65.9,"score_display":"65.9%","score_unit":"percent","tools":true,"reasoning_effort":"High","harness":"Muse Glimmer high-reasoning evaluation","evaluator":"Meta","evaluation_date":"2026-08-10","source":"https://huggingface.co/meta-models/Muse-Glimmer-30B","notes":null},
 {"id":"tq-20260906-benchfill-glimmer-swe-pro","model_slug":"muse-glimmer","category":"coding","benchmark_name":"SWE-bench Pro","benchmark_version":"Public","score_numeric":51.2,"score_display":"51.2%","score_unit":"percent","tools":true,"reasoning_effort":"High","harness":"ScaleAI SWE-bench Pro","evaluator":"Meta","evaluation_date":"2026-08-10","source":"https://huggingface.co/meta-models/Muse-Glimmer-30B","notes":null},
 {"id":"tq-20260906-benchfill-glimmer-swe-verified","model_slug":"muse-glimmer","category":"coding","benchmark_name":"SWE-bench Verified","benchmark_version":null,"score_numeric":76.0,"score_display":"76.0%","score_unit":"percent","tools":true,"reasoning_effort":"High","harness":"SWE-bench Verified","evaluator":"Meta","evaluation_date":"2026-08-10","source":"https://huggingface.co/meta-models/Muse-Glimmer-30B","notes":null},
 {"id":"tq-20260906-benchfill-glimmer-terminal21","model_slug":"muse-glimmer","category":"coding","benchmark_name":"Terminal-Bench 2.1","benchmark_version":"Terminus-2","score_numeric":51.7,"score_display":"51.7%","score_unit":"percent","tools":true,"reasoning_effort":"High","harness":"Terminus-2","evaluator":"Artificial Analysis","evaluation_date":"2026-08-10","source":"https://huggingface.co/meta-models/Muse-Glimmer-30B","notes":null},
 {"id":"tq-20260906-benchfill-glimmer-aime2026","model_slug":"muse-glimmer","category":"math_reasoning","benchmark_name":"AIME","benchmark_version":"2026","score_numeric":94.7,"score_display":"94.7%","score_unit":"percent","tools":false,"reasoning_effort":"High","harness":"Muse Glimmer high-reasoning evaluation","evaluator":"Meta","evaluation_date":"2026-08-10","source":"https://huggingface.co/meta-models/Muse-Glimmer-30B","notes":null},
 {"id":"tq-20260906-benchfill-glimmer-gpqa","model_slug":"muse-glimmer","category":"knowledge","benchmark_name":"GPQA Diamond","benchmark_version":null,"score_numeric":83.5,"score_display":"83.5%","score_unit":"percent","tools":false,"reasoning_effort":"High","harness":"AA evaluation","evaluator":"Artificial Analysis","evaluation_date":"2026-08-10","source":"https://huggingface.co/meta-models/Muse-Glimmer-30B","notes":null},
 {"id":"tq-20260906-benchfill-glimmer-hle","model_slug":"muse-glimmer","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":null,"score_numeric":22.0,"score_display":"22.0%","score_unit":"percent","tools":false,"reasoning_effort":"High","harness":"Text evaluation","evaluator":"Artificial Analysis","evaluation_date":"2026-08-10","source":"https://huggingface.co/meta-models/Muse-Glimmer-30B","notes":null}
]
$benchmarks$::jsonb) AS b(
  id text,model_slug text,category text,benchmark_name text,benchmark_version text,
  score_numeric double precision,score_display text,score_unit text,tools boolean,
  reasoning_effort text,harness text,evaluator text,evaluation_date text,source text,notes text
)
ON CONFLICT (id) DO UPDATE SET
  model_slug=EXCLUDED.model_slug,
  category=EXCLUDED.category,
  benchmark_name=EXCLUDED.benchmark_name,
  benchmark_version=EXCLUDED.benchmark_version,
  score_numeric=EXCLUDED.score_numeric,
  score_display=EXCLUDED.score_display,
  score_unit=EXCLUDED.score_unit,
  tools=EXCLUDED.tools,
  reasoning_effort=EXCLUDED.reasoning_effort,
  harness=EXCLUDED.harness,
  evaluator=EXCLUDED.evaluator,
  evaluation_date=EXCLUDED.evaluation_date,
  source=EXCLUDED.source,
  notes=EXCLUDED.notes,
  updated_at=NOW();

-- Bring existing materialized comparisons up to date without clobbering editorial overrides.
-- A cell is changed only when it is empty or still exactly matches the pre-backfill auto value.
WITH old_rendered AS (
  SELECT
    b.model_slug,
    b.benchmark_name,
    b.evaluation_date,
    b.id,
    CASE
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
      THEN b.score_display || ' (' || concat_ws('; ',
        CASE
          WHEN b.benchmark_version IS NOT NULL
           AND lower(trim(b.benchmark_version)) <> 'public'
           AND position(lower(b.benchmark_version) in lower(b.benchmark_name)) = 0
          THEN b.benchmark_version
        END,
        CASE WHEN b.tools IS TRUE THEN 'tools' WHEN b.tools IS FALSE THEN 'no tools' END,
        b.reasoning_effort
      ) || ')'
      ELSE b.score_display
    END AS rendered
  FROM model_benchmarks AS b
  WHERE b.id NOT LIKE 'tq-20260906-benchfill-%'
),
new_rendered AS (
  SELECT
    b.model_slug,
    b.benchmark_name,
    b.evaluation_date,
    b.id,
    CASE
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
      THEN b.score_display || ' (' || concat_ws('; ',
        CASE
          WHEN b.benchmark_version IS NOT NULL
           AND lower(trim(b.benchmark_version)) <> 'public'
           AND position(lower(b.benchmark_version) in lower(b.benchmark_name)) = 0
          THEN b.benchmark_version
        END,
        CASE WHEN b.tools IS TRUE THEN 'tools' WHEN b.tools IS FALSE THEN 'no tools' END,
        b.reasoning_effort
      ) || ')'
      ELSE b.score_display
    END AS rendered
  FROM model_benchmarks AS b
),
old_dedup AS (
  SELECT model_slug,benchmark_name,rendered,MIN(evaluation_date) AS evaluation_date,MIN(id) AS id
  FROM old_rendered
  GROUP BY model_slug,benchmark_name,rendered
),
new_dedup AS (
  SELECT model_slug,benchmark_name,rendered,MIN(evaluation_date) AS evaluation_date,MIN(id) AS id
  FROM new_rendered
  GROUP BY model_slug,benchmark_name,rendered
),
old_grouped AS (
  SELECT model_slug,benchmark_name,
    string_agg(rendered, ' · ' ORDER BY evaluation_date NULLS LAST, id) AS value
  FROM old_dedup
  GROUP BY model_slug,benchmark_name
),
new_grouped AS (
  SELECT model_slug,benchmark_name,
    string_agg(rendered, ' · ' ORDER BY evaluation_date NULLS LAST, id) AS value
  FROM new_dedup
  GROUP BY model_slug,benchmark_name
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
            COALESCE(
              (
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
                              WHEN new_a.value IS NULL THEN COALESCE(row_entry.row_value->>1, '')
                              WHEN COALESCE(row_entry.row_value->>1, '') = '' THEN new_a.value
                              WHEN old_a.value IS NOT NULL
                               AND COALESCE(row_entry.row_value->>1, '') = old_a.value THEN new_a.value
                              ELSE COALESCE(row_entry.row_value->>1, '')
                            END
                          ),
                          false
                        ),
                        '{2}',
                        to_jsonb(
                          CASE
                            WHEN new_b.value IS NULL THEN COALESCE(row_entry.row_value->>2, '')
                            WHEN COALESCE(row_entry.row_value->>2, '') = '' THEN new_b.value
                            WHEN old_b.value IS NOT NULL
                             AND COALESCE(row_entry.row_value->>2, '') = old_b.value THEN new_b.value
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
                LEFT JOIN old_grouped AS old_a
                  ON old_a.model_slug = c.metadata->>'modelA'
                 AND lower(old_a.benchmark_name) = lower(COALESCE(row_entry.row_value->>0, ''))
                LEFT JOIN new_grouped AS new_a
                  ON new_a.model_slug = c.metadata->>'modelA'
                 AND lower(new_a.benchmark_name) = lower(COALESCE(row_entry.row_value->>0, ''))
                LEFT JOIN old_grouped AS old_b
                  ON old_b.model_slug = c.metadata->>'modelB'
                 AND lower(old_b.benchmark_name) = lower(COALESCE(row_entry.row_value->>0, ''))
                LEFT JOIN new_grouped AS new_b
                  ON new_b.model_slug = c.metadata->>'modelB'
                 AND lower(new_b.benchmark_name) = lower(COALESCE(row_entry.row_value->>0, ''))
              ),
              '[]'::jsonb
            ),
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
  GROUP BY c.id,c.metadata
)
UPDATE content_items AS c
SET blocks=rebuilt.blocks,updated_at=NOW()
FROM rebuilt
WHERE c.id=rebuilt.id
  AND c.blocks IS DISTINCT FROM rebuilt.blocks;
