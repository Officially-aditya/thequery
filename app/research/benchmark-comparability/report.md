# AI Benchmark Scores Change. The Model Often Doesn't.

By [Addy](https://www.thequery.in/about) · October 4, 2026

A model name and a benchmark name do not always identify one score. In TheQuery's October 4 data release, 63 model-and-benchmark groups contain different recorded numeric results. That is a reason to read the conditions attached to each result before making a comparison. It is not evidence that 63 results are wrong.

The release contains 105 model records from 15 developers and 837 reported benchmark observations. We audited the catalog's structure and checked three primary-source pages for concrete examples of how evaluation conditions change the meaning of a number. TheQuery did not run these benchmarks or independently reproduce the vendors' results.

The strongest finding is also the most practical: a benchmark score needs a record of how it was obtained. Some differences are explained by tool access, some by the software surrounding the model, and some by a changed scoring process. A larger number can describe a different test setup without establishing that the underlying model improved.

## What the catalog audit actually measures

The release preserves the website-produced snapshot timestamp, October 2, 2026 at 10:20:11.074 UTC. We downloaded it on October 4 and generated the companion CSV from that same JSON, retaining ISO-formatted evaluation dates. The earlier GitHub release contained 95 models and 676 observations. The new release adds 10 model records and 161 observations; that growth does not by itself establish stronger verification.

The 837 observations cover 84 model records and 86 distinct benchmark names. Every observation records a source URL, a numeric score, a unit, and an evaluator. There are 80 distinct recorded source URLs. These counts describe the selected catalog, rather than the entire market or 837 independent experiments. Some model records are aliases or related variants, and some observations repeat a result in another comparison table.

To find cases worth examining, we grouped observations by exact model identifier, exact benchmark name, and score unit. We treated `%` and `percent` as the same unit. We deliberately kept versions, dates, evaluators, and other conditions inside those groups so that differences remained visible. This produced 706 groups, including 113 with multiple observations, 63 with different numeric scores, and 29 with both tool-enabled and tool-disabled observations.

These are retrieval groups, not matched experimental cohorts. Two rows in a group may use different benchmark versions or evaluation procedures. Different names may also describe related tests that this grouping does not join. The accompanying [audit script and observation-level evidence](https://github.com/Officially-aditya/thequery-ai-data/tree/snapshot-2026-10-04/audit) make that choice inspectable.

## An empty catalog field is not a vendor omission

The structured fields are unevenly populated. The table below counts a field as recorded when it is neither null nor empty. A recorded `false` for tool access counts as present, because it tells the reader that the recorded run did not use tools.

| Field in the catalog | Recorded observations | Total observations |
| --- | ---: | ---: |
| Source URL | 837 | 837 |
| Evaluator | 837 | 837 |
| Evaluation date | 749 | 837 |
| Evaluation harness | 471 | 837 |
| Tool access | 430 | 837 |
| Reasoning effort | 382 | 837 |
| Separate benchmark-version field | 258 | 837 |

The final row requires particular care. A version may appear in the benchmark name, in notes, or in the original source rather than in the separate field. Our text check found 202 observations with an empty version field and a version-like token in the benchmark name. That check is a heuristic: a token can identify a subset or another qualifier, so it is not a verified count of complete version records.

This is closer to checking a library catalog than inspecting every book on its shelves. An empty catalog entry does not prove the information is absent from the book. We therefore make no claim that vendors withheld the missing fields, and we do not use field completeness to score vendor transparency. Some fields may also be inapplicable to a particular evaluation. Filling every cell is not a sufficient test of comparability.

## Qwen reports two different HLE conditions

The [Qwen3.5-122B-A10B model card](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) reports 25.3 for Humanity's Last Exam under its reasoning condition and 47.5 under its tool condition. The catalog retains both observations, including their different tool flags. Their reported scores differ by 22.2 percentage points, while the model identifier remains the same.

Tool access lets a model use external capabilities while answering, rather like letting an exam candidate consult reference material. Qwen also describes a strategy that prunes earlier tool responses when accumulated context reaches a threshold. These are parts of the evaluated system. The two table entries do not isolate the causal effect of tools or establish equal costs. They demonstrate why a score stripped of its condition can describe the wrong comparison.

## DeepSeek publishes eight harness results for one model

A harness is the software that runs an evaluation and mediates the model's actions. For a coding agent, it is comparable to the workbench and equipment supplied alongside the worker. The [DeepSeek-V4.1-Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) makes this concrete by reporting the same model across eight agent scaffolds.

| Agent scaffold | DeepSWE v1.1, resolved (%) | Terminal-Bench 2.1, Pass@1 (%) |
| --- | ---: | ---: |
| Claude Code | 69.8 | 88.0 |
| Codex | 65.6 | 84.1 |
| OpenCode | 65.5 | 85.0 |
| Pi | 66.2 | 86.1 |
| mini-SWE | 74.2 | 90.3 |
| DSH Minimal | 72.6 | 90.6 |
| DSH Standard | 70.5 | 85.8 |
| DSH PTC | 67.6 | 85.8 |

The reported DeepSWE range is 65.5 to 74.2, a difference of 8.7 points. Terminal-Bench ranges from 84.1 to 90.6, a difference of 6.5 points. The source specifies maximum reasoning effort, sample counts, context limits, and other settings, including disabled network access for Terminal-Bench. These remain DeepSeek's reported results, without independent reproduction by TheQuery.

Our catalog stores 74.2 with the mini-SWE harness and mentions 72.6 in notes. The complete eight-scaffold table exists in the source. This is a concrete example of why a catalog can preserve provenance while still representing only part of the available evaluation detail.

## A changed grader can revise an older model's score

Anthropic's [Claude Sonnet 5 announcement](https://www.anthropic.com/news/claude-sonnet-5) explicitly explains a revision to Sonnet 4.6's Humanity's Last Exam results. It reports 34.6% without tools and 46.8% with tools after changing the grader, and says this accounts for differences from the earlier launch post. The catalog's corresponding records retain those numbers and note the updated grader.

A grader is the mechanism that decides whether an answer receives credit. Changing it is like changing the marking scheme after an exam: the reported mark can move without the candidate sitting a new course. Anthropic separately explains a change to its OSWorld-Verified evaluation procedure. A date and an evaluator help trace those revisions, but the methodological explanation is needed to understand them. Treating all historical scores as results from one unchanged procedure would lose that distinction.

## What this evidence supports, and what remains unknown

The three source checks were selected to examine specific mechanisms: tool access, agent scaffolds, and revised evaluation procedures. They are purposive examples, not a random sample of the catalog. We verified the cited table entries and explanations; we did not revalidate every observation associated with those pages, check all 80 source URLs, or execute any benchmark. The [source-check record](https://github.com/Officially-aditya/thequery-ai-data/blob/snapshot-2026-10-04/audit/source-checks.json) identifies the observations and limits of each check.

The 63 groups with different scores support a narrower finding: retaining multiple observations preserves information that a single-score model table can discard. They do not establish 63 contradictions, a rate of misleading reporting, or the proportion of AI benchmarks that are unreliable. The selective catalog and its grouping rules do not support those population-wide conclusions.

Nor does this analysis identify a universally best model. Choosing the maximum result from each group would silently mix tools, dates, harnesses, and evaluators. Reasoning effort, the amount of computation allocated to working through a problem, adds another condition; it is like giving one candidate more working time. Matching a benchmark name cannot compensate for unequal conditions or unrecorded budgets.

## How to use and cite the release

For a comparison, start with the original source and the observation's full conditions. Match benchmark versions and subsets, scoring units, tool access, reasoning settings, harnesses, and evaluators where the evidence allows. If a condition is unknown, keep it unknown. A justified comparison may involve fewer models than a broad leaderboard, but its result has a clearer meaning.

The [versioned October 4 release](https://github.com/Officially-aditya/thequery-ai-data/tree/snapshot-2026-10-04) includes JSON, CSV, the audit script, computed summaries, source checks, and this report. Running the script requires no network access. It verifies unique observation IDs, model references, and numeric scores, then regenerates the catalog counts and groups. The summary records SHA-256 hashes, file fingerprints that help readers check they are using the same data. It does not automate verification of the primary sources.

Cite TheQuery, this report, the October 4 release, and the preserved October 2 snapshot timestamp when using the catalog analysis. Cite the original evaluator and source as well when using an individual benchmark score. Corrections should identify the observation ID, the source, and the proposed change through [TheQuery's research page](https://www.thequery.in/research#citation).

The score needs its conditions. The model name is not enough.

*Sources:*

- [TheQuery AI model catalog, October 4 release](https://github.com/Officially-aditya/thequery-ai-data/tree/snapshot-2026-10-04) - TheQuery
- [Qwen3.5-122B-A10B model card](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) - Qwen
- [DeepSeek-V4.1-Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) - DeepSeek
- [Introducing Claude Sonnet 5](https://www.anthropic.com/news/claude-sonnet-5) - Anthropic

**Previously on TheQuery:** [How to Audit AI Benchmark Claims: A Practical Guide](https://www.thequery.in/guides/how-to-check-ai-benchmark-claims) and [RAG Works in Theory. Here's Why It Fails in Production.](https://www.thequery.in/guides/rag-fails-in-production)
