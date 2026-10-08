import type { ModelBenchmarkCategory } from "./models";

export interface BenchmarkRow {
  model_slug: string;
  category: ModelBenchmarkCategory;
  benchmark_name: string;
  benchmark_version: string | null;
  score_numeric: number | null;
  score_display: string;
  tools: boolean | null;
  reasoning_effort: string | null;
  harness: string | null;
  evaluator: string | null;
  source: string | null;
}

const effortLevels = ["none", "minimal", "low", "medium", "high", "xhigh", "max"];
// The older AA-Omniscience catalog label also stores hallucination rate.
const lowerIsBetter = new Set(["AA-Omniscience", "AA-Omniscience Hallucination Rate"]);

function effortRank(effort: string | null): number {
  return effortLevels.indexOf(effort?.trim().toLowerCase() ?? "");
}

export function highestReasoningEffort(rows: BenchmarkRow[]): string | null {
  const rank = rows.reduce((highest, row) => Math.max(highest, effortRank(row.reasoning_effort)), -1);
  return effortLevels[rank] ?? null;
}

export function selectBestBenchmarks(rows: BenchmarkRow[]): BenchmarkRow[] {
  const best = new Map<string, BenchmarkRow>();
  for (const row of rows) {
    const current = best.get(row.benchmark_name);
    if (!current) {
      best.set(row.benchmark_name, row);
      continue;
    }
    const score = row.score_numeric;
    if (score === null || !Number.isFinite(score)) continue;
    const previous = current.score_numeric;
    const better = previous === null || !Number.isFinite(previous)
      || (lowerIsBetter.has(row.benchmark_name) ? score < previous : score > previous);
    const tiedAtHigherEffort = score === previous && effortRank(row.reasoning_effort) > effortRank(current.reasoning_effort);
    if (better || tiedAtHigherEffort) best.set(row.benchmark_name, row);
  }
  return [...best.values()];
}

export function benchmarkValue(row: BenchmarkRow, highestEffort: string | null): string {
  const qualifiers: string[] = [];
  if (
    row.benchmark_version
    && row.benchmark_version.trim().toLowerCase() !== "public"
    && !row.benchmark_name.toLowerCase().includes(row.benchmark_version.toLowerCase())
  ) {
    qualifiers.push(row.benchmark_version);
  }
  if (row.tools === true) qualifiers.push("tools");
  if (row.tools === false) qualifiers.push("no tools");
  if (row.reasoning_effort && row.reasoning_effort.trim().toLowerCase() !== highestEffort) {
    qualifiers.push(row.reasoning_effort);
  }
  return qualifiers.length > 0 ? `${row.score_display} (${qualifiers.join("; ")})` : row.score_display;
}
