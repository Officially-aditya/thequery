export const benchmarkColumns = ["id", "model_slug", "category", "benchmark_name", "benchmark_version", "score_numeric", "score_display", "score_unit", "tools", "reasoning_effort", "harness", "evaluator", "evaluation_date", "source", "notes"] as const;

function csvCell(value: unknown): string {
  let text = value == null ? "" : String(value);
  // Keep spreadsheet applications from interpreting source text as formulas.
  if (typeof value === "string" && /^[\s]*[=+@-]/.test(text)) text = `'${text}`;
  return `"${text.replace(/"/g, '""')}"`;
}

export function benchmarkCsv(rows: Record<string, unknown>[]): string {
  return [benchmarkColumns.join(","), ...rows.map(row => benchmarkColumns.map(key => csvCell(row[key])).join(","))].join("\r\n") + "\r\n";
}
