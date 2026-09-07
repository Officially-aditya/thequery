import type { ContentBlock, Source, SpecTableBlock } from "./content-types";
import type { ModelBenchmarkCategory, ModelCatalogEntry } from "./models";

interface ComparisonSection {
  title: string;
  labels: string[];
}

const sections: ComparisonSection[] = [
  {
    title: "Specifications",
    labels: [
      "Developer",
      "Release date",
      "API model ID",
      "Context window",
      "Max output",
      "Knowledge cutoff",
      "Reasoning / effort",
    ],
  },
  {
    title: "Pricing",
    labels: [
      "Input / 1M tokens",
      "Cached input / 1M",
      "Cache write / 1M",
      "Output / 1M tokens",
      "Batch / flex discount",
      "Long-context surcharge",
    ],
  },
  {
    title: "Capabilities & access",
    labels: [
      "Text input",
      "Image / vision input",
      "Audio input",
      "Video input",
      "File / document input",
      "Text output",
      "Image output",
      "Audio output",
      "Video output",
      "Tool / function calling",
      "Computer use",
      "API access",
      "Product access",
      "Weights / license",
    ],
  },
  {
    title: "Model behavior",
    labels: [
      "Primary focus",
      "Long-horizon work",
      "Agent orchestration",
      "User collaboration",
      "Efficiency / generation change",
      "Safety / approvals",
    ],
  },
  {
    title: "Coding",
    labels: [
      "SWE-bench Verified",
      "SWE-bench Pro",
      "Multi-SWE-Bench",
      "FrontierCode 1.1 Main",
      "FrontierCode 1.1 Extended",
      "DeepSWE v1.1",
      "SWE-Atlas Codebase QnA",
      "NL2Repo",
      "VIBE-Pro",
      "Terminal-Bench 2.0",
      "Terminal-Bench 2.1",
      "Terminal-Bench 3.0",
      "Terminal-Bench 4.0",
      "Terminal-Bench Science 0.1",
      "Terminal-Bench",
      "MLE-Bench",
      "LiveCodeBench",
      "CursorBench",
    ],
  },
  {
    title: "Math & reasoning",
    labels: ["AIME", "HMMT", "ARC-AGI", "FrontierMath", "FrontierMath Tier 4 (v2)", "MRCR v2 256K–512K", "MRCR v2 512K–1M"],
  },
  {
    title: "Knowledge",
    labels: ["GPQA Diamond", "Humanity's Last Exam", "HLE-Verified", "MMLU-Pro"],
  },
  {
    title: "Agentic & computer use",
    labels: [
      "OSWorld",
      "OSWorld-Verified",
      "OSWorld 2.0",
      "BrowseComp",
      "DeepSearchQA",
      "GDPval-AA",
      "GDPval-AA v2",
      "AutomationBench",
      "Agentic IF Index",
      "Agents' Last Exam",
      "ApexBench",
      "Arena Search",
      "τ²-bench Telecom",
      "MCP Atlas",
      "Toolathlon",
      "MCP / tool-use benchmark",
    ],
  },
];

const benchmarkSection: Record<ModelBenchmarkCategory, string> = {
  coding: "Coding",
  math_reasoning: "Math & reasoning",
  knowledge: "Knowledge",
  agentic_computer_use: "Agentic & computer use",
  multimodal: "Multimodal",
  professional: "Professional",
  other: "Other benchmarks",
};

function sectionBlock(
  title: string,
  labels: string[],
  modelA: ModelCatalogEntry,
  modelB: ModelCatalogEntry,
  index: number,
): SpecTableBlock | null {
  const rows = labels
    .map((label) => [
      label,
      modelA.comparisonData[label] ?? "",
      modelB.comparisonData[label] ?? "",
    ])
    .filter((row) => row[1] || row[2]);

  if (rows.length === 0) return null;

  return {
    id: `database-comparison-${index + 1}`,
    type: "spec_table",
    title,
    columns: [modelA.name, modelB.name],
    rows,
  };
}

function comparisonSections(modelA: ModelCatalogEntry, modelB: ModelCatalogEntry): ComparisonSection[] {
  const dynamic = new Map<string, string[]>();
  const predefined = new Set(sections.flatMap((section) => section.labels));

  for (const benchmark of [...modelA.benchmarks, ...modelB.benchmarks]) {
    if (predefined.has(benchmark.name)) continue;
    const title = benchmarkSection[benchmark.category];
    const labels = dynamic.get(title) ?? [];
    if (!labels.includes(benchmark.name)) labels.push(benchmark.name);
    dynamic.set(title, labels);
  }

  const merged = sections.map((section) => ({
    ...section,
    labels: [...section.labels, ...(dynamic.get(section.title) ?? [])],
  }));
  const knownTitles = new Set(merged.map((section) => section.title));

  for (const title of ["Multimodal", "Professional", "Other benchmarks"]) {
    const labels = dynamic.get(title);
    if (labels?.length && !knownTitles.has(title)) merged.push({ title, labels });
  }

  return merged;
}

export function buildModelComparisonBlocks(modelA: ModelCatalogEntry, modelB: ModelCatalogEntry): ContentBlock[] {
  return comparisonSections(modelA, modelB).flatMap((section, index) => {
    const block = sectionBlock(section.title, section.labels, modelA, modelB, index);
    return block ? [block] : [];
  });
}

export function modelComparisonSources(modelA: ModelCatalogEntry, modelB: ModelCatalogEntry): Source[] {
  return Array.from(
    new Map([...modelA.sources, ...modelB.sources].map((source) => [source.url, source])).values(),
  );
}
