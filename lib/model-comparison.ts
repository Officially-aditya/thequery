import type { ContentBlock, Source, SpecTableBlock } from "./content-types";
import type { ModelCatalogEntry } from "./models";

const sections: Array<{ title: string; labels: string[] }> = [
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
    title: "Coding",
    labels: [
      "SWE-bench Verified",
      "SWE-bench Pro",
      "FrontierCode 1.1 Main",
      "FrontierCode 1.1 Extended",
      "DeepSWE v1.1",
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
    labels: ["AIME", "HMMT", "ARC-AGI", "FrontierMath", "FrontierMath Tier 4 (v2)"],
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
      "GDPval-AA v2",
      "AutomationBench",
      "Agents' Last Exam",
      "MCP Atlas",
      "MCP / tool-use benchmark",
    ],
  },
];

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

export function buildModelComparisonBlocks(modelA: ModelCatalogEntry, modelB: ModelCatalogEntry): ContentBlock[] {
  const specBlocks = sections.flatMap((section, index) => {
    const block = sectionBlock(section.title, section.labels, modelA, modelB, index);
    return block ? [block] : [];
  });

  return [
    ...specBlocks,
    {
      id: "database-comparison-bottom-line",
      type: "markdown",
      content: "## Bottom line\n\nThis comparison is generated from TheQuery's verified model catalog. Empty or undisclosed fields are omitted, and benchmark conditions are preserved when the source reports them.",
    },
  ];
}

export function modelComparisonSources(modelA: ModelCatalogEntry, modelB: ModelCatalogEntry): Source[] {
  return Array.from(
    new Map([...modelA.sources, ...modelB.sources].map((source) => [source.url, source])).values(),
  );
}
