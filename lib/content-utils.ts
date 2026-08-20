import type { ChartBlock, ComparisonTableBlock, ContentBlock, Source } from "./content-types";

const markdownLinkPattern = /^\s*[-*]\s+\[([^\]]+)]\(([^)\s]+)(?:\s+[^)]*)?\)(?:\s*[-—–:]\s*.*)?\s*$/;

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function asText(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function uniqueText(values: string[]): string[] {
  return [...new Set(values.filter(Boolean))];
}

function isSourceHeading(line: string): boolean {
  const normalized = line
    .trim()
    .replace(/^#{1,6}\s+/, "")
    .replace(/\*/g, "")
    .replace(/:$/, "")
    .trim()
    .toLowerCase();
  return normalized === "sources" || normalized === "sources and further reading" || normalized === "references & resources";
}

export function slugify(value: string): string {
  return value
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "");
}

export function normalizeSources(value: unknown): Source[] {
  if (!Array.isArray(value)) return [];

  return value.flatMap((entry) => {
    const source = asRecord(entry);
    const title = asText(source?.title);
    const url = asText(source?.url);

    if (!title || !/^https?:\/\//i.test(url)) return [];
    return [{ title, url }];
  });
}

export function normalizeBlocks(value: unknown): ContentBlock[] {
  if (!Array.isArray(value)) return [];

  return value.flatMap<ContentBlock>((entry, index): ContentBlock[] => {
    const block = asRecord(entry);
    const id = asText(block?.id) || `block-${index + 1}`;

    if (block?.type === "markdown") {
      const content = typeof block.content === "string" ? block.content : "";
      return content.trim() ? [{ id, type: "markdown" as const, content }] : [];
    }

    if (block?.type === "comparison_table") {
      const columns = Array.isArray(block.columns)
        ? uniqueText(block.columns.map(asText))
        : [];
      const rows = Array.isArray(block.rows)
        ? block.rows.flatMap((row) => {
            if (!Array.isArray(row)) return [];
            const cells = row.map((cell) => (cell == null ? "" : String(cell).trim()));
            return cells.some(Boolean) ? [cells] : [];
          })
        : [];

      if (columns.length === 0) return [];
      const comparison: ComparisonTableBlock = {
        id,
        type: "comparison_table",
        columns,
        rows,
      };
      const title = asText(block.title);
      const caption = asText(block.caption);
      const sourceNote = asText(block.sourceNote);
      if (title) comparison.title = title;
      if (caption) comparison.caption = caption;
      if (sourceNote) comparison.sourceNote = sourceNote;
      return [comparison];
    }

    if (block?.type === "chart") {
      const data = Array.isArray(block.data)
        ? block.data.flatMap((row) => {
            const record = asRecord(row);
            if (!record) return [];
            const normalized = Object.fromEntries(
              Object.entries(record).flatMap(([key, cell]) => {
                if (typeof cell === "string" || typeof cell === "number" || cell === null) {
                  return [[key, cell]];
                }
                return [];
              }),
            );
            return Object.keys(normalized).length ? [normalized] : [];
          })
        : [];
      const title = asText(block.title);

      if (!title || data.length === 0) return [];
      const chart: ChartBlock = {
        id,
        type: "chart",
        title,
        data,
      };
      if (block.chartType === "line" || block.chartType === "bar") chart.chartType = block.chartType;
      const description = asText(block.description);
      const sourceNote = asText(block.sourceNote);
      if (description) chart.description = description;
      if (sourceNote) chart.sourceNote = sourceNote;
      return [chart];
    }

    return [];
  });
}

export function markdownBlock(content = ""): ContentBlock {
  return {
    id: "markdown-1",
    type: "markdown",
    content,
  };
}

export function extractSources(markdown: string): { content: string; sources: Source[] } {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const headingIndex = lines.findIndex((line) => isSourceHeading(line));

  if (headingIndex < 0) return { content: markdown.trim(), sources: [] };

  let endIndex = headingIndex + 1;
  while (endIndex < lines.length) {
    const line = lines[endIndex];
    const trimmed = line.trim();
    if (/^#{1,6}\s+/.test(trimmed) || /^\*\*[^*]+\*\*:?\s*$/.test(trimmed)) break;
    endIndex += 1;
  }

  const sources = lines
    .slice(headingIndex + 1, endIndex)
    .flatMap((line) => {
      const match = line.match(markdownLinkPattern);
      return match ? [{ title: match[1].trim(), url: match[2].trim() }] : [];
    });

  if (sources.length === 0) return { content: markdown.trim(), sources: [] };

  return {
    content: [...lines.slice(0, headingIndex), ...lines.slice(endIndex)]
      .join("\n")
      .replace(/\n{3,}/g, "\n\n")
      .trim(),
    sources,
  };
}
