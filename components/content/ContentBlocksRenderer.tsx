"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import MarkdownRenderer, { type GlossaryLink } from "@/components/MarkdownRenderer";
import type { ChartBlock, ComparisonTableBlock, ContentBlock, Source } from "@/lib/content-types";

const chartColors = ["#2563eb", "#0d9488", "#d97706", "#7c3aed", "#dc2626"];

export function SourcesList({ sources }: { sources: Source[] }) {
  if (sources.length === 0) return null;

  return (
    <section className="mt-10 border-t border-border pt-6" aria-labelledby="sources-heading">
      <h2 id="sources-heading" className="font-serif text-sm font-semibold text-text-muted mb-3">
        Sources
      </h2>
      <ol className="space-y-2 list-decimal pl-5">
        {sources.map((source, index) => (
          <li key={`${source.url}-${index}`} className="pl-1">
            <a
              href={source.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-accent hover:text-accent-hover transition-colors"
            >
              {source.title}
            </a>
          </li>
        ))}
      </ol>
    </section>
  );
}

function ComparisonTable({ block }: { block: ComparisonTableBlock }) {
  return (
    <figure className="my-8 overflow-x-auto rounded-lg border border-border">
      {block.title ? <h2 className="px-4 pt-4 font-serif text-lg font-semibold text-text-primary">{block.title}</h2> : null}
      {block.caption ? <p className="px-4 pt-1 text-sm text-text-secondary">{block.caption}</p> : null}
      <table className="w-full min-w-[480px] border-collapse text-left text-sm">
        <thead className="bg-bg-secondary">
          <tr>
            {block.columns.map((column) => (
              <th key={column} scope="col" className="border-t border-border px-4 py-3 font-semibold text-text-primary">
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {block.rows.map((row, rowIndex) => (
            <tr key={`${block.id}-${rowIndex}`} className="border-t border-border">
              {block.columns.map((_, cellIndex) => (
                <td key={cellIndex} className="px-4 py-3 align-top text-text-secondary">
                  {row[cellIndex] ?? ""}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {block.sourceNote ? <figcaption className="px-4 py-3 text-xs text-text-muted">Source: {block.sourceNote}</figcaption> : null}
    </figure>
  );
}

function GenericChart({ block }: { block: ChartBlock }) {
  const firstRow = block.data[0] ?? {};
  const labelKey = Object.keys(firstRow).find((key) => key.toLowerCase() === "label") ?? Object.keys(firstRow)[0];
  const series = Object.keys(firstRow).filter(
    (key) => key !== labelKey && block.data.some((row) => typeof row[key] === "number"),
  );

  if (!labelKey || series.length === 0) return null;

  const Chart = block.chartType === "line" ? LineChart : BarChart;
  return (
    <figure className="my-8 rounded-xl border border-border bg-bg-secondary p-4 sm:p-6" aria-label={block.title}>
      <h2 className="font-serif text-xl font-semibold text-text-primary">{block.title}</h2>
      {block.description ? <p className="mt-2 text-sm leading-relaxed text-text-secondary">{block.description}</p> : null}
      <div className="mt-5 h-[300px] w-full" role="img" aria-label={block.title}>
        <ResponsiveContainer width="100%" height="100%">
          <Chart data={block.data} margin={{ top: 8, right: 16, left: -16, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey={labelKey} tick={{ fill: "var(--text-muted)", fontSize: 12 }} />
            <YAxis tick={{ fill: "var(--text-muted)", fontSize: 12 }} />
            <Tooltip />
            <Legend />
            {series.map((key, index) =>
              block.chartType === "line" ? (
                <Line key={key} type="monotone" dataKey={key} stroke={chartColors[index % chartColors.length]} strokeWidth={2} />
              ) : (
                <Bar key={key} dataKey={key} fill={chartColors[index % chartColors.length]} radius={[4, 4, 0, 0]} />
              ),
            )}
          </Chart>
        </ResponsiveContainer>
      </div>
      {block.sourceNote ? <figcaption className="mt-3 text-xs text-text-muted">Source: {block.sourceNote}</figcaption> : null}
    </figure>
  );
}

export default function ContentBlocksRenderer({
  blocks,
  sources = [],
  glossaryTerms = [],
  disableMath = false,
}: {
  blocks: ContentBlock[];
  sources?: Source[];
  glossaryTerms?: GlossaryLink[];
  disableMath?: boolean;
}) {
  return (
    <>
      {blocks.map((block) => {
        if (block.type === "markdown") {
          return <MarkdownRenderer key={block.id} content={block.content} glossaryTerms={glossaryTerms} disableMath={disableMath} />;
        }
        if (block.type === "comparison_table") return <ComparisonTable key={block.id} block={block} />;
        return <GenericChart key={block.id} block={block} />;
      })}
      <SourcesList sources={sources} />
    </>
  );
}
