"use client";

import React from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getHorizontalBarChartHeight } from "./chartSizing";

const COLORS = {
  background: "#12161f",
  panel: "#1b2230",
  border: "#2a3241",
  text: "#e9eaec",
  muted: "#8b93a3",
  qwen: "#4fb0a5",
  opus: "#5b6472",
  flagged: "#d98c4a",
};

const textData = [
  { benchmark: "SWE-bench Pro", qwen: 61.7, opus: 53.4 },
  { benchmark: "QwenSWEBench", qwen: 79, opus: 63.8 },
  { benchmark: "GPQA Diamond", qwen: 89.2, opus: 91.3 },
  { benchmark: "HLE", qwen: 30.8, opus: 40 },
  { benchmark: "LiveCodeBench v6", qwen: 90.3, opus: 88.8 },
];

const visionData = [
  { benchmark: "OSWorld-Verified", qwen: 84.3, opus: 72.7, flagged: false },
  { benchmark: "AndroidWorld", qwen: 81.9, opus: 62, flagged: false },
  { benchmark: "ERQA", qwen: 65.5, opus: 40.8, flagged: false },
  { benchmark: "MathVision", qwen: 94.6, opus: 65.5, flagged: true },
  { benchmark: "BabyVision", qwen: 85.6, opus: 12.6, flagged: true },
];

type BenchmarkRow = (typeof textData)[number] | (typeof visionData)[number];

interface TooltipEntry {
  color?: string;
  name?: string;
  value?: number | string;
}

interface ChartTooltipProps {
  active?: boolean;
  payload?: readonly TooltipEntry[];
  label?: React.ReactNode;
}

function ChartTooltip({ active, payload, label }: ChartTooltipProps) {
  if (!active || !payload?.length) return null;

  return (
    <div
      className="rounded-md px-3 py-2 font-sans text-xs shadow-xl"
      style={{
        backgroundColor: COLORS.panel,
        border: `1px solid ${COLORS.border}`,
        color: COLORS.text,
      }}
    >
      <div className="mb-1 font-semibold">{label}</div>
      {payload.map((entry, index) => (
        <div key={`${entry.name ?? "score"}-${index}`} style={{ color: entry.color }}>
          {entry.name}: {entry.value}
        </div>
      ))}
    </div>
  );
}

function BenchmarkPanel({
  title,
  subtitle,
  data,
  flagged = false,
}: {
  title: string;
  subtitle: string;
  data: readonly BenchmarkRow[];
  flagged?: boolean;
}) {
  return (
    <section
      className="rounded-lg p-4 sm:p-5"
      style={{ backgroundColor: COLORS.panel, border: `1px solid ${COLORS.border}` }}
      aria-label={title}
    >
      <div className="font-sans text-sm font-semibold" style={{ color: COLORS.text }}>
        {title}
      </div>
      <div className="mb-4 mt-1 font-sans text-xs leading-relaxed" style={{ color: COLORS.muted }}>
        {subtitle}
      </div>
      <div
        className="w-full"
        style={{ height: getHorizontalBarChartHeight(data.length) }}
        role="img"
        aria-label={`${title}: ${data.map((row) => `${row.benchmark}, Qwen ${row.qwen}, Opus ${row.opus}`).join("; ")}`}
      >
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            layout="vertical"
            margin={{ left: 4, right: 32, top: 4, bottom: 4 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke={COLORS.border}
              horizontal={false}
            />
            <XAxis
              type="number"
              domain={[0, 100]}
              stroke={COLORS.muted}
              tick={{ fontSize: 11, fill: COLORS.muted }}
            />
            <YAxis
              type="category"
              dataKey="benchmark"
              stroke={COLORS.muted}
              width={118}
              tick={{ fontSize: 10, fill: COLORS.muted }}
            />
            <Tooltip
              content={<ChartTooltip />}
              cursor={{ fill: "rgba(255,255,255,0.03)" }}
            />
            <Bar
              dataKey="qwen"
              name="Qwen3.8-27B"
              fill={COLORS.qwen}
              radius={[0, 4, 4, 0]}
              barSize={14}
            >
              {flagged
                ? data.map((row) => (
                    <Cell
                      key={row.benchmark}
                      fill={"flagged" in row && row.flagged ? COLORS.flagged : COLORS.qwen}
                    />
                  ))
                : null}
              <LabelList
                dataKey="qwen"
                position="right"
                style={{ fill: COLORS.text, fontSize: 10 }}
              />
            </Bar>
            <Bar
              dataKey="opus"
              name="Opus 4.6 Max"
              fill={COLORS.opus}
              radius={[0, 4, 4, 0]}
              barSize={14}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

export default function Qwen27BChart() {
  return (
    <figure
      className="my-10 w-full rounded-xl p-5 sm:p-7 xl:my-0"
      style={{ backgroundColor: COLORS.background, border: `1px solid ${COLORS.border}` }}
      aria-labelledby="qwen-27b-chart-title"
    >
      <div className="font-sans text-[11px] uppercase tracking-[0.18em]" style={{ color: COLORS.muted }}>
        TheQuery data
      </div>
      <div
        id="qwen-27b-chart-title"
        className="mt-2 font-serif text-2xl font-semibold sm:text-3xl"
        style={{ color: COLORS.text }}
      >
        Qwen3.8-27B vs. Opus 4.6 Max
      </div>
      <p className="mb-6 mt-3 text-sm leading-relaxed" style={{ color: COLORS.muted }}>
        A close fight on text. A sweep on vision, with two bars carrying an asterisk.
      </p>

      <div className="grid gap-5">
        <BenchmarkPanel
          title="Text benchmarks"
          subtitle="Same testing conditions for both models"
          data={textData}
        />
        <BenchmarkPanel
          title="Vision benchmarks"
          subtitle="Orange bars: Qwen's score used code interpreter access, Opus's did not"
          data={visionData}
          flagged
        />
      </div>

      <figcaption className="mt-4 font-sans text-[11px] leading-relaxed" style={{ color: COLORS.muted }}>
        Source: Qwen3.8-27B model card, Hugging Face, August 2026. Orange bars mark scores reported with code interpreter access.
      </figcaption>

      <div className="sr-only">
        <table>
          <caption>Qwen3.8-27B and Opus 4.6 Max benchmark comparison</caption>
          <thead>
            <tr>
              <th scope="col">Benchmark</th>
              <th scope="col">Qwen3.8-27B</th>
              <th scope="col">Opus 4.6 Max</th>
            </tr>
          </thead>
          <tbody>
            {[...textData, ...visionData].map((row) => (
              <tr key={row.benchmark}>
                <th scope="row">{row.benchmark}</th>
                <td>{row.qwen}</td>
                <td>{row.opus}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </figure>
  );
}
