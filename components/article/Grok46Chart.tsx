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

const COLORS = {
  background: "#12161f",
  panel: "#1b2230",
  border: "#2a3241",
  text: "#e9eaec",
  muted: "#8b93a3",
  grok: "#4fb0a5",
  other: "#5b6472",
  gap: "#d14f4f",
};

const indexData = [
  { model: "Fable 5 Max", score: 62 },
  { model: "Grok 4.6", score: 61 },
  { model: "GPT-5.6 Sol Max", score: 61 },
  { model: "Grok 4.5", score: 56 },
];

const terminalData = [
  { model: "GPT-5.6 Sol Max", score: 34.6 },
  { model: "Fable 5 Max", score: 34.1 },
  { model: "Grok 4.6", score: 26 },
  { model: "Grok 4.5", score: 15.7 },
];

type ChartRow = (typeof indexData)[number] | (typeof terminalData)[number];

interface TooltipEntry {
  color?: string;
  payload?: ChartRow;
}

interface ChartTooltipProps {
  active?: boolean;
  payload?: readonly TooltipEntry[];
  label?: React.ReactNode;
}

function ChartTooltip({ active, payload, label }: ChartTooltipProps) {
  const row = payload?.[0]?.payload;
  if (!active || !row) return null;

  return (
    <div
      className="rounded-md px-3 py-2 font-sans text-xs shadow-xl"
      style={{
        backgroundColor: COLORS.panel,
        border: `1px solid ${COLORS.border}`,
        color: COLORS.text,
      }}
    >
      <div className="font-semibold">{label}</div>
      <div style={{ color: payload?.[0]?.color || COLORS.text }}>{row.score}</div>
    </div>
  );
}

function BenchmarkPanel({
  title,
  subtitle,
  data,
  domain,
  unit,
  highlightColor,
}: {
  title: string;
  subtitle: string;
  data: readonly ChartRow[];
  domain: [number, number];
  unit?: string;
  highlightColor: string;
}) {
  return (
    <div className="rounded-lg p-4" style={{ backgroundColor: COLORS.panel, border: `1px solid ${COLORS.border}` }}>
      <h2 className="text-sm font-semibold" style={{ color: COLORS.text }}>{title}</h2>
      <p className="mb-3 mt-1 text-xs" style={{ color: COLORS.muted }}>{subtitle}</p>
      <div className="h-[220px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ left: 4, right: 28, top: 4, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} horizontal={false} />
            <XAxis type="number" domain={domain} stroke={COLORS.muted} tick={{ fontSize: 11, fill: COLORS.muted }} unit={unit} />
            <YAxis type="category" dataKey="model" stroke={COLORS.muted} width={112} tick={{ fontSize: 11, fill: COLORS.muted }} />
            <Tooltip content={<ChartTooltip />} cursor={{ fill: "rgba(255,255,255,0.03)" }} />
            <Bar dataKey="score" radius={[0, 4, 4, 0]} barSize={24}>
              {data.map((entry) => (
                <Cell key={entry.model} fill={entry.model === "Grok 4.6" ? highlightColor : COLORS.other} />
              ))}
              <LabelList
                dataKey="score"
                position="right"
                formatter={(value) => (unit ? `${value}%` : value)}
                style={{ fill: COLORS.text, fontSize: 11 }}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default function Grok46Chart() {
  return (
    <figure
      className="my-10 w-full rounded-xl p-5 sm:p-7 xl:my-0"
      style={{ backgroundColor: COLORS.background, border: `1px solid ${COLORS.border}` }}
      aria-labelledby="grok-46-chart-title"
    >
      <div className="font-sans text-[11px] uppercase tracking-[0.18em]" style={{ color: COLORS.muted }}>
        TheQuery data
      </div>
      <div id="grok-46-chart-title" className="mt-2 font-serif text-2xl font-semibold" style={{ color: COLORS.text }}>
        Grok 4.6: Tied on Average, Not Tied Everywhere
      </div>
      <p className="mb-6 mt-3 text-sm leading-relaxed" style={{ color: COLORS.muted }}>
        The composite score and the hardest individual benchmark tell different stories.
      </p>

      <div className="grid gap-5">
        <BenchmarkPanel
          title="AA Intelligence Index (composite)"
          subtitle="Nine benchmarks blended into one score"
          data={indexData}
          domain={[0, 65]}
          highlightColor={COLORS.grok}
        />
        <BenchmarkPanel
          title="Terminal-Bench v3.0"
          subtitle="The row the launch post's prose skips"
          data={terminalData}
          domain={[0, 40]}
          unit="%"
          highlightColor={COLORS.gap}
        />
      </div>

      <div className="mt-5 rounded-lg p-4" style={{ backgroundColor: COLORS.panel, border: `1px solid ${COLORS.border}` }}>
        <h2 className="text-sm font-semibold" style={{ color: COLORS.text }}>Efficiency on AA-Briefcase, at comparable quality</h2>
        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          <div>
            <div className="font-serif text-2xl" style={{ color: COLORS.grok }}>~53 turns / ~0.5B tokens</div>
            <p className="text-xs" style={{ color: COLORS.muted }}>Grok 4.6</p>
          </div>
          <div>
            <div className="font-serif text-2xl" style={{ color: COLORS.other }}>~103 turns / ~2.0B tokens</div>
            <p className="text-xs" style={{ color: COLORS.muted }}>Claude Opus 5 (max)</p>
          </div>
        </div>
      </div>

      <figcaption className="mt-4 text-xs leading-relaxed" style={{ color: COLORS.muted }}>
        Source: Artificial Analysis (AA-Briefcase, AA Intelligence Index) and SpaceXAI&apos;s Grok 4.6 launch table (Terminal-Bench v3.0).
      </figcaption>
    </figure>
  );
}
