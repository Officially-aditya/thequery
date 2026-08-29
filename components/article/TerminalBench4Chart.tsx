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
  leader: "#d98c4a",
  other: "#5b6472",
};

const data = [
  { model: "Opus 5 + Claude Code", agent: "Claude Code", score: 51.8 },
  { model: "Fable 5 + Claude Code", agent: "Claude Code", score: 44.5 },
  { model: "GLM-5.3 + Claude Code", agent: "Claude Code", score: 41.8 },
  { model: "GPT-5.6 Sol + Codex", agent: "Codex", score: 37.3 },
  { model: "Opus 4.8 + Claude Code", agent: "Claude Code", score: 23.6 },
  { model: "GPT-5.6 Terra + Codex", agent: "Codex", score: 21.5 },
  { model: "Grok 4.6 + Grok Build", agent: "Grok Build", score: 20.3 },
  { model: "GPT-5.6 Luna + Codex", agent: "Codex", score: 17.3 },
  { model: "Grok 4.5 + Grok Build", agent: "Grok Build", score: 12.4 },
  { model: "Sonnet 5 + Claude Code", agent: "Claude Code", score: 12.4 },
];

interface TooltipEntry {
  color?: string;
  payload?: (typeof data)[number];
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
      <div style={{ color: COLORS.muted }}>{row.agent}</div>
      <div style={{ color: payload?.[0]?.color || COLORS.text }}>
        Resolution rate: {row.score}%
      </div>
    </div>
  );
}

export default function TerminalBench4Chart() {
  const accessibleLabel = data
    .map((entry) => `${entry.model} ${entry.score}%`)
    .join(", ");

  return (
    <figure
      className="my-10 w-full rounded-xl p-5 sm:p-7 xl:my-0"
      style={{ backgroundColor: COLORS.background, border: `1px solid ${COLORS.border}` }}
      aria-labelledby="terminal-bench-4-chart-title"
    >
      <div className="font-sans text-[11px] uppercase tracking-[0.18em]" style={{ color: COLORS.muted }}>
        TheQuery data
      </div>
      <div
        id="terminal-bench-4-chart-title"
        className="mt-2 font-serif text-2xl font-semibold sm:text-3xl"
        style={{ color: COLORS.text }}
      >
        Terminal-Bench 4.0 Resolution Rates
      </div>
      <p className="mb-6 mt-3 max-w-2xl text-sm leading-relaxed" style={{ color: COLORS.muted }}>
        The score belongs to each model and agent pairing, not to the model alone.
      </p>

      <div
        className="w-full"
        style={{ height: getHorizontalBarChartHeight(data.length) }}
        role="img"
        aria-label={`Terminal-Bench 4.0 resolution rates: ${accessibleLabel}.`}
      >
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ left: 4, right: 34, top: 4, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} horizontal={false} />
            <XAxis
              type="number"
              domain={[0, 60]}
              stroke={COLORS.muted}
              tick={{ fontSize: 11, fill: COLORS.muted }}
              unit="%"
            />
            <YAxis
              type="category"
              dataKey="model"
              stroke={COLORS.muted}
              width={148}
              tick={{ fontSize: 10, fill: COLORS.muted }}
            />
            <Tooltip content={<ChartTooltip />} cursor={{ fill: "rgba(255,255,255,0.03)" }} />
            <Bar dataKey="score" radius={[0, 4, 4, 0]} barSize={24}>
              {data.map((entry, index) => (
                <Cell key={entry.model} fill={index === 0 ? COLORS.leader : COLORS.other} />
              ))}
              <LabelList
                dataKey="score"
                position="right"
                formatter={(value) => `${value}%`}
                style={{ fill: COLORS.text, fontSize: 11 }}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 font-sans text-[11px] leading-relaxed" style={{ color: COLORS.muted }}>
        Source: Terminal-Bench 4.0 official leaderboard. Whiskers in the source table represent 95% confidence intervals.
      </div>

      <div className="sr-only">
        <table>
          <caption>Terminal-Bench 4.0 resolution rates</caption>
          <thead>
            <tr>
              <th scope="col">Model and agent</th>
              <th scope="col">Resolution rate</th>
            </tr>
          </thead>
          <tbody>
            {data.map((entry) => (
              <tr key={entry.model}>
                <td>{entry.model}</td>
                <td>{entry.score}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </figure>
  );
}
