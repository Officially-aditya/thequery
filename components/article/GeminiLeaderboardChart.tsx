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
  neutral: "#4fb0a5",
  flagged: "#d14f4f",
};

const data = [
  { model: "Claude Fable 5", lab: "Anthropic", score: 60 },
  { model: "GPT-5.6 Sol", lab: "OpenAI", score: 59 },
  { model: "Kimi K3", lab: "Moonshot AI", score: 57 },
  { model: "Claude Opus 4.8", lab: "Anthropic", score: 56 },
  { model: "Grok 4.5", lab: "SpaceXAI", score: 54 },
  { model: "GLM-5.2", lab: "Z AI", score: 51 },
  { model: "Muse Spark 1.1", lab: "Meta", score: 51 },
  { model: "Gemini 3.6 Flash", lab: "Google", score: 50 },
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
      <div style={{ color: COLORS.muted }}>{row.lab}</div>
      <div style={{ color: payload?.[0]?.color || COLORS.text }}>
        Index score: {row.score}
      </div>
    </div>
  );
}

export default function GeminiLeaderboardChart() {
  return (
    <figure
      className="my-10 w-full rounded-xl p-5 sm:p-7 xl:my-0"
      style={{
        backgroundColor: COLORS.background,
        border: `1px solid ${COLORS.border}`,
      }}
      aria-labelledby="gemini-leaderboard-title"
    >
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <div
            className="font-sans text-[11px] uppercase tracking-[0.18em]"
            style={{ color: COLORS.muted }}
          >
            TheQuery data
          </div>
          <div
            id="gemini-leaderboard-title"
            className="mt-2 font-serif text-2xl font-semibold sm:text-3xl"
            style={{ color: COLORS.text }}
          >
            Artificial Analysis Intelligence Index
          </div>
        </div>
        <div
          className="w-fit rounded-full px-3 py-1.5 font-sans text-xs font-semibold"
          style={{
            backgroundColor: "rgba(209,79,79,0.14)",
            border: "1px solid rgba(209,79,79,0.45)",
            color: "#f1a3a3",
          }}
        >
          Gemini 3.6 Flash: 50
        </div>
      </div>

      <div
        className="mb-7 mt-3 max-w-2xl font-sans text-sm leading-relaxed"
        style={{ color: COLORS.muted }}
      >
        Six labs field at least one model above 50. Google&apos;s newest release
        stops at 50 and sits outside the top ten.
      </div>

      <div
        className="rounded-lg p-3 sm:p-5"
        style={{
          backgroundColor: COLORS.panel,
          border: `1px solid ${COLORS.border}`,
        }}
      >
        <div
          className="h-[390px] w-full"
          role="img"
          aria-label="Selected Artificial Analysis Intelligence Index scores. Claude Fable 5 scores 60, GPT-5.6 Sol 59, Kimi K3 57, Claude Opus 4.8 56, Grok 4.5 54, GLM-5.2 and Muse Spark 1.1 each 51, and Gemini 3.6 Flash 50."
        >
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={data}
              layout="vertical"
              margin={{ left: 4, right: 36, top: 4, bottom: 4 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke={COLORS.border}
                horizontal={false}
              />
              <XAxis
                type="number"
                domain={[0, 65]}
                stroke={COLORS.muted}
                tick={{ fontSize: 11, fill: COLORS.muted }}
              />
              <YAxis
                type="category"
                dataKey="model"
                stroke={COLORS.muted}
                width={122}
                tick={{ fontSize: 11, fill: COLORS.muted }}
              />
              <Tooltip
                content={<ChartTooltip />}
                cursor={{ fill: "rgba(255,255,255,0.03)" }}
              />
              <Bar dataKey="score" radius={[0, 4, 4, 0]} barSize={25}>
                {data.map((entry) => (
                  <Cell
                    key={entry.model}
                    fill={entry.lab === "Google" ? COLORS.flagged : COLORS.neutral}
                  />
                ))}
                <LabelList
                  dataKey="score"
                  position="right"
                  style={{ fill: COLORS.text, fontSize: 11 }}
                />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div
          className="mt-3 font-sans text-[11px] leading-relaxed"
          style={{ color: COLORS.muted }}
        >
          Source: Artificial Analysis Intelligence Index, July 21, 2026.
          Selected models shown; higher is better.
        </div>
      </div>

      <table className="sr-only">
        <caption>Selected Artificial Analysis Intelligence Index scores</caption>
        <thead>
          <tr>
            <th scope="col">Model</th>
            <th scope="col">Lab</th>
            <th scope="col">Score</th>
          </tr>
        </thead>
        <tbody>
          {data.map((entry) => (
            <tr key={entry.model}>
              <td>{entry.model}</td>
              <td>{entry.lab}</td>
              <td>{entry.score}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  );
}
