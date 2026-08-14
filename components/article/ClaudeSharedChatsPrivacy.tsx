"use client";

import React from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
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
  accent: "#4fb0a5",
};

const youthData = [
  { year: "A year earlier", pct: 13.1 },
  { year: "This year", pct: 19.2 },
];

const stats = [
  {
    value: "32%",
    label: "US adults who used an AI chatbot for health information or advice in the past year",
    source: "KFF",
  },
  {
    value: "13%",
    label: "of all US adults who uploaded personal medical information into an AI tool",
    source: "KFF",
  },
  {
    value: "77%",
    label: "of adults concerned about the privacy of medical information shared with AI",
    source: "KFF",
  },
  {
    value: "66M+",
    label: "Americans, about 1 in 4 adults, who have used AI for health advice",
    source: "West Health / Gallup",
  },
];

interface TooltipEntry {
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
      <div style={{ color: COLORS.muted }}>{label}</div>
      <div style={{ color: COLORS.accent }}>{payload[0].value}% of ages 12–21</div>
    </div>
  );
}

export default function ClaudeSharedChatsPrivacy() {
  return (
    <figure
      className="my-10 w-full rounded-xl p-5 sm:p-7 xl:my-0"
      style={{
        backgroundColor: COLORS.background,
        border: `1px solid ${COLORS.border}`,
      }}
      aria-labelledby="claude-shared-chats-privacy-title"
    >
      <div
        className="font-sans text-[11px] uppercase tracking-[0.18em]"
        style={{ color: COLORS.muted }}
      >
        TheQuery data
      </div>
      <div
        id="claude-shared-chats-privacy-title"
        className="mt-2 font-serif text-2xl font-semibold sm:text-3xl"
        style={{ color: COLORS.text }}
      >
        How Ordinary This Has Become
      </div>
      <div
        className="mb-7 mt-3 max-w-2xl font-sans text-sm leading-relaxed"
        style={{ color: COLORS.muted }}
      >
        What people already bring to AI chat, before any link becomes searchable.
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        {stats.map((stat) => (
          <section
            key={stat.label}
            className="rounded-lg p-4 sm:p-5"
            style={{
              backgroundColor: COLORS.panel,
              border: `1px solid ${COLORS.border}`,
            }}
          >
            <div
              className="font-serif text-3xl font-semibold"
              style={{ color: COLORS.accent }}
            >
              {stat.value}
            </div>
            <div
              className="mt-2 font-sans text-sm leading-relaxed"
              style={{ color: COLORS.text }}
            >
              {stat.label}
            </div>
            <div
              className="mt-3 font-sans text-[11px]"
              style={{ color: COLORS.muted }}
            >
              Source: {stat.source}
            </div>
          </section>
        ))}
      </div>

      <section
        className="mt-4 rounded-lg p-4 sm:p-5"
        style={{
          backgroundColor: COLORS.panel,
          border: `1px solid ${COLORS.border}`,
        }}
      >
        <div className="font-sans text-sm font-semibold" style={{ color: COLORS.text }}>
          Ages 12–21 using AI chatbots for mental health advice
        </div>
        <div className="mb-4 mt-1 font-sans text-xs" style={{ color: COLORS.muted }}>
          Nearly two out of three did not tell anyone.
        </div>
        <div
          className="w-full"
          style={{ height: getHorizontalBarChartHeight(youthData.length) }}
          role="img"
          aria-label="The share of people ages 12 to 21 using AI chatbots for mental health advice rose from 13.1 percent a year earlier to 19.2 percent this year."
        >
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={youthData}
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
                domain={[0, 25]}
                stroke={COLORS.muted}
                tick={{ fontSize: 11, fill: COLORS.muted }}
                unit="%"
              />
              <YAxis
                type="category"
                dataKey="year"
                stroke={COLORS.muted}
                width={108}
                tick={{ fontSize: 11, fill: COLORS.muted }}
              />
              <Tooltip
                content={<ChartTooltip />}
                cursor={{ fill: "rgba(255,255,255,0.03)" }}
              />
              <Bar
                dataKey="pct"
                fill={COLORS.accent}
                radius={[0, 4, 4, 0]}
                barSize={30}
              >
                <LabelList
                  dataKey="pct"
                  position="right"
                  formatter={(value) => `${value}%`}
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
          Source: RAND / JAMA Pediatrics, 2026. The 19.2% result is compared
          with a similar RAND survey a year earlier.
        </div>
      </section>

      <div className="sr-only">
        <table>
          <caption>How ordinary it has become to share sensitive information with AI chatbots</caption>
          <tbody>
            {stats.map((stat) => (
              <tr key={stat.label}>
                <th scope="row">{stat.label}</th>
                <td>{stat.value}</td>
                <td>{stat.source}</td>
              </tr>
            ))}
            {youthData.map((entry) => (
              <tr key={entry.year}>
                <th scope="row">Ages 12–21 using AI chatbots for mental health advice: {entry.year}</th>
                <td>{entry.pct}%</td>
                <td>RAND / JAMA Pediatrics</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </figure>
  );
}
