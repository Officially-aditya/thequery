"use client";

import React from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
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
  genuine: "#4fb0a5",
  headline: "#d98c4a",
};

const membershipData = [
  { period: "Apr 2 (founding)", members: 22 },
  { period: "Jul 14 (operational)", members: 40 },
];

const volumeData = [
  { category: "x402 headline", value: 24_200_000, color: COLORS.headline },
  { category: "DefiLlama metric", value: 572_000, color: COLORS.genuine },
];

interface TooltipEntry {
  color?: string;
  name?: string;
  value?: number | string;
}

interface ChartTooltipProps {
  active?: boolean;
  payload?: readonly TooltipEntry[];
  label?: React.ReactNode;
  currency?: boolean;
}

function ChartTooltip({
  active,
  payload,
  label,
  currency = false,
}: ChartTooltipProps) {
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
      {label ? (
        <div className="mb-1" style={{ color: COLORS.muted }}>
          {label}
        </div>
      ) : null}
      {payload.map((entry, index) => {
        const numericValue = Number(entry.value ?? 0);
        const value = currency
          ? `USD ${numericValue.toLocaleString("en-US")}`
          : numericValue.toLocaleString("en-US");

        return (
          <div key={`${entry.name ?? "value"}-${index}`} style={{ color: entry.color || COLORS.text }}>
            {entry.name}: {value}
          </div>
        );
      })}
    </div>
  );
}

function Panel({
  title,
  subtitle,
  children,
  source,
}: {
  title: string;
  subtitle: string;
  children: React.ReactNode;
  source: string;
}) {
  return (
    <section
      className="min-w-0 rounded-lg p-4 sm:p-5"
      style={{
        backgroundColor: COLORS.panel,
        border: `1px solid ${COLORS.border}`,
      }}
    >
      <div className="font-sans text-sm font-semibold" style={{ color: COLORS.text }}>
        {title}
      </div>
      <div className="mb-4 mt-1 font-sans text-xs" style={{ color: COLORS.muted }}>
        {subtitle}
      </div>
      {children}
      <div className="mt-3 font-sans text-[11px] leading-relaxed" style={{ color: COLORS.muted }}>
        Source: {source}
      </div>
    </section>
  );
}

export default function X402RealityCheck() {
  return (
    <figure
      className="relative left-1/2 my-10 w-[min(960px,calc(100vw-2rem))] -translate-x-1/2 rounded-xl p-5 sm:p-7"
      style={{
        backgroundColor: COLORS.background,
        border: `1px solid ${COLORS.border}`,
      }}
      aria-labelledby="x402-reality-check-title"
    >
      <div className="font-sans text-[11px] uppercase tracking-[0.18em]" style={{ color: COLORS.muted }}>
        TheQuery data
      </div>
      <div
        id="x402-reality-check-title"
        className="mt-2 font-serif text-2xl font-semibold sm:text-3xl"
        style={{ color: COLORS.text }}
      >
        x402: Infrastructure vs. Reality
      </div>
      <div className="mb-7 mt-2 max-w-2xl font-sans text-sm leading-relaxed" style={{ color: COLORS.muted }}>
        Foundation membership rose from 22 to 40 in 103 days. Reported payment
        volume changes dramatically depending on what the tracker counts.
      </div>

      <div className="grid gap-5 md:grid-cols-2">
        <Panel
          title="x402 Foundation members"
          subtitle="Founding cohort to operational launch"
          source="CoinDesk; x402 Foundation"
        >
          <div className="h-[220px]" role="img" aria-label="x402 Foundation membership rose from 22 in April 2026 to 40 in July 2026">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={membershipData}
                layout="vertical"
                margin={{ left: 4, right: 20, top: 4, bottom: 4 }}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke={COLORS.border}
                  horizontal={false}
                />
                <XAxis
                  type="number"
                  domain={[0, 40]}
                  allowDecimals={false}
                  stroke={COLORS.muted}
                  tick={{ fontSize: 11, fill: COLORS.muted }}
                />
                <YAxis
                  type="category"
                  dataKey="period"
                  stroke={COLORS.muted}
                  width={128}
                  tick={{ fontSize: 11, fill: COLORS.muted }}
                />
                <Tooltip
                  content={<ChartTooltip />}
                  cursor={{ fill: "rgba(255,255,255,0.03)" }}
                />
                <Bar
                  dataKey="members"
                  name="Members"
                  fill={COLORS.genuine}
                  radius={[0, 4, 4, 0]}
                  barSize={34}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel
          title="30-day volume: two methodologies"
          subtitle="The smaller figure is narrower, not wash-adjusted"
          source="x402 dashboard; CoinDesk citing DefiLlama"
        >
          <div className="h-[220px]" role="img" aria-label="x402 reported 24.2 million US dollars in 30-day volume while DefiLlama tracked a narrower 572 thousand US dollar metric">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={volumeData}
                layout="vertical"
                margin={{ left: 4, right: 20, top: 4, bottom: 4 }}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke={COLORS.border}
                  horizontal={false}
                />
                <XAxis
                  type="number"
                  domain={[0, 24_200_000]}
                  stroke={COLORS.muted}
                  tick={{ fontSize: 11, fill: COLORS.muted }}
                  tickFormatter={(value) => `USD ${Math.round(Number(value) / 1_000_000)}M`}
                />
                <YAxis
                  type="category"
                  dataKey="category"
                  stroke={COLORS.muted}
                  width={112}
                  tick={{ fontSize: 11, fill: COLORS.muted }}
                />
                <Tooltip
                  content={<ChartTooltip currency />}
                  cursor={{ fill: "rgba(255,255,255,0.03)" }}
                />
                <Bar
                  dataKey="value"
                  name="30-day volume"
                  radius={[0, 4, 4, 0]}
                  barSize={34}
                >
                  {volumeData.map((entry) => (
                    <Cell key={entry.category} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      </div>

      <div
        className="mt-5 rounded-lg p-4 font-sans text-sm leading-relaxed sm:p-5"
        style={{
          backgroundColor: COLORS.panel,
          border: `1px solid ${COLORS.border}`,
          color: COLORS.text,
        }}
      >
        As of July 19, 2026, neither Shopify nor Adyen had publicly announced a
        consumer-facing x402 checkout product.
        <div className="mt-2 text-[11px]" style={{ color: COLORS.muted }}>
          Source: Major Matters x402 Adoption Tracker
        </div>
      </div>

      <table className="sr-only">
        <caption>x402 Foundation membership and reported payment volume</caption>
        <thead>
          <tr>
            <th scope="col">Measure</th>
            <th scope="col">Period or methodology</th>
            <th scope="col">Value</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Foundation members</td>
            <td>April 2, 2026 founding cohort</td>
            <td>22</td>
          </tr>
          <tr>
            <td>Foundation members</td>
            <td>July 14, 2026 operational launch</td>
            <td>40</td>
          </tr>
          <tr>
            <td>30-day volume</td>
            <td>x402 headline</td>
            <td>USD 24.2 million</td>
          </tr>
          <tr>
            <td>30-day volume</td>
            <td>DefiLlama narrower metric</td>
            <td>USD 572,000</td>
          </tr>
        </tbody>
      </table>
    </figure>
  );
}
