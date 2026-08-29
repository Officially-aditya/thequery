import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

test("Terminal-Bench 4 uses the dedicated model-plus-agent chart", async () => {
  const [page, chart] = await Promise.all([
    readFile(path.join(root, "app/articles/[slug]/page.tsx"), "utf8"),
    readFile(path.join(root, "components/article/TerminalBench4Chart.tsx"), "utf8"),
  ]);

  assert.match(page, /import TerminalBench4Chart/);
  assert.match(page, /terminal-bench-4-agent-not-model/);
  assert.match(page, /Here is the current Terminal-Bench 4\.0 leaderboard/);
  assert.match(chart, /Terminal-Bench 4\.0 Resolution Rates/);
  assert.match(chart, /role="img"/);
});
