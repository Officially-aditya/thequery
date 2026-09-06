import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");
const source = (file: string) => readFile(path.join(root, file), "utf8");

test("normalized model_benchmarks override legacy comparison_data benchmark strings", async () => {
  const models = await source("lib/models.ts");
  assert.match(models, /if \(values\.length > 0\) next\[name\] = values\.join\(" · "\)/);
  assert.doesNotMatch(models, /if \(!next\[name\]/);
});
