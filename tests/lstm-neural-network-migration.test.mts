import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

test("LSTM and neural network database updates are registered and rendered from their bodies", async () => {
  const [runner, migration] = await Promise.all([
    readFile(path.join(import.meta.dirname, "../scripts/migrate.mjs"), "utf8"),
    readFile(path.join(import.meta.dirname, "../db/migrations/006_update_lstm_and_neural_network.sql"), "utf8"),
  ]);

  assert.match(runner, /006_update_lstm_and_neural_network/);
  assert.match(migration, /## How an LSTM cell works/);
  assert.match(migration, /## How a neural network learns/);
  assert.match(migration, /'content', glossary_updates\.body/);
  assert.match(migration, /item\.slug = glossary_updates\.slug/);
});
