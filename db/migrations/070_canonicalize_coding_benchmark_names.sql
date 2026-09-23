-- Canonicalize minority base+version benchmark spellings to the catalog-wide
-- embedded labels so database-generated (x vs y) comparisons render one row
-- per benchmark instead of two.
--
-- Root cause: most of the catalog stores the version inside benchmark_name
-- with benchmark_version NULL (e.g. 'DeepSWE v1.1', 'Terminal-Bench 2.1',
-- 'FrontierCode 1.1 Main'), which is also what lib/model-comparison.ts,
-- components/admin/admin-client.ts and the tests use as predefined labels.
-- A minority of rows from 015 (comprehensive catalog), 040 (DeepSeek V4.1
-- Flash), 048 (Grok 4.7), 052 (Opus 5.5 comparison) and 057 (Sol clean table)
-- stored the base name plus a separate benchmark_version instead
-- (e.g. benchmark_name 'DeepSWE' with version 'v1.1'). lib/models.ts groups
-- by exact benchmark_name, so e.g. GPT-6 Luna's 'DeepSWE v1.1' and DeepSeek
-- V4.1 Flash's 'DeepSWE'+'v1.1' rendered as two DeepSWE rows for the same
-- comparison. Same class of dedupe as 067 (Sol/Luna DeepSWE + FrontierCode)
-- and 068 (Agents' Last Exam), extended to the remaining outliers.
--
-- Convention (unchanged): full benchmark_name including version in the label
-- column, score plus tools/effort qualifiers in each model column, e.g.
-- "DeepSWE v1.1 | 74.2% (max) | 66.6% (tools; max)". lib/models.ts already
-- omits benchmark_version from qualifiers when it is contained in the name,
-- so moving the version into the name keeps rendered values stable.
--
-- Families intentionally NOT touched here:
-- - CursorBench stays base+version ('CursorBench' + '3.2'/'3.2.0'/'4.0'):
--   every row uses that spelling, so they already merge into one grouped row.
-- - AutomationBench, GDPval-AA, OSWorld and other versioned families keep
--   their current mixed spellings; follow-up if duplicates show there.
-- - Terminal-Bench '2' and 'Hard' have no embedded counterpart and stay as is.

-- 1. DeepSWE base + 1.1/v1.1 -> the catalog-wide 'DeepSWE v1.1' label.
UPDATE model_benchmarks SET
  benchmark_name = 'DeepSWE v1.1',
  benchmark_version = NULL,
  updated_at = NOW()
WHERE benchmark_name = 'DeepSWE'
  AND benchmark_version IN ('1.1', 'v1.1');

-- 2. DeepSWE 1.0 (Grok 4.5 era) keeps its own versioned label.
UPDATE model_benchmarks SET
  benchmark_name = 'DeepSWE 1.0',
  benchmark_version = NULL,
  updated_at = NOW()
WHERE benchmark_name = 'DeepSWE'
  AND benchmark_version = '1.0';

-- 3. Redundant version on already-embedded DeepSWE rows (016): display is
-- unchanged because lib/models.ts already omits a version contained in the
-- name from qualifiers.
UPDATE model_benchmarks SET
  benchmark_version = NULL,
  updated_at = NOW()
WHERE benchmark_name = 'DeepSWE v1.1'
  AND benchmark_version = 'v1.1';

-- 4. Terminal-Bench base + numeric version -> embedded per-release labels.
UPDATE model_benchmarks SET
  benchmark_name = 'Terminal-Bench 2.0',
  benchmark_version = NULL,
  updated_at = NOW()
WHERE benchmark_name = 'Terminal-Bench'
  AND benchmark_version = '2.0';

UPDATE model_benchmarks SET
  benchmark_name = 'Terminal-Bench 2.1',
  benchmark_version = NULL,
  updated_at = NOW()
WHERE benchmark_name = 'Terminal-Bench'
  AND benchmark_version = '2.1';

UPDATE model_benchmarks SET
  benchmark_name = 'Terminal-Bench 3.0',
  benchmark_version = NULL,
  updated_at = NOW()
WHERE benchmark_name = 'Terminal-Bench'
  AND benchmark_version = '3.0';

UPDATE model_benchmarks SET
  benchmark_name = 'Terminal-Bench 4.0',
  benchmark_version = NULL,
  updated_at = NOW()
WHERE benchmark_name = 'Terminal-Bench'
  AND benchmark_version = '4.0';

-- 5. Redundant version on already-embedded Terminal-Bench 2.1 rows (016).
UPDATE model_benchmarks SET
  benchmark_version = NULL,
  updated_at = NOW()
WHERE benchmark_name = 'Terminal-Bench 2.1'
  AND benchmark_version = '2.1';

-- 6. FrontierCode base + split -> embedded Main / Extended labels.
UPDATE model_benchmarks SET
  benchmark_name = 'FrontierCode 1.1 Main',
  benchmark_version = NULL,
  updated_at = NOW()
WHERE benchmark_name = 'FrontierCode'
  AND benchmark_version = '1.1 Main';

UPDATE model_benchmarks SET
  benchmark_name = 'FrontierCode 1.1 Extended',
  benchmark_version = NULL,
  updated_at = NOW()
WHERE benchmark_name = 'FrontierCode'
  AND benchmark_version = '1.1 Extended';

-- 7. Gemini 3.7 Flash blog row duplicates the model-card Main row for the
-- same model and score (43.6, unit %). Fold it into the Main label and
-- normalize the display so the merged row renders a single value.
UPDATE model_benchmarks SET
  benchmark_name = 'FrontierCode 1.1 Main',
  benchmark_version = NULL,
  score_display = '43.6%',
  updated_at = NOW()
WHERE id = 'gemini-3-7-flash-frontiercode-1-1-43-6-none-none-none-google-deepmind';

-- 8. NL2Repo alias -> the NL2Repo-Bench label used by 040 and the authored
-- DeepSeek comparison table (predefined label is added in this pass).
UPDATE model_benchmarks SET
  benchmark_name = 'NL2Repo-Bench',
  updated_at = NOW()
WHERE benchmark_name = 'NL2Repo';
