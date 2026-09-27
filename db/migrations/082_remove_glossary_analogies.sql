-- Remove analogy field from all glossary entries.
--
-- Only strips the `analogy` key from metadata JSON; all other columns
-- and all other metadata keys are left untouched.
UPDATE content_items
SET metadata = metadata - 'analogy'
WHERE kind = 'glossary'
  AND metadata ? 'analogy';
