-- Remove an unverified incident from this living glossary definition and
-- provide a primary reference for its agent loop, costs, and safeguards.
-- Keep the original publication date and all unrelated content intact.

WITH corrected AS (
  SELECT id,
    CASE WHEN strpos(body, ' In production, an agent that called the wrong API or misinterpreted search results caused thousands of incorrect refund transactions.') > 0
      THEN REPLACE(body,
        ' In production, an agent that called the wrong API or misinterpreted search results caused thousands of incorrect refund transactions.',
        '') || chr(10) || chr(10) || '**Editorial note (October 13, 2026):** An earlier version included an unsourced example claiming thousands of incorrect refund transactions. That example has been removed.'
      ELSE body
    END AS body
  FROM content_items
  WHERE kind = 'glossary' AND parent_slug = '' AND slug = 'ai-agent'
)
UPDATE content_items AS item
SET body = corrected.body,
    blocks = jsonb_set(item.blocks, '{0,content}', to_jsonb(corrected.body)),
    sources = CASE
      WHEN item.sources @> '[{"url":"https://www.anthropic.com/engineering/building-effective-agents"}]'::jsonb
        THEN item.sources
      ELSE item.sources || '[{"title":"Building effective agents (Anthropic)","url":"https://www.anthropic.com/engineering/building-effective-agents"}]'::jsonb
    END,
    updated_at = NOW()
FROM corrected
WHERE item.id = corrected.id
  AND (
    item.body IS DISTINCT FROM corrected.body
    OR NOT item.sources @> '[{"url":"https://www.anthropic.com/engineering/building-effective-agents"}]'::jsonb
  );
