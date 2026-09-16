-- Split the Codex guide's closing Caveats callout into two paragraphs:
-- the compatibility sentence moves to its own paragraph inside the quote.
WITH old_new(old_text, new_text) AS (
  SELECT
    $old$are not accepted from project-level `.codex/config.toml`. Custom-provider behavior can also vary$old$::text,
    $new$are not accepted from project-level `.codex/config.toml`.
>
> Custom-provider behavior can also vary$new$::text
)
UPDATE content_items
SET
  body = replace(content_items.body, old_new.old_text, old_new.new_text),
  blocks = jsonb_build_array(
    jsonb_build_object(
      'id', 'markdown-1',
      'type', 'markdown',
      'content', replace(content_items.body, old_new.old_text, old_new.new_text)
    )
  ),
  updated_at = NOW()
FROM old_new
WHERE kind = 'guide'
  AND slug = 'how-to-use-other-models-in-codex'
  AND parent_slug = '';
