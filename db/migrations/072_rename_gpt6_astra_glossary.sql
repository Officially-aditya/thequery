UPDATE content_items
SET
  title = 'GPT 6 Astra',
  updated_at = NOW()
WHERE kind = 'glossary'
  AND slug = 'gpt-6-astra';
