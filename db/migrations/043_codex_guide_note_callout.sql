-- Restyle the closing callout of the Codex provider guide:
-- Caveats -> NOTE with the prose in capitals (code spans untouched).
WITH old_new(old_text, new_text) AS (
  SELECT
    $old$> **Caveats:** Current Codex versions require the Responses API for custom providers; older configurations using `wire_api = "chat"` no longer work. Provider settings such as `model_provider` and `model_providers` are not accepted from project-level `.codex/config.toml`. Custom-provider behavior can also vary between Codex releases and authentication modes, and current issues have been reported around the `/model` picker, changing providers inside an existing session, custom model catalogs, and some advanced features such as MCP. Provider compatibility should therefore be checked against both the current Codex documentation and the provider's own API documentation before relying on a specific configuration.$old$::text,
    $new$> **NOTE:** CURRENT CODEX VERSIONS REQUIRE THE RESPONSES API FOR CUSTOM PROVIDERS; OLDER CONFIGURATIONS USING `wire_api = "chat"` NO LONGER WORK. PROVIDER SETTINGS SUCH AS `model_provider` AND `model_providers` ARE NOT ACCEPTED FROM PROJECT-LEVEL `.codex/config.toml`. CUSTOM-PROVIDER BEHAVIOR CAN ALSO VARY BETWEEN CODEX RELEASES AND AUTHENTICATION MODES, AND CURRENT ISSUES HAVE BEEN REPORTED AROUND THE `/model` PICKER, CHANGING PROVIDERS INSIDE AN EXISTING SESSION, CUSTOM MODEL CATALOGS, AND SOME ADVANCED FEATURES SUCH AS MCP. PROVIDER COMPATIBILITY SHOULD THEREFORE BE CHECKED AGAINST BOTH THE CURRENT CODEX DOCUMENTATION AND THE PROVIDER'S OWN API DOCUMENTATION BEFORE RELYING ON A SPECIFIC CONFIGURATION.$new$::text
)
UPDATE content_items
SET
  body = replace(content_items.body, old_new.old_text, old_new.new_text),
  blocks = replace(blocks::text, old_new.old_text, old_new.new_text)::jsonb,
  updated_at = NOW()
FROM old_new
WHERE kind = 'guide'
  AND slug = 'how-to-use-other-models-in-codex'
  AND parent_slug = ''
  AND content_items.body LIKE '%' || old_new.old_text || '%';
