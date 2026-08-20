import assert from "node:assert/strict";
import test from "node:test";
import { isConfiguredAdminEmail, normalizeAdminEmail } from "../lib/admin-auth-config.ts";

test("admin email matching is normalized and only permits the configured account", () => {
  assert.equal(normalizeAdminEmail(" Addy@TheQuery.in "), "addy@thequery.in");
  assert.equal(isConfiguredAdminEmail("ADDY@thequery.in", "addy@thequery.in"), true);
  assert.equal(isConfiguredAdminEmail("editor@thequery.in", "addy@thequery.in"), false);
  assert.equal(isConfiguredAdminEmail("addy@thequery.in", undefined), false);
});
