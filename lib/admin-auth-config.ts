export function normalizeAdminEmail(value: string | null | undefined): string {
  return typeof value === "string" ? value.trim().toLowerCase() : "";
}

export function isConfiguredAdminEmail(
  candidate: string | null | undefined,
  configured: string | null | undefined,
): boolean {
  const configuredEmail = normalizeAdminEmail(configured);
  return Boolean(configuredEmail) && normalizeAdminEmail(candidate) === configuredEmail;
}
