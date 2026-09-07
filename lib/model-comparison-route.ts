export function publicModelSlug(slug: string): string {
  const normalized = slug.trim().toLowerCase();
  // Anthropic's internal catalog slugs carry the vendor prefix, but the product
  // family is the useful search term (for example, fable-5-1 rather than
  // claude-fable-5-1). Keep all other canonical model slugs unchanged.
  return normalized.startsWith("claude-") ? normalized.slice("claude-".length) : normalized;
}

export function canonicalComparisonModels(modelA: string, modelB: string): [string, string] {
  const pair = [modelA, modelB]
    .map((slug) => ({ slug, publicSlug: publicModelSlug(slug) }))
    .sort((a, b) => a.publicSlug.localeCompare(b.publicSlug));
  return [pair[0].slug, pair[1].slug];
}

export function canonicalComparisonSlug(modelA: string, modelB: string): string {
  const [first, second] = canonicalComparisonModels(modelA, modelB);
  return `${publicModelSlug(first)}-vs-${publicModelSlug(second)}`;
}

export function resolveComparisonSlug(slug: string, modelSlugs: string[]): [string, string] | null {
  const normalized = slug.trim().toLowerCase();
  const byPublicSlug = new Map(modelSlugs.map((modelSlug) => [publicModelSlug(modelSlug), modelSlug]));

  for (const [leftPublicSlug, leftModelSlug] of byPublicSlug) {
    const prefix = `${leftPublicSlug}-vs-`;
    if (!normalized.startsWith(prefix)) continue;
    const rightModelSlug = byPublicSlug.get(normalized.slice(prefix.length));
    if (rightModelSlug && rightModelSlug !== leftModelSlug) return [leftModelSlug, rightModelSlug];
  }

  return null;
}
