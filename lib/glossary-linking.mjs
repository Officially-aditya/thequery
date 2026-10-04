import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import rehypeRaw from "rehype-raw";

/** @typedef {{name: string, slug: string}} GlossaryLink */

export const remarkPlugins = [remarkGfm, remarkMath];
export const rehypePlugins = [rehypeRaw, rehypeKatex, rehypeHighlight];

/** @param {GlossaryLink[]} terms */
function createGlossaryMatcher(terms) {
  const sorted = [...terms].sort((a, b) => b.name.length - a.name.length);
  const byName = new Map(sorted.map((term) => [term.name.toLowerCase(), term]));
  const escaped = sorted.map((term) => term.name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  return {
    regex: terms.length ? new RegExp(`\\b(${escaped.join("|")})\\b`, "gi") : null,
    byName,
  };
}

/**
 * Shared by the renderer and link map: only direct paragraph/list text is eligible.
 * @param {{terms: GlossaryLink[], onMatch?: (slug: string) => void}} options
 */
export function rehypeGlossaryLinks({ terms, onMatch }) {
  const { regex, byName } = createGlossaryMatcher(terms);
  const linkedTerms = new Set();
  /** @param {import("hast").Root} tree */
  return (tree) => {
    if (!regex) return;
    /** @param {import("hast").Root | import("hast").Element} node */
    function visit(node) {
      if (node.type === "element" && (node.tagName === "p" || node.tagName === "li")) {
        node.children = node.children.flatMap((child) => {
          if (child.type !== "text") return [child];
          regex.lastIndex = 0;
          /** @type {import("hast").ElementContent[]} */
          const parts = [];
          let lastIndex = 0;
          let match;
          while ((match = regex.exec(child.value)) !== null) {
            const termKey = match[0].toLowerCase();
            const term = byName.get(termKey);
            if (!term) continue;
            onMatch?.(term.slug);
            if (linkedTerms.has(termKey)) continue;
            linkedTerms.add(termKey);
            if (match.index > lastIndex) parts.push({ type: "text", value: child.value.slice(lastIndex, match.index) });
            parts.push({
              type: "element",
              tagName: "a",
              properties: {
                href: `/glossary/${term.slug}`,
                className: ["text-accent", "underline", "decoration-dotted", "underline-offset-2", "hover:decoration-solid"],
                title: term.name,
              },
              children: [{ type: "text", value: match[0] }],
            });
            lastIndex = match.index + match[0].length;
          }
          if (lastIndex < child.value.length) parts.push({ type: "text", value: child.value.slice(lastIndex) });
          return parts;
        });
      }
      for (const child of node.children) {
        if (child.type === "element" && child.tagName !== "a") visit(child);
      }
    }
    visit(tree);
  };
}

function isMathLikeInner(inner) {
  const trimmed = inner.trim();
  if (!inner || inner.length > 500 || inner.includes("$") || inner.includes("\n\n")) return false;
  // Thousands separators and currency context words mean money, not math.
  if (/\d,\d/.test(inner)) return false;
  if (/\b(min|hrs?|input|output|tokens?|price|pricing|list|promo|cached?|peak|audio|video|text|image|month|year|per\b|bill|million)\b/i.test(inner)) return false;
  // A lone variable ($a$) or bare number ($0.01$) with an explicit closing
  // delimiter is author-intended math; currency never closes the pair.
  if (/^[a-zA-Z]$/.test(trimmed)) return true;
  if (/^\d+(\.\d+)?$/.test(trimmed)) return true;
  // Digits/operators only (e.g. the "2/" in "$2/$6") is a currency fragment.
  if (/^[\d\s.,/$+\-*]+$/.test(trimmed)) return false;
  if (/\\|\^|_|\*|=/.test(inner)) return true;
  if (/\b(sqrt|tanh|sigmoid|phi|sigma|exp|sum|frac|log|logits|tau|softmax|gelu|silu|swish|relu|beta|Phi)\b/.test(inner)) return true;
  // Slash with letters after it (2/pi, logits / tau, 1/sqrt) is math;
  // a bare trailing slash (2/) or digits-only is currency ($2/$6).
  if (/\/\s*[a-zA-Z\\(]/.test(inner)) return true;
  return false;
}

function escapeCurrencyAmounts(markdown) {
  // Protect fenced code, inline code, display math, and \(...\) / \[...\]
  // so $ amounts inside them are never touched. MathJax (like GateOverflow)
  // uses $...$ inline and $$...$$ display, so single-$ math starting with a
  // digit ($0.5 \cdot x ...$) must survive while currency ($30,000) stays escaped.
  const protectedPattern = /(```[\s\S]*?```|`[^`\n]*?`|\$\$[\s\S]*?\$\$|\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\])/g;
  const segments = markdown.split(protectedPattern);
  for (let i = 0; i < segments.length; i += 1) {
    const segment = segments[i];
    if (segment === undefined || segment === "") continue;
    // Odd indices are the protected matches.
    if (i % 2 === 1) continue;
    // Split out $...$ candidates so math-like spans (even those starting
    // with a digit) are never currency-escaped, while plain-text $ before
    // a digit ($30,000, $200) still is.
    const parts = segment.split(/(\$[^$\n]*?\$)/g);
    for (let j = 0; j < parts.length; j += 1) {
      const part = parts[j];
      if (part === undefined) continue;
      if (j % 2 === 1) {
        const inner = part.slice(1, -1);
        parts[j] = isMathLikeInner(inner) ? part : part.replace(/\$/g, "\\$");
      } else {
        parts[j] = part.replace(/(?<!\\)\$(?=\d)/g, "\\$");
      }
    }
    segments[i] = parts.join("");
  }
  return segments.join("");
}

function normalizeLatexDelimiters(markdown) {
  return markdown
    .replace(/\\\[([\s\S]*?)\\\]/g, (_, inner) => `$$${inner}$$`)
    .replace(/\\\(([\s\S]*?)\\\)/g, (_, inner) => `$${inner}$`);
}

export function prepareGlossaryMarkdown(content) {
  return normalizeLatexDelimiters(escapeCurrencyAmounts(content));
}

/** @param {string} content @param {GlossaryLink[]} terms */
export function getGlossaryMentions(content, terms) {
  /** @type {Map<string, number>} */
  const mentions = new Map();
  Markdown({
    children: prepareGlossaryMarkdown(content),
    remarkPlugins,
    rehypePlugins: [...rehypePlugins, [rehypeGlossaryLinks, {
      terms,
      onMatch: (slug) => mentions.set(slug, (mentions.get(slug) ?? 0) + 1),
    }]],
  });
  return mentions;
}
