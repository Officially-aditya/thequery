"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeMathjaxChtml from "rehype-mathjax/chtml";
import rehypeHighlight from "rehype-highlight";
import rehypeRaw from "rehype-raw";
import type { Components } from "react-markdown";
import type { Pluggable } from "unified";
import React from "react";
import ImageLightbox from "@/components/content/ImageLightbox";

export interface GlossaryLink {
  name: string;
  slug: string;
}

interface GlossaryMatcher {
  regex: RegExp | null;
  byName: Map<string, GlossaryLink>;
}

function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .replace(/(^-|-$)/g, "");
}

function HeadingWithId({ level, children }: { level: number; children: React.ReactNode }) {
  const text = extractText(children);
  const id = slugify(text);
  const Tag = `h${level}` as "h1" | "h2" | "h3" | "h4" | "h5" | "h6";
  return <Tag id={id}>{children}</Tag>;
}

function extractText(node: React.ReactNode): string {
  if (typeof node === "string") return node;
  if (typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(extractText).join("");
  if (node && typeof node === "object" && "props" in node) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return extractText((node as any).props.children);
  }
  return "";
}

function createGlossaryMatcher(terms: GlossaryLink[]): GlossaryMatcher {
  if (terms.length === 0) return { regex: null, byName: new Map() };

  const sorted = [...terms].sort((a, b) => b.name.length - a.name.length);
  const byName = new Map(sorted.map((term) => [term.name.toLowerCase(), term]));
  const escaped = sorted.map((term) => term.name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  return {
    regex: new RegExp(`\\b(${escaped.join("|")})\\b`, "gi"),
    byName,
  };
}

function linkifyText(
  text: string,
  matcher: GlossaryMatcher,
  linkedTerms: Set<string>,
): React.ReactNode[] {
  const { regex, byName } = matcher;
  if (!regex) return [text];

  regex.lastIndex = 0;
  const parts: React.ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(text)) !== null) {
    const matchedText = match[0];
    const termKey = matchedText.toLowerCase();
    const term = byName.get(termKey);
    if (!term || linkedTerms.has(termKey)) continue;

    linkedTerms.add(termKey);
    if (match.index > lastIndex) parts.push(text.slice(lastIndex, match.index));
    parts.push(
      <a
        key={`gl-${match.index}`}
        href={`/glossary/${term.slug}`}
        className="text-accent underline decoration-dotted underline-offset-2 hover:decoration-solid"
        title={term.name}
      >
        {matchedText}
      </a>
    );
    lastIndex = match.index + matchedText.length;
  }

  if (lastIndex < text.length) parts.push(text.slice(lastIndex));
  return parts.length > 0 ? parts : [text];
}

function buildComponents(glossaryTerms: GlossaryLink[]): Components {
  const linkedTerms = new Set<string>();
  // Sorting, escaping, regex construction, and name indexing used to happen
  // for every paragraph/list text node. Compile once for the whole render.
  const glossaryMatcher = createGlossaryMatcher(glossaryTerms);

  return {
    h1: ({ children }) => <HeadingWithId level={1}>{children}</HeadingWithId>,
    h2: ({ children }) => <HeadingWithId level={2}>{children}</HeadingWithId>,
    h3: ({ children }) => <HeadingWithId level={3}>{children}</HeadingWithId>,
    small: ({ children }) => (
      <small className="mb-6 block text-xs leading-relaxed text-text-muted">
        {children}
      </small>
    ),
    img: ({ src, alt, ...props }) => {
      if (typeof src !== "string" || !src) return null;
      return (
        <ImageLightbox src={src} alt={alt || ""}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={src}
            alt={alt || ""}
            loading="lazy"
            style={{ maxWidth: "100%", height: "auto" }}
            {...props}
          />
        </ImageLightbox>
      );
    },
    p: ({ children }) => {
      if (glossaryTerms.length === 0) return <p>{children}</p>;
      const processed = React.Children.map(children, (child) => {
        if (typeof child === "string") {
          return <>{linkifyText(child, glossaryMatcher, linkedTerms)}</>;
        }
        return child;
      });
      return <p>{processed}</p>;
    },
    li: ({ children }) => {
      if (glossaryTerms.length === 0) return <li>{children}</li>;
      const processed = React.Children.map(children, (child) => {
        if (typeof child === "string") {
          return <>{linkifyText(child, glossaryMatcher, linkedTerms)}</>;
        }
        return child;
      });
      return <li>{processed}</li>;
    },
  };
}

function isMathLikeInner(inner: string): boolean {
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

function escapeCurrencyAmounts(markdown: string): string {
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

function normalizeLatexDelimiters(markdown: string): string {
  return markdown
    .replace(/\\\[([\s\S]*?)\\\]/g, (_, inner) => `$$${inner}$$`)
    .replace(/\\\(([\s\S]*?)\\\)/g, (_, inner) => `$${inner}$`);
}

export default function MarkdownRenderer({
  content,
  glossaryTerms = [],
}: {
  content: string;
  glossaryTerms?: GlossaryLink[];
}) {
  const components = buildComponents(glossaryTerms);
  const remarkPlugins = [remarkGfm, remarkMath];
  // MathJax CHTML (same engine as GateOverflow): renders $...$ inline and
  // $$...$$ display at compile time, no client-side typesetting needed.
  // Fonts load from the MathJax CDN; CSS is emitted inline by MathJax.
  // mathjax-full stays external (see next.config.ts serverExternalPackages)
  // so its runtime package.json version lookup resolves from node_modules
  // instead of breaking Turbopack SSR prerendering.
  const rehypePlugins: Pluggable[] = [
    rehypeRaw,
    [
      rehypeMathjaxChtml,
      {
        chtml: {
          fontURL:
            "https://cdn.jsdelivr.net/npm/mathjax@3/es5/output/chtml/fonts/woff-v2",
        },
      },
    ],
    rehypeHighlight,
  ];
  const renderedContent = normalizeLatexDelimiters(escapeCurrencyAmounts(content));

  return (
    <div className="prose-custom">
      <ReactMarkdown
        remarkPlugins={remarkPlugins}
        rehypePlugins={rehypePlugins}
        components={components}
      >
        {renderedContent}
      </ReactMarkdown>
    </div>
  );
}
