"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import rehypeRaw from "rehype-raw";
import type { Components } from "react-markdown";
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

function escapeCurrencyAmounts(markdown: string): string {
  return markdown.replace(/(?<!\\)\$(?=\d)/g, "\\$");
}

export default function MarkdownRenderer({
  content,
  glossaryTerms = [],
  disableMath = false,
}: {
  content: string;
  glossaryTerms?: GlossaryLink[];
  disableMath?: boolean;
}) {
  const components = buildComponents(glossaryTerms);
  const remarkPlugins = disableMath ? [remarkGfm] : [remarkGfm, remarkMath];
  const rehypePlugins = disableMath
    ? [rehypeRaw, rehypeHighlight]
    : [rehypeRaw, rehypeKatex, rehypeHighlight];
  const renderedContent = escapeCurrencyAmounts(content);

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
