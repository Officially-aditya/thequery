"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import type { Components } from "react-markdown";
import React from "react";

export interface GlossaryLink {
  name: string;
  slug: string;
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

function linkifyText(text: string, terms: GlossaryLink[]): React.ReactNode[] {
  if (terms.length === 0) return [text];

  // Build regex matching all term names, longest first to avoid partial matches
  const sorted = [...terms].sort((a, b) => b.name.length - a.name.length);
  const escaped = sorted.map((t) => t.name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const regex = new RegExp(`\\b(${escaped.join("|")})\\b`, "gi");

  const parts: React.ReactNode[] = [];
  let lastIndex = 0;
  const linked = new Set<string>(); // only link first occurrence per render
  let match: RegExpExecArray | null;

  while ((match = regex.exec(text)) !== null) {
    const matchedText = match[0];
    const termKey = matchedText.toLowerCase();

    // Find the matching term
    const term = sorted.find((t) => t.name.toLowerCase() === termKey);
    if (!term || linked.has(termKey)) {
      continue;
    }
    linked.add(termKey);

    // Add text before match
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }

    // Add linked term
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

  // Add remaining text
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return parts.length > 0 ? parts : [text];
}

function buildComponents(glossaryTerms: GlossaryLink[]): Components {
  return {
    h1: ({ children }) => <HeadingWithId level={1}>{children}</HeadingWithId>,
    h2: ({ children }) => <HeadingWithId level={2}>{children}</HeadingWithId>,
    h3: ({ children }) => <HeadingWithId level={3}>{children}</HeadingWithId>,
    // Auto-link glossary terms in paragraph text nodes
    p: ({ children }) => {
      if (glossaryTerms.length === 0) return <p>{children}</p>;
      const processed = React.Children.map(children, (child) => {
        if (typeof child === "string") {
          return <>{linkifyText(child, glossaryTerms)}</>;
        }
        return child;
      });
      return <p>{processed}</p>;
    },
    li: ({ children }) => {
      if (glossaryTerms.length === 0) return <li>{children}</li>;
      const processed = React.Children.map(children, (child) => {
        if (typeof child === "string") {
          return <>{linkifyText(child, glossaryTerms)}</>;
        }
        return child;
      });
      return <li>{processed}</li>;
    },
  };
}

export default function MarkdownRenderer({
  content,
  glossaryTerms = [],
}: {
  content: string;
  glossaryTerms?: GlossaryLink[];
}) {
  const components = buildComponents(glossaryTerms);

  return (
    <div className="prose-custom">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeHighlight]}
        components={components}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
