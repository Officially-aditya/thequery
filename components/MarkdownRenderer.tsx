"use client";

import ReactMarkdown from "react-markdown";
import { remarkPlugins, rehypePlugins, rehypeGlossaryLinks, prepareGlossaryMarkdown } from "@/lib/glossary-linking.mjs";
import type { Components } from "react-markdown";
import type { Pluggable } from "unified";
import React from "react";
import ImageLightbox from "@/components/content/ImageLightbox";

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

function buildComponents(): Components {
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
  };
}

export default function MarkdownRenderer({
  content,
  glossaryTerms = [],
}: {
  content: string;
  glossaryTerms?: GlossaryLink[];
}) {
  const components = buildComponents();
  const glossaryPlugins: Pluggable[] = [...rehypePlugins, [rehypeGlossaryLinks, { terms: glossaryTerms }]];
  const renderedContent = prepareGlossaryMarkdown(content);

  return (
    <div className="prose-custom">
      <ReactMarkdown
        remarkPlugins={remarkPlugins}
        rehypePlugins={glossaryPlugins}
        components={components}
      >
        {renderedContent}
      </ReactMarkdown>
    </div>
  );
}
