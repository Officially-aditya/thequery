"use client";

import CoverImage from "@/components/content/CoverImage";

const fieldClass = "w-full rounded-md border border-border bg-bg-primary px-3 py-2 text-sm text-text-primary outline-none focus:border-accent";

interface CoverImageFieldsProps {
  title: string;
  coverImageUrl: string;
  coverImageAlt: string;
  onChange: (next: { coverImageUrl: string; coverImageAlt: string }) => void;
}

export default function CoverImageFields({
  title,
  coverImageUrl,
  coverImageAlt,
  onChange,
}: CoverImageFieldsProps) {
  return (
    <section className="rounded-xl border border-border p-4">
      <h2 className="font-serif text-lg font-semibold text-text-primary">Cover image</h2>
      <p className="mt-1 text-sm text-text-secondary">Optional. Use an HTTPS image URL or a site-relative path such as <code>/images/cover.jpg</code>.</p>
      <div className="mt-4 grid gap-4 sm:grid-cols-2">
        <label className="sm:col-span-2 text-sm font-medium text-text-secondary">
          Image URL
          <input
            className={`${fieldClass} mt-1`}
            type="url"
            value={coverImageUrl}
            onChange={(event) => onChange({ coverImageUrl: event.target.value, coverImageAlt })}
            placeholder="https://images.example.com/article-cover.jpg"
          />
        </label>
        <label className="sm:col-span-2 text-sm font-medium text-text-secondary">
          Alt text <span className="font-normal text-text-muted">(optional)</span>
          <input
            className={`${fieldClass} mt-1`}
            value={coverImageAlt}
            onChange={(event) => onChange({ coverImageUrl, coverImageAlt: event.target.value })}
            placeholder="Describe the image for readers using assistive technology"
          />
        </label>
      </div>
      <CoverImage
        src={coverImageUrl}
        alt={coverImageAlt}
        title={title}
        className="mt-4 max-w-2xl overflow-hidden rounded-lg border border-border bg-bg-secondary"
        loading="lazy"
      />
    </section>
  );
}
