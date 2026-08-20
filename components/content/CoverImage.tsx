/* eslint-disable @next/next/no-img-element */

interface CoverImageProps {
  src?: string | null;
  alt?: string | null;
  title: string;
  className?: string;
  imageClassName?: string;
  loading?: "eager" | "lazy";
}

export default function CoverImage({
  src,
  alt,
  title,
  className = "mb-8 overflow-hidden rounded-xl border border-border bg-bg-secondary",
  imageClassName = "aspect-[16/9] w-full object-cover",
  loading = "eager",
}: CoverImageProps) {
  if (!src) return null;

  return (
    <figure className={className}>
      <img
        src={src}
        alt={alt || `${title || "Content"} cover image`}
        className={imageClassName}
        loading={loading}
        decoding="async"
      />
    </figure>
  );
}
