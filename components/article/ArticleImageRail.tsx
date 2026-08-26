import Image from "next/image";
import ImageLightbox from "@/components/content/ImageLightbox";

interface ArticleImageRailProps {
  src: string;
  alt: string;
  width: number;
  height: number;
  caption: string;
}

export default function ArticleImageRail({
  src,
  alt,
  width,
  height,
  caption,
}: ArticleImageRailProps) {
  return (
    <figure className="my-10 overflow-hidden rounded-xl border border-border bg-white xl:my-0">
      <ImageLightbox
        src={src}
        alt={alt}
        triggerClassName="block w-full cursor-pointer border-0 bg-transparent p-0 text-left"
      >
        <Image
          src={src}
          alt={alt}
          width={width}
          height={height}
          sizes="(min-width: 1536px) 400px, calc(100vw - 2rem)"
          className="h-auto w-full"
        />
      </ImageLightbox>
      <figcaption className="border-t border-border bg-bg-secondary px-4 py-3 text-xs leading-relaxed text-text-muted">
        {caption} Click the image for the full-size chart.
      </figcaption>
    </figure>
  );
}
