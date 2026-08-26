"use client";

import type { ReactNode } from "react";
import { useEffect, useState } from "react";

interface ImageLightboxProps {
  src: string;
  alt: string;
  children: ReactNode;
  triggerClassName?: string;
}

export default function ImageLightbox({
  src,
  alt,
  children,
  triggerClassName = "inline-block max-w-full cursor-zoom-in border-0 bg-transparent p-0 text-left",
}: ImageLightboxProps) {
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    if (!isOpen) return;

    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setIsOpen(false);
    };

    document.addEventListener("keydown", closeOnEscape);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    return () => {
      document.removeEventListener("keydown", closeOnEscape);
      document.body.style.overflow = previousOverflow;
    };
  }, [isOpen]);

  return (
    <>
      <button
        type="button"
        className={triggerClassName}
        onClick={() => setIsOpen(true)}
        aria-label={`Open ${alt || "image"} in a larger view`}
      >
        {children}
      </button>

      {isOpen ? (
        <div
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 p-4 sm:p-8"
          role="dialog"
          aria-modal="true"
          aria-label={`Image preview: ${alt || "image"}`}
          onClick={(event) => {
            if (event.target === event.currentTarget) setIsOpen(false);
          }}
        >
          <button
            type="button"
            className="absolute right-4 top-4 rounded-full bg-black/60 px-3 py-1 text-2xl leading-none text-white transition-colors hover:bg-black/80"
            onClick={() => setIsOpen(false)}
            aria-label="Close image preview"
          >
            &times;
          </button>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={src}
            alt={alt}
            className="max-h-[calc(100vh-2rem)] max-w-full object-contain sm:max-h-[calc(100vh-4rem)]"
            onClick={(event) => event.stopPropagation()}
          />
        </div>
      ) : null}
    </>
  );
}
