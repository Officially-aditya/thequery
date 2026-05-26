"use client";

import { useEffect, useRef, useState } from "react";
import type {
  MouseEvent as ReactMouseEvent,
  PointerEvent as ReactPointerEvent,
} from "react";

type Section = {
  id: number;
  label: string;
  top: number;
  progress: number;
  displayProgress: number;
};

type Placement =
  | { mode: "center" }
  | { mode: "right"; left: number };

const tickCount = 34;
const trackInsetPercent = 2;

function clamp(value: number) {
  return Math.min(100, Math.max(0, value));
}

function getTickIndex(progress: number) {
  return Math.min(
    tickCount - 1,
    Math.max(0, Math.round((clamp(progress) / 100) * (tickCount - 1)))
  );
}

function getTickDisplayProgress(index: number) {
  const usableRange = 100 - trackInsetPercent * 2;
  return trackInsetPercent + (index / (tickCount - 1)) * usableRange;
}

function getTickProgress(progress: number) {
  return getTickDisplayProgress(getTickIndex(progress));
}

function getScrollMetrics() {
  const el = document.documentElement;
  const maxScroll = el.scrollHeight - el.clientHeight;
  const progress = maxScroll > 0 ? (el.scrollTop / maxScroll) * 100 : 0;
  return { maxScroll, progress: clamp(progress) };
}

function getSectionHeadings() {
  const subheadings = Array.from(
    document.querySelectorAll<HTMLElement>("main h2")
  );

  return subheadings.length > 0
    ? subheadings
    : Array.from(document.querySelectorAll<HTMLElement>("main h1"));
}

function getHeadingTop(heading: HTMLElement) {
  const maxScroll = Math.max(
    0,
    document.documentElement.scrollHeight - window.innerHeight
  );
  const headingTop = heading.getBoundingClientRect().top + window.scrollY;

  return Math.min(maxScroll, Math.max(0, headingTop - 88));
}

function getActiveSection(sections: Section[], currentY: number) {
  for (let i = sections.length - 1; i >= 0; i -= 1) {
    if (sections[i].top <= currentY) {
      return sections[i];
    }
  }

  return sections[0];
}

export default function ReadingProgress() {
  const [progress, setProgress] = useState(0);
  const [activeSection, setActiveSection] = useState("Start");
  const [activeSectionId, setActiveSectionId] = useState(0);
  const [sections, setSections] = useState<Section[]>([]);
  const [isReady, setIsReady] = useState(false);
  const [isEngaged, setIsEngaged] = useState(false);
  const [placement, setPlacement] = useState<Placement>({ mode: "center" });
  const sectionsRef = useRef<Section[]>([]);
  const shellRef = useRef<HTMLDivElement | null>(null);
  const railRef = useRef<HTMLDivElement | null>(null);
  const hideTimerRef = useRef<number | null>(null);

  const showDetail = isReady && isEngaged;
  const thumbPosition = Math.min(98, Math.max(2, progress));
  const sectionTicks = new Map<number, boolean>();

  sections.forEach((section) => {
    const tickIndex = getTickIndex(section.progress);
    sectionTicks.set(
      tickIndex,
      sectionTicks.get(tickIndex) || section.id === activeSectionId
    );
  });

  useEffect(() => {
    const collectSections = () => {
      const maxScroll = Math.max(
        1,
        document.documentElement.scrollHeight - window.innerHeight
      );
      const headings = getSectionHeadings();

      const nextSections = headings
        .map((heading, index) => {
          const top = getHeadingTop(heading);
          const progress = clamp((top / maxScroll) * 100);

          return {
            id: index,
            displayProgress: getTickProgress(progress),
            progress,
            top,
            label: heading.textContent?.trim() || "Reading",
          };
        })
        .filter((section) => section.label.length > 0);

      sectionsRef.current = nextSections.length
        ? nextSections
        : [
            {
              id: 0,
              label: document.title.replace(" | TheQuery", ""),
              displayProgress: 0,
              progress: 0,
              top: 0,
            },
          ];
      setSections(sectionsRef.current);
    };

    const updatePlacement = () => {
      const frame = document.querySelector<HTMLElement>("[data-reading-frame]");
      const shell = shellRef.current;

      if (!frame || !shell) {
        setPlacement({ mode: "center" });
        return;
      }

      const frameRect = frame.getBoundingClientRect();
      const railWidth = shell.getBoundingClientRect().width;
      const gap = 24;
      const margin = 16;
      const left = frameRect.right + gap;
      const availableRightSpace = window.innerWidth - left;

      if (availableRightSpace >= railWidth + margin) {
        setPlacement({ mode: "right", left });
      } else {
        setPlacement({ mode: "center" });
      }
    };

    const handleScroll = () => {
      const { maxScroll, progress: nextProgress } = getScrollMetrics();
      const currentY = window.scrollY + 1;
      const currentSection = getActiveSection(sectionsRef.current, currentY);

      setProgress(nextProgress);
      setActiveSection(currentSection?.label || "Reading");
      setActiveSectionId(currentSection?.id ?? 0);
      setIsReady(maxScroll > 80);
      setIsEngaged(true);

      if (hideTimerRef.current) {
        window.clearTimeout(hideTimerRef.current);
      }

      hideTimerRef.current = window.setTimeout(() => {
        setIsEngaged(false);
      }, 1200);
    };

    const handleResize = () => {
      collectSections();
      updatePlacement();
      handleScroll();
    };

    collectSections();
    updatePlacement();
    handleScroll();

    window.addEventListener("scroll", handleScroll, { passive: true });
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("scroll", handleScroll);
      window.removeEventListener("resize", handleResize);

      if (hideTimerRef.current) {
        window.clearTimeout(hideTimerRef.current);
      }
    };
  }, []);

  const scrollToClientX = (clientX: number) => {
    const rail = railRef.current;
    if (!rail) return;

    const rect = rail.getBoundingClientRect();
    const nextProgress = clamp(((clientX - rect.left) / rect.width) * 100);
    const maxScroll = document.documentElement.scrollHeight - window.innerHeight;

    window.scrollTo({
      top: (nextProgress / 100) * maxScroll,
      behavior: "auto",
    });
  };

  const scrollToSection = (section: Section) => {
    const heading = getSectionHeadings()[section.id];
    const top = heading ? getHeadingTop(heading) : section.top;

    setActiveSection(section.label);
    setActiveSectionId(section.id);
    setIsEngaged(true);

    window.scrollTo({
      top,
      behavior: "smooth",
    });
  };

  const getNearestSectionFromClientX = (clientX: number) => {
    const rail = railRef.current;
    if (!rail || sections.length === 0) return null;

    const rect = rail.getBoundingClientRect();
    const clickProgress = clamp(((clientX - rect.left) / rect.width) * 100);

    return sections.reduce((nearest, section) => {
      const nearestDistance = Math.abs(nearest.displayProgress - clickProgress);
      const sectionDistance = Math.abs(section.displayProgress - clickProgress);
      return sectionDistance < nearestDistance ? section : nearest;
    }, sections[0]);
  };

  const handleRailClick = (event: ReactMouseEvent<HTMLDivElement>) => {
    const nearestSection = getNearestSectionFromClientX(event.clientX);

    if (nearestSection) {
      scrollToSection(nearestSection);
    }
  };

  const handlePointerDown = (event: ReactPointerEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsEngaged(true);
    scrollToClientX(event.clientX);

    const handlePointerMove = (moveEvent: PointerEvent) => {
      scrollToClientX(moveEvent.clientX);
    };

    const handlePointerUp = () => {
      setIsEngaged(false);
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    };

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
  };

  return (
    <div
      ref={shellRef}
      className={`scroll-progress ${
        isReady ? "scroll-progress--ready" : ""
      } ${placement.mode === "right" ? "scroll-progress--side" : ""}`}
      style={
        placement.mode === "right" ? { left: `${placement.left}px` } : undefined
      }
      onMouseEnter={() => setIsEngaged(true)}
      onMouseLeave={() => setIsEngaged(false)}
    >
      <div
        className={`scroll-progress__card ${
          showDetail ? "scroll-progress__card--visible" : ""
        }`}
      >
        <span className="scroll-progress__eyebrow">{"// Reading now"}</span>
        <code>{activeSection}</code>
      </div>

      <div
        ref={railRef}
        className="scroll-progress__rail"
        onClick={handleRailClick}
      >
        <div className="scroll-progress__ticks" aria-hidden="true">
          {Array.from({ length: tickCount }, (_, index) => {
            const isSectionTick = sectionTicks.has(index);
            const isActiveTick = sectionTicks.get(index);

            return (
              <span
                key={index}
                style={{ left: `${getTickDisplayProgress(index)}%` }}
                className={
                  isActiveTick
                    ? "scroll-progress__tick--section scroll-progress__tick--active"
                    : isSectionTick
                      ? "scroll-progress__tick--section"
                      : ""
                }
              />
            );
          })}
        </div>
      </div>

      <button
        type="button"
        className="scroll-progress__thumb"
        style={{ left: `${thumbPosition}%` }}
        onPointerDown={handlePointerDown}
        onFocus={() => setIsEngaged(true)}
        onBlur={() => setIsEngaged(false)}
        aria-label={`Reading progress ${Math.round(progress)} percent`}
      />
    </div>
  );
}
