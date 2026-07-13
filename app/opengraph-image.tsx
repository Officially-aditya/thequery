import { ImageResponse } from "next/og";

export const alt = "TheQuery - AI knowledge from first principles";
export const size = {
  width: 1200,
  height: 630,
};
export const contentType = "image/png";

export default function OpenGraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          background: "#f7f3ea",
          color: "#172033",
          padding: "64px 72px",
          border: "18px solid #002147",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "18px",
              fontSize: 42,
              fontWeight: 700,
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                width: 74,
                height: 74,
                borderRadius: 14,
                background: "#002147",
                color: "#f7f3ea",
                fontFamily: "monospace",
                fontSize: 34,
              }}
            >
              &gt;_
            </div>
            TheQuery
          </div>
          <div
            style={{
              display: "flex",
              color: "#526078",
              fontSize: 24,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
            }}
          >
            thequery.in
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
          <div
            style={{
              display: "flex",
              maxWidth: 940,
              fontFamily: "serif",
              fontSize: 76,
              fontWeight: 700,
              lineHeight: 1.08,
              letterSpacing: "-0.035em",
            }}
          >
            AI knowledge from first principles.
          </div>
          <div
            style={{
              display: "flex",
              color: "#526078",
              fontSize: 30,
              lineHeight: 1.35,
            }}
          >
            Glossary, guides, books, and sharp analysis for people building with AI.
          </div>
        </div>

        <div
          style={{
            display: "flex",
            width: "100%",
            height: 8,
            borderRadius: 999,
            background: "#c8612c",
          }}
        />
      </div>
    ),
    size,
  );
}
