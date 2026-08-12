import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async redirects() {
    return [
      // Preserve equity from article and glossary URLs that were renamed or
      // removed while keeping every editorial link useful to readers.
      { source: "/glossary/gemma", destination: "/articles/google-gemma-4-open-source-ai-apache", permanent: true },
      { source: "/glossary/gpt-2", destination: "/glossary/large-language-model", permanent: true },
      { source: "/glossary/razorpay", destination: "/articles/razorpay-agent-studio-ai-payments-platform", permanent: true },
      { source: "/glossary/agent-framework", destination: "/glossary/ai-agent", permanent: true },
      { source: "/glossary/multi-agent-system", destination: "/glossary/agent-orchestration", permanent: true },
      { source: "/articles/minimax-m3-glm52-deepseek-v4-chinese-labs-competing", destination: "/articles/real-ai-race-minimax-m3-glm-52-deepseek-v4", permanent: true },
      { source: "/articles/grok-45-opus-class-claim-cursor-acquisition", destination: "/articles/grok-4-5-opus-adjacent-cursor-acquisition", permanent: true },
      { source: "/articles/claude-mythos-apple-m5-security-exploit", destination: "/articles/claude-mythos-apple-m5-security-mie", permanent: true },
      { source: "/guides/how-to-build-ai-agents-2026", destination: "/guides/you-want-to-build-an-ai-agent-here-is-where-to-actually-start", permanent: true },
      { source: "/articles/minimax-m3-open-weight-frontier-coding", destination: "/articles/minimax-m3-open-weight-frontier-coding-weights-not-out", permanent: true },
      { source: "/articles/deepswe-benchmark-claude-opus-loophole-gpt55", destination: "/articles/deepswe-benchmark-claude-loophole-gpt-55-coding-leader", permanent: true },
      { source: "/articles/project-glasswing-anthropic-mythos", destination: "/articles/anthropic-project-glasswing-mythos-cybersecurity", permanent: true },
      { source: "/articles/chatgpt-images-2-industry-shift", destination: "/articles/the-image-that-doesnt-look-like-ai-anymore", permanent: true },
      { source: "/articles/vibevoice-distribution-lag-startup-lesson", destination: "/articles/microsoft-vibevoice-open-voice-ai-distribution-gap", permanent: true },
      { source: "/articles/qwen36-27b-local-first-paradigm", destination: "/articles/qwen-37-max-closed-source-alibaba-playbook", permanent: true },
      { source: "/articles/subquadratic-12-million-context-window", destination: "/articles/transformer-9-year-ceiling-subquadratic-subq", permanent: true },
      { source: "/articles/gemini-omni-35-flash-google-io-2026", destination: "/articles/gemini-omni-gemini-3-5-flash-google-io-2026", permanent: true },
      { source: "/articles/open-source-ai-race-gemma-4", destination: "/articles/google-gemma-4-open-source-ai-apache", permanent: true },
      { source: "/articles/claude-opus-47-race-no-finish-line", destination: "/articles/claude-opus-4-7-proved-the-race-has-no-finish-line", permanent: true },
      { source: "/articles/claude-code-moat-disappeared", destination: "/articles/the-day-claude-codes-moat-disappeared", permanent: true },
      { source: "/articles/mcp-anthropic-standard-tools", destination: "/glossary/mcp", permanent: true },
      { source: "/articles/web3-iot-ai-operating-system", destination: "/articles/the-next-layer-how-ai-is-moving-from-your-screen-to-your-world", permanent: true },
    ];
  },
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "X-Frame-Options", value: "SAMEORIGIN" },
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          {
            key: "Permissions-Policy",
            value: "camera=(), microphone=(), geolocation=()",
          },
        ],
      },
    ];
  },
};

export default nextConfig;
