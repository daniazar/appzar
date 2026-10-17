---
title: "Google Flow MCP: Reverse-Engineering Google Labs Flow into a Two-Tier tRPC & Playwright MCP Server"
date: "2026-09-14"
excerpt: "How I built a 36-tool Model Context Protocol (MCP) server for Google Labs Flow (Veo 3.1 & Imagen 4), evolving from fragile DOM automation to a verified Tier-1 direct HTTP/tRPC client with spend ledgers and sharp-frame extraction."
category: "AI & Developer Tools"
tags: ["MCP", "Google Flow", "Veo", "Playwright", "TypeScript", "tRPC", "Video Generation"]
coverImage: "/images/covers/google-flow-mcp-trpc-video-server.webp"
---

**Google Labs Flow** (`labs.google/fx/tools/flow`) is one of the most capable interfaces for generating cinema-grade video with **Veo 3.1** (`veo_3_1_t2v_fast_ultra`) and **Imagen 4**, especially when paired with a Google AI Ultra subscription. However, Flow was built strictly as a web application—meaning autonomous coding agents, filmmaking pipelines like **CineZero**, and local scripts had no programmatic SDK to drive it.

To bridge that gap, I built **[google-flow-mcp](https://github.com/daniazar/google-flow-mcp)**—a 36-tool **Model Context Protocol (MCP)** server in TypeScript that exposes Google Flow's entire project, keyframe, video generation, transition, upscaling, and budget lifecycle to any MCP-compatible AI agent.

![Google Labs Flow session driven autonomously by google-flow-mcp during CineZero's Lighthouse production](/assets/flow-mcp/flow-session.webp)

---

## Evolving from DOM Clicking to a Two-Tier Architecture

Early prototypes of browser automation against rich creative web apps always hit the same wall: React/Lit virtualized grids recycle DOM nodes on scroll, hidden file inputs report success before an upload finishes processing, and UI redesigns break CSS selectors overnight.

To make `google-flow-mcp` production-grade, I designed a **Two-Tier Execution Engine** (`src/flow-manager.ts` and `src/http-client.ts`):

### Tier 1: Direct HTTP / tRPC & AISandbox API (`FlowHttpClient`)
By building an automated network-traffic discovery tool (`flow_discover_api` and `flow_verify_http_tier`), I mapped Flow's internal tRPC procedures (`https://labs.google/fx/api/trpc/*`) and Google's `aisandbox-pa.googleapis.com/v1` endpoints.
* **Zero UI Flakiness**: Listing projects (`flow_list_projects`), querying media libraries (`flow_list_media`), checking credit balances (`flow_budget`), and polling generation statuses happen over direct authenticated HTTP calls using the browser context's cookies and bearer tokens—executing in milliseconds without touching the DOM.

### Tier 2: Persistent Playwright Chromium Fallback (`FlowManager`)
For actions that require live browser session state or complex canvas/upload interactions, `FlowManager` maintains a persistent, logged-in Chromium profile (`~/.flow-mcp/chrome-profile`) and verifies every state transition before returning:

![Google Flow project workspace managed via google-flow-mcp](/assets/flow-mcp/flow-project.webp)

---

## Hard-Won Engineering Lessons from Real Productions

Building an MCP server that survives real multi-shot film productions surfaced several subtle failure modes that `google-flow-mcp` solves directly:

1. **Verifying Chip Assignment by Media ID, Not Upload Return**:
   In Flow's UI, uploading an image via `flow_upload_frame` can return HTTP 200 while leaving the `Start` frame chip empty in the prompt bar—causing Veo to silently generate an unconditioned text-to-video clip and waste credits. `google-flow-mcp` splits upload from assignment (`flow_assign_frame`) and inspects the prompt state to verify the exact media UUID is attached before allowing generation to proceed.
2. **Sharpest-Frame Seam Chaining with FFmpeg**:
   When chaining two 8-second Veo clips together, `google-flow-mcp` uses FFmpeg (`src/ffmpeg.ts`) to extract candidate frames from the tail of Clip $N$, scores them for sharpness, and feeds the cleanest frame into Clip $N+1$.
3. **Built-In Credit Budget & Spend Ledger (`src/ledger.ts`)**:
   Every tool call that consumes Flow credits (`flow_generate_video`, `flow_generate_transition`, `flow_upscale`) logs the pre-call and post-call credit delta to a persistent local ledger (`flow_ledger` and `flow_budget`), ensuring an autonomous agent can never silently drain a user's monthly quota.

---

## Source Code

The complete 36-tool MCP server, tRPC discovery harness, and Vitest test suite are available on GitHub at **[daniazar/google-flow-mcp](https://github.com/daniazar/google-flow-mcp)**.
