# Project Context: Compare Chunker

## Project Overview

**Compare Chunker** is a React-based tool designed to visualize and compare different text chunking strategies for Retrieval-Augmented Generation (RAG) pipelines. It serves as a "lab" for AI engineers to understand how different splitters (Markdown, Recursive Character, Semantic) fragment context and to tune parameters like chunk size and overlap.

**Key Features:**
*   **Tri-Splitter Comparison:** Visualizes MarkdownHeader, RecursiveCharacter, and Semantic splitting side-by-side.
*   **Chunk-to-Source Heatmap:** Highlights the original text span when hovering over a chunk.
*   **Local-First Architecture:** Runs all processing, including embeddings (via `@xenova/transformers`), directly in the browser. No data leaves the client.
*   **Retrieval Simulation:** Simulates cosine similarity search to test retrieval quality.

## Tech Stack

*   **Frontend Framework:** React 19 + Vite + TypeScript
*   **Text Processing:** LangChain (client-side splitters)
*   **Embeddings:** `@xenova/transformers` (ONNX models running in-browser)
*   **Build Tooling:** Vite, ESLint

## Architecture & Design Decisions

Refer to `ARCHITECTURE.md` for deep dives.

1.  **Client-Side Processing:** All text splitting and embedding generation happens in the browser. This ensures privacy, zero latency, and zero operational cost.
2.  **Performance Optimization:**
    *   Extensive use of `useMemo` to decouple expensive calculations (metric generation) from render cycles.
    *   Asynchronous `runSplit` function using `Promise.all` to run splitters concurrently without freezing the UI.
3.  **State Management:**
    *   `Promise.allSettled` is used to handle multiple splitter operations, ensuring robustness if one fails.
    *   Race condition handling prevents stale results from overwriting new ones during rapid input changes.

## Building and Running

**Prerequisites:** Node.js (v20+ recommended)

| Command | Description |
| :--- | :--- |
| `npm install` | Install dependencies (LangChain, transformers.js, React, Vite) |
| `npm run dev` | Start the local development server at `http://localhost:5173` |
| `npm run build` | Run TypeScript checks and build for production |
| `npm run lint` | Run ESLint to enforce code style |
| `npm run preview` | Preview the production build locally |

## Directory Structure

*   **`src/`**: Source code.
    *   `App.tsx`: Main application logic, state management, and UI composition.
    *   `main.tsx`: Entry point.
*   **`public/`**: Static assets.
*   **`ARCHITECTURE.md`**: Detailed architectural decisions.
*   **`AGENTS.md` / `CASE_STUDIES.md`**: Project-specific documentation/context.

## Development Conventions

*   **Code Style:** strict TypeScript, ESLint with React hooks rules.
*   **Async Logic:** Use `Promise.all` / `Promise.allSettled` for parallel processing. Handle race conditions in `useEffect` hooks.
*   **Testing:** Currently relying on manual testing (visual comparison), with Vitest planned for the future.
