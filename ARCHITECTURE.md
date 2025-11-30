# Architecture & Technical Decisions

This document outlines the high-level architectural decisions for the Compare Chunker application, focusing on the rationale behind client-side processing, state management strategies, and asynchronous flow control.

## 1. Client-Side LangChain Processing

The core design philosophy of this application is **"Local First"**. We deliberately chose to run LangChain's text splitters (`RecursiveCharacterTextSplitter`, `MarkdownTextSplitter`) directly in the browser rather than offloading this work to a backend API.

### Justification

*   **Privacy**: User data (documents, code snippets, sensitive logs) never leaves the client's machine. There is no risk of data leakage during transit or storage on our servers.
*   **Latency**: By eliminating network round-trips, feedback is immediate. Users can adjust chunk sizes and overlaps and see the visual impact instantly, creating a tight feedback loop essential for tuning RAG pipelines.
*   **Zero Cost**: Leveraging the user's device for compute eliminates the need for scalable backend infrastructure to handle text processing, resulting in zero operational costs for the chunking logic.

## 2. State & Performance Management

Handling large text documents and visualizing hundreds of chunks requires careful performance optimization to maintain 60fps UI responsiveness. We utilize React's `useMemo` hook extensively in `App.tsx` to decouple expensive calculations from the render cycle.

### Efficient Metric Recalculation

We use `useMemo` to derive state that depends on the input text or configuration, ensuring these are only recalculated when their specific dependencies change:

*   **`workingText`**: Memoized to handle switching between "Markdown" and "Raw HTML" modes. It cleans HTML only when the source or cleaning toggle changes, not on every render.
*   **`totalChars`**: Derived from `workingText`. Calculating string length is cheap, but memoizing it establishes a clear dependency chain.
*   **`qualityStats`**: The calculation of standard deviation and broken chunk percentages is computationally intensive (looping through all chunks). This is memoized so that UI updates (like toggling a panel) do not trigger a re-analysis of the chunk quality.

```typescript
// Example from App.tsx
const qualityStats = useMemo<QualityStats>(() => buildQualityStats(results), [results]);
```

This ensures that while the user interacts with the UI (e.g., hovering over chunks, switching tabs), the application does not waste cycles re-computing static metrics.

## 3. Asynchronous Flow: `runSplit`

To prevent the main thread from freezing during the processing of large documents, the chunking logic is encapsulated in an asynchronous `runSplit` function.

### Parallel Execution

The `runSplit` function orchestrates the execution of multiple splitters concurrently. We use `Promise.all` to run the `RecursiveCharacterTextSplitter` and `MarkdownTextSplitter` in parallel.

```typescript
async function runSplit(text: string, chunkSize: number, chunkOverlap: number) {
  // ... splitter initialization ...

  const [recursiveDocs, markdownDocs] = await Promise.all([
    recursiveSplitter.createDocuments([text]),
    markdownSplitter.createDocuments([text]),
  ]);

  // ... post-processing ...
}
```

### Promise Handling

In the main `useEffect` hook, we use `Promise.allSettled` to manage the lifecycle of these operations along with the semantic chunking (which runs via a separate worker/pipeline).

*   **Robustness**: `Promise.allSettled` ensures that if one splitter fails (e.g., a semantic embedding error), the others still return results. The UI can display partial results rather than crashing entirely.
*   **Race Condition Handling**: A `cancelled` flag pattern is used within the `useEffect` to ignore results from stale promises if the user has already changed the input (e.g., typing fast), preventing "glitching" where old results overwrite newer ones.
