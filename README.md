# Why Does Chunking Break My RAG?

Context fragmentation is the silent killer of retrieval-augmented generation. When a splitter slices a paragraph mid-thought, you embed and retrieve fragments that no longer contain the evidence a model needs to answer coherently. The result is hallucinated endings, contradictory citations, and "no idea" responses even though the source text is in your vector store. Compare Chunker exists to make that failure mode tangible and to demonstrate mitigation strategies the moment someone asks, "Why did my RAG bot miss this?"

## The Problem

- **Fragmented context** - naive fixed-width chunks cut sentences in half, so similarity search retrieves shards that describe symptoms without remedies or requirements without constraints.
- **Strategy blind spots** - teams tune only chunk size or overlap, ignoring structure-aware or semantic splitters that better respect document intent.
- **Dirty inputs** - logs, scraped HTML, and CMS exports ship with scripts, tables, and inline styles; Markdown splitters choke before chunking even begins.
- **No quantitative signal** - most demos rely on "looks good" eyeballing instead of metrics that prove one strategy preserves intent better than another.

## How Compare Chunker Demonstrates Mastery

| Capability | Why it matters for AI Engineers |
| --- | --- |
| **Tri-splitter lab** (MarkdownHeaderSplitter, RecursiveCharacterTextSplitter, MiniLM semantic grouping) | Shows how structural versus embedding-informed splitting changes the boundaries you index. |
| **Chunk-to-source heatmap** | Hover any chunk to spotlight the exact span in the raw document so you can see precisely where context fractures. |
| **Retrieval simulation** | Type a query, run cosine similarity per strategy, and instantly see which chunks fire and whether the answer arrived fragmented. |
| **Quality metrics dashboard** | Broken-sentence percentage, average chunk length, and standard deviation provide hard numbers for trade-off discussions. |
| **Dirty-input lab** | Paste raw HTML, toggle cleaning, and watch Markdown splitters fail or recover; demonstrates ETL awareness before embedding. |
| **Snapshot persistence and A/B testing** | Save configurations, reload after refresh, and compare variants side-by-side to mimic how real RAG teams iterate. |
| **MiniLM preload plus cache insight** | Pre-warms the 85 MB embedding model on mount and surfaces cache state, signaling observability instincts beyond UI polish. |

## Workflow Walkthrough

1. **Load or paste source text** - either pristine Markdown or noisy HTML. Cleaning helpers strip tags and decode entities so you can contrast before vs after.
2. **Tune splitters** - adjust chunk size, overlap, recursion type, and semantic cohesion threshold. The lab recomputes in real time.
3. **Inspect fragmentation** - each column keeps chunks color-coded, counts overlaps, and lets you click to highlight the originating text region.
4. **Quantify quality** - the metrics panel recalculates broken edges and variance for every new configuration.
5. **Simulate retrieval** - enter a user question, run cosine similarity with the cached MiniLM embeddings, and review per-strategy hit lists plus fragmentation alerts.
6. **Save snapshots for A/B** - persist Variant A/B, compare deltas (chunk counts, cleanliness, retrieval scores), and narrate the business trade-offs confidently.

## Tech Stack Highlights

- React + Vite + TypeScript with strict typing, flat ESLint, and hot reloads for rapid UX iteration.
- LangChain splitters client-side to mirror production RAG pipelines without a server dependency.
- `@xenova/transformers` MiniLM embeddings (ONNX) to keep semantic chunking and retrieval entirely in-browser.
- LocalStorage snapshotting plus JSON session logs so every UX session and CLI conversation is auditable.

## Running the Lab Locally

```bash
npm install     # sync deps (LangChain, transformers.js, Vite)
npm run dev     # start the playground on http://localhost:5173
npm run build   # type-check + production bundle (transformers chunk > 500 kB)
npm run lint    # enforce React, Hooks, and TS rules
```

## Talking Points For Your Portfolio

- **Context fragmentation elevator pitch** - show the retrieval simulation where the Markdown splitter produces two hits for a single answer while recursive or semantic keeps the narrative intact.
- **Data quality first** - walk through the raw HTML input, demonstrate how cleaning changes the broken-edge metric, and relate that to real ETL pipelines.
- **Instrumentation mindset** - highlight the MiniLM preload/cache chip and metrics dashboard to prove you care about latency, observability, and scientific comparison.
- **Iteration culture** - use the snapshot board to explain how RAG teams run experiments, capture baselines, and justify configuration changes with numbers.

## Roadmap Ideas

- Add latency tracking per splitter or embedding path to tie UX perception to measurable performance.
- Layer in Vitest plus Testing Library once the chunking helpers stabilize to keep regressions out of production.
- Offer exportable JSON or CSV reports so consultants can attach empirical evidence directly to client proposals.

---

Compare Chunker is more than a demo. It is a narrative device to prove you understand why chunking breaks RAG systems and how to diagnose, visualize, and remediate that risk like an AI Engineer.
