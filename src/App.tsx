import { useEffect, useMemo, useState } from "react";
import type { ClipboardEvent, DragEvent } from "react";
import {
  MarkdownTextSplitter,
  RecursiveCharacterTextSplitter,
} from "@langchain/textsplitters";
import "./App.css";

type ChunkCard = {
  id: number;
  content: string;
  lineLabel?: string;
  charCount: number;
  cleanBoundary: boolean;
  start: number;
  end: number;
  splitter: "markdown" | "recursive";
};

type Insight = {
  id: string;
  title: string;
  summary: string;
  bestFor: string;
  watchFor: string;
};

type SectionKey = "insights" | "upload" | "markdown" | "recursive";

const ARTICLE_INSIGHTS: Insight[] = [
  {
    id: "01",
    title: "Fixed-size sliding window",
    summary:
      "Deterministic character windows deliver uniform chunks for quick baselines when you just need predictable budgets.",
    bestFor: "Fast baselines or corpora without reliable structure cues.",
    watchFor: "Mid-sentence breaks torpedo recall because semantics are ignored.",
  },
  {
    id: "02",
    title: "Recursive structure-aware",
    summary:
      "LangChain's recursive splitter walks separator lists (headers, paragraphs, sentences) before falling back to characters.",
    bestFor: "Mixed-format docs like release notes or changelogs that blend prose, bullets, and code.",
    watchFor: "If separators are sparse, it degrades into raw character chopping.",
  },
  {
    id: "03",
    title: "Sentence/paragraph splits",
    summary:
      "Lean on author semantics so each chunk preserves a complete idea without fiddling endlessly with chunk knobs.",
    bestFor: "Human-authored essays, newsletters, or reports with trustworthy punctuation.",
    watchFor: "Transcripts/logs lacking punctuation either explode chunk sizes or lose meaning.",
  },
  {
    id: "04",
    title: "Content-aware layering",
    summary:
      "Chain multiple splitters: carve high-level sections first, then tailor chunking logic inside each block.",
    bestFor: "Regulatory filings, API docs, or patterned corpora where sections repeat predictably.",
    watchFor: "Extra complexity + latency when heuristics don't match the document's shape.",
  },
  {
    id: "05",
    title: "Code-aware splitters",
    summary:
      "LangChain's language-specific splitters keep whole functions/classes intact so embeddings retain call-graph intent.",
    bestFor: "Source QA bots, doc generators, or review copilots that need full function scope.",
    watchFor: "Auto-generated or minified files can still overshoot limits and not every language is covered.",
  },
  {
    id: "06",
    title: "Semantic chunking",
    summary:
      "Embed first, then cluster by meaning to locate topic boundaries instead of arbitrary character counts.",
    bestFor: "Dense research papers, specs, or long support tickets with subtle topic shifts.",
    watchFor: "You pay the cost of embedding upfront and model drift can change distance metrics.",
  },
  {
    id: "07",
    title: "Agentic chunking",
    summary:
      "Delegate chunking to an LLM agent that can read the doc, pick splitters, and size overlaps per objective.",
    bestFor: "High-value corpora (earnings calls, compliance docs) where bespoke chunks justify latency and spend.",
    watchFor: "Expensive, slower, and less deterministic unless the agent is tightly constrained.",
  },
];

const SAMPLE_MD = `# Chunking Strategy: why cuts matter

Modern retrieval pipelines live and die by context. When a chunk boundary slices through a sentence, we lose intent: prompts receive fragments instead of facts.

## What an AI engineer cares about
- predictable recall for long-form knowledge
- enough overlap to reassemble meaning
- respecting the document's own structure (headers, lists, code blocks)

### Bad split example
The spacecraft entered Mars orbit. The atmospheric model predicted a...
<-- boundary here -->
... 42% chance of dust storms. Mission control inferred the wrong risk.

### Markdown-aware splitting
Headers describe hierarchical meaning. Keeping them attached to their paragraphs makes embeddings richer and reranking easier.

### Recursive character splitting
Useful when content is noisy or unstructured, but it can still sever phrases if chunk sizes are too aggressive.

## Takeaways
- Tune chunk size and overlap to the retrieval budget.
- Prefer structure-preserving splitters when the file already has signal.
- Inspect the edges of chunks; that's where context usually leaks.`;

const punctuationEnd = /[.!?…:;)]/;
const strongStart = /^[A-Z0-9#>`*-]/;

function assessBoundary(text: string) {
  const trimmed = text.trim();
  if (!trimmed) return false;
  const startClean = strongStart.test(trimmed);
  const endChar = trimmed.at(-1) ?? "";
  const endClean =
    punctuationEnd.test(endChar) || trimmed.endsWith("```") || trimmed.endsWith('"');
  return startClean && endClean;
}

async function runSplit(
  text: string,
  chunkSize: number,
  chunkOverlap: number,
) {
  const recursiveSplitter = new RecursiveCharacterTextSplitter({
    chunkSize,
    chunkOverlap,
    keepSeparator: true,
  });

  const markdownSplitter = new MarkdownTextSplitter({
    chunkSize,
    chunkOverlap,
    keepSeparator: true,
  });

  const [recursiveDocs, markdownDocs] = await Promise.all([
    recursiveSplitter.createDocuments([text]),
    markdownSplitter.createDocuments([text]),
  ]);

  const computeRanges = (docs: any[]) => {
    const ranges: { start: number; end: number }[] = [];
    let start = 0;
    for (const doc of docs) {
      const len = doc.pageContent.length;
      let end = start + len;
      let slice = text.slice(start, end);
      if (slice !== doc.pageContent) {
        // fallback: search near the expected window, then globally
        const nearby = text.indexOf(
          doc.pageContent,
          Math.max(0, start - chunkOverlap * 2),
        );
        const foundStart = nearby !== -1 ? nearby : text.indexOf(doc.pageContent);
        start = foundStart === -1 ? -1 : foundStart;
        end = start === -1 ? -1 : start + len;
        slice = text.slice(start, end);
      }
      ranges.push({ start, end });
      // advance using splitter logic: next chunk starts at prior start + len - overlap
      if (start === -1) {
        start = end; // best effort
      } else {
        start = Math.max(0, start + len - chunkOverlap);
      }
    }
    return ranges;
  };

  const mdRanges = computeRanges(markdownDocs);
  const rcRanges = computeRanges(recursiveDocs);

  const convert = (docs: any[], splitter: "markdown" | "recursive", ranges: { start: number; end: number }[]): ChunkCard[] =>
    docs.map((doc, idx) => {
      const loc = doc.metadata?.loc?.lines;
      const lineLabel =
        loc && typeof loc.from === "number"
          ? `Lines ${loc.from}–${loc.to}`
          : undefined;
      return {
        id: idx,
        content: doc.pageContent,
        charCount: doc.pageContent.length,
        lineLabel,
        cleanBoundary: assessBoundary(doc.pageContent),
        start: ranges[idx]?.start ?? -1,
        end: ranges[idx]?.end ?? -1,
        splitter,
      };
    });

  return {
    recursive: convert(recursiveDocs, "recursive", rcRanges),
    markdown: convert(markdownDocs, "markdown", mdRanges),
  };
}

function App() {
  const [source, setSource] = useState<string>(SAMPLE_MD);
  const [fileName, setFileName] = useState<string>("sample-chunking.md");
  const [chunkSize, setChunkSize] = useState<number>(900);
  const [overlap, setOverlap] = useState<number>(120);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [results, setResults] = useState<{
    recursive: ChunkCard[];
    markdown: ChunkCard[];
  }>({ recursive: [], markdown: [] });
  const [focusRange, setFocusRange] = useState<{
    start: number;
    end: number;
    splitter: "markdown" | "recursive";
    label: string;
  } | null>(null);
  const [collapsedSections, setCollapsedSections] = useState<Record<SectionKey, boolean>>({
    insights: false,
    upload: false,
    markdown: false,
    recursive: false,
  });
  const toggleSection = (section: SectionKey) => {
    setCollapsedSections((prev) => ({
      ...prev,
      [section]: !prev[section],
    }));
  };

  useEffect(() => {
    let cancelled = false;
    setIsLoading(true);
    runSplit(source, chunkSize, overlap)
      .then((payload) => {
        if (!cancelled) setResults(payload);
      })
      .catch((err) => {
        console.error("Split error", err);
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [source, chunkSize, overlap]);

  useEffect(() => {
    setOverlap((prev) => Math.max(0, Math.min(prev, chunkSize - 20)));
  }, [chunkSize]);

  const totalChars = useMemo(() => source.length, [source]);

  const handleFile = async (file?: File) => {
    if (!file) return;
    const text = await file.text();
    setSource(text);
    setFileName(file.name);
  };

  const handleDrop = async (e: DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    const file = e.dataTransfer?.files?.[0];
    await handleFile(file);
  };

  const handlePaste = async (e: ClipboardEvent<HTMLTextAreaElement>) => {
    if (e.clipboardData?.files?.length) {
      e.preventDefault();
      await handleFile(e.clipboardData.files[0]);
    }
  };

  const highlightColor =
    focusRange?.splitter === "markdown"
      ? "var(--accent)"
      : "var(--amber)";

  const renderSource = () => {
    if (!focusRange || focusRange.start < 0 || focusRange.end < 0)
      return <pre className="source-view">{source}</pre>;
    return (
      <pre className="source-view">
        <span>{source.slice(0, focusRange.start)}</span>
        <span className="highlight" style={{ backgroundColor: highlightColor }}>
          {source.slice(focusRange.start, focusRange.end)}
        </span>
        <span>{source.slice(focusRange.end)}</span>
      </pre>
    );
  };

  return (
    <div className="page">
      <header className="hero">
        <div>
          <p className="eyebrow">Chunking Strategy • AI Engineer playbook</p>
          <h1>Context survives—or snaps—when you slice Markdown.</h1>
          <p className="lede">
            Compare <strong>RecursiveCharacterTextSplitter</strong> vs{" "}
            <strong>MarkdownTextSplitter</strong> (LangChain JS). Upload a large
            .md, tweak chunk size/overlap, and inspect where edges cut sentences
            in half.
          </p>
          <div className="metrics">
            <div>
              <span className="label">Active file</span>
              <p className="value">{fileName}</p>
            </div>
            <div>
              <span className="label">Characters</span>
              <p className="value">{totalChars.toLocaleString()}</p>
            </div>
            <div>
              <span className="label">Chunks</span>
              <p className="value">
                MD {results.markdown.length} • RC {results.recursive.length}
              </p>
            </div>
          </div>
        </div>
      </header>

      <section className="insights">
        <div className="card insights-card">
          <div className="panel-head">
            <div>
              <p className="section-title">Field guide: 7 chunking plays</p>
            </div>
            <div className="panel-head-actions">
              <span className="chip subtle">Educational insights</span>
              <button
                type="button"
                className="collapse-toggle"
                onClick={() => toggleSection("insights")}
                aria-expanded={!collapsedSections.insights}
                aria-controls="insights-panel"
              >
                {collapsedSections.insights ? "Expand" : "Collapse"}
              </button>
            </div>
          </div>
          <div
            id="insights-panel"
            className="insight-grid collapsible-content"
            hidden={collapsedSections.insights}
            aria-hidden={collapsedSections.insights}
          >
            {ARTICLE_INSIGHTS.map((insight) => (
              <article key={insight.id} className="insight">
                <div className="insight-head">
                  <span className="chip subtle">{insight.id}</span>
                  <h3>{insight.title}</h3>
                </div>
                <p className="insight-summary">{insight.summary}</p>
                <div className="insight-meta">
                  <p>
                    <span className="meta-label">Best for</span>
                    {insight.bestFor}
                  </p>
                  <p>
                    <span className="meta-label">Watch for</span>
                    {insight.watchFor}
                  </p>
                </div>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="panel-grid compact">
        <div className="card upload">
          <div className="panel-head">
            <div>
              <p className="section-title">Load Markdown</p>
              <p className="muted">
                Drop a file, paste, or edit inline. Everything stays in the
                browser.
              </p>
            </div>
            <div className="panel-head-actions">
              <button
                type="button"
                className="collapse-toggle"
                onClick={() => toggleSection("upload")}
                aria-expanded={!collapsedSections.upload}
                aria-controls="upload-panel"
              >
                {collapsedSections.upload ? "Expand" : "Collapse"}
              </button>
            </div>
          </div>
          <div
            id="upload-panel"
            className="collapsible-stack collapsible-content"
            hidden={collapsedSections.upload}
            aria-hidden={collapsedSections.upload}
          >
            <div className="upload-row">
              <label
                className="dropzone"
                onDrop={handleDrop}
                onDragOver={(e) => e.preventDefault()}
              >
                <input
                  type="file"
                  accept=".md, text/markdown, .txt"
                  onChange={(e) => handleFile(e.target.files?.[0])}
                />
                <div>
                  <p className="big">Drop markdown</p>
                  <p className="muted">or click</p>
                </div>
              </label>
              <div className="controls mini">
                <div className="control tight">
                  <label htmlFor="chunkSize">Chunk size</label>
                  <div className="slider-row">
                    <input
                      id="chunkSize"
                      type="range"
                      min={300}
                      max={1800}
                      step={50}
                      value={chunkSize}
                      onChange={(e) => setChunkSize(Number(e.target.value))}
                    />
                    <span className="chip">{chunkSize}</span>
                  </div>
                </div>
                <div className="control tight">
                  <label htmlFor="overlap">Overlap</label>
                  <div className="slider-row">
                    <input
                      id="overlap"
                      type="range"
                      min={0}
                      max={Math.max(0, chunkSize - 20)}
                      step={10}
                      value={overlap}
                      onChange={(e) => setOverlap(Number(e.target.value))}
                    />
                    <span className="chip">{overlap}</span>
                  </div>
                </div>
              </div>
            </div>
            <div className="source-pane">
              <div className="source-meta">
                <span className="chip subtle">{fileName}</span>
                {focusRange && (
                  <span className="chip" style={{ background: highlightColor }}>
                    Highlight: {focusRange.label}
                  </span>
                )}
              </div>
              <textarea
                className="editor mini"
                value={source}
                onChange={(e) => setSource(e.target.value)}
                onPaste={handlePaste}
                spellCheck={false}
                aria-label="Markdown editor"
              />
              {renderSource()}
            </div>
          </div>
        </div>
      </section>

      <section className="results">
        <div className="column">
          <div className="column-header">
            <div>
              <p className="eyebrow">MarkdownTextSplitter</p>
              <p className="muted">
                Honors Markdown structure (headers, fences) to keep semantic
                blocks intact.
              </p>
            </div>
            <button
              type="button"
              className="collapse-toggle"
              onClick={() => toggleSection("markdown")}
              aria-expanded={!collapsedSections.markdown}
              aria-controls="markdown-panel"
            >
              {collapsedSections.markdown ? "Expand" : "Collapse"}
            </button>
          </div>
          <div
            id="markdown-panel"
            className="chunk-list collapsible-content"
            hidden={collapsedSections.markdown}
            aria-hidden={collapsedSections.markdown}
          >
            {results.markdown.map((chunk) => (
              <article
                key={`md-${chunk.id}`}
                className={`chunk ${chunk.cleanBoundary ? "clean" : "risky"}`}
                onMouseEnter={() =>
                  setFocusRange({
                    start: chunk.start,
                    end: chunk.end,
                    splitter: "markdown",
                    label: `MD #${chunk.id + 1}`,
                  })
                }
                onMouseLeave={() => setFocusRange(null)}
                onClick={() =>
                  setFocusRange({
                    start: chunk.start,
                    end: chunk.end,
                    splitter: "markdown",
                    label: `MD #${chunk.id + 1}`,
                  })
                }
              >
                <div className="chunk-meta">
                  <span className="chip subtle">#{chunk.id + 1}</span>
                  <span className="chip subtle">
                    {chunk.charCount.toLocaleString()} chars
                  </span>
                  {chunk.lineLabel && (
                    <span className="chip subtle">{chunk.lineLabel}</span>
                  )}
                  <span className={`chip ${chunk.cleanBoundary ? "ok" : "warn"}`}>
                    {chunk.cleanBoundary ? "Edge looks clean" : "Likely mid-sentence"}
                  </span>
                </div>
                <pre>{chunk.content}</pre>
              </article>
            ))}
          </div>
        </div>

        <div className="column">
          <div className="column-header">
            <div>
              <p className="eyebrow">RecursiveCharacterTextSplitter</p>
              <p className="muted">
                Splits by separators in order; solid for mixed content but can
                still slice sentences.
              </p>
            </div>
            <button
              type="button"
              className="collapse-toggle"
              onClick={() => toggleSection("recursive")}
              aria-expanded={!collapsedSections.recursive}
              aria-controls="recursive-panel"
            >
              {collapsedSections.recursive ? "Expand" : "Collapse"}
            </button>
          </div>
          <div
            id="recursive-panel"
            className="chunk-list collapsible-content"
            hidden={collapsedSections.recursive}
            aria-hidden={collapsedSections.recursive}
          >
            {results.recursive.map((chunk) => (
              <article
                key={`rc-${chunk.id}`}
                className={`chunk ${chunk.cleanBoundary ? "clean" : "risky"}`}
                onMouseEnter={() =>
                  setFocusRange({
                    start: chunk.start,
                    end: chunk.end,
                    splitter: "recursive",
                    label: `RC #${chunk.id + 1}`,
                  })
                }
                onMouseLeave={() => setFocusRange(null)}
                onClick={() =>
                  setFocusRange({
                    start: chunk.start,
                    end: chunk.end,
                    splitter: "recursive",
                    label: `RC #${chunk.id + 1}`,
                  })
                }
              >
                <div className="chunk-meta">
                  <span className="chip subtle">#{chunk.id + 1}</span>
                  <span className="chip subtle">
                    {chunk.charCount.toLocaleString()} chars
                  </span>
                  {chunk.lineLabel && (
                    <span className="chip subtle">{chunk.lineLabel}</span>
                  )}
                  <span className={`chip ${chunk.cleanBoundary ? "ok" : "warn"}`}>
                    {chunk.cleanBoundary ? "Edge looks clean" : "Likely mid-sentence"}
                  </span>
                </div>
                <pre>{chunk.content}</pre>
              </article>
            ))}
          </div>
        </div>
      </section>

      {isLoading && <div className="loading">Re-chunking…</div>}
    </div>
  );
}

export default App;
