import { useEffect, useMemo, useState } from "react";
import type { ClipboardEvent, DragEvent } from "react";
import {
  MarkdownTextSplitter,
  RecursiveCharacterTextSplitter,
} from "@langchain/textsplitters";
import salesLetterDoc from "../README.md?url";
import caseStudiesDoc from "../CASE_STUDIES.md?url";
import "./App.css";

type ChunkCard = {
  id: number;
  content: string;
  lineLabel?: string;
  charCount: number;
  cleanBoundary: boolean;
  start: number;
  end: number;
  splitter: "markdown" | "recursive" | "semantic";
};

type Insight = {
  id: string;
  title: string;
  summary: string;
  bestFor: string;
  watchFor: string;
};

type SectionKey = "insights" | "upload" | "markdown" | "recursive" | "semantic";

type InputMode = "markdown" | "rawHtml";

type SemanticOptions = {
  text: string;
  chunkSize: number;
  similarity: number;
};

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

const DOC_LINKS = [
  {
    id: "sales-letter",
    title: "Sales letter: Why chunking breaks my RAG",
    description: "Answers the core \"Why\" with the portfolio-ready narrative in README.md.",
    badge: "Playbook",
    href: salesLetterDoc,
  },
  {
    id: "case-studies",
    title: "Case studies: Context fragmentation in the wild",
    description: "Quantifies script splits, orphaned lists, and semantic fixes via CASE_STUDIES.md.",
    badge: "Data evidence",
    href: caseStudiesDoc,
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

const SAMPLE_RAW_HTML = `<div class="release-note">
  <p><strong>🚧 Deployment log</strong></p>
  <p>The crawler dumped raw <span style="color:#f66;">HTML</span> with inline scripts. Here's a slice:</p>
  <ul>
    <li><code>&lt;div class="summary"&gt;</code>Mission log shows <em>dust storms</em></li>
    <li><code>&lt;span data-ts="03:42"&gt;</code>Engineers flagged boundary issues</li>
  </ul>
  <script type="text/javascript">console.log('noisy inline widget')</script>
  <p>Without cleanup, Markdown splitters see tags instead of prose.</p>
</div>`;

const punctuationEnd = /[.!?…:;)]/;
const strongStart = /^[A-Z0-9#>`*-]/;
const DEFAULT_SEMANTIC_THRESHOLD = 0.82;
const MIN_SEMANTIC_THRESHOLD = 0.5;
const MAX_SEMANTIC_THRESHOLD = 0.98;
const SEMANTIC_BATCH_SIZE = 8;
const RETRIEVAL_TOP_K = 3;
const RETRIEVAL_SCORE_FLOOR = 0.58;
const MODEL_CACHE_KEY = "compare-chunker-minilm-ready";
const MODEL_SIZE_MB = 85;

function assessBoundary(text: string) {
  const trimmed = text.trim();
  if (!trimmed) return false;
  const startClean = strongStart.test(trimmed);
  const endChar = trimmed.at(-1) ?? "";
  const endClean =
    punctuationEnd.test(endChar) || trimmed.endsWith("```") || trimmed.endsWith('"');
  return startClean && endClean;
}

type SentenceSegment = {
  text: string;
  start: number;
  end: number;
};

type SemanticBuilder = {
  start: number;
  end: number;
  vector: number[];
  count: number;
  sentenceStart: number;
  sentenceEnd: number;
};

type EmbeddingResult =
  | number
  | Float32Array
  | number[]
  | {
      data?: Float32Array | number[];
    };

type LangChainDoc = {
  pageContent: string;
  metadata?: {
    loc?: {
      lines?: {
        from?: number;
        to?: number;
      };
    };
  };
};

type ModelStatus = {
  state: "idle" | "preloading" | "ready" | "error";
  origin: "unknown" | "cache" | "network";
  message?: string;
};

type ChunkVectors = Record<ChunkCard["splitter"], number[][]>;

type QualityMetric = {
  sampleSize: number;
  brokenPercent: number | null;
  stdDeviation: number | null;
  meanSize: number | null;
};

type QualityStats = Record<ChunkCard["splitter"], QualityMetric>;

type RetrievalResult = {
  scores: number[];
  topIndices: number[];
  fragmented: boolean;
  contiguous: boolean;
  bestScore: number;
};

type RetrievalMatches = Record<ChunkCard["splitter"], RetrievalResult | null>;

const RETRIEVAL_SPLITTERS: ChunkCard["splitter"][] = [
  "markdown",
  "recursive",
  "semantic",
];

const SPLITTER_LABELS: Record<ChunkCard["splitter"], string> = {
  markdown: "Markdown",
  recursive: "Recursive",
  semantic: "Semantic",
};

type Snapshot = {
  id: string;
  name: string;
  createdAt: number;
  config: {
    chunkSize: number;
    overlap: number;
    semanticThreshold: number;
    inputMode: InputMode;
    shouldCleanRawHtml: boolean;
    retrievalQuery: string;
  };
  files: {
    source: string;
    rawSource: string;
    fileName: string;
    rawFileName: string;
  };
  results: Record<ChunkCard["splitter"], ChunkCard[]>;
  qualityStats: QualityStats;
  retrievalMatches: RetrievalMatches;
  summary: {
    workingChars: number;
  };
};

const SNAPSHOT_STORAGE_KEY = "compare-chunker-snapshots";

type FeatureExtractionPipeline = (
  input: string | string[],
  options?: Record<string, unknown>,
) => Promise<EmbeddingResult | EmbeddingResult[]>;

let embeddingPipelinePromise: Promise<FeatureExtractionPipeline> | null = null;

function clampSemanticSimilarity(value: number) {
  if (Number.isNaN(value)) return DEFAULT_SEMANTIC_THRESHOLD;
  return Math.min(MAX_SEMANTIC_THRESHOLD, Math.max(MIN_SEMANTIC_THRESHOLD, value));
}

async function loadEmbeddingPipeline() {
  if (!embeddingPipelinePromise) {
    embeddingPipelinePromise = (async () => {
      const transformers = await import("@xenova/transformers");
      const pipelineFactory = transformers.pipeline as unknown as (
        task: string,
        model?: string,
      ) => Promise<FeatureExtractionPipeline>;
      if (transformers.env) {
        transformers.env.allowLocalModels = false;
        transformers.env.useBrowserCache = true;
      }
      return pipelineFactory("feature-extraction", "Xenova/all-MiniLM-L6-v2");
    })();
  }
  return embeddingPipelinePromise;
}

function tensorToArray(tensor: EmbeddingResult) {
  if (typeof tensor === "number") return [tensor];
  if (Array.isArray(tensor)) return tensor;
  if (tensor instanceof Float32Array) return Array.from(tensor);
  if (tensor && typeof tensor === "object" && "data" in tensor) {
    const payload = tensor.data;
    if (Array.isArray(payload)) return payload;
    if (payload instanceof Float32Array) return Array.from(payload);
  }
  return [] as number[];
}

function ensureEmbeddingArray(value: EmbeddingResult | EmbeddingResult[]) {
  return Array.isArray(value) ? value : [value];
}

async function embedTexts(texts: string[]) {
  if (!texts.length) return [] as number[][];
  const pipeline = await loadEmbeddingPipeline();
  const vectors: number[][] = [];
  for (let i = 0; i < texts.length; i += SEMANTIC_BATCH_SIZE) {
    const slice = texts.slice(i, i + SEMANTIC_BATCH_SIZE);
    const output = await pipeline(slice, { pooling: "mean", normalize: true });
    const tensors = ensureEmbeddingArray(output);
    tensors.forEach((tensor) => {
      vectors.push(tensorToArray(tensor));
    });
  }
  return vectors;
}

function analyzeFragmentation(indices: number[]) {
  if (indices.length <= 1) {
    return { fragmented: false, contiguous: true };
  }
  const sorted = [...indices].sort((a, b) => a - b);
  let contiguous = true;
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i] !== sorted[i - 1] + 1) {
      contiguous = false;
      break;
    }
  }
  return { fragmented: true, contiguous };
}

function computeQualityMetric(chunks: ChunkCard[]): QualityMetric {
  if (!chunks.length) {
    return {
      sampleSize: 0,
      brokenPercent: null,
      stdDeviation: null,
      meanSize: null,
    };
  }
  const sampleSize = chunks.length;
  const brokenCount = chunks.reduce(
    (count, chunk) => count + (chunk.cleanBoundary ? 0 : 1),
    0,
  );
  const totalSize = chunks.reduce((sum, chunk) => sum + chunk.charCount, 0);
  const meanSize = totalSize / sampleSize;
  const variance =
    chunks.reduce((sum, chunk) => {
      const diff = chunk.charCount - meanSize;
      return sum + diff * diff;
    }, 0) / sampleSize;
  return {
    sampleSize,
    brokenPercent: (brokenCount / sampleSize) * 100,
    stdDeviation: Math.sqrt(Math.max(variance, 0)),
    meanSize,
  };
}

function buildQualityStats(results: {
  markdown: ChunkCard[];
  recursive: ChunkCard[];
  semantic: ChunkCard[];
}): QualityStats {
  return {
    markdown: computeQualityMetric(results.markdown),
    recursive: computeQualityMetric(results.recursive),
    semantic: computeQualityMetric(results.semantic),
  };
}

function cloneChunks(chunks: ChunkCard[]): ChunkCard[] {
  return chunks.map((chunk) => ({ ...chunk }));
}

function cloneRetrievalResult(result: RetrievalResult | null): RetrievalResult | null {
  if (!result) return null;
  return {
    scores: [...result.scores],
    topIndices: [...result.topIndices],
    fragmented: result.fragmented,
    contiguous: result.contiguous,
    bestScore: result.bestScore,
  };
}

function cloneRetrievalMatches(matches: RetrievalMatches): RetrievalMatches {
  return {
    markdown: cloneRetrievalResult(matches.markdown),
    recursive: cloneRetrievalResult(matches.recursive),
    semantic: cloneRetrievalResult(matches.semantic),
  };
}

function createSnapshotId() {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `snap-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

function cloneQualityStats(stats: QualityStats): QualityStats {
  const cloneMetric = (metric: QualityMetric): QualityMetric => ({
    sampleSize: metric.sampleSize,
    brokenPercent: metric.brokenPercent,
    stdDeviation: metric.stdDeviation,
    meanSize: metric.meanSize,
  });
  return {
    markdown: cloneMetric(stats.markdown),
    recursive: cloneMetric(stats.recursive),
    semantic: cloneMetric(stats.semantic),
  };
}

function formatSnapshotTimestamp(value: number) {
  return new Date(value).toLocaleString();
}

function truncateText(value: string, max = 120) {
  if (!value) return "";
  if (value.length <= max) return value;
  return `${value.slice(0, max)}…`;
}

function segmentTextIntoSentences(text: string): SentenceSegment[] {
  const trimmed = text.trim();
  if (!trimmed) return [];
  const segments: SentenceSegment[] = [];
  const hasSegmenter =
    typeof Intl !== "undefined" &&
    "Segmenter" in Intl &&
    typeof Intl.Segmenter === "function";
  if (hasSegmenter) {
    const SegmenterCtor = (Intl as typeof Intl & {
      Segmenter: typeof Intl.Segmenter;
    }).Segmenter;
    const segmenter = new SegmenterCtor("en", { granularity: "sentence" });
    let fallbackIndex = 0;
    for (const entry of segmenter.segment(text)) {
      const segmentText: string = entry.segment ?? "";
      if (!segmentText.trim()) {
        fallbackIndex += segmentText.length;
        continue;
      }
      const providedIndex = typeof entry.index === "number" ? entry.index : -1;
      const computedIndex = text.indexOf(segmentText, fallbackIndex);
      const index = providedIndex >= 0 ? providedIndex : computedIndex >= 0 ? computedIndex : fallbackIndex;
      fallbackIndex = index + segmentText.length;
      if (!segmentText.trim()) continue;
      segments.push({
        text: segmentText,
        start: index,
        end: index + segmentText.length,
      });
    }
  }

  if (!segments.length) {
    const sentenceRegex = /[^.!?…]+[.!?…]?\s*/g;
    let match: RegExpExecArray | null;
    while ((match = sentenceRegex.exec(text)) !== null) {
      const fragment = match[0];
      if (!fragment.trim()) continue;
      segments.push({
        text: fragment,
        start: match.index,
        end: match.index + fragment.length,
      });
    }
  }

  if (!segments.length) {
    segments.push({ text, start: 0, end: text.length });
  }

  return segments;
}

const BLOCK_TAG_REGEX = /<\/?(p|div|section|article|header|footer|main|aside|nav|li|ul|ol|h[1-6]|blockquote|pre|code|table|tr|td|th)[^>]*>/gi;
const BREAK_TAG_REGEX = /<(br|hr)\s*\/?>/gi;

function decodeEntities(value: string) {
  return value.replace(/&(nbsp|amp|quot|lt|gt);/gi, (_, entity) => {
    const lower = String(entity).toLowerCase();
    if (lower === "nbsp") return " ";
    if (lower === "amp") return "&";
    if (lower === "quot") return '"';
    if (lower === "lt") return "<";
    if (lower === "gt") return ">";
    return _;
  });
}

function scrubScriptsAndStyles(value: string) {
  return value
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ");
}

function cleanDirtyHtml(value: string) {
  if (!value) return "";
  let text = scrubScriptsAndStyles(value);
  text = text.replace(BREAK_TAG_REGEX, "\n");
  text = text.replace(BLOCK_TAG_REGEX, "\n");
  text = text.replace(/<[^>]+>/g, " ");
  text = decodeEntities(text);
  text = text
    .replace(/\r\n?/g, "\n")
    .replace(/\t+/g, " ")
    .replace(/[ ]{2,}/g, " ")
    .replace(/[ ]+\n/g, "\n")
    .replace(/\n[ ]+/g, "\n")
    .replace(/\n{3,}/g, "\n\n");
  return text.trim();
}

function cosineSimilarity(a: number[] | null, b: number[] | null) {
  if (!a || !b) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    const av = a[i];
    const bv = b[i];
    dot += av * bv;
    normA += av * av;
    normB += bv * bv;
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  if (!denom) return 0;
  const score = dot / denom;
  return Number.isFinite(score) ? score : 0;
}

function blendVectors(base: number[], incoming: number[], weight: number) {
  if (!base.length) return incoming.slice();
  if (!incoming.length) return base.slice();
  const result = new Array(base.length);
  for (let i = 0; i < base.length; i++) {
    result[i] = (base[i] * weight + incoming[i]) / (weight + 1);
  }
  return result;
}

async function generateSemanticChunks({ text, chunkSize, similarity }: SemanticOptions) {
  const sentences = segmentTextIntoSentences(text);
  if (!sentences.length) return [];
  const pipeline = await loadEmbeddingPipeline();
  const embeddings: number[][] = [];
  for (let i = 0; i < sentences.length; i += SEMANTIC_BATCH_SIZE) {
    const batchSegments = sentences.slice(i, i + SEMANTIC_BATCH_SIZE);
    const payload = batchSegments.map((seg) => seg.text);
    const output = await pipeline(payload, { pooling: "mean", normalize: true });
    const tensors = ensureEmbeddingArray(output);
    tensors.forEach((tensor) => {
      embeddings.push(tensorToArray(tensor));
    });
  }

  const chunks: ChunkCard[] = [];
  const threshold = clampSemanticSimilarity(similarity);
  const targetSize = Math.max(300, chunkSize);
  const hardLimit = Math.round(targetSize * 1.4);
  let builder: SemanticBuilder | null = null;

  const flushBuilder = () => {
    if (!builder) return;
    const content = text.slice(builder.start, builder.end);
    const label =
      builder.sentenceStart === builder.sentenceEnd
        ? `Sentence ${builder.sentenceStart + 1}`
        : `Sentences ${builder.sentenceStart + 1}–${builder.sentenceEnd + 1}`;
    chunks.push({
      id: chunks.length,
      content,
      charCount: content.length,
      lineLabel: label,
      cleanBoundary: assessBoundary(content),
      start: builder.start,
      end: builder.end,
      splitter: "semantic",
    });
    builder = null;
  };

  sentences.forEach((segment, idx) => {
    const embedding = embeddings[idx];
    if (!embedding) return;
    if (!builder) {
      builder = {
        start: segment.start,
        end: segment.end,
        vector: embedding.slice(),
        count: 1,
        sentenceStart: idx,
        sentenceEnd: idx,
      };
      return;
    }
    const similarityScore = cosineSimilarity(builder.vector, embedding);
    const currentLength = builder.end - builder.start;
    const projectedLength = segment.end - builder.start;
    const exceededTarget = currentLength >= targetSize;
    const exceedsHardLimit = projectedLength >= hardLimit;
    const shouldSplit =
      similarityScore < threshold ||
      exceedsHardLimit ||
      (exceededTarget && similarityScore < Math.min(0.97, threshold + 0.08));
    if (shouldSplit) {
      flushBuilder();
      builder = {
        start: segment.start,
        end: segment.end,
        vector: embedding.slice(),
        count: 1,
        sentenceStart: idx,
        sentenceEnd: idx,
      };
      return;
    }
    builder.end = segment.end;
    builder.vector = blendVectors(builder.vector, embedding, builder.count);
    builder.count += 1;
    builder.sentenceEnd = idx;
  });

  flushBuilder();
  return chunks;
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

  const computeRanges = (docs: LangChainDoc[]) => {
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

  const convert = (
    docs: LangChainDoc[],
    splitter: "markdown" | "recursive",
    ranges: { start: number; end: number }[],
  ): ChunkCard[] =>
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
  const [rawSource, setRawSource] = useState<string>(SAMPLE_RAW_HTML);
  const [rawFileName, setRawFileName] = useState<string>("sample-raw.html");
  const [inputMode, setInputMode] = useState<InputMode>("markdown");
  const [shouldCleanRawHtml, setShouldCleanRawHtml] = useState<boolean>(true);
  const [chunkSize, setChunkSize] = useState<number>(900);
  const [overlap, setOverlap] = useState<number>(120);
  const [semanticThreshold, setSemanticThreshold] = useState<number>(DEFAULT_SEMANTIC_THRESHOLD);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [results, setResults] = useState<{
    recursive: ChunkCard[];
    markdown: ChunkCard[];
    semantic: ChunkCard[];
  }>({ recursive: [], markdown: [], semantic: [] });
  const [chunkVectors, setChunkVectors] = useState<ChunkVectors>({
    markdown: [],
    recursive: [],
    semantic: [],
  });
  const [focusRange, setFocusRange] = useState<{
    start: number;
    end: number;
    splitter: "markdown" | "recursive" | "semantic";
    label: string;
  } | null>(null);
  const [semanticError, setSemanticError] = useState<string | null>(null);
  const [retrievalQuery, setRetrievalQuery] = useState<string>("");
  const [retrievalMatches, setRetrievalMatches] = useState<RetrievalMatches>({
    markdown: null,
    recursive: null,
    semantic: null,
  });
  const [isRetrievalLoading, setIsRetrievalLoading] = useState<boolean>(false);
  const [retrievalError, setRetrievalError] = useState<string | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatus>({
    state: "idle",
    origin: "unknown",
  });
  const [collapsedSections, setCollapsedSections] = useState<Record<SectionKey, boolean>>({
    insights: false,
    upload: false,
    markdown: false,
    recursive: false,
    semantic: false,
  });
  const [snapshots, setSnapshots] = useState<Snapshot[]>([]);
  const [snapshotName, setSnapshotName] = useState<string>("");
  const [snapshotError, setSnapshotError] = useState<string | null>(null);
  const [variantAId, setVariantAId] = useState<string | null>(null);
  const [variantBId, setVariantBId] = useState<string | null>(null);
  const [snapshotsLoaded, setSnapshotsLoaded] = useState<boolean>(false);
  const toggleSection = (section: SectionKey) => {
    setCollapsedSections((prev) => ({
      ...prev,
      [section]: !prev[section],
    }));
  };

  const cleanedRawText = useMemo(() => cleanDirtyHtml(rawSource), [rawSource]);

  const workingText = useMemo(() => {
    if (inputMode === "markdown") return source;
    return shouldCleanRawHtml ? cleanedRawText : rawSource;
  }, [inputMode, source, shouldCleanRawHtml, cleanedRawText, rawSource]);

  const activeFileName = inputMode === "markdown" ? fileName : rawFileName;

  const totalChars = useMemo(() => workingText.length, [workingText]);

  const qualityStats = useMemo<QualityStats>(() => buildQualityStats(results), [results]);

  useEffect(() => {
    let cancelled = false;
    const trimmed = workingText.trim();
    if (!trimmed) {
      setResults({ recursive: [], markdown: [], semantic: [] });
      setSemanticError(null);
      setIsLoading(false);
      return () => {
        cancelled = true;
      };
    }
    setIsLoading(true);
    setSemanticError(null);
    Promise.allSettled([
      runSplit(workingText, chunkSize, overlap),
      generateSemanticChunks({ text: workingText, chunkSize, similarity: semanticThreshold }),
    ])
      .then(([structureResult, semanticResult]) => {
        if (cancelled) return;
        if (structureResult.status === "rejected") {
          console.error("Split error", structureResult.reason);
          setResults({ recursive: [], markdown: [], semantic: [] });
        } else {
          const semanticChunks =
            semanticResult.status === "fulfilled" ? semanticResult.value : [];
          if (semanticResult.status === "rejected") {
            console.error("Semantic chunk error", semanticResult.reason);
            const reason =
              semanticResult.reason instanceof Error
                ? semanticResult.reason.message
                : "Embedding step failed. Try again or reload the page.";
            setSemanticError(reason);
          }
          setResults({
            recursive: structureResult.value.recursive,
            markdown: structureResult.value.markdown,
            semantic: semanticChunks,
          });
        }
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [workingText, chunkSize, overlap, semanticThreshold]);

  useEffect(() => {
    setOverlap((prev) => Math.max(0, Math.min(prev, chunkSize - 20)));
  }, [chunkSize]);

  useEffect(() => {
    setChunkVectors({ markdown: [], recursive: [], semantic: [] });
    setRetrievalMatches({ markdown: null, recursive: null, semantic: null });
    setRetrievalError(null);
  }, [workingText, chunkSize, overlap, semanticThreshold]);

  useEffect(() => {
    if (typeof window === "undefined") return () => {};
    let cancelled = false;
    const cached = window.localStorage.getItem(MODEL_CACHE_KEY);
    const origin: ModelStatus["origin"] = cached ? "cache" : "network";
    setModelStatus({ state: "preloading", origin });
    loadEmbeddingPipeline()
      .then(() => {
        if (cancelled) return;
        try {
          window.localStorage.setItem(MODEL_CACHE_KEY, "1");
        } catch (error) {
          console.warn("Unable to persist MiniLM cache marker", error);
        }
        setModelStatus({ state: "ready", origin: origin === "cache" ? "cache" : "network" });
      })
      .catch((error) => {
        if (cancelled) return;
        console.error("MiniLM preload failed", error);
        setModelStatus({
          state: "error",
          origin,
          message: error instanceof Error ? error.message : "Unknown error",
        });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const stored = window.localStorage.getItem(SNAPSHOT_STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored) as Snapshot[];
        if (Array.isArray(parsed)) {
          setSnapshots(parsed);
        }
      }
    } catch (error) {
      console.error("Failed to load snapshots", error);
    } finally {
      setSnapshotsLoaded(true);
    }
  }, []);

  useEffect(() => {
    if (!snapshotsLoaded || typeof window === "undefined") return;
    try {
      window.localStorage.setItem(
        SNAPSHOT_STORAGE_KEY,
        JSON.stringify(snapshots),
      );
      setSnapshotError((prev) =>
        prev && prev.startsWith("Unable to persist") ? null : prev,
      );
    } catch (error) {
      console.error("Failed to persist snapshots", error);
      setSnapshotError("Unable to persist snapshots. Storage might be full.");
    }
  }, [snapshots, snapshotsLoaded]);

  useEffect(() => {
    if (!snapshots.length) {
      setVariantAId(null);
      return;
    }
    setVariantAId((prev) => {
      if (prev && snapshots.some((snapshot) => snapshot.id === prev)) {
        return prev;
      }
      return snapshots[0]?.id ?? null;
    });
  }, [snapshots]);

  useEffect(() => {
    if (!snapshots.length) {
      setVariantBId(null);
      return;
    }
    setVariantBId((prev) => {
      const currentA = variantAId;
      if (prev && prev !== currentA && snapshots.some((snapshot) => snapshot.id === prev)) {
        return prev;
      }
      const fallback = snapshots.find((snapshot) => snapshot.id !== currentA)?.id ?? null;
      return fallback;
    });
  }, [snapshots, variantAId]);


  const handleFile = async (file?: File, target: InputMode = inputMode) => {
    if (!file) return;
    const text = await file.text();
    if (target === "rawHtml") {
      setRawSource(text);
      setRawFileName(file.name);
    } else {
      setSource(text);
      setFileName(file.name);
    }
  };

  const handleDrop = async (e: DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    const file = e.dataTransfer?.files?.[0];
    await handleFile(file, inputMode);
  };

  const handleMarkdownPaste = async (e: ClipboardEvent<HTMLTextAreaElement>) => {
    if (e.clipboardData?.files?.length) {
      e.preventDefault();
      await handleFile(e.clipboardData.files[0], "markdown");
    }
  };

  const handleRawPaste = async (e: ClipboardEvent<HTMLTextAreaElement>) => {
    if (e.clipboardData?.files?.length) {
      e.preventDefault();
      await handleFile(e.clipboardData.files[0], "rawHtml");
    }
  };

  const ensureChunkVectors = async (
    splitter: ChunkCard["splitter"],
    chunks: ChunkCard[],
  ) => {
    if (!chunks.length) return [] as number[][];
    const cached = chunkVectors[splitter];
    if (cached.length === chunks.length && cached.length) return cached;
    const texts = chunks.map((chunk) => chunk.content);
    const vectors = await embedTexts(texts);
    setChunkVectors((prev) => ({
      ...prev,
      [splitter]: vectors,
    }));
    return vectors;
  };

  const runRetrievalSimulation = async () => {
    const trimmed = retrievalQuery.trim();
    if (!trimmed) {
      setRetrievalError("Enter a query to simulate retrieval.");
      return;
    }
    if (
      !results.markdown.length &&
      !results.recursive.length &&
      !results.semantic.length
    ) {
      setRetrievalError("Chunks are still loading. Try again in a moment.");
      return;
    }
    setIsRetrievalLoading(true);
    setRetrievalError(null);
    try {
      const queryVectors = await embedTexts([trimmed]);
      const queryVector = queryVectors[0];
      if (!queryVector) {
        throw new Error("Query embedding failed. Reload and try again.");
      }
      const matchPayload: RetrievalMatches = {
        markdown: null,
        recursive: null,
        semantic: null,
      };
      for (const splitter of RETRIEVAL_SPLITTERS) {
        const chunks = results[splitter];
        if (!chunks.length) {
          matchPayload[splitter] = {
            scores: [],
            topIndices: [],
            fragmented: false,
            contiguous: true,
            bestScore: 0,
          };
          continue;
        }
        const chunkVecs = await ensureChunkVectors(splitter, chunks);
        const scores = chunks.map((_, idx) =>
          cosineSimilarity(chunkVecs[idx] ?? null, queryVector),
        );
        const sorted = scores
          .map((score, idx) => ({ score, idx }))
          .sort((a, b) => b.score - a.score);
        const filtered = sorted.filter((entry) => entry.score >= RETRIEVAL_SCORE_FLOOR);
        if (!filtered.length && sorted.length) {
          filtered.push(sorted[0]);
        }
        const top = filtered.slice(0, RETRIEVAL_TOP_K);
        const topIndices = top.map((entry) => entry.idx);
        const { fragmented, contiguous } = analyzeFragmentation(topIndices);
        matchPayload[splitter] = {
          scores,
          topIndices,
          fragmented,
          contiguous,
          bestScore: top[0]?.score ?? 0,
        };
      }
      setRetrievalMatches(matchPayload);
    } catch (error) {
      console.error("Retrieval simulation failed", error);
      const message =
        error instanceof Error
          ? error.message
          : "Embedding step failed. Try again or reload.";
      setRetrievalError(message);
    } finally {
      setIsRetrievalLoading(false);
    }
  };

  const formatScore = (value: number | null | undefined) => {
    if (value === undefined || value === null || Number.isNaN(value)) return "--";
    return `${Math.round(value * 100)}%`;
  };

  const formatPercentMetric = (value: number | null) => {
    if (value === null || Number.isNaN(value)) return "--";
    const decimals = value < 10 ? 1 : 0;
    return `${value.toFixed(decimals)}%`;
  };

  const formatCharsMetric = (value: number | null) => {
    if (value === null || Number.isNaN(value)) return "--";
    return `${Math.round(value).toLocaleString()} chars`;
  };

  const formatDeltaValue = (
    a: number | null | undefined,
    b: number | null | undefined,
    { suffix = "", decimals = 0 }: { suffix?: string; decimals?: number } = {},
  ) => {
    if (
      a === null ||
      a === undefined ||
      b === null ||
      b === undefined ||
      Number.isNaN(a) ||
      Number.isNaN(b)
    ) {
      return "--";
    }
    const delta = a - b;
    const precision = decimals ?? 0;
    const magnitude = Math.abs(delta);
    const formatted =
      precision > 0
        ? magnitude.toFixed(precision)
        : Math.round(magnitude).toString();
    if (Number(formatted) === 0) {
      return `0${suffix}`;
    }
    const prefix = delta > 0 ? "+" : "-";
    return `${prefix}${formatted}${suffix}`;
  };

  const getSnapshotInputLabel = (snapshot: Snapshot) => {
    if (snapshot.config.inputMode === "markdown") return "Markdown";
    return snapshot.config.shouldCleanRawHtml ? "Cleaned HTML" : "Raw HTML";
  };

  const getChunkCount = (snapshot: Snapshot | null, splitter: ChunkCard["splitter"]) =>
    snapshot ? snapshot.results[splitter].length : null;

  const getBrokenPercent = (snapshot: Snapshot | null, splitter: ChunkCard["splitter"]) =>
    snapshot ? snapshot.qualityStats[splitter].brokenPercent : null;

  const getBestScore = (snapshot: Snapshot | null, splitter: ChunkCard["splitter"]) =>
    snapshot?.retrievalMatches[splitter]?.bestScore ?? null;

  const formatCount = (value: number | null) =>
    value === null ? "--" : value.toLocaleString();

  const toPercent = (value: number | null) =>
    value === null ? null : value * 100;

  const renderVariantColumn = (slot: "A" | "B", snapshot: Snapshot | null) => {
    if (!snapshot) {
      return (
        <div key={slot} className="ab-column empty">
          <span className="chip subtle">Variant {slot}</span>
          <p className="muted">Assign a snapshot to this slot to unlock comparisons.</p>
        </div>
      );
    }
    const retrievalSummary = snapshot.config.retrievalQuery.trim()
      ? truncateText(snapshot.config.retrievalQuery.trim(), 90)
      : null;
    return (
      <div key={slot} className="ab-column">
        <div className="variant-head">
          <span className="chip subtle">Variant {slot}</span>
          <h3>{snapshot.name}</h3>
          <p className="muted">{formatSnapshotTimestamp(snapshot.createdAt)}</p>
        </div>
        <div className="variant-tags">
          <span className="chip subtle">{getSnapshotInputLabel(snapshot)}</span>
          <span className="chip subtle">
            {snapshot.summary.workingChars.toLocaleString()} chars
          </span>
          <span className="chip subtle">Size {snapshot.config.chunkSize}</span>
          <span className="chip subtle">Overlap {snapshot.config.overlap}</span>
          <span className="chip subtle">
            Sem {snapshot.config.semanticThreshold.toFixed(2)}
          </span>
        </div>
        <div className="variant-splitter-grid">
          {RETRIEVAL_SPLITTERS.map((splitter) => {
            const chunkCount = snapshot.results[splitter].length;
            const brokenPercent = snapshot.qualityStats[splitter].brokenPercent;
            const bestScore = snapshot.retrievalMatches[splitter]?.bestScore ?? null;
            return (
              <div key={`${slot}-${splitter}`} className="variant-splitter">
                <span className="chip subtle">{SPLITTER_LABELS[splitter]}</span>
                <p className="metric-value">
                  {chunkCount.toLocaleString()} chunk
                  {chunkCount === 1 ? "" : "s"}
                </p>
                <p className="metric-footnote">
                  Broken {formatPercentMetric(brokenPercent)}
                </p>
                <p className="metric-footnote">
                  Best score {formatScore(bestScore)}
                </p>
              </div>
            );
          })}
        </div>
        <div className="variant-query">
          <span className="meta-label">Query</span>
          <p className={retrievalSummary ? "" : "muted"}>
            {retrievalSummary ? `“${retrievalSummary}”` : "No retrieval simulation saved."}
          </p>
        </div>
      </div>
    );
  };

  const getRetrievalMeta = (
    splitter: ChunkCard["splitter"],
    chunkId: number,
  ) => {
    const summary = retrievalMatches[splitter];
    if (!summary) return { isRetrieved: false, score: null };
    return {
      isRetrieved: summary.topIndices.includes(chunkId),
      score: summary.scores[chunkId] ?? null,
    };
  };

  const modelStatusLabel = useMemo(() => {
    if (modelStatus.state === "error") return "MiniLM preload failed";
    if (modelStatus.state === "ready") {
      return modelStatus.origin === "cache"
        ? "MiniLM cached"
        : `MiniLM ready (~${MODEL_SIZE_MB} MB warmup)`;
    }
    if (modelStatus.state === "preloading") {
      return modelStatus.origin === "cache"
        ? "Priming MiniLM cache"
        : `Downloading ~${MODEL_SIZE_MB} MB`;
    }
    return "Waiting for MiniLM";
  }, [modelStatus]);

  const modelStatusTone =
    modelStatus.state === "error"
      ? "warn"
      : modelStatus.state === "ready"
        ? "ok"
        : "subtle";

  const modelStatusDetail =
    modelStatus.state === "error" ? modelStatus.message ?? null : null;

  const highlightColor = focusRange
    ? focusRange.splitter === "markdown"
      ? "var(--accent)"
      : focusRange.splitter === "semantic"
        ? "var(--iris)"
        : "var(--amber)"
    : "var(--accent)";

  const renderWorkingSource = (text: string) => {
    if (!focusRange || focusRange.start < 0 || focusRange.end < 0)
      return <pre className="source-view">{text}</pre>;
    return (
      <pre className="source-view">
        <span>{text.slice(0, focusRange.start)}</span>
        <span className="highlight" style={{ backgroundColor: highlightColor }}>
          {text.slice(focusRange.start, focusRange.end)}
        </span>
        <span>{text.slice(focusRange.end)}</span>
      </pre>
    );
  };

  const canSaveSnapshot = snapshotsLoaded && !isLoading && Boolean(workingText.trim());

  const handleSnapshotSave = () => {
    setSnapshotError(null);
    if (!snapshotsLoaded) {
      setSnapshotError("Snapshots are still loading. Please wait a second.");
      return;
    }
    if (!workingText.trim()) {
      setSnapshotError("Load or paste content before saving a snapshot.");
      return;
    }
    if (isLoading) {
      setSnapshotError("Wait for the chunk pass to finish before saving.");
      return;
    }
    const trimmedName = snapshotName.trim();
    const timestampLabel = new Date().toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
    const fallbackName = `${activeFileName} @ ${timestampLabel}`;
    const snapshotLabel = trimmedName || fallbackName;
    const snapshotRecord: Snapshot = {
      id: createSnapshotId(),
      name: snapshotLabel,
      createdAt: Date.now(),
      config: {
        chunkSize,
        overlap,
        semanticThreshold,
        inputMode,
        shouldCleanRawHtml,
        retrievalQuery,
      },
      files: {
        source,
        rawSource,
        fileName,
        rawFileName,
      },
      results: {
        markdown: cloneChunks(results.markdown),
        recursive: cloneChunks(results.recursive),
        semantic: cloneChunks(results.semantic),
      },
      qualityStats: cloneQualityStats(qualityStats),
      retrievalMatches: cloneRetrievalMatches(retrievalMatches),
      summary: {
        workingChars: workingText.length,
      },
    };
    setSnapshots((prev) => [snapshotRecord, ...prev]);
    if (!variantAId) {
      setVariantAId(snapshotRecord.id);
    } else if (!variantBId) {
      setVariantBId(snapshotRecord.id);
    }
    setSnapshotName("");
  };

  const loadSnapshotById = (id: string) => {
    const snapshot = snapshots.find((entry) => entry.id === id);
    if (!snapshot) return;
    const { config, files } = snapshot;
    setSource(files.source);
    setRawSource(files.rawSource);
    setFileName(files.fileName);
    setRawFileName(files.rawFileName);
    setInputMode(config.inputMode);
    setShouldCleanRawHtml(config.shouldCleanRawHtml);
    setChunkSize(config.chunkSize);
    setOverlap(config.overlap);
    setSemanticThreshold(config.semanticThreshold);
    setRetrievalQuery(config.retrievalQuery);
    setFocusRange(null);
  };

  const deleteSnapshot = (id: string) => {
    setSnapshots((prev) => prev.filter((snapshot) => snapshot.id !== id));
    if (variantAId === id) {
      setVariantAId(null);
    }
    if (variantBId === id) {
      setVariantBId(null);
    }
  };

  const assignVariant = (slot: "A" | "B", id: string) => {
    if (!snapshots.some((snapshot) => snapshot.id === id)) return;
    if (slot === "A") {
      setVariantAId(id);
      if (variantBId === id) {
        setVariantBId(null);
      }
    } else {
      setVariantBId(id);
      if (variantAId === id) {
        setVariantAId(null);
      }
    }
  };

  const variantA = variantAId ? snapshots.find((entry) => entry.id === variantAId) ?? null : null;
  const variantB = variantBId ? snapshots.find((entry) => entry.id === variantBId) ?? null : null;
  const bestScoreMdA = getBestScore(variantA, "markdown");
  const bestScoreMdB = getBestScore(variantB, "markdown");
  const bestScoreRcA = getBestScore(variantA, "recursive");
  const bestScoreRcB = getBestScore(variantB, "recursive");
  const bestScoreSemA = getBestScore(variantA, "semantic");
  const bestScoreSemB = getBestScore(variantB, "semantic");

  const comparisonRows = [
    {
      label: "Chunk size",
      valueA: variantA ? `${variantA.config.chunkSize} chars` : "--",
      valueB: variantB ? `${variantB.config.chunkSize} chars` : "--",
      delta: formatDeltaValue(
        variantA?.config.chunkSize ?? null,
        variantB?.config.chunkSize ?? null,
        { suffix: " chars" },
      ),
    },
    {
      label: "Overlap",
      valueA: variantA ? `${variantA.config.overlap} chars` : "--",
      valueB: variantB ? `${variantB.config.overlap} chars` : "--",
      delta: formatDeltaValue(
        variantA?.config.overlap ?? null,
        variantB?.config.overlap ?? null,
        { suffix: " chars" },
      ),
    },
    {
      label: "Semantic cohesion",
      valueA: variantA ? variantA.config.semanticThreshold.toFixed(2) : "--",
      valueB: variantB ? variantB.config.semanticThreshold.toFixed(2) : "--",
      delta: formatDeltaValue(
        variantA?.config.semanticThreshold ?? null,
        variantB?.config.semanticThreshold ?? null,
        { decimals: 2 },
      ),
    },
    {
      label: "Working chars",
      valueA: variantA ? variantA.summary.workingChars.toLocaleString() : "--",
      valueB: variantB ? variantB.summary.workingChars.toLocaleString() : "--",
      delta: formatDeltaValue(
        variantA?.summary.workingChars ?? null,
        variantB?.summary.workingChars ?? null,
        { suffix: " chars" },
      ),
    },
    {
      label: "Markdown chunks",
      valueA: formatCount(getChunkCount(variantA, "markdown")),
      valueB: formatCount(getChunkCount(variantB, "markdown")),
      delta: formatDeltaValue(
        getChunkCount(variantA, "markdown"),
        getChunkCount(variantB, "markdown"),
        { suffix: " chunks" },
      ),
    },
    {
      label: "Markdown broken edges",
      valueA: formatPercentMetric(getBrokenPercent(variantA, "markdown")),
      valueB: formatPercentMetric(getBrokenPercent(variantB, "markdown")),
      delta: formatDeltaValue(
        getBrokenPercent(variantA, "markdown"),
        getBrokenPercent(variantB, "markdown"),
        { suffix: "%", decimals: 1 },
      ),
    },
    {
      label: "Recursive chunks",
      valueA: formatCount(getChunkCount(variantA, "recursive")),
      valueB: formatCount(getChunkCount(variantB, "recursive")),
      delta: formatDeltaValue(
        getChunkCount(variantA, "recursive"),
        getChunkCount(variantB, "recursive"),
        { suffix: " chunks" },
      ),
    },
    {
      label: "Recursive broken edges",
      valueA: formatPercentMetric(getBrokenPercent(variantA, "recursive")),
      valueB: formatPercentMetric(getBrokenPercent(variantB, "recursive")),
      delta: formatDeltaValue(
        getBrokenPercent(variantA, "recursive"),
        getBrokenPercent(variantB, "recursive"),
        { suffix: "%", decimals: 1 },
      ),
    },
    {
      label: "Semantic chunks",
      valueA: formatCount(getChunkCount(variantA, "semantic")),
      valueB: formatCount(getChunkCount(variantB, "semantic")),
      delta: formatDeltaValue(
        getChunkCount(variantA, "semantic"),
        getChunkCount(variantB, "semantic"),
        { suffix: " chunks" },
      ),
    },
    {
      label: "Semantic broken edges",
      valueA: formatPercentMetric(getBrokenPercent(variantA, "semantic")),
      valueB: formatPercentMetric(getBrokenPercent(variantB, "semantic")),
      delta: formatDeltaValue(
        getBrokenPercent(variantA, "semantic"),
        getBrokenPercent(variantB, "semantic"),
        { suffix: "%", decimals: 1 },
      ),
    },
    {
      label: "Retrieval • Markdown",
      valueA: formatScore(bestScoreMdA),
      valueB: formatScore(bestScoreMdB),
      delta: formatDeltaValue(toPercent(bestScoreMdA), toPercent(bestScoreMdB), {
        suffix: "%",
      }),
    },
    {
      label: "Retrieval • Recursive",
      valueA: formatScore(bestScoreRcA),
      valueB: formatScore(bestScoreRcB),
      delta: formatDeltaValue(toPercent(bestScoreRcA), toPercent(bestScoreRcB), {
        suffix: "%",
      }),
    },
    {
      label: "Retrieval • Semantic",
      valueA: formatScore(bestScoreSemA),
      valueB: formatScore(bestScoreSemB),
      delta: formatDeltaValue(toPercent(bestScoreSemA), toPercent(bestScoreSemB), {
        suffix: "%",
      }),
    },
  ];

  return (
    <div className="page">
      <header className="hero">
        <div>
          <p className="eyebrow">Chunking Strategy • AI Engineer playbook</p>
          <h1>Context survives—or snaps—when you slice Markdown.</h1>
          <p className="lede">
            Compare <strong>RecursiveCharacterTextSplitter</strong>,{" "}
            <strong>MarkdownTextSplitter</strong>, and a new{" "}
            <strong>semantic embedder</strong> (MiniLM via Transformers.js).
            Upload a large .md, tweak chunk size/overlap, and inspect where
            edges cut sentences—or let embeddings decide when topics shift.
          </p>
          <div className="metrics">
            <div>
              <span className="label">Active file</span>
              <p className="value">{activeFileName}</p>
            </div>
            <div>
              <span className="label">Characters</span>
              <p className="value">{totalChars.toLocaleString()}</p>
            </div>
            <div>
              <span className="label">Chunks</span>
              <p className="value">
                MD {results.markdown.length} • RC {results.recursive.length} • SEM {results.semantic.length}
              </p>
            </div>
          </div>
        </div>
      </header>

      <section className="docs-links">
        <div className="card doc-card">
          <div className="panel-head">
            <div>
              <p className="section-title">Docs & case studies</p>
              <p className="muted">
                Skim the sales letter for narrative framing or dive into the quantified failures before demoing the lab.
              </p>
            </div>
            <div className="panel-head-actions">
              <span className="chip subtle">Portfolio ready</span>
            </div>
          </div>
          <div className="doc-link-grid">
            {DOC_LINKS.map((doc) => (
              <a
                key={doc.id}
                className="doc-link"
                href={doc.href}
                target="_blank"
                rel="noopener noreferrer"
              >
                <span className="chip subtle">{doc.badge}</span>
                <strong>{doc.title}</strong>
                <p>{doc.description}</p>
              </a>
            ))}
          </div>
        </div>
      </section>

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
              <p className="section-title">Load text & clean it</p>
              <p className="muted">
                Drop Markdown or raw HTML. Flip into dirty mode to feel how cleaning protects chunk quality.
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
                  accept=".md,.markdown,.txt,.html,text/markdown,text/plain,text/html"
                  onChange={(e) => handleFile(e.target.files?.[0], inputMode)}
                />
                <div>
                  <p className="big">Drop Markdown / HTML</p>
                  <p className="muted">routes to the active tab</p>
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
                <div className="control tight">
                  <label htmlFor="semanticThreshold">
                    Semantic cohesion
                    <span className="muted inline-hint">Higher = stricter merges</span>
                  </label>
                  <div className="slider-row">
                    <input
                      id="semanticThreshold"
                      type="range"
                      min={MIN_SEMANTIC_THRESHOLD}
                      max={MAX_SEMANTIC_THRESHOLD}
                      step={0.01}
                      value={semanticThreshold}
                      onChange={(e) =>
                        setSemanticThreshold(Number(e.target.value))
                      }
                    />
                    <span className="chip">{semanticThreshold.toFixed(2)}</span>
                  </div>
                </div>
              </div>
            </div>
            <div className="source-pane">
              <div className="source-meta">
                <span className="chip subtle">{activeFileName}</span>
                <span className="chip subtle">
                  {inputMode === "markdown"
                    ? "Markdown assumed clean"
                    : shouldCleanRawHtml
                      ? "Raw HTML → cleaned"
                      : "Raw HTML (dirty)"}
                </span>
                {focusRange && (
                  <span className="chip" style={{ background: highlightColor }}>
                    Highlight: {focusRange.label}
                  </span>
                )}
              </div>
              <div
                className="input-mode-toggle"
                role="tablist"
                aria-label="Input mode"
              >
                <button
                  type="button"
                  className={`mode-pill${inputMode === "markdown" ? " active" : ""}`}
                  onClick={() => setInputMode("markdown")}
                  aria-pressed={inputMode === "markdown"}
                >
                  Markdown
                </button>
                <button
                  type="button"
                  className={`mode-pill${inputMode === "rawHtml" ? " active" : ""}`}
                  onClick={() => setInputMode("rawHtml")}
                  aria-pressed={inputMode === "rawHtml"}
                >
                  Raw HTML / dirty text
                </button>
              </div>
              {inputMode === "markdown" ? (
                <>
                  <textarea
                    className="editor mini"
                    value={source}
                    onChange={(e) => setSource(e.target.value)}
                    onPaste={handleMarkdownPaste}
                    spellCheck={false}
                    aria-label="Markdown editor"
                  />
                  {renderWorkingSource(workingText)}
                </>
              ) : (
                <div className="dirty-lab">
                  <div className="dirty-column">
                    <label htmlFor="rawInput">Raw HTML or dirty text</label>
                    <textarea
                      id="rawInput"
                      className="editor mini"
                      value={rawSource}
                      onChange={(e) => setRawSource(e.target.value)}
                      onPaste={handleRawPaste}
                      spellCheck={false}
                      aria-label="Raw HTML editor"
                    />
                    <label className="clean-toggle">
                      <input
                        type="checkbox"
                        checked={shouldCleanRawHtml}
                        onChange={(e) => setShouldCleanRawHtml(e.target.checked)}
                      />
                      Clean before chunking (strip tags, flatten lists)
                    </label>
                    <p className="muted">
                      Disable cleaning to watch MarkdownTextSplitter trip over literal tags, then re-enable to showcase your ETL fix.
                    </p>
                  </div>
                  <div className="dirty-column preview-column">
                    <div className="dirty-preview-head">
                      <span className="chip subtle">Working text</span>
                      <span className={`chip ${shouldCleanRawHtml ? "ok" : "warn"}`}>
                        {shouldCleanRawHtml ? "Cleaned for chunking" : "As-is (noisy)"}
                      </span>
                      <span className="chip subtle">
                        {workingText.length.toLocaleString()} chars
                      </span>
                    </div>
                    {renderWorkingSource(workingText)}
                    {!shouldCleanRawHtml && (
                      <div className="note warn mini">
                        MarkdownTextSplitter sees raw tags; compare quality metrics before toggling cleaning back on.
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="card retrieval-card">
          <div className="panel-head">
            <div>
              <p className="section-title">Retrieval simulation</p>
              <p className="muted">
                Embed an ad-hoc query and preview which chunks surface per strategy.
              </p>
            </div>
          </div>
          <div className="collapsible-stack">
            <div className="control tight">
              <label htmlFor="retrievalQuery">Ad-hoc query</label>
              <textarea
                id="retrievalQuery"
                className="editor mini"
                placeholder="e.g. Dust storm risk on Mars orbit insertion"
                value={retrievalQuery}
                onChange={(e) => setRetrievalQuery(e.target.value)}
                spellCheck={false}
              />
            </div>
            <div className="simulate-row">
              <button
                type="button"
                className="primary-btn"
                onClick={runRetrievalSimulation}
                disabled={isRetrievalLoading || !retrievalQuery.trim()}
              >
                {isRetrievalLoading ? "Simulating…" : "Run simulation"}
              </button>
              <span className={`chip ${modelStatusTone} model-chip`}>
                {modelStatusLabel}
              </span>
              {modelStatusDetail && <span className="error-text">{modelStatusDetail}</span>}
              {retrievalError && <span className="error-text">{retrievalError}</span>}
            </div>
            <div className="retrieval-summary">
              {RETRIEVAL_SPLITTERS.map((splitter) => {
                const summary = retrievalMatches[splitter];
                const chunkList = summary?.topIndices ?? [];
                const statusLabel = summary
                  ? chunkList.length === 0
                    ? "No hits"
                    : summary.fragmented
                      ? summary.contiguous
                        ? "Fragmented (adjacent)"
                        : "Fragmented"
                      : "Contained"
                  : null;
                return (
                  <div key={splitter} className="summary-line">
                    <span className="chip subtle summary-label">
                      {SPLITTER_LABELS[splitter]}
                    </span>
                    {summary ? (
                      <div className="summary-meta">
                        <span className="chip match">
                          {chunkList.length
                            ? `Top ${formatScore(summary.bestScore)}`
                            : "No hits"}
                        </span>
                        <span
                          className={`chip ${
                            chunkList.length === 0
                              ? "subtle"
                              : summary.fragmented
                                ? "warn"
                                : "ok"
                          }`}
                        >
                          {statusLabel}
                        </span>
                        <p className="summary-copy">
                          {chunkList.length
                            ? `Chunks ${chunkList
                                .map((idx) => `#${idx + 1}`)
                                .join(", ")}`
                            : "Below score floor"}
                        </p>
                      </div>
                    ) : (
                      <p className="summary-copy muted">
                        Run the simulation to preview recall.
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        <div className="card quality-card">
          <div className="panel-head">
            <div>
              <p className="section-title">Quality metrics</p>
              <p className="muted">
                Quantify sentence breaks and chunk-size variance so you can cite data, not vibes.
              </p>
            </div>
          </div>
          <div className="quality-grid">
            {RETRIEVAL_SPLITTERS.map((splitter) => {
              const stats = qualityStats[splitter];
              const chunkLabel = stats.sampleSize
                ? `${stats.sampleSize} chunk${stats.sampleSize === 1 ? "" : "s"}`
                : "Awaiting chunks";
              return (
                <article key={`quality-${splitter}`} className="quality-tile">
                  <div className="tile-head">
                    <span className="chip subtle">{SPLITTER_LABELS[splitter]}</span>
                    <span className="chip subtle">{chunkLabel}</span>
                  </div>
                  <div className="metric-pair">
                    <div>
                      <p className="metric-label">Broken edges</p>
                      <p className="metric-value">
                        {formatPercentMetric(stats.brokenPercent)}
                      </p>
                    </div>
                    <div>
                      <p className="metric-label">Chunk size σ</p>
                      <p className="metric-value">
                        {formatCharsMetric(stats.stdDeviation)}
                      </p>
                    </div>
                  </div>
                  <p className="metric-footnote">
                    {stats.meanSize !== null
                      ? `Avg ${formatCharsMetric(stats.meanSize)}`
                      : "Waiting for chunking"}
                  </p>
                </article>
              );
            })}
          </div>
        </div>
      </section>

      <section className="ab-section">
        <div className="card snapshot-card">
          <div className="panel-head">
            <div>
              <p className="section-title">Snapshots & persistence</p>
              <p className="muted">
                Capture the current text, chunk settings, quality metrics, and retrieval summary so you can reload or compare after a refresh.
              </p>
            </div>
          </div>
          <div className="snapshot-form">
            <input
              type="text"
              placeholder={`e.g. ${activeFileName} baseline`}
              value={snapshotName}
              onChange={(e) => {
                setSnapshotName(e.target.value);
                if (snapshotError) setSnapshotError(null);
              }}
            />
            <button
              type="button"
              className="primary-btn"
              onClick={handleSnapshotSave}
              disabled={!canSaveSnapshot}
            >
              Save snapshot
            </button>
          </div>
          <div className="snapshot-hint">
            <span className="chip subtle">
              {snapshots.length ? `${snapshots.length} saved` : "No snapshots yet"}
            </span>
            <span className="muted">Stored locally in this browser.</span>
          </div>
          {snapshotError && <div className="note warn mini">{snapshotError}</div>}
          {snapshots.length ? (
            <div className="snapshot-list">
              {snapshots.map((snapshot) => {
                const variantBadge =
                  snapshot.id === variantAId
                    ? "Variant A"
                    : snapshot.id === variantBId
                      ? "Variant B"
                      : null;
                const inputLabel = getSnapshotInputLabel(snapshot);
                const queryPreview = snapshot.config.retrievalQuery.trim()
                  ? truncateText(snapshot.config.retrievalQuery.trim(), 80)
                  : null;
                return (
                  <article key={snapshot.id} className="snapshot-item">
                    <div className="snapshot-item-head">
                      <div>
                        <p className="snapshot-name">{snapshot.name}</p>
                        <p className="muted snapshot-date">
                          {formatSnapshotTimestamp(snapshot.createdAt)}
                        </p>
                      </div>
                      <div className="snapshot-tag-group">
                        {variantBadge && <span className="chip match">{variantBadge}</span>}
                        <span className="chip subtle">{inputLabel}</span>
                        <span className="chip subtle">Size {snapshot.config.chunkSize}</span>
                        <span className="chip subtle">Overlap {snapshot.config.overlap}</span>
                        <span className="chip subtle">
                          Sem {snapshot.config.semanticThreshold.toFixed(2)}
                        </span>
                      </div>
                    </div>
                    <div className="snapshot-meta">
                      <p>
                        <span className="meta-label">Chars</span>
                        {snapshot.summary.workingChars.toLocaleString()}
                      </p>
                      <p>
                        <span className="meta-label">Chunks</span>
                        MD {snapshot.results.markdown.length} • RC {snapshot.results.recursive.length} • SEM {snapshot.results.semantic.length}
                      </p>
                    </div>
                    <p className={`snapshot-query ${queryPreview ? "" : "muted"}`}>
                      <span className="meta-label">Query</span>
                      {queryPreview ? ` “${queryPreview}”` : " None saved"}
                    </p>
                    <div className="snapshot-actions">
                      <button type="button" className="ghost-btn" onClick={() => loadSnapshotById(snapshot.id)}>
                        Load & activate
                      </button>
                      <button
                        type="button"
                        className={`pill-btn${variantAId === snapshot.id ? " active" : ""}`}
                        onClick={() => assignVariant("A", snapshot.id)}
                      >
                        {variantAId === snapshot.id ? "Pinned A" : "Set A"}
                      </button>
                      <button
                        type="button"
                        className={`pill-btn${variantBId === snapshot.id ? " active" : ""}`}
                        onClick={() => assignVariant("B", snapshot.id)}
                      >
                        {variantBId === snapshot.id ? "Pinned B" : "Set B"}
                      </button>
                      <button
                        type="button"
                        className="text-btn danger"
                        onClick={() => deleteSnapshot(snapshot.id)}
                      >
                        Delete
                      </button>
                    </div>
                  </article>
                );
              })}
            </div>
          ) : (
            <p className="muted empty-copy">
              Save a snapshot after chunking finishes to persist this configuration and unlock the A/B dashboard.
            </p>
          )}
        </div>

        <div className="card ab-card">
          <div className="panel-head">
            <div>
              <p className="section-title">A/B testing board</p>
              <p className="muted">
                Pin two snapshots to compare chunk math, edge cleanliness, and retrieval scores side-by-side.
              </p>
            </div>
          </div>
          <div className="ab-columns">
            {renderVariantColumn("A", variantA)}
            {renderVariantColumn("B", variantB)}
          </div>
          <div className="delta-table">
            <div className="comparison-row heading">
              <span>Metric</span>
              <span>Variant A</span>
              <span>Variant B</span>
              <span>Δ (A − B)</span>
            </div>
            {comparisonRows.map((row) => (
              <div key={row.label} className="comparison-row">
                <span>{row.label}</span>
                <span>{row.valueA}</span>
                <span>{row.valueB}</span>
                <span>{row.delta}</span>
              </div>
            ))}
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
            {results.markdown.map((chunk) => {
              const { isRetrieved, score } = getRetrievalMeta("markdown", chunk.id);
              return (
                <article
                  key={`md-${chunk.id}`}
                  className={`chunk ${chunk.cleanBoundary ? "clean" : "risky"}${
                    isRetrieved ? " retrieved" : ""
                  }`}
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
                  {retrievalMatches.markdown && (
                    <span className={`chip ${isRetrieved ? "match" : "subtle"}`}>
                      {isRetrieved ? "Retrieved" : "Score"} {formatScore(score)}
                    </span>
                  )}
                </div>
                <pre>{chunk.content}</pre>
                </article>
              );
            })}
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
            {results.recursive.map((chunk) => {
              const { isRetrieved, score } = getRetrievalMeta("recursive", chunk.id);
              return (
                <article
                  key={`rc-${chunk.id}`}
                  className={`chunk ${chunk.cleanBoundary ? "clean" : "risky"}${
                    isRetrieved ? " retrieved" : ""
                  }`}
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
                  {retrievalMatches.recursive && (
                    <span className={`chip ${isRetrieved ? "match" : "subtle"}`}>
                      {isRetrieved ? "Retrieved" : "Score"} {formatScore(score)}
                    </span>
                  )}
                </div>
                <pre>{chunk.content}</pre>
                </article>
              );
            })}
          </div>
        </div>

        <div className="column">
          <div className="column-header">
            <div>
              <p className="eyebrow">Semantic embeddings</p>
              <p className="muted">
                Adjacent sentences stay together only when their MiniLM vectors
                stay similar; great for topic-driven slices.
              </p>
            </div>
            <button
              type="button"
              className="collapse-toggle"
              onClick={() => toggleSection("semantic")}
              aria-expanded={!collapsedSections.semantic}
              aria-controls="semantic-panel"
            >
              {collapsedSections.semantic ? "Expand" : "Collapse"}
            </button>
          </div>
          <div
            id="semantic-panel"
            className="chunk-list collapsible-content"
            hidden={collapsedSections.semantic}
            aria-hidden={collapsedSections.semantic}
          >
            <div className="note info">
              MiniLM (~85{"\u00a0"}MB) now preloads when the page opens and stays
              cached locally; check the chip above to see whether this session
              streamed new weights or reused the cache.
            </div>
            {semanticError && <div className="note warn">{semanticError}</div>}
            {results.semantic.map((chunk) => {
              const { isRetrieved, score } = getRetrievalMeta("semantic", chunk.id);
              return (
                <article
                  key={`sem-${chunk.id}`}
                  className={`chunk ${chunk.cleanBoundary ? "clean" : "risky"}${
                    isRetrieved ? " retrieved" : ""
                  }`}
                onMouseEnter={() =>
                  setFocusRange({
                    start: chunk.start,
                    end: chunk.end,
                    splitter: "semantic",
                    label: `SEM #${chunk.id + 1}`,
                  })
                }
                onMouseLeave={() => setFocusRange(null)}
                onClick={() =>
                  setFocusRange({
                    start: chunk.start,
                    end: chunk.end,
                    splitter: "semantic",
                    label: `SEM #${chunk.id + 1}`,
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
                  {retrievalMatches.semantic && (
                    <span className={`chip ${isRetrieved ? "match" : "subtle"}`}>
                      {isRetrieved ? "Retrieved" : "Score"} {formatScore(score)}
                    </span>
                  )}
                </div>
                <pre>{chunk.content}</pre>
                </article>
              );
            })}
          </div>
        </div>
      </section>

      {isLoading && <div className="loading">Re-chunking…</div>}
    </div>
  );
}

export default App;
