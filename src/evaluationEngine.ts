export type ChunkCard = {
  id: number;
  content: string;
  lineLabel?: string;
  charCount: number;
  cleanBoundary: boolean;
  start: number;
  end: number;
  splitter: "markdown" | "recursive" | "semantic";
};

export type GoldStandardPair = {
  id: string;
  question: string;
  answerSnippet: string;
  category?: string;
  difficulty?: "easy" | "medium" | "hard";
};

export type QuestionEvaluationResult = {
  questionId: string;
  question: string;
  answerSnippet: string;
  results: Record<ChunkCard["splitter"], {
    found: boolean;
    rank: number | null; // Position where answer was found (1-indexed), null if not found
    retrievedChunkIds: number[];
    scores: number[];
  }>;
};

export type EvaluationMetrics = {
  hitRate: number; // % of questions with answer in top-K
  mrr: number; // Mean Reciprocal Rank
  precision: number; // Relevant retrieved / Total retrieved
  recall: number; // Relevant retrieved / Total relevant
};

export type StrategyEvaluationMetrics = Record<ChunkCard["splitter"], EvaluationMetrics>;

type ChunkVectors = Record<ChunkCard["splitter"], number[][]>;

/**
 * Compute cosine similarity between two vectors
 */
function cosineSimilarity(a: number[], b: number[]): number {
  if (!a || !b || a.length === 0 || b.length === 0) return 0;

  let dot = 0;
  let normA = 0;
  let normB = 0;
  const minLen = Math.min(a.length, b.length);

  for (let i = 0; i < minLen; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  return denominator === 0 ? 0 : dot / denominator;
}

/**
 * Check if a text snippet contains the answer (fuzzy matching)
 */
function containsAnswer(chunkText: string, answerSnippet: string): boolean {
  if (!chunkText || !answerSnippet) return false;

  // Normalize both texts for comparison
  const normalizedChunk = chunkText.toLowerCase().trim();
  const normalizedAnswer = answerSnippet.toLowerCase().trim();

  // Direct substring match
  if (normalizedChunk.includes(normalizedAnswer)) {
    return true;
  }

  // Check if answer words appear in sequence (with some flexibility)
  const answerWords = normalizedAnswer.split(/\s+/).filter(w => w.length > 2);

  if (answerWords.length === 0) return false;

  // For short answers (1-3 words), require exact match
  if (answerWords.length <= 3) {
    return normalizedChunk.includes(normalizedAnswer);
  }

  // For longer answers, check if most words appear in order
  let matchCount = 0;
  let lastIndex = -1;

  for (const word of answerWords) {
    const index = normalizedChunk.indexOf(word, lastIndex + 1);
    if (index > lastIndex) {
      matchCount++;
      lastIndex = index;
    }
  }

  // Require at least 70% of words to match in order
  return matchCount >= answerWords.length * 0.7;
}

/**
 * Run retrieval for a single question against a strategy
 */
function retrieveForQuestion(
  queryVector: number[],
  chunkVectors: number[][],
  chunks: ChunkCard[],
  answerSnippet: string,
  topK: number,
  scoreFloor: number
): {
  found: boolean;
  rank: number | null;
  retrievedChunkIds: number[];
  scores: number[];
} {
  // Compute similarities
  const similarities = chunkVectors.map((chunkVec) =>
    cosineSimilarity(queryVector, chunkVec)
  );

  // Create scored chunks
  const scored = similarities.map((score, idx) => ({ idx, score }));

  // Filter by score threshold and sort descending
  const filtered = scored
    .filter(s => s.score >= scoreFloor)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);

  // Check each retrieved chunk for the answer
  let found = false;
  let rank: number | null = null;

  for (let i = 0; i < filtered.length; i++) {
    const chunkIdx = filtered[i].idx;
    const chunk = chunks[chunkIdx];

    if (chunk && containsAnswer(chunk.content, answerSnippet)) {
      found = true;
      rank = i + 1; // 1-indexed rank
      break;
    }
  }

  return {
    found,
    rank,
    retrievedChunkIds: filtered.map(s => s.idx),
    scores: filtered.map(s => s.score),
  };
}

/**
 * Run batch evaluation of all gold standard pairs against all strategies
 */
export async function runBatchEvaluation(
  goldStandardPairs: GoldStandardPair[],
  results: Record<ChunkCard["splitter"], ChunkCard[]>,
  chunkVectors: ChunkVectors,
  embedTexts: (texts: string[]) => Promise<number[][]>,
  topK: number = 3,
  scoreFloor: number = 0.58
): Promise<QuestionEvaluationResult[]> {
  if (goldStandardPairs.length === 0) {
    throw new Error("No gold standard pairs to evaluate");
  }

  // Embed all questions at once for efficiency
  const questions = goldStandardPairs.map(pair => pair.question);
  const questionVectors = await embedTexts(questions);

  // Evaluate each question against each strategy
  const evaluationResults: QuestionEvaluationResult[] = [];

  for (let i = 0; i < goldStandardPairs.length; i++) {
    const pair = goldStandardPairs[i];
    const queryVector = questionVectors[i];

    const result: QuestionEvaluationResult = {
      questionId: pair.id,
      question: pair.question,
      answerSnippet: pair.answerSnippet,
      results: {
        markdown: retrieveForQuestion(
          queryVector,
          chunkVectors.markdown,
          results.markdown,
          pair.answerSnippet,
          topK,
          scoreFloor
        ),
        recursive: retrieveForQuestion(
          queryVector,
          chunkVectors.recursive,
          results.recursive,
          pair.answerSnippet,
          topK,
          scoreFloor
        ),
        semantic: retrieveForQuestion(
          queryVector,
          chunkVectors.semantic,
          results.semantic,
          pair.answerSnippet,
          topK,
          scoreFloor
        ),
      },
    };

    evaluationResults.push(result);
  }

  return evaluationResults;
}

/**
 * Calculate evaluation metrics from batch results
 */
export function calculateMetrics(
  evaluationResults: QuestionEvaluationResult[]
): StrategyEvaluationMetrics {
  const strategies: ChunkCard["splitter"][] = ["markdown", "recursive", "semantic"];
  const metrics: Partial<StrategyEvaluationMetrics> = {};

  for (const strategy of strategies) {
    let hits = 0;
    let totalReciprocalRank = 0;
    let totalRelevantRetrieved = 0;
    let totalRetrieved = 0;
    const totalRelevant = evaluationResults.length; // Each question has at least 1 relevant chunk

    for (const result of evaluationResults) {
      const strategyResult = result.results[strategy];

      // Hit Rate: did we find the answer?
      if (strategyResult.found) {
        hits++;
      }

      // MRR: reciprocal of rank (if found)
      if (strategyResult.rank !== null) {
        totalReciprocalRank += 1 / strategyResult.rank;
      }

      // Precision & Recall: count relevant chunks
      totalRetrieved += strategyResult.retrievedChunkIds.length;

      if (strategyResult.found) {
        totalRelevantRetrieved += 1; // Found the answer = 1 relevant chunk retrieved
      }
    }

    const questionCount = evaluationResults.length;

    metrics[strategy] = {
      hitRate: questionCount > 0 ? hits / questionCount : 0,
      mrr: questionCount > 0 ? totalReciprocalRank / questionCount : 0,
      precision: totalRetrieved > 0 ? totalRelevantRetrieved / totalRetrieved : 0,
      recall: totalRelevant > 0 ? totalRelevantRetrieved / totalRelevant : 0,
    };
  }

  return metrics as StrategyEvaluationMetrics;
}

/**
 * Format metrics for display (percentages and decimals)
 */
export function formatMetric(value: number, type: "percentage" | "decimal"): string {
  if (type === "percentage") {
    return `${(value * 100).toFixed(1)}%`;
  } else {
    return value.toFixed(3);
  }
}

/**
 * Determine which strategy performs best for a given metric
 */
export function getBestStrategy(
  metrics: StrategyEvaluationMetrics,
  metricKey: keyof EvaluationMetrics
): ChunkCard["splitter"] | null {
  const strategies: ChunkCard["splitter"][] = ["markdown", "recursive", "semantic"];

  let bestStrategy: ChunkCard["splitter"] | null = null;
  let bestValue = -Infinity;

  for (const strategy of strategies) {
    const value = metrics[strategy][metricKey];
    if (value > bestValue) {
      bestValue = value;
      bestStrategy = strategy;
    }
  }

  return bestStrategy;
}

/**
 * Calculate delta between two metric values
 */
export function calculateMetricDelta(
  current: number,
  previous: number
): { value: number; direction: "up" | "down" | "neutral" } {
  const delta = current - previous;
  const direction = delta > 0.001 ? "up" : delta < -0.001 ? "down" : "neutral";

  return { value: delta, direction };
}
