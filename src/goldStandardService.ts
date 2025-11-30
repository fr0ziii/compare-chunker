import Anthropic from "@anthropic-ai/sdk";

export type GoldStandardPair = {
  id: string;
  question: string;
  answerSnippet: string;
  category?: string;
  difficulty?: "easy" | "medium" | "hard";
};

export type LLMProvider = "local" | "anthropic";

export type LLMConfig = {
  provider: LLMProvider;
  apiKey?: string;
  temperature?: number;
  pairCount?: number;
};

type Text2TextGenerationPipeline = (
  input: string,
  options?: Record<string, unknown>,
) => Promise<{ generated_text: string }[]>;

let localLLMPipelinePromise: Promise<Text2TextGenerationPipeline> | null = null;

/**
 * Load the local LLM pipeline (Flan-T5-small) using Transformers.js
 */
async function loadLocalLLM(): Promise<Text2TextGenerationPipeline> {
  if (!localLLMPipelinePromise) {
    localLLMPipelinePromise = (async () => {
      const transformers = await import("@xenova/transformers");
      const pipelineFactory = transformers.pipeline as unknown as (
        task: string,
        model?: string,
      ) => Promise<Text2TextGenerationPipeline>;

      if (transformers.env) {
        transformers.env.allowLocalModels = false;
        transformers.env.useBrowserCache = true;
      }

      // Using Flan-T5-small for text generation (77M params, ~300MB)
      // Alternative: "Xenova/flan-t5-base" for better quality but larger size (~900MB)
      return pipelineFactory("text2text-generation", "Xenova/flan-t5-small");
    })();
  }
  return localLLMPipelinePromise;
}

/**
 * Generate Q&A pairs using local LLM (Flan-T5)
 */
async function generateWithLocalLLM(
  text: string,
  count: number,
  temperature: number
): Promise<GoldStandardPair[]> {
  try {
    const pipeline = await loadLocalLLM();

    // Truncate text if too long for local model (max ~512 tokens)
    const maxChars = 2000;
    const truncatedText = text.length > maxChars
      ? text.slice(0, maxChars) + "..."
      : text;

    // Prompt engineering for Flan-T5
    const prompt = `Generate ${count} question-answer pairs from this document. Format each as "Q: [question] A: [answer]" on separate lines.

Document:
${truncatedText}

Question-answer pairs:`;

    const result = await pipeline(prompt, {
      max_new_tokens: 500,
      temperature,
      do_sample: temperature > 0,
      top_k: 50,
      top_p: 0.9,
    });

    // Parse the generated text into structured pairs
    const generatedText = result[0]?.generated_text || "";
    return parseQAPairsFromText(generatedText, "local");

  } catch (error) {
    console.error("Local LLM generation failed:", error);
    throw new Error(
      `Failed to generate with local LLM: ${error instanceof Error ? error.message : "Unknown error"}`
    );
  }
}

/**
 * Generate Q&A pairs using Anthropic Claude API
 */
async function generateWithAnthropicAPI(
  text: string,
  count: number,
  temperature: number,
  apiKey?: string
): Promise<GoldStandardPair[]> {
  if (!apiKey || apiKey.trim() === "") {
    throw new Error("Anthropic API key is required. Please set it in the settings.");
  }

  try {
    const client = new Anthropic({
      apiKey,
      dangerouslyAllowBrowser: true, // Required for client-side usage
    });

    // Truncate text if too long (max ~100k tokens for Claude)
    const maxChars = 100000;
    const truncatedText = text.length > maxChars
      ? text.slice(0, maxChars) + "..."
      : text;

    const prompt = `You are a helpful assistant that generates high-quality question-answer pairs for evaluating document chunking and retrieval systems.

Your task: Generate exactly ${count} diverse question-answer pairs from the following document. Each pair should:
1. Test understanding of specific facts, concepts, or relationships in the document
2. Have clear, concise answers that can be found in the document
3. Vary in difficulty (easy factual recall, medium comprehension, hard inference)
4. Cover different topics/sections of the document

Return ONLY a valid JSON array with this exact structure (no markdown formatting, no backticks):
[
  {
    "question": "What is...",
    "answerSnippet": "The answer is...",
    "category": "factual|conceptual|inference",
    "difficulty": "easy|medium|hard"
  }
]

Document:
${truncatedText}`;

    const message = await client.messages.create({
      model: "claude-3-5-sonnet-20241022",
      max_tokens: 2000,
      temperature,
      messages: [
        {
          role: "user",
          content: prompt,
        },
      ],
    });

    // Extract the text content from Claude's response
    const responseText = message.content
      .filter((block) => block.type === "text")
      .map((block) => (block as { type: "text"; text: string }).text)
      .join("\n");

    // Parse JSON response
    return parseQAPairsFromJSON(responseText, "anthropic");

  } catch (error) {
    console.error("Anthropic API generation failed:", error);
    if (error instanceof Error && error.message.includes("401")) {
      throw new Error("Invalid API key. Please check your Anthropic API key in settings.");
    }
    throw new Error(
      `Failed to generate with Anthropic API: ${error instanceof Error ? error.message : "Unknown error"}`
    );
  }
}

/**
 * Parse Q&A pairs from JSON response (Anthropic)
 */
function parseQAPairsFromJSON(responseText: string, source: string): GoldStandardPair[] {
  try {
    // Remove markdown code blocks if present
    let cleaned = responseText.trim();
    if (cleaned.startsWith("```json")) {
      cleaned = cleaned.replace(/^```json\s*/, "").replace(/```\s*$/, "");
    } else if (cleaned.startsWith("```")) {
      cleaned = cleaned.replace(/^```\s*/, "").replace(/```\s*$/, "");
    }

    const parsed = JSON.parse(cleaned);

    if (!Array.isArray(parsed)) {
      throw new Error("Response is not an array");
    }

    return parsed.map((item, index) => ({
      id: `${source}-${Date.now()}-${index}`,
      question: item.question || "",
      answerSnippet: item.answerSnippet || item.answer || "",
      category: item.category,
      difficulty: item.difficulty,
    })).filter(pair => pair.question && pair.answerSnippet);

  } catch (error) {
    console.error("JSON parsing failed:", error);
    console.error("Response text:", responseText);
    throw new Error(
      `Failed to parse LLM response as JSON. Please try again or switch to local LLM.`
    );
  }
}

/**
 * Parse Q&A pairs from plain text response (Local LLM)
 * Expected format: "Q: [question] A: [answer]" on separate lines
 */
function parseQAPairsFromText(responseText: string, source: string): GoldStandardPair[] {
  try {
    const pairs: GoldStandardPair[] = [];
    const lines = responseText.split("\n").filter(line => line.trim());

    let currentQuestion = "";

    for (const line of lines) {
      const trimmed = line.trim();

      if (trimmed.startsWith("Q:") || trimmed.startsWith("Question:")) {
        currentQuestion = trimmed.replace(/^(Q:|Question:)\s*/i, "");
      } else if (trimmed.startsWith("A:") || trimmed.startsWith("Answer:")) {
        const answer = trimmed.replace(/^(A:|Answer:)\s*/i, "");
        if (currentQuestion && answer) {
          pairs.push({
            id: `${source}-${Date.now()}-${pairs.length}`,
            question: currentQuestion,
            answerSnippet: answer,
            category: "general",
            difficulty: "medium",
          });
          currentQuestion = "";
        }
      }
    }

    // If no pairs found, try a simpler pattern
    if (pairs.length === 0) {
      // Try to find numbered patterns like "1. What is... Answer: ..."
      const matches = responseText.matchAll(/(\d+)\.\s*([^?]+\?)\s*(?:Answer:)?\s*([^1-9]+)/gi);
      for (const match of matches) {
        if (match[2] && match[3]) {
          pairs.push({
            id: `${source}-${Date.now()}-${pairs.length}`,
            question: match[2].trim(),
            answerSnippet: match[3].trim(),
            category: "general",
            difficulty: "medium",
          });
        }
      }
    }

    return pairs;

  } catch (error) {
    console.error("Text parsing failed:", error);
    throw new Error("Failed to parse Q&A pairs from local LLM response");
  }
}

/**
 * Main function to generate Q&A pairs using the configured LLM provider
 */
export async function generateQAPairs(
  text: string,
  config: LLMConfig
): Promise<GoldStandardPair[]> {
  const count = config.pairCount || 10;
  const temperature = config.temperature || 0.7;

  if (!text || text.trim().length < 100) {
    throw new Error("Document is too short. Please provide at least 100 characters of text.");
  }

  if (config.provider === "anthropic") {
    return generateWithAnthropicAPI(text, count, temperature, config.apiKey);
  } else if (config.provider === "local") {
    return generateWithLocalLLM(text, count, temperature);
  } else {
    throw new Error(`Unknown LLM provider: ${config.provider}`);
  }
}

/**
 * Validate API key format (basic check)
 */
export function validateApiKey(provider: LLMProvider, apiKey: string): boolean {
  if (provider === "local") {
    return true; // No API key needed for local
  }

  if (provider === "anthropic") {
    // Anthropic keys start with "sk-ant-"
    return apiKey.startsWith("sk-ant-") && apiKey.length > 20;
  }

  return false;
}
