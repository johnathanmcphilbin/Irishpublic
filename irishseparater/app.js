import { env, pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const ABairEndpoint = "https://recognition.abair.ie/v3-5/transcribe";
const englishModelId = "Xenova/whisper-tiny.en";

env.allowLocalModels = false;

const strongIrishWords = new Set([
  "agus",
  "bhfuil",
  "bhí",
  "cónaí",
  "ceart",
  "chuig",
  "duit",
  "dom",
  "focal",
  "gaeilge",
  "go",
  "go raibh",
  "maith",
  "agat",
  "agam",
  "anseo",
  "anois",
  "inniu",
  "leis",
  "liom",
  "mhaith",
  "mé",
  "muid",
  "ní",
  "seo",
  "sin",
  "tá",
  "thabhairt",
  "uait",
]);

const mediumIrishWords = new Set([
  "ag",
  "an",
  "ar",
  "as",
  "atá",
  "de",
  "do",
  "faoi",
  "is",
  "le",
  "mar",
  "mo",
  "ó",
  "sa",
  "siad",
  "sibh",
  "tú",
]);

const englishWords = new Set([
  "a",
  "about",
  "and",
  "are",
  "be",
  "can",
  "do",
  "english",
  "for",
  "hello",
  "how",
  "i",
  "if",
  "in",
  "is",
  "it",
  "my",
  "of",
  "on",
  "please",
  "recording",
  "speech",
  "test",
  "that",
  "the",
  "this",
  "to",
  "we",
  "what",
  "with",
  "you",
]);

const recordButton = document.querySelector("#recordButton");
const resetButton = document.querySelector("#resetButton");
const statusText = document.querySelector("#statusText");
const timerText = document.querySelector("#timerText");
const englishModelStatus = document.querySelector("#englishModelStatus");
const irishMeter = document.querySelector("#irishMeter");
const englishMeter = document.querySelector("#englishMeter");
const irishPercentLabel = document.querySelector("#irishPercentLabel");
const englishPercentLabel = document.querySelector("#englishPercentLabel");
const irishConfidenceValue = document.querySelector("#irishConfidenceValue");
const coverageValue = document.querySelector("#coverageValue");
const tokenCountValue = document.querySelector("#tokenCountValue");
const durationValue = document.querySelector("#durationValue");
const summaryText = document.querySelector("#summaryText");
const comparisonText = document.querySelector("#comparisonText");
const mergedTranscriptText = document.querySelector("#mergedTranscriptText");
const abairTranscriptText = document.querySelector("#abairTranscriptText");
const englishTranscriptText = document.querySelector("#englishTranscriptText");
const irishWordList = document.querySelector("#irishWordList");
const englishWordList = document.querySelector("#englishWordList");
const uncertainWordList = document.querySelector("#uncertainWordList");

let mediaRecorder = null;
let mediaStream = null;
let audioChunks = [];
let recordingStartedAt = 0;
let timerId = null;
let englishTranscriber = null;
let englishTranscriberPromise = null;

function setEnglishModelStatus(message) {
  englishModelStatus.textContent = message;
}

function formatClock(seconds) {
  const wholeSeconds = Math.max(0, Math.floor(seconds));
  const minutes = String(Math.floor(wholeSeconds / 60)).padStart(2, "0");
  const remainder = String(wholeSeconds % 60).padStart(2, "0");
  return `${minutes}:${remainder}`;
}

function updateTimer() {
  if (!recordingStartedAt) {
    timerText.textContent = "00:00";
    return;
  }

  const elapsedSeconds = (Date.now() - recordingStartedAt) / 1000;
  timerText.textContent = formatClock(elapsedSeconds);
}

function resetResults() {
  irishMeter.style.width = "0%";
  englishMeter.style.width = "0%";
  irishPercentLabel.textContent = "0%";
  englishPercentLabel.textContent = "0%";
  irishConfidenceValue.textContent = "0%";
  coverageValue.textContent = "0%";
  tokenCountValue.textContent = "0";
  durationValue.textContent = "0.0s";
  summaryText.textContent = "Record audio to generate an estimate.";
  comparisonText.textContent = "Waiting for a recording.";
  mergedTranscriptText.textContent = "No merged transcript yet.";
  abairTranscriptText.textContent = "No Abair transcript yet.";
  englishTranscriptText.textContent = "No English transcript yet.";
  irishWordList.textContent = "No Irish words detected yet.";
  englishWordList.textContent = "No English words detected yet.";
  uncertainWordList.textContent = "No uncertain words yet.";
  irishWordList.className = "token-list empty-state";
  englishWordList.className = "token-list empty-state";
  uncertainWordList.className = "token-list empty-state";
}

function setStatus(message) {
  statusText.textContent = message;
}

function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-záéíóú'-\s]/gi, " ")
    .split(/\s+/)
    .map((token) => token.trim())
    .filter(Boolean);
}

function normalizeToken(token) {
  return token
    .toLowerCase()
    .replace(/^[^a-záéíóú]+|[^a-záéíóú]+$/gi, "")
    .replace(/['’-]+/g, "-");
}

function extractDisplayTokens(text) {
  return text.match(/[A-Za-zÁÉÍÓÚáéíóú'-]+/g) || [];
}

function scoreIrishToken(token) {
  let score = 0;

  if (/[áéíóú]/.test(token)) {
    score += 3;
  }

  if (strongIrishWords.has(token)) {
    score += 3;
  } else if (mediumIrishWords.has(token)) {
    score += 1.5;
  }

  if (/^(bhf|bh|ch|dh|fh|gh|mh|ph|sh|th)/.test(token)) {
    score += 1;
  }

  if (/^(n-|t-)/.test(token)) {
    score += 1;
  }

  if (token.endsWith("aidh") || token.endsWith("acht") || token.endsWith("eacht")) {
    score += 1;
  }

  return score;
}

function scoreEnglishToken(token) {
  let score = 0;

  if (englishWords.has(token)) {
    score += 2;
  }

  if (/[wqjkx]/.test(token)) {
    score += 0.75;
  }

  if (token.endsWith("ing") || token.endsWith("tion") || token.endsWith("ed")) {
    score += 0.75;
  }

  return score;
}

function classifyToken(token) {
  const irishScore = scoreIrishToken(token);
  const englishScore = scoreEnglishToken(token);
  const totalScore = irishScore + englishScore;

  let label = "uncertain";
  let confidence = 0;

  if (totalScore > 0) {
    const dominantScore = Math.max(irishScore, englishScore);
    const secondaryScore = Math.min(irishScore, englishScore);
    const separation = (dominantScore - secondaryScore) / totalScore;
    const strength = Math.min(1, totalScore / 4);
    confidence = Math.round((0.65 * separation + 0.35 * strength) * 100);

    if (irishScore > englishScore) {
      label = confidence >= 35 ? "irish" : "uncertain";
    } else if (englishScore > irishScore) {
      label = confidence >= 35 ? "english" : "uncertain";
    }
  }

  return {
    token,
    irishScore,
    englishScore,
    label,
    confidence,
  };
}

function classifyEnglishTranscriptToken(token) {
  const normalized = normalizeToken(token);
  const irishScore = scoreIrishToken(normalized);
  const englishScore = Math.max(scoreEnglishToken(normalized), 2);
  const englishLead = englishScore - irishScore;

  if (irishScore >= englishScore + 1.5) {
    return {
      token,
      irishScore,
      englishScore,
      label: "uncertain",
      confidence: 38,
      source: "english",
    };
  }

  return {
    token,
    irishScore,
    englishScore,
    label: englishLead > 0 ? "english" : "uncertain",
    confidence: Math.max(42, Math.min(94, Math.round(60 + englishLead * 10))),
    source: "english",
  };
}

function analyseWordBreakdown(wordBreakdown, durationSeconds) {
  const tokens = wordBreakdown.map((item) => normalizeToken(item.token)).filter(Boolean);
  const matchedTokens = wordBreakdown.filter((item) => item.irishScore > 0 || item.englishScore > 0).length;
  const irishScore = wordBreakdown.reduce((total, item) => total + item.irishScore, 0);
  const englishScore = wordBreakdown.reduce((total, item) => total + item.englishScore, 0);
  const combinedScore = irishScore + englishScore;
  const neutralSplit = combinedScore === 0;
  const irishShare = neutralSplit ? 0 : irishScore / combinedScore;
  const englishShare = neutralSplit ? 0 : englishScore / combinedScore;
  const coverage = wordBreakdown.length === 0 ? 0 : matchedTokens / wordBreakdown.length;
  const separation = neutralSplit ? 0 : Math.abs(irishShare - englishShare);
  const durationFactor = Math.min(1, durationSeconds / 10);
  const tokenFactor = Math.min(1, wordBreakdown.length / 20);
  const confidence = Math.round((0.45 * coverage + 0.35 * separation + 0.1 * durationFactor + 0.1 * tokenFactor) * 100);

  return {
    tokens,
    wordBreakdown,
    irishPercent: Math.round(irishShare * 100),
    englishPercent: Math.round(englishShare * 100),
    confidence,
    coverage: Math.round(coverage * 100),
  };
}

function renderTokenList(container, items, emptyMessage, className) {
  if (!items.length) {
    container.textContent = emptyMessage;
    container.className = "token-list empty-state";
    return;
  }

  container.className = "token-list";
  container.replaceChildren(
    ...items.map((item) => {
      const chip = document.createElement("div");
      chip.className = `token-chip ${className}`;

      const word = document.createElement("span");
      word.className = "token-word";
      word.textContent = item.token;

      const meta = document.createElement("span");
      meta.className = "token-meta";
      meta.textContent = `${item.confidence}% confidence`;

      chip.append(word, meta);
      return chip;
    })
  );
}

function analyseTranscript(transcript, durationSeconds) {
  const tokens = tokenize(transcript);
  const displayTokens = extractDisplayTokens(transcript);
  let irishScore = 0;
  let englishScore = 0;
  let matchedTokens = 0;
  const wordBreakdown = [];

  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index];
    const classification = classifyToken(token);
    const irishTokenScore = classification.irishScore;
    const englishTokenScore = classification.englishScore;

    if (irishTokenScore > 0 || englishTokenScore > 0) {
      matchedTokens += 1;
    }

    irishScore += irishTokenScore;
    englishScore += englishTokenScore;
    wordBreakdown.push({
      ...classification,
      token: displayTokens[index] || token,
    });
  }

  const combinedScore = irishScore + englishScore;
  const neutralSplit = combinedScore === 0;
  const irishShare = neutralSplit ? 0 : irishScore / combinedScore;
  const englishShare = neutralSplit ? 0 : englishScore / combinedScore;
  const coverage = tokens.length === 0 ? 0 : matchedTokens / tokens.length;
  const separation = neutralSplit ? 0 : Math.abs(irishShare - englishShare);
  const durationFactor = Math.min(1, durationSeconds / 10);
  const tokenFactor = Math.min(1, tokens.length / 20);
  const confidence = Math.round((0.45 * coverage + 0.35 * separation + 0.1 * durationFactor + 0.1 * tokenFactor) * 100);

  return {
    tokens,
    wordBreakdown,
    irishPercent: Math.round(irishShare * 100),
    englishPercent: Math.round(englishShare * 100),
    confidence,
    coverage: Math.round(coverage * 100),
  };
}

function tokenSimilarity(left, right) {
  const normalizedLeft = normalizeToken(left);
  const normalizedRight = normalizeToken(right);

  if (!normalizedLeft || !normalizedRight) {
    return -1;
  }

  if (normalizedLeft === normalizedRight) {
    return 2;
  }

  if (normalizedLeft.replace(/-/g, "") === normalizedRight.replace(/-/g, "")) {
    return 1.5;
  }

  if (
    normalizedLeft.length > 3 &&
    normalizedRight.length > 3 &&
    (normalizedLeft.startsWith(normalizedRight) || normalizedRight.startsWith(normalizedLeft))
  ) {
    return 0.5;
  }

  return -1;
}

function alignTokenSequences(leftTokens, rightTokens) {
  const rows = leftTokens.length + 1;
  const columns = rightTokens.length + 1;
  const gapPenalty = -1;
  const scores = Array.from({ length: rows }, () => Array(columns).fill(0));
  const moves = Array.from({ length: rows }, () => Array(columns).fill(""));

  for (let row = 1; row < rows; row += 1) {
    scores[row][0] = row * gapPenalty;
    moves[row][0] = "up";
  }

  for (let column = 1; column < columns; column += 1) {
    scores[0][column] = column * gapPenalty;
    moves[0][column] = "left";
  }

  for (let row = 1; row < rows; row += 1) {
    for (let column = 1; column < columns; column += 1) {
      const diagonal = scores[row - 1][column - 1] + tokenSimilarity(leftTokens[row - 1], rightTokens[column - 1]);
      const up = scores[row - 1][column] + gapPenalty;
      const left = scores[row][column - 1] + gapPenalty;
      const bestScore = Math.max(diagonal, up, left);
      scores[row][column] = bestScore;

      if (bestScore === diagonal) {
        moves[row][column] = "diag";
      } else if (bestScore === up) {
        moves[row][column] = "up";
      } else {
        moves[row][column] = "left";
      }
    }
  }

  const alignment = [];
  let row = leftTokens.length;
  let column = rightTokens.length;

  while (row > 0 || column > 0) {
    const move = moves[row][column];

    if (move === "diag") {
      alignment.push([row - 1, column - 1]);
      row -= 1;
      column -= 1;
    } else if (move === "up") {
      alignment.push([row - 1, null]);
      row -= 1;
    } else {
      alignment.push([null, column - 1]);
      column -= 1;
    }
  }

  return alignment.reverse();
}

function stitchTranscript(tokens) {
  return tokens.join(" ").replace(/\s+([,.;!?])/g, "$1").trim();
}

function findTrustedEnglishCutoff(alignment, abairTokens, englishTokens) {
  let trustedEnglishCutoff = -1;
  let matchedAnchorCount = 0;
  let divergenceStreak = 0;

  for (const [abairIndex, englishIndex] of alignment) {
    if (englishIndex === null) {
      continue;
    }

    const englishToken = englishTokens[englishIndex];
    const englishItem = classifyEnglishTranscriptToken(englishToken);
    const similarity = abairIndex === null ? -1 : tokenSimilarity(abairTokens[abairIndex], englishToken);
    const strongAnchor = similarity >= 0.5;
    const englishLooksClean = englishItem.label === "english" && englishItem.confidence >= 55;

    if (strongAnchor) {
      matchedAnchorCount += 1;
      divergenceStreak = 0;
      trustedEnglishCutoff = englishIndex;
      continue;
    }

    if (matchedAnchorCount < 2 && englishLooksClean) {
      trustedEnglishCutoff = englishIndex;
      continue;
    }

    divergenceStreak += 1;

    if (matchedAnchorCount >= 2 && divergenceStreak >= 2) {
      break;
    }
  }

  return trustedEnglishCutoff;
}

function findHandoffIndices(alignment, trustedEnglishCutoff) {
  let handoffEnglishIndex = trustedEnglishCutoff;
  let handoffAbairIndex = -1;

  for (const [abairIndex, englishIndex] of alignment) {
    if (englishIndex === null || englishIndex > trustedEnglishCutoff) {
      continue;
    }

    if (abairIndex !== null) {
      handoffAbairIndex = abairIndex;
    }
  }

  return {
    handoffEnglishIndex,
    handoffAbairIndex,
  };
}

function dedupeBoundary(englishPrefix, abairSuffix) {
  if (englishPrefix.length === 0 || abairSuffix.length === 0) {
    return { englishPrefix, abairSuffix };
  }

  const lastEnglish = englishPrefix[englishPrefix.length - 1];
  const firstAbair = abairSuffix[0];

  if (tokenSimilarity(lastEnglish.token, firstAbair.token) >= 1.5) {
    return {
      englishPrefix,
      abairSuffix: abairSuffix.slice(1),
    };
  }

  return { englishPrefix, abairSuffix };
}

function buildMergedConversation(abairTranscript, englishTranscript) {
  const abairTokens = extractDisplayTokens(abairTranscript);
  const englishTokens = englishTranscript.words?.length
    ? englishTranscript.words.map((item) => item.token)
    : extractDisplayTokens(englishTranscript.text || "");
  const abairBreakdown = abairTokens.map((token) => {
    const normalized = normalizeToken(token);
    return {
      ...classifyToken(normalized),
      token,
      source: "abair",
    };
  });
  const englishBreakdown = englishTokens.map((token) => classifyEnglishTranscriptToken(token));
  const alignment = alignTokenSequences(abairTokens, englishTokens);
  const trustedEnglishCutoff = findTrustedEnglishCutoff(alignment, abairTokens, englishTokens);
  const { handoffEnglishIndex, handoffAbairIndex } = findHandoffIndices(alignment, trustedEnglishCutoff);

  let englishPrefix = [];
  let abairSuffix = [];

  if (handoffEnglishIndex >= 0) {
    englishPrefix = englishBreakdown
      .slice(0, handoffEnglishIndex + 1)
      .filter((item) => item.label === "english" && item.confidence >= 55)
      .map((item) => ({
        ...item,
        source: "english-prefix",
      }));
  }

  if (handoffAbairIndex >= 0) {
    abairSuffix = abairBreakdown.slice(handoffAbairIndex + 1).map((item) => ({
      ...item,
      source: "abair-suffix",
    }));
  } else {
    abairSuffix = abairBreakdown.map((item) => ({
      ...item,
      source: "abair-only",
    }));
  }

  const dedupedBoundary = dedupeBoundary(englishPrefix, abairSuffix);
  englishPrefix = dedupedBoundary.englishPrefix;
  abairSuffix = dedupedBoundary.abairSuffix;

  const mergedBreakdown = [...englishPrefix, ...abairSuffix];
  const acceptedEnglishOnlyCount = englishPrefix.length;
  const ignoredUntrustedEnglishCount = Math.max(0, englishBreakdown.length - englishPrefix.length);
  const ignoredEnglishOnlyCount = 0;
  const abairReplacementCount = abairSuffix.length;

  const mergedTranscript = stitchTranscript(mergedBreakdown.map((item) => item.token));

  return {
    mergedTranscript,
    mergedBreakdown,
    abairReplacementCount,
    ignoredEnglishOnlyCount,
    ignoredUntrustedEnglishCount,
    acceptedEnglishOnlyCount,
    trustedEnglishCutoff,
    abairTokens: abairTokens.length,
    englishTokens: englishTokens.length,
  };
}

function buildComparisonSummary({
  abairTranscript,
  englishTranscript,
  mergedConversation,
  errors,
}) {
  const fragments = [];

  if (abairTranscript) {
    fragments.push(`Abair returned ${mergedConversation.abairTokens} words.`);
  } else {
    fragments.push("Abair did not return a usable transcript.");
  }

  if (englishTranscript.text) {
    fragments.push(`The local English recognizer returned ${mergedConversation.englishTokens} words.`);
  } else {
    fragments.push("The local English recognizer did not return a usable transcript.");
  }

  if (mergedConversation.acceptedEnglishOnlyCount > 0) {
    fragments.push(`${mergedConversation.acceptedEnglishOnlyCount} English words were kept before the drift point.`);
  }

  if (mergedConversation.abairReplacementCount > 0) {
    fragments.push(`${mergedConversation.abairReplacementCount} Abair words were used from the handoff onward.`);
  }

  if (mergedConversation.ignoredEnglishOnlyCount > 0) {
    fragments.push(`${mergedConversation.ignoredEnglishOnlyCount} weak English-only extras were ignored.`);
  }

  if (mergedConversation.ignoredUntrustedEnglishCount > 0) {
    fragments.push(`${mergedConversation.ignoredUntrustedEnglishCount} English words after the drift point were ignored and Abair was preferred instead.`);
  }

  if (errors.length > 0) {
    fragments.push(errors.join(" "));
  }

  return fragments.join(" ");
}

function renderAnalysis({
  abairTranscript,
  englishTranscript,
  mergedTranscript,
  mergedBreakdown,
  durationSeconds,
  comparisonSummary,
}) {
  const analysis = analyseWordBreakdown(mergedBreakdown, durationSeconds);
  const roundedDuration = durationSeconds.toFixed(1);
  const irishPercent = analysis.irishPercent;
  const englishPercent = analysis.englishPercent;
  const irishWords = analysis.wordBreakdown.filter((item) => item.label === "irish");
  const englishWords = analysis.wordBreakdown.filter((item) => item.label === "english");
  const uncertainWords = analysis.wordBreakdown.filter((item) => item.label === "uncertain");

  irishMeter.style.width = `${irishPercent}%`;
  englishMeter.style.width = `${englishPercent}%`;
  irishPercentLabel.textContent = `${irishPercent}%`;
  englishPercentLabel.textContent = `${englishPercent}%`;
  irishConfidenceValue.textContent = `${analysis.confidence}%`;
  coverageValue.textContent = `${analysis.coverage}%`;
  tokenCountValue.textContent = String(analysis.tokens.length);
  durationValue.textContent = `${roundedDuration}s`;
  comparisonText.textContent = comparisonSummary;
  mergedTranscriptText.textContent = mergedTranscript || "No merged transcript returned.";
  abairTranscriptText.textContent = abairTranscript || "No Abair transcript returned.";
  englishTranscriptText.textContent = englishTranscript || "No English transcript returned.";
  renderTokenList(irishWordList, irishWords, "No Irish words detected.", "irish-token");
  renderTokenList(englishWordList, englishWords, "No English words detected.", "english-token");
  renderTokenList(uncertainWordList, uncertainWords, "No uncertain words.", "uncertain-token");

  if (!analysis.tokens.length) {
    summaryText.textContent = "The recognizers did not produce enough usable words to estimate the language mix.";
    return;
  }

  summaryText.textContent = `Estimated Irish usage: ${irishPercent}%. Estimated English usage: ${englishPercent}%. Irish-confidence score: ${analysis.confidence}%. Transcript coverage: ${analysis.coverage}%. Irish words: ${irishWords.length}. English words: ${englishWords.length}. Uncertain words: ${uncertainWords.length}.`;
}

async function decodeAudioToMono(blob) {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;

  if (!AudioContextClass) {
    throw new Error("This browser does not support AudioContext, so the English recognizer cannot run.");
  }

  const audioContext = new AudioContextClass();

  try {
    const audioBuffer = await audioContext.decodeAudioData(await blob.arrayBuffer());
    const mono = new Float32Array(audioBuffer.length);

    for (let channelIndex = 0; channelIndex < audioBuffer.numberOfChannels; channelIndex += 1) {
      const channelData = audioBuffer.getChannelData(channelIndex);

      for (let sampleIndex = 0; sampleIndex < audioBuffer.length; sampleIndex += 1) {
        mono[sampleIndex] += channelData[sampleIndex] / audioBuffer.numberOfChannels;
      }
    }

    return {
      samples: mono,
      sampleRate: audioBuffer.sampleRate,
    };
  } finally {
    if (typeof audioContext.close === "function") {
      await audioContext.close();
    }
  }
}

async function resampleAudio(samples, inputRate, targetRate) {
  if (inputRate === targetRate) {
    return samples;
  }

  const OfflineAudioContextClass = window.OfflineAudioContext || window.webkitOfflineAudioContext;

  if (!OfflineAudioContextClass) {
    throw new Error("This browser does not support OfflineAudioContext, so the English recognizer cannot resample audio.");
  }

  const frameCount = Math.ceil(samples.length * targetRate / inputRate);
  const offlineContext = new OfflineAudioContextClass(1, frameCount, targetRate);
  const buffer = offlineContext.createBuffer(1, samples.length, inputRate);
  buffer.copyToChannel(samples, 0);

  const source = offlineContext.createBufferSource();
  source.buffer = buffer;
  source.connect(offlineContext.destination);
  source.start(0);

  const rendered = await offlineContext.startRendering();
  return new Float32Array(rendered.getChannelData(0));
}

function normalizeWhisperChunks(chunks) {
  const words = [];

  for (const chunk of chunks || []) {
    const rawToken = typeof chunk.text === "string" ? chunk.text.trim() : "";

    if (!rawToken) {
      continue;
    }

    for (const token of extractDisplayTokens(rawToken)) {
      words.push({
        token,
        timestamp: Array.isArray(chunk.timestamp) ? chunk.timestamp : null,
      });
    }
  }

  return words;
}

async function getEnglishTranscriber() {
  if (englishTranscriber) {
    return englishTranscriber;
  }

  if (!englishTranscriberPromise) {
    setEnglishModelStatus("Loading local model...");
    englishTranscriberPromise = pipeline("automatic-speech-recognition", englishModelId, {
      dtype: "fp32",
      device: "wasm",
      progress_callback(update) {
        if (typeof update.progress === "number") {
          setEnglishModelStatus(`Loading local model... ${Math.round(update.progress)}%`);
          return;
        }

        if (update.status === "done") {
          setEnglishModelStatus("Local model ready");
        }
      },
    })
      .then((transcriber) => {
        englishTranscriber = transcriber;
        setEnglishModelStatus("Local model ready");
        return transcriber;
      })
      .catch((error) => {
        englishTranscriberPromise = null;
        const message = error instanceof Error ? error.message : "Unknown load error";
        setEnglishModelStatus(`Local model failed: ${message}`);
        throw error;
      });
  }

  return englishTranscriberPromise;
}

async function transcribeWithEnglish(audioBlob) {
  const transcriber = await getEnglishTranscriber();
  const decodedAudio = await decodeAudioToMono(audioBlob);
  const audio = await resampleAudio(decodedAudio.samples, decodedAudio.sampleRate, 16000);
  const output = await transcriber(audio, {
    chunk_length_s: 20,
    stride_length_s: 4,
    return_timestamps: "word",
  });

  return {
    text: (output.text || "").trim(),
    words: normalizeWhisperChunks(output.chunks),
  };
}

async function transcribeWithAbair(audioBlob) {
  const formData = new FormData();
  formData.append("file", audioBlob, "conversation.webm");
  formData.append("captpunct", "true");

  const response = await fetch(ABairEndpoint, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(`Abair request failed: ${response.status} ${message}`);
  }

  const payload = await response.json();
  return payload.text || payload.transcription || payload.result || "";
}

function cleanupRecorder() {
  if (timerId) {
    window.clearInterval(timerId);
    timerId = null;
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }

  mediaRecorder = null;
  audioChunks = [];
  recordingStartedAt = 0;
  updateTimer();
}

async function startRecording() {
  resetResults();
  setStatus("Requesting microphone...");
  recordButton.disabled = true;

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    const preferredType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : "audio/webm";

    mediaRecorder = new MediaRecorder(mediaStream, { mimeType: preferredType });
    audioChunks = [];

    mediaRecorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
    });

    mediaRecorder.addEventListener("stop", async () => {
      const durationSeconds = (Date.now() - recordingStartedAt) / 1000;
      setStatus("Running Abair and English recognizer...");
      recordButton.disabled = true;
      resetButton.disabled = true;

      try {
        const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || "audio/webm" });
        const [abairResult, englishResult] = await Promise.allSettled([
          transcribeWithAbair(audioBlob),
          transcribeWithEnglish(audioBlob),
        ]);
        const abairTranscript = abairResult.status === "fulfilled" ? abairResult.value.trim() : "";
        const englishTranscript = englishResult.status === "fulfilled"
          ? englishResult.value
          : { text: "", words: [] };
        const errors = [];

        if (abairResult.status === "rejected") {
          errors.push(abairResult.reason instanceof Error ? `Abair failed: ${abairResult.reason.message}` : "Abair failed.");
        }

        if (englishResult.status === "rejected") {
          errors.push(englishResult.reason instanceof Error ? `English recognizer failed: ${englishResult.reason.message}` : "English recognizer failed.");
        }

        if (!abairTranscript && !englishTranscript.text) {
          throw new Error(errors.join(" ") || "Both transcription paths failed.");
        }

        const mergedConversation = buildMergedConversation(abairTranscript, englishTranscript);
        const comparisonSummary = buildComparisonSummary({
          abairTranscript,
          englishTranscript,
          mergedConversation,
          errors,
        });

        setStatus(errors.length ? "Analysis complete with partial fallback" : "Analysis complete");
        renderAnalysis({
          abairTranscript,
          englishTranscript: englishTranscript.text,
          mergedTranscript: mergedConversation.mergedTranscript || abairTranscript || englishTranscript.text,
          mergedBreakdown: mergedConversation.mergedBreakdown,
          durationSeconds,
          comparisonSummary,
        });
      } catch (error) {
        setStatus("Transcription failed");
        summaryText.textContent = error instanceof Error ? error.message : "Unknown transcription error.";
        comparisonText.textContent = "The two-transcriber pass did not complete.";
        mergedTranscriptText.textContent = "No merged transcript returned.";
        abairTranscriptText.textContent = "No Abair transcript returned.";
        englishTranscriptText.textContent = "No English transcript returned.";
      } finally {
        recordButton.disabled = false;
        resetButton.disabled = false;
        recordButton.textContent = "Start recording";
        recordButton.classList.remove("recording");
        cleanupRecorder();
      }
    });

    mediaRecorder.start();
    recordingStartedAt = Date.now();
    timerId = window.setInterval(updateTimer, 250);
    updateTimer();
    setStatus("Recording...");
    recordButton.textContent = "Stop recording";
    recordButton.classList.add("recording");
  } catch (error) {
    cleanupRecorder();
    setStatus("Microphone access failed");
    summaryText.textContent = error instanceof Error ? error.message : "Could not access microphone.";
  } finally {
    recordButton.disabled = false;
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    setStatus("Stopping recording...");
    mediaRecorder.stop();
  }
}

recordButton.addEventListener("click", async () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    stopRecording();
    return;
  }

  await startRecording();
});

resetButton.addEventListener("click", () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    stopRecording();
  }

  cleanupRecorder();
  setStatus("Idle");
  recordButton.textContent = "Start recording";
  recordButton.classList.remove("recording");
  resetResults();
});

resetResults();
getEnglishTranscriber().catch(() => {
  setStatus("Idle");
});