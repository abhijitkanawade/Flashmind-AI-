// FlashMind AI - /api/chat
// Groq (OpenAI-compatible) with image understanding, deep mode and streaming.
// Env: GROQ_API_KEY (required). Optional: GROQ_TEXT_MODEL, GROQ_VISION_MODEL.

const GROQ_URL = "https://api.groq.com/openai/v1/chat/completions";
const TEXT_MODEL = process.env.GROQ_TEXT_MODEL || "openai/gpt-oss-120b";
const VISION_MODEL =
  process.env.GROQ_VISION_MODEL || "meta-llama/llama-4-scout-17b-16e-instruct";

const MAX_HISTORY_MESSAGES = 40;   // turns kept from the client
const MAX_TEXT = 40000;            // characters per message
const CONTEXT_BUDGET = 150000;     // characters of history sent upstream (oldest dropped first)
const MAX_IMAGES = 4;
const MAX_IMAGE_CHARS = 3_000_000; // ~2.2MB of image data per image
const IMAGE_RE = /^data:image\/(png|jpe?g|webp|gif);base64,[A-Za-z0-9+/=]+$/;

const MAX_TOKENS = { normal: 4096, deep: 8192 };

const TONES = {
  balanced: "Be clear, helpful and well organised.",
  concise: "Keep answers short and to the point.",
  detailed: "Give thorough, step-by-step answers with examples.",
  friendly: "Be warm, casual and encouraging.",
};

const DEEP_NOTE =
  " This may be a heavy task. Understand the goal first, break it into steps, work through each step carefully, check your result, and then give a complete answer. Do not cut the answer short. For code, give complete, runnable files rather than fragments.";

function buildSystem(tone, custom, deep) {
  let s =
    "You are FlashMind AI, a capable and accurate AI assistant. " +
    (TONES[tone] || TONES.balanced) +
    (deep ? DEEP_NOTE : "") +
    " Use Markdown for formatting (lists, **bold**, fenced code blocks)." +
    " When the user attaches images, look at them carefully and describe or answer based on what you actually see; say so if something is unclear.";
  if (typeof custom === "string" && custom.trim()) {
    s += "\n\nUser's custom instructions:\n" + custom.trim().slice(0, 1000);
  }
  return s;
}

function readBody(req) {
  if (req.body && typeof req.body === "object") return req.body;
  try {
    return JSON.parse(req.body || "{}");
  } catch {
    return {};
  }
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }
  const apiKey = (process.env.GROQ_API_KEY || "").trim();
  if (!apiKey) {
    return res.status(500).json({ error: "Server is missing GROQ_API_KEY." });
  }

  const body = readBody(req);
  let history = Array.isArray(body.messages) ? body.messages : [];
  if (!history.length && typeof body.message === "string") {
    history = [{ role: "user", content: body.message }];
  }
  // Ignore malformed entries instead of crashing on them.
  history = history
    .filter((m) => m && typeof m === "object" && (m.role === "user" || m.role === "assistant"))
    .slice(-MAX_HISTORY_MESSAGES);

  const last = history[history.length - 1];
  if (!last || last.role !== "user") {
    return res.status(400).json({ error: "Last message must be from the user." });
  }

  // Validate the newest message's images.
  let images = [];
  if (Array.isArray(last.images) && last.images.length) {
    images = last.images.slice(0, MAX_IMAGES);
    for (const img of images) {
      if (typeof img !== "string" || !IMAGE_RE.test(img) || img.length > MAX_IMAGE_CHARS) {
        return res.status(400).json({ error: "One of the images is invalid or too large." });
      }
    }
  }
  if (!images.length && !String(last.content || "").trim()) {
    return res.status(400).json({ error: "The message is empty." });
  }

  // Build turns. Only the newest user message carries image data; older
  // images are mentioned in text so the payload stays small.
  const turns = history.map((m, i) => {
    const text = String(m.content || "").slice(0, MAX_TEXT);
    const isLast = i === history.length - 1;
    if (isLast && images.length) {
      return { role: "user", text: text.trim() || "What is in this image?", images };
    }
    const note =
      m.role === "user" && m.imageCount
        ? `\n[The user attached ${m.imageCount} image(s) earlier in the conversation.]`
        : "";
    return { role: m.role, text: (text + note) || "(empty)" };
  });

  // Keep the newest turns that fit the budget; always keep the latest one.
  const kept = [];
  let used = 0;
  for (let i = turns.length - 1; i >= 0; i--) {
    const size = turns[i].text.length;
    if (kept.length && used + size > CONTEXT_BUDGET) break;
    kept.unshift(turns[i]);
    used += size;
  }
  // The conversation sent upstream must start with a user message.
  while (kept.length && kept[0].role !== "user") kept.shift();

  const deep = body.mode === "deep";
  const hasImages = images.length > 0;
  const messages = [{ role: "system", content: buildSystem(body.tone, body.custom, deep) }];
  for (const t of kept) {
    if (t.images) {
      messages.push({
        role: "user",
        content: [
          { type: "text", text: t.text },
          ...t.images.map((url) => ({ type: "image_url", image_url: { url } })),
        ],
      });
    } else {
      messages.push({ role: t.role, content: t.text });
    }
  }

  // Stop the upstream request if the visitor closes the page or taps Stop.
  const ac = new AbortController();
  res.on("close", () => {
    if (!res.writableEnded) ac.abort();
  });

  let upstream;
  try {
    upstream = await fetch(GROQ_URL, {
      method: "POST",
      signal: ac.signal,
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: hasImages ? VISION_MODEL : TEXT_MODEL,
        messages,
        temperature: deep ? 0.4 : 0.7,
        max_tokens: deep ? MAX_TOKENS.deep : MAX_TOKENS.normal,
        stream: true,
      }),
    });
  } catch (e) {
    if (ac.signal.aborted) return;
    return res.status(502).json({ error: "Could not reach the AI service." });
  }

  if (!upstream.ok || !upstream.body) {
    let detail = "";
    try {
      const j = await upstream.json();
      detail = j?.error?.message || "";
    } catch {}
    return res
      .status(upstream.status === 429 ? 429 : 502)
      .json({ error: detail || `AI service error (${upstream.status}).` });
  }

  res.statusCode = 200;
  res.setHeader("Content-Type", "text/plain; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("X-Accel-Buffering", "no");

  let finishReason = null;
  try {
    const reader = upstream.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        const t = line.trim();
        if (!t.startsWith("data:")) continue;
        const data = t.slice(5).trim();
        if (!data || data === "[DONE]") continue;
        try {
          const choice = JSON.parse(data)?.choices?.[0];
          if (choice?.delta?.content) res.write(choice.delta.content);
          if (choice?.finish_reason) finishReason = choice.finish_reason;
        } catch {}
      }
    }
  } catch (e) {
    if (!ac.signal.aborted && !res.writableEnded) {
      res.write("\n\n⚠️ The response was interrupted. Send “continue” to pick up where it stopped.");
    }
  }

  if (finishReason === "length" && !res.writableEnded) {
    res.write("\n\n_Stopped at the length limit. Send “continue” for the rest._");
  }
  if (!res.writableEnded) res.end();
}
