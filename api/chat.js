// FlashMind AI - /api/chat
// Groq (OpenAI-compatible) with image understanding + streaming.
// Env: GROQ_API_KEY (required). Optional: GROQ_TEXT_MODEL, GROQ_VISION_MODEL.

const TEXT_MODEL = process.env.GROQ_TEXT_MODEL || "openai/gpt-oss-120b";
const VISION_MODEL =
  process.env.GROQ_VISION_MODEL || "meta-llama/llama-4-scout-17b-16e-instruct";

const MAX_MESSAGES = 20;
const MAX_TEXT = 8000;
const MAX_IMAGES = 4;
const MAX_IMAGE_CHARS = 3_000_000; // ~2.2MB of image data per image
const IMAGE_RE = /^data:image\/(png|jpe?g|webp|gif);base64,[A-Za-z0-9+/=]+$/;

const TONES = {
  balanced: "Be clear, helpful and well organised.",
  concise: "Keep answers short and to the point.",
  detailed: "Give thorough, step-by-step answers with examples.",
  friendly: "Be warm, casual and encouraging.",
};

function buildSystem(tone, custom) {
  let s =
    "You are FlashMind AI, a helpful AI assistant. " +
    (TONES[tone] || TONES.balanced) +
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
  if (!process.env.GROQ_API_KEY) {
    return res.status(500).json({ error: "Server is missing GROQ_API_KEY." });
  }

  const body = readBody(req);
  let history = Array.isArray(body.messages) ? body.messages : [];
  if (!history.length && typeof body.message === "string") {
    history = [{ role: "user", content: body.message }];
  }
  history = history.slice(-MAX_MESSAGES);
  if (!history.length || history[history.length - 1].role !== "user") {
    return res.status(400).json({ error: "Last message must be from the user." });
  }

  let hasImages = false;
  const messages = [{ role: "system", content: buildSystem(body.tone, body.custom) }];

  for (let i = 0; i < history.length; i++) {
    const m = history[i];
    const role = m.role === "assistant" ? "assistant" : "user";
    const text = String(m.content || "").slice(0, MAX_TEXT);
    const isLast = i === history.length - 1;

    // Only the newest user message carries real image data; older ones are
    // mentioned in text so the payload stays small.
    if (role === "user" && isLast && Array.isArray(m.images) && m.images.length) {
      const imgs = m.images.slice(0, MAX_IMAGES);
      for (const img of imgs) {
        if (typeof img !== "string" || !IMAGE_RE.test(img) || img.length > MAX_IMAGE_CHARS) {
          return res.status(400).json({ error: "One of the images is invalid or too large." });
        }
      }
      hasImages = true;
      messages.push({
        role,
        content: [
          { type: "text", text: text || "What is in this image?" },
          ...imgs.map((url) => ({ type: "image_url", image_url: { url } })),
        ],
      });
    } else {
      const note =
        role === "user" && m.imageCount
          ? `\n[The user attached ${m.imageCount} image(s) earlier in the conversation.]`
          : "";
      messages.push({ role, content: (text + note) || "(empty)" });
    }
  }

  let upstream;
  try {
    upstream = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.GROQ_API_KEY.trim()}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: hasImages ? VISION_MODEL : TEXT_MODEL,
        messages,
        temperature: 0.7,
        max_tokens: 2048,
        stream: true,
      }),
    });
  } catch (e) {
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

  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for await (const chunk of upstream.body) {
      buffer += decoder.decode(chunk, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        const t = line.trim();
        if (!t.startsWith("data:")) continue;
        const data = t.slice(5).trim();
        if (data === "[DONE]") continue;
        try {
          const delta = JSON.parse(data)?.choices?.[0]?.delta?.content;
          if (delta) res.write(delta);
        } catch {}
      }
    }
  } catch (e) {
    res.write("\n\n⚠️ The response was interrupted.");
  }
  res.end();
}