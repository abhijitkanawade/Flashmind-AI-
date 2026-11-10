export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    return res.status(500).json({ error: "API Key सापडली नाही. कृपया Vercel मध्ये ANTHROPIC_API_KEY तपासा." });
  }

  try {
    const { message, messages } = req.body || {};

    const chatMessages = Array.isArray(messages) && messages.length > 0
      ? messages
      : [{ role: "user", content: message || "Hello" }];

    const systemPrompt = "You are FlashMind AI, a smart, helpful, friendly, and fast AI assistant. Give clear, well-structured, and accurate answers.";

    // जर Key 'gsk_' ने सुरू होत असेल तर
    if (apiKey.trim().startsWith("gsk_")) {
      const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${apiKey.trim()}`
        },
        body: JSON.stringify({
          model: "llama-3.3-70b-versatile",
          messages: [
            { role: "system", content: systemPrompt },
            ...chatMessages
          ],
          temperature: 0.7,
          max_tokens: 1024
        })
      });

      const data = await response.json();
      if (!response.ok) {
        return res.status(response.status).json({ error: data.error?.message || "AI कडून उत्तर मिळण्यात अडचण आली." });
      }

      const reply = data.choices?.[0]?.message?.content || "माफ करा, उत्तर मिळाले नाही.";
      return res.status(200).json({ reply });
    }

    // Claude AI (Anthropic API) साठी
    const models = ["claude-sonnet-4-20250514", "claude-3-5-sonnet-latest", "claude-3-haiku-20240307"];
    let lastError = "Claude API कडून उत्तर मिळण्यात अडचण आली.";

    for (const model of models) {
      const response = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-api-key": apiKey.trim(),
          "anthropic-version": "2023-06-01"
        },
        body: JSON.stringify({
          model: model,
          max_tokens: 1024,
          system: systemPrompt,
          messages: chatMessages
        })
      });

      const data = await response.json();
      if (response.ok) {
        const reply = data.content?.[0]?.text || "माफ करा, उत्तर मिळाले नाही.";
        return res.status(200).json({ reply });
      }
      lastError = data.error?.message || lastError;
    }

    return res.status(500).json({ error: lastError });

  } catch (error) {
    return res.status(500).json({ error: "Server Error: " + error.message });
  }
}
