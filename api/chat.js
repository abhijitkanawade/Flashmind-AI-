export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({
      error: "Method not allowed"
    });
  }

  const apiKey = process.env.GROQ_API_KEY;

  if (!apiKey) {
    return res.status(500).json({
      error: "GROQ_API_KEY not found in Vercel Environment Variables."
    });
  }

  try {
    const { message, messages } = req.body || {};

    const chatMessages = Array.isArray(messages) && messages.length
      ? messages
      : [{ role: "user", content: message || "Hello" }];

    const response = await fetch(
      "https://api.groq.com/openai/v1/chat/completions",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${apiKey.trim()}`
        },
        body: JSON.stringify({
          model: "openai/gpt-oss-120b",
          messages: [
            {
              role: "system",
              content: "You are FlashMind AI, a smart, friendly and helpful AI assistant. Give accurate, clear and well-structured answers."
            },
            ...chatMessages
          ],
          max_tokens: 2048
        })
      }
    );

    const data = await response.json();

    if (!response.ok) {
      return res.status(response.status).json({
        error: data.error?.message || "Groq API request failed."
      });
    }

    const reply =
      data.choices?.[0]?.message?.content;

    if (!reply) {
      return res.status(500).json({
        error: "Empty response from Groq."
      });
    }

    return res.status(200).json({ reply });

  } catch (error) {
    console.error("FlashMind Error:", error);

    return res.status(500).json({
      error: "Server Error: " + error.message
    });
  }
}
