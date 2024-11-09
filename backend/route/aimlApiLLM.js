const axios = require('axios');
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

class AimlApiLLM {
  constructor(apiKey, modelName = 'o1-mini') {
    this.apiKey = apiKey;
    this.modelName = modelName;
  }

  async generateResponse(prompt, previousPrompt) {
    const url = 'https://api.aimlapi.com/chat/completions';
    const retries = 5;

    // Context setting for chatbot's role and capabilities
    const systemPrompt = `
      You are a doctor assistant chatbot.`;

    // Combine previous prompt and the new one
    const combinedPrompt = `Previous Prompt: ${systemPrompt}\nUser Prompt: ${prompt}`;

    for (let attempt = 0; attempt < retries; attempt++) {
      try {
        const response = await axios.post(
          url,
          {
            model: this.modelName,
            messages: [
                {
                  role: "user",
                  content: combinedPrompt,
                },
            ],
            max_tokens: 256,
            stream: false,
          },
          {
            headers: {
              Authorization: `Bearer ${this.apiKey}`,
              'Content-Type': 'application/json',
            },
          }
        );
        console.log(40);
        console.log(response.data.choices[0].message);
        return response.data.choices[0].message.content;
      } catch (error) {
        console.error(error);
      }
    }

    throw new Error('Exceeded maximum retries due to rate limits.');
  }
}

module.exports = AimlApiLLM;
