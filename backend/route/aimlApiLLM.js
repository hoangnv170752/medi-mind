const axios = require('axios');
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

class AimlApiLLM {
  constructor(apiKey, modelName = 'o1-preview') {
    this.apiKey = apiKey;
    this.modelName = modelName;
  }

  async generateResponse(prompt, previousPrompt) {
    const url = 'https://api.aimlapi.com/chat/completions';
    const retries = 5;

    // Context setting for chatbot's role and capabilities
    const context = `
      You are a doctor assistant chatbot with the ability to gather information from medical histories, 
      and internet resources. Use this knowledge to assist medical employees and patients.`;

    // Combine previous prompt and the new one
    const combinedPrompt = `${context}\nPrevious Prompt: ${previousPrompt}\nUser Prompt: ${prompt}`;

    for (let attempt = 0; attempt < retries; attempt++) {
      try {
        const response = await axios.post(
          url,
          {
            model: this.modelName,
            messages: [{ role: 'user', content: combinedPrompt }],
            max_tokens: 512,
            stream: false,
          },
          {
            headers: {
              Authorization: `Bearer ${this.apiKey}`,
              'Content-Type': 'application/json',
            },
          }
        );
        return response.data.choices[0].message.content;
      } catch (error) {
        console.error(error);
      }
    }

    throw new Error('Exceeded maximum retries due to rate limits.');
  }
}

module.exports = AimlApiLLM;
