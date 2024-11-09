const axios = require('axios');
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

class ChatGptLLM {
  constructor(apiKey, modelName = 'o1-mini') {
    this.apiKey = apiKey;
    this.modelName = modelName;
  }

  async generateResponse(prompt) {
    const url = 'https://api.aimlapi.com/chat/completions';
    const retries = 5;
  
    const persona = `
    You are a medical chatbot assistant that helps patients identify possible diseases based on symptoms. 
    You provide clear, accurate explanations about diseases, including symptoms and treatments.
  `;
  
  const context = `
    The user will describe symptoms they have. Your task is to analyze these symptoms and suggest potential diseases, formatted in HTML.
  `;
  
  const task = `
    Your response should:
    1. Explain the potential disease(s) based on the symptoms.
    2. Include relevant data or details.
    3. Provide possible treatments, if applicable.
    4. Format the response in HTML for easy readability.
  `;
  
    const combinedPrompt = `
      Persona: ${persona}
      Context: ${context}
      Task: ${task}
      User's Symptoms: ${prompt}
    `;
  
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
        return response.data.choices[0].message.content;
      } catch (error) {
        console.error(error);
      }
    }
  
    throw new Error('Exceeded maximum retries due to rate limits.');
  }
  
}

module.exports = ChatGptLLM;
