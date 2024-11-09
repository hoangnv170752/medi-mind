const express = require('express');
const AimlApiLLM = require('./aimlApiLLM');
const ChatGptLLM = require('./chatGptLLM');
const LlamaLLM = require('./llamaLLM');
async function createLlamaLLM(apiToken) {
  const { default: LlamaAI } = await import('llamaai');
  return new LlamaAI(apiToken, { model: 'llama3.1-70b' });
}

const router = express.Router();
const API_KEY = '1d525acd15344646a522ba71620f23df';
const API_OPEN_AI = process.env.API_AIMLLLM;
const API_LLAMA = process.env.API_LLAMA;

const aiml_llm = new AimlApiLLM(API_KEY);
const chatGptLLM = new ChatGptLLM(API_OPEN_AI);

router.post('/', async (req, res) => {
  const { prompt, patient_data, chat_history } = req.body;

  try {
    const full_prompt = `Patient Data: ${patient_data}\nUser: ${prompt}`;
    
    const responseContent = await aiml_llm.generateResponse(full_prompt);
    
    return res.json({ response: responseContent });
  } catch (error) {
    console.error('Error generating response:', error);
    return res.status(500).json({ error: 'Failed to generate response' });
  }
});

router.post('/chat-gpt', async (req, res) => {
  const { prompt, patient_data } = req.body;

  try {
    const full_prompt = `Patient Data: ${patient_data}\nUser: ${prompt}`;
    
    const responseContent = await chatGptLLM.generateResponse(full_prompt);
    
    return res.json({ response: responseContent });
  } catch (error) {
    console.error('Error generating response:', error);
    return res.status(500).json({ error: 'Failed to generate response' });
  }
});

router.post('/chat-llama', async (req, res) => {
  const { prompt, patient_data, chat_history } = req.body;
  const llamaAPI = await createLlamaLLM(API_LLAMA);

  try {
    // const responseContent = await llama_llm.generateResponse(prompt, patient_data, chat_history);
    const apiRequestJson = {
      messages: [
        { role: "user", content: prompt }
      ],
      functions: [
        {
          name: "information_extraction", 
          description: "Extract the relevant information from the passage.",  
          parameters: {
            type: "object",
            properties: {
              sentiment: {
                type: "string",
                description: "The sentiment encountered in the passage"
              },
              aggressiveness: {
                type: "integer",
                description: "A 0-10 score of how aggressive the passage is"
              },
              language: {
                type: "string",
                description: "The language of the passage"
              }
            },
            required: ["sentiment", "aggressiveness", "language"]
          }
        }
      ],
      stream: false,  
      function_call: "information_extraction" 
    };
    
    
    llamaAPI.run(apiRequestJson)
    .then(response => {
      console.log(response);
      return res.json({ response: response });

    })
    .catch(error => {
      return res.json({ response: error });
    });
  } catch (error) {
    console.error('Error generating response:', error);
    return res.status(500).json({ error: 'Failed to generate response' });
  }
});

module.exports = router;
