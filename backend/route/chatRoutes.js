const express = require('express');
const AimlApiLLM = require('./aimlApiLLM');

const router = express.Router();
const API_KEY = '1d525acd15344646a522ba71620f23df';
const aiml_llm = new AimlApiLLM(API_KEY);

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

module.exports = router;
