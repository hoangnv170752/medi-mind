const axios = require('axios');
const csv = require('csv-parser');
const fs = require('fs');
const readlineSync = require('readline-sync');
const ragcy = require('ragcy');

const API_KEY = process.env.RAGCY;
const client = ragcy.init({ key: API_KEY });

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

class AimlApiLLM {
  constructor(apiKey, modelName = 'o1-mini') {
    this.apiKey = apiKey;
    this.modelName = modelName;
  }

  async readCSVData(filePath) {
    return new Promise((resolve, reject) => {
      let data = '';
      fs.createReadStream(filePath)
        .pipe(csv())
        .on('data', (row) => {
          data += JSON.stringify(row) + '\n';
        })
        .on('end', () => resolve(data))
        .on('error', reject);
    });
  }

  async getRagcySuggestion(prompt, corpusId) {
    try {
      const response = await client.getSuggestions({
        corpusId: corpusId,
        query: prompt,
      });

      return response.suggestions[0].text;
    } catch (error) {
      console.error('Error fetching Ragcy suggestion:', error);
      return '';
    }
  }

  async generateResponse(prompt, previousPrompt) {
    const url = 'https://api.aimlapi.com/chat/completions';
    const retries = 5;

    const systemPrompt = `You are a doctor assistant chatbot.`;

    const combinedPrompt = `Previous Prompt: ${systemPrompt}\nUser Prompt: ${prompt}`;

    // const ragcySuggestion = await this.getRagcySuggestion(prompt, 'your_corpus_id_here');

    const diseaseMedicineData = await this.readCSVData('./data/disease_medicine.csv');
    const diseaseSymptomsData = await this.readCSVData('./data/disease_symptoms.csv');
    const patientData = await this.readCSVData('./data/patient_data.csv');

    const ragPrompt = `
      Disease and Medicine Information: ${diseaseMedicineData}
      Disease Symptoms Information: ${diseaseSymptomsData}
      Patient Data: ${patientData}
      Ragcy Suggestion: ${ragcySuggestion}
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
        if (response.data.choices && response.data.choices[0].message) {
          return {
            responseChatbot: response.data.choices[0].message.content,
            suggestRagcy: 'Would you like to ask Ragcy for a suggestion?',
          };
        } else {
          throw new Error('Invalid response structure');
        }
      } catch (error) {
        console.error('Error during API call:', error.message);
        if (attempt === retries - 1) {
          throw new Error('Exceeded maximum retries due to API call failure.');
        }
        await sleep(2000);
      }
    }
  }
}

module.exports = AimlApiLLM;
