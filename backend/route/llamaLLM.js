async function createLlamaLLM(apiToken) {
    const { default: LlamaAI } = await import('llamaai');
    return new LlamaAI(apiToken, { model: 'llama3.1-70b' });
  }
  
  class LlamaLLM {
    constructor(apiToken) {
      this.apiToken = apiToken;
      this.llama = null;
    }
  
    async init() {
      if (!this.llama) {
        this.llama = await createLlamaLLM(this.apiToken); // Initialize with model
      }
    }
  
    async generateResponse(prompt) {
      await this.init();
  
      const apiRequestJson = {
        messages: [
          { role: "user", content: prompt }
        ],
        functions: [
          {
            name: "get_medical_info",
            description: "Retrieve information about a disease, its symptoms, and possible treatments.",
            parameters: {
              type: "object",
              properties: {
                disease: {
                  type: "string",
                  description: "The name of the disease or condition, e.g., hypertension, diabetes."
                },
                include_symptoms: {
                  type: "boolean",
                  description: "Whether to include a list of symptoms associated with the disease."
                },
                include_treatments: {
                  type: "boolean",
                  description: "Whether to include treatment options."
                },
                include_medication_info: {
                  type: "boolean",
                  description: "Whether to include information on commonly prescribed medications for the disease."
                }
              },
              required: ["disease"]
            }
          }
        ],
        stream: false,
        function_call: "get_medical_info"
      };
  
      try {
        const response = await this.llama.run(apiRequestJson);
        return response.choices[0].message.content;
      } catch (error) {
        console.error("Error fetching response:", error);
        throw error;
      }
    }
  }
  
  module.exports = LlamaLLM;
  