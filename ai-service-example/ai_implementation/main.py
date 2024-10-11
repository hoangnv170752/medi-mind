from fastapi import FastAPI
from pydantic import BaseModel
from AimlApiLLM import AimlApiLLM  
from fastapi.middleware.cors import CORSMiddleware 

app = FastAPI()
# Allow CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You may want to restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class ChatRequest(BaseModel):
    prompt: str
    patient_data: str
    chat_history: list[str]  # Accepting previous chat messages as a list

class ChatResponse(BaseModel):
    response: str

# Replace with your actual API key
API_KEY = "1d525acd15344646a522ba71620f23df"
aiml_llm = AimlApiLLM(api_key=API_KEY)

@app.post("/chat/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Construct the full prompt with patient data and chat history
    full_prompt = "Previous Messages:\n" + "\n".join(request.chat_history) + \
                  f"\nPatient Data: {request.patient_data}\nUser: {request.prompt}"
    
    # Generate a response
    response_content = aiml_llm.generate_response(full_prompt)
    
    return ChatResponse(response=response_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
