import requests
import time
import os
class AimlApiLLM:
    def __init__(self, api_key: str, model_name: str = "o1-preview"):
        self.api_key = api_key
        self.model_name = model_name

    def generate_response(self, prompt: str) -> str:
        url = "https://api.aimlapi.com/chat/completions"
        retries = 5
        
        for attempt in range(retries):
            response = requests.post(
                url=url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 512,
                    "stream": False,
                },
            )

            try:
                response.raise_for_status()  # Raise an error for bad responses
                return response.json()["choices"][0]["message"]["content"]  # Extract response content
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:
                    # Too many requests; wait and retry
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Rate limit exceeded. Waiting for {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    # Raise other HTTP errors
                    raise e
        
        raise Exception("Exceeded maximum retries due to rate limits.")

if __name__ == "__main__":
    api_key = "1d525acd15344646a522ba71620f23df"  # Replace with your actual API key
    aiml_llm = AimlApiLLM(api_key=api_key)
    
    prompt = "What kind of model are you?"
    response = aiml_llm.generate_response(prompt)
    print("Generated Response:", response)
