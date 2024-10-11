import requests
import time

# URL of the FastAPI application
BASE_URL = "http://127.0.0.1:8000/chat/"

def test_chat():
    # Define the input data for the test
    input_data = {
        "prompt": "What should I do for my headache?",
        "patient_data": "Patient has a history of migraines and has taken ibuprofen.",
        "chat_history": ["User: How do I treat my asthma?", "Assistant: You can use an inhaler..."]
    }

    # Make a POST request to the FastAPI endpoint
    response = requests.post(BASE_URL, json=input_data)

    # Print the response
    print("Response Status Code:", response.status_code)
    print("Response JSON:", response.json())

if __name__ == "__main__":
    # Optionally wait for the server to start
    time.sleep(1)  # Wait for 1 second before sending the request
    test_chat()
