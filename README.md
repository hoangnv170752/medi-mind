# Application Setup Guide

## Project Structure
```
medi-mind/
├── ai-service-example/
│   ├── AimlApiLLM.py
│   ├── index.html
│   ├── main.py      # Entry point for the chat API
│   ├── requirements.txt
│   ├── styles.css
│   └── test.py
└── ...
```

## 1. Run the Frontend Application

### Install Dependencies
Navigate to the frontend directory and install the required packages:

```bash
# Using Yarn
yarn install

# Or using NPM
npm install
```

### Start the Frontend
Run the application with one of the following commands:

```bash
# Using Yarn
yarn run dev

# Or using NPM
npm run dev
```

### Create an Account
- Press the **Login** button and register a new account.
- Ensure the account has a **patient** role.

---

## 2. Run the AI Service

### Directory Overview
The AI service is located in the `ai-service-example` directory and contains the following files:
- **AimlApiLLM.py**: Contains the AIML API logic.
- **index.html**: Frontend interface for the chat.
- **main.py**: Entry point for the chat API.
- **requirements.txt**: Dependencies for the AI service.
- **styles.css**: Styles for the frontend interface.
- **test.py**: Additional tests or functionalities.

### Install Dependencies
Ensure you have Python and pip installed. Then navigate to the `ai-service-example` directory and install the required packages:

```bash
pip install -r requirements.txt
```

### Start the AI Service
Run the service using the following command:

```bash
python main.py
```

---

Follow these steps to successfully set up and run both the frontend application and the AI service! If you encounter any issues, check the console for error messages for troubleshooting.
