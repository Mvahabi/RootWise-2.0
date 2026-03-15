<img src="images/project.png" alt="Alt text" />

## Some Background
RootWise, inspired and pioneered by [Lily Faris](https://lilymfaris.com/), is a project that explores how AI can make food systems more accessible, sustainable, and human-centered. The goal is to experiment with traceable, transparent, and user-focused RAG AI that uses functional medicine to connect food choices with well-being, while supporting zero-waste habits and local food knowledge.

ZoneWise is the latest developed feature, a comprehensive health analytics dashboard that ingests physiological metrics—such as heart rate zones—and delivers real-time insights through interactive visualizations and chat-based exploration. Similar to RootWise, ZoneWise is personalized to each user’s records, enabling tailored insights, feedback, and context-aware assistance through the integrated chatbot. 

While the broader initiative has been a collaborative effort with Vihan Patil and Johnathan Harding, [Mahyar Vahabi](https://mvahabi.github.io/portfolio/) independently led the development of the ZoneWise system architecture and implementation. 


## Setup

### 0. Gathering API Tokens

1. **NGC API Key (NVIDIA GPU Cloud)**  
   1. Sign in at: https://ngc.nvidia.com  
   2. Click your user icon (top right).  
   3. Navigate to **Setup → API Key**.  
   4. Click **Generate Key** and copy the key securely.  
   5. Embedding model endpoint: https://build.nvidia.com/nvidia/nv-embedqa-e5-v5  

2. **OpenAI API Key**  
   1. Go to: https://platform.openai.com/api-keys  
   2. Sign in to your account.  
   3. Click **Create new secret key**.  
   4. Copy and store the key securely. 
   5. Ensure your account/org has access to the chat model you set in OPENAI_CHAT_MODEL.

### 1. Evironmental Variables
First create a .env file in your root directory
```bash
# API Tokens
NGC_API_KEY="your-ngc-api-key"
OPENAI_API_KEY="your-openai-api-key"

# Database
DATABASE_URL="{CONTACT MAHYAR VAHABI FOR THIS INFO}"

# Chat model (must match an actual OpenAI model ID your account can use)
OPENAI_CHAT_MODEL="gpt-4.1-mini"

# Auth
JWT_SECRET="something_extremely_secret_and_long"
JWT_ALG="HS256"
JWT_EXPIRE_HOURS="1"

# Backend server
ROOTWISE_BACKEND_URL=http://127.0.0.1:8000
```

### 2. Creating Virtual Environment
Before starting, ensure the following are installed:

- **⚠️ Recommended: Python 3.11+**
- **Node.js + npm**

There are two ways to install backend dependencies:
#### ✅ Option A — VSCode (Recommended)

1. Open any Python file inside the `backend` folder.

2. In VSCode, click the **Python Interpreter** (bottom-right corner).

3. Select **"+ Create Virtual Environment…"**

4. Choose:
   - **Venv**
   - A **Python 3.11+ interpreter**

5. Wait for VSCode to create `.venv` and install dependencies.

6. Continue to **Step 3**.

#### Option B — Manual Setup (Terminal)
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
```

### 3. Run the backend (Terminal 1):
```bash
# if installations via Option A: 
source .venv/bin/activate

# if installations via Option B: 
source venv/bin/activate 

cd backend
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 4. Run the frontend (Terminal 2):
```bash
# if installations via Option A: 
source .venv/bin/activate

# if installations via Option B: 
source venv/bin/activate 

cd frontend
npm ci
npm run dev
```

### 5. Optional: Run the Agentic Service (Terminal 3)

RootWise now supports an experimental `agentic mode` for capstone testing. This runs in a **separate Python service** so the main backend can keep its current FastAPI/Starlette/Uvicorn versions while Google ADK uses its own dependency set.

If you only want the classic RootWise experience, you can skip this step.

```bash
# if installations via Option A: 
source .venv/bin/activate

# if installations via Option B: 
source venv/bin/activate 

cd backend
uvicorn app.main:app --host 127.0.0.1 --port 8100 --reload
```

### 6. Local Development Summary

For classic mode only:

- Terminal 1: main backend on `127.0.0.1:8000`
- Terminal 2: frontend on `127.0.0.1:5173`

For classic + agentic comparison:

- Terminal 1: main backend on `127.0.0.1:8000`
- Terminal 2: frontend on `127.0.0.1:5173`
- Terminal 3: agentic service on `127.0.0.1:8100`

### 7. Open this server
http://localhost:5173


### Demo
https://youtu.be/CIY4ehuy7zg

