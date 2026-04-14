<div align="center">
  <img src="https://img.icons8.com/color/96/000000/google-meet.png" alt="Google Meet Logo" width="80" />
  <h1>Meet AI Scribe 🎙️✨</h1>
  <p><strong>A fully automated, headless AI bot that joins Google Meet, records audio, transcribes in 11+ languages, and generates structured meeting summaries.</strong></p>

  [![Frontend](https://img.shields.io/badge/Live_Frontend-Vercel-000000?style=for-the-badge&logo=vercel)](https://google-meet-ai-scribe.vercel.app)
  [![Backend](https://img.shields.io/badge/Live_Backend-Render-46E3B7?style=for-the-badge&logo=render)](https://google-meet-ai-scribe.onrender.com)
</div>

<hr>

## 🚀 Overview

**Meet AI Scribe** is a sophisticated full-stack application designed to take the busywork out of virtual meetings. By simply providing a Google Meet URL, a headless bot automatically joins the meeting, records the audio, securely processes the file, and runs it through state-of-the-art AI models to provide high-quality transcriptions and structured summaries. 

It handles multiple Indian languages, seamless translation, and persists meeting history in the cloud securely behind Firebase user authentication.

## ✨ Features

- **🤖 Automated Bot Integration:** Utilizes Recall.ai to deploy an invisible scribe bot that automatically joins and records Google Meet calls.
- **🗣️ Multilingual Transcription:** Transcribes English natively using **Groq Whisper Large v3**, and supports 11 Indian regional languages using **Sarvam AI (saarika:v2.5)** with automatic chunking for long meetings.
- **🌐 In-App Translation:** Translate meeting summaries into any supported language immediately after the meeting using Sarvam's `mayura:v1` translation models.
- **📝 Intelligent Summaries:** Uses Hugging Face's open-source 120B parameter chat models to detect meeting titles automatically and generate structured notes (Overview, Decisions Made, Action Items).
- **🔒 Secure Cloud Storage:** Transcripts and summaries are safely stored in **AWS S3** and isolated per user.
- **🔐 Authentication:** User login, registration, and isolated meeting history powered by **Firebase Auth**.
- **🎨 Stunning UI:** A breathtaking, highly responsive React frontend featuring glassmorphism and a dynamic, interactive particle physics background.

## 🏗️ Architecture

- **Frontend:** React, Vite, Vanilla CSS.
- **Backend:** Python, FastAPI, Uvicorn (Server).
- **Audio Processing:** `ffmpeg` for automatic audio extraction and sampling normalization.
- **Database / Storage:** AWS S3 for documents, Firebase for User Identity.
- **AI/API Providers:** Recall.ai, Groq, Sarvam AI, Hugging Face.

## 🛠️ Local Development Setup

### Prerequisites
- Node.js (v18+)
- Python 3.10+
- Accounts/API keys for Firebase, AWS, Groq, Sarvam AI, Huggingface, and Recall.ai.

### 1. Clone the repository
```bash
git clone https://github.com/rc2308/Google-Meet-Ai-Scribe.git
cd Google-Meet-Ai-Scribe
```

### 2. Backend Setup
1. Open terminal and navigate to the `backend` folder:
   ```bash
   cd backend
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your environment variables. Copy the `.env.example` file and rename it to `.env`. Fill in your real API keys.
4. Start the FastAPI server:
   ```bash
   uvicorn server:app --host 0.0.0.0 --port 3000 --reload
   ```

### 3. Frontend Setup
1. Open a new terminal window and navigate to the `frontend` folder:
   ```bash
   cd frontend
   ```
2. Install Node dependencies:
   ```bash
   npm install
   ```
3. Set up your environment variables. Copy `.env.example` to `.env` and fill in your Firebase project credentials.
4. Start the Vite development server:
   ```bash
   npm run dev
   ```

### 4. Running the App
Open your browser and navigate to `http://localhost:5173`. 
Ensure both the frontend and backend servers are running simultaneously.

---

## 🌍 Platform Deployment 

The project is fully prepared for cloud deployment, utilizing separated environments:

### Frontend (Vercel)
The frontend is deployed as a static React app on Vercel. 
During deployment, the `VITE_API_URL` environment variable is explicitly set to point to the live Render backend URL so the frontend can communicate with the backend seamlessly.

### Backend (Render)
The backend is deployed as a Web Service on Render.
**Start Command:** `uvicorn server:app --host 0.0.0.0 --port $PORT`
Render gracefully handles python requirements, including the auto-installation of the background `ffmpeg` dependency required by Whisper/Sarvam for audio conversions.

---

## 📄 License
This project is open-source and available under the MIT License.
