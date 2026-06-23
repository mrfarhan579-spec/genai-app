# 🤖 Local LLM Playground — Streamlit + Ollama

A fully **local**, **private** AI chat interface built with **Streamlit** connected to a locally hosted Large Language Model via **Ollama**.

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 💬 **Chat Input** | Quick single-line chat input for fast queries |
| 📝 **Text Box Input** | Multi-line text area for longer, structured queries |
| 🔄 **Real-Time Streaming** | Token-by-token response streaming from the LLM |
| 📜 **Conversation History** | Full session-based chat history panel |
| 🗑️ **Reset Button** | Clear all conversation history with one click |
| ⚙️ **Model Parameters** | Control Temperature, Top-P, and System Prompt |
| 📥 **Model Downloader** | Pull any Ollama model directly from the UI |
| 🟢 **Connection Status** | Live indicator showing Ollama server status |
| 📊 **Session Stats** | Track number of messages and queries per session |
| 🔒 **100% Local & Private** | No data ever leaves your machine |

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running

### 1. Clone the Repository
```bash
git clone https://github.com/mrfarhan579-spec/genai-app.git
cd genai-app
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the Ollama Server
```bash
ollama serve
```

### 4. Pull a Model (Recommended for 4GB RAM)
```bash
# Fastest - 350MB
ollama pull qwen2.5:0.5b

# Better intelligence - 1.3GB  
ollama pull llama3.2:1b
```

### 5. Run the Streamlit App
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501** 🎉

---

## 📁 Project Structure

```
genai-app/
│
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM Backend**: Ollama (local inference server)
- **Model**: Qwen2.5:0.5b (default — lightweight, runs on 4GB RAM)
- **Language**: Python 3.12

---

## 📸 Usage

1. Open the app at `http://localhost:8501`
2. Choose **Input Mode** in the sidebar:
   - **Chat Input** — quick one-line messages
   - **Text Box** — multi-line detailed queries
3. Adjust **Temperature** and **Top-P** sliders for response creativity
4. Type your query and send!
5. Use the **Reset Conversation** button to start fresh

---

*Built with ❤️ using Streamlit + Ollama*
