import streamlit as st
import requests
from ollama import Client

# ─────────────────────────────────────────────
#  Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Chat Interface | Ollama",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  Custom CSS Styling (Premium Dark Theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Outfit', 'Segoe UI', sans-serif !important; }

    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #0f1923 50%, #0d1117 100%);
        color: #c9d1d9;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
        border-right: 1px solid #30363d;
    }

    h1 { color: #58a6ff !important; text-shadow: 0 0 30px rgba(88,166,255,0.3); }
    h2, h3, h4 { color: #79c0ff !important; }

    /* Chat messages thread */
    .chat-bubble-user {
        background: rgba(56, 139, 253, 0.15) !important;
        border: 1px solid #388bfd !important;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        color: #e6edf3;
    }
    .chat-bubble-ai {
        background: rgba(46, 160, 67, 0.1) !important;
        border: 1px solid #2ea043;
        border-left: 4px solid #2ea043;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        color: #e6edf3;
    }

    /* Text area input box */
    .stTextArea textarea {
        background: rgba(22, 27, 34, 0.95) !important;
        border: 2px solid #388bfd !important;
        border-radius: 12px !important;
        color: #e6edf3 !important;
        font-size: 16px !important;
        padding: 14px !important;
        line-height: 1.6 !important;
    }
    .stTextArea textarea:focus {
        border-color: #58a6ff !important;
        box-shadow: 0 0 0 4px rgba(88,166,255,0.2) !important;
    }

    /* Response display box */
    .response-box {
        background: rgba(22, 27, 34, 0.9);
        border: 1px solid #2ea043;
        border-left: 4px solid #2ea043;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 12px 0;
        min-height: 80px;
        color: #e6edf3;
        font-size: 16px;
        line-height: 1.8;
    }

    /* Send button */
    .send-btn button {
        background: linear-gradient(135deg, #238636, #2ea043) !important;
        color: white !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 8px 32px !important;
        border-radius: 10px !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }

    /* Status badges */
    .status-connected {
        display: inline-block;
        background: rgba(46,160,67,0.15);
        border: 1px solid #2ea043;
        color: #3fb950;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 500;
    }
    .status-disconnected {
        display: inline-block;
        background: rgba(218,54,51,0.15);
        border: 1px solid #da3633;
        color: #f85149;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 500;
    }

    /* History items */
    .hist-user {
        background: rgba(56,139,253,0.08);
        border-left: 3px solid #388bfd;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 6px;
        font-size: 13px;
        color: #c9d1d9;
    }
    .hist-ai {
        background: rgba(46,160,67,0.08);
        border-left: 3px solid #2ea043;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 6px;
        font-size: 13px;
        color: #c9d1d9;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Session State Initialization
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_response" not in st.session_state:
    st.session_state.last_response = ""
if "ollama_host_config" not in st.session_state:
    st.session_state.ollama_host_config = "http://127.0.0.1:11434"

# ─────────────────────────────────────────────
#  Ollama Connection
# ─────────────────────────────────────────────
OLLAMA_HOST = st.session_state.ollama_host_config.strip()
client = Client(host=OLLAMA_HOST)

is_connected = False
available_models = []

try:
    # Use standard HTTP request with short timeout to prevent UI freeze
    r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=1.5)
    if r.status_code == 200:
        is_connected = True
        available_models = [m['name'] for m in r.json().get('models', [])]
except Exception:
    is_connected = False

# ─────────────────────────────────────────────
#  SIDEBAR — History Panel + Reset
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📜 Conversation History")
    st.caption("Past queries and model responses")
    st.divider()

    # History Display
    if st.session_state.messages:
        for msg in st.session_state.messages:
            preview = msg["content"][:60] + ("..." if len(msg["content"]) > 60 else "")
            if msg["role"] == "user":
                st.markdown(f'<div class="hist-user">🧑 <b>You:</b> {preview}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="hist-ai">🤖 <b>AI:</b> {preview}</div>', unsafe_allow_html=True)
    else:
        st.info("No conversation history yet.", icon="💬")

    st.divider()

    # Reset Button
    if st.button("🗑️ Reset Conversation", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.last_response = ""
        st.toast("✅ Chat history cleared!", icon="🧹")
        st.rerun()

    st.divider()

    # Connection Status & Configuration
    st.markdown("**🔌 Connection & Backend Config**")
    
    # Text input to dynamically modify host URL (supports localhost or public tunnel/ngrok URLs)
    new_host = st.text_input(
        "Ollama Host URL:",
        key="ollama_host_config",
        placeholder="e.g. http://127.0.0.1:11434"
    )
    
    if is_connected:
        st.markdown('<span class="status-connected">🟢 Connected</span>', unsafe_allow_html=True)
        st.caption(f"Connected to backend: `{OLLAMA_HOST}`")
    else:
        st.markdown('<span class="status-disconnected">🔴 Disconnected</span>', unsafe_allow_html=True)
        st.warning("Cannot communicate with the LLM backend. Please make sure Ollama server is running locally on port 11434.")
        st.info("💡 **Streamlit Cloud Note:** If you are running this app on Streamlit Cloud, you must expose your local Ollama port (11434) using a tunnel (e.g. `ngrok http 11434`) and paste the ngrok URL above.")
        
        if st.button("🔄 Retry Connection", use_container_width=True):
            st.rerun()

    # Model Settings
    if is_connected and available_models:
        st.divider()
        st.markdown("**⚙️ Settings**")
        selected_model = st.selectbox("Select Model", options=available_models, index=0)
        temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
    else:
        selected_model = "qwen2.5:0.5b"
        temperature = 0.7

# ─────────────────────────────────────────────
#  MAIN AREA
# ─────────────────────────────────────────────
st.markdown("# 🤖 Local LLM Chat Application")
st.caption("Seamless communication between Streamlit frontend and local Ollama backend.")
st.divider()

# ─────────────────────────────────────────────
#  RESPONSE AREA (Shows current or last output)
# ─────────────────────────────────────────────
st.markdown("### 💬 Response Area")

# We create a placeholder where the response will be streamed
response_placeholder = st.empty()

# Show default or last message
if st.session_state.last_response:
    response_placeholder.markdown(
        f'<div class="response-box">{st.session_state.last_response}</div>',
        unsafe_allow_html=True
    )
else:
    response_placeholder.markdown(
        '<div class="response-box" style="color: #6e7681; font-style: italic;">Response will appear here...</div>',
        unsafe_allow_html=True
    )

st.divider()

# ─────────────────────────────────────────────
#  TEXT INPUT BOX (User Query)
# ─────────────────────────────────────────────
st.markdown("### ✏️ Text Input Box")

# Form to hold the text area and send button
with st.form(key="chat_form", clear_on_submit=True):
    user_query = st.text_area(
        label="Ask anything:",
        placeholder="Type your question here and click Send...",
        height=120,
        key="query_input",
        label_visibility="collapsed"
    )
    
    col_submit, col_info = st.columns([1, 4])
    with col_submit:
        submitted = st.form_submit_button("🚀 Send Query", use_container_width=True)
    with col_info:
        st.caption("Press button to send your query to the local LLM.")

# ─────────────────────────────────────────────
#  API Connection and Message Processing
# ─────────────────────────────────────────────
if submitted and user_query.strip():
    if not is_connected:
        st.error("❌ Cannot communicate with the LLM backend. Please make sure Ollama server is running locally on port 11434.")
    else:
        # Add user query to conversation history
        st.session_state.messages.append({"role": "user", "content": user_query.strip()})

        # Build context from previous messages
        api_messages = [{"role": "system", "content": "You are a helpful, friendly, and concise AI assistant."}]
        for msg in st.session_state.messages:
            api_messages.append({"role": msg["role"], "content": msg["content"]})

        # Stream the response from backend
        full_response = ""
        try:
            with st.spinner("⏳ LLM is thinking..."):
                stream = client.chat(
                    model=selected_model,
                    messages=api_messages,
                    options={"temperature": temperature},
                    stream=True
                )
                for chunk in stream:
                    token = chunk.get("message", {}).get("content", "")
                    full_response += token
                    response_placeholder.markdown(
                        f'<div class="response-box">{full_response}▌</div>',
                        unsafe_allow_html=True
                    )
            
            # Final output formatting
            response_placeholder.markdown(
                f'<div class="response-box">{full_response}</div>',
                unsafe_allow_html=True
            )
            
            # Save assistant response
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.last_response = full_response
            
            # Rerun to update history in sidebar
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error communicating with LLM backend: {e}")

# ─────────────────────────────────────────────
#  CONVERSATION HISTORY THREAD (Full back-and-forth)
# ─────────────────────────────────────────────
if st.session_state.messages:
    st.markdown("### 📋 Conversation Thread")
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-bubble-user">🧑 <b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble-ai">🤖 <b>AI:</b> {msg["content"]}</div>', unsafe_allow_html=True)

# Footer
st.divider()
st.caption("🔒 100% Local & Private | Streamlit + Ollama Integration")
