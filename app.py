import streamlit as st
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
#  Custom CSS Styling
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

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: rgba(22, 27, 34, 0.8) !important;
        border: 1px solid #30363d !important;
        border-radius: 12px !important;
        padding: 12px !important;
        margin: 6px 0 !important;
    }

    /* Big text area input box */
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
    .stTextArea textarea::placeholder {
        color: #6e7681 !important;
        font-size: 15px !important;
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
        font-size: 15px;
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
    .send-btn button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 15px rgba(46,160,67,0.4) !important;
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

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Session State
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_response" not in st.session_state:
    st.session_state.last_response = ""

# ─────────────────────────────────────────────
#  Ollama Client & Connection
# ─────────────────────────────────────────────
OLLAMA_HOST = "http://localhost:11434"
client = Client(host=OLLAMA_HOST)

is_connected = False
available_models = []
try:
    models_response = client.list()
    available_models = [m['model'] for m in models_response.get('models', [])]
    is_connected = True
except Exception:
    is_connected = False

# ─────────────────────────────────────────────
#  SIDEBAR — History Panel + Reset + Settings
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📜 Conversation History")
    st.caption("Your queries and AI responses")
    st.divider()

    # Conversation History Panel
    if st.session_state.messages:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                preview = msg["content"][:80] + ("..." if len(msg["content"]) > 80 else "")
                st.markdown(f'<div class="hist-user">🧑 <b>You:</b> {preview}</div>', unsafe_allow_html=True)
            else:
                preview = msg["content"][:80] + ("..." if len(msg["content"]) > 80 else "")
                st.markdown(f'<div class="hist-ai">🤖 <b>AI:</b> {preview}</div>', unsafe_allow_html=True)
    else:
        st.info("No conversation yet. Type a question below!", icon="💬")

    # Reset Button
    st.divider()
    if st.button("🗑️ Reset Conversation", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.last_response = ""
        st.toast("✅ Conversation cleared!", icon="🧹")
        st.rerun()

    # Connection Status
    st.divider()
    st.markdown("**🔌 Status**")
    if is_connected:
        st.markdown('<span class="status-connected">🟢 Connected</span>', unsafe_allow_html=True)
        st.caption(f"Server: `{OLLAMA_HOST}`")
    else:
        st.markdown('<span class="status-disconnected">🔴 Disconnected</span>', unsafe_allow_html=True)
        if st.button("🔄 Retry", use_container_width=True):
            st.rerun()

    # Model Settings
    if is_connected and available_models:
        st.divider()
        st.markdown("**⚙️ Model**")
        selected_model = st.selectbox("Select Model", options=available_models, index=0)
        temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
    else:
        selected_model = "qwen2.5:0.5b"
        temperature = 0.7

# ─────────────────────────────────────────────
#  MAIN AREA — Title
# ─────────────────────────────────────────────
st.markdown("# 🤖 Local LLM Chat Interface")
st.caption("Interactive Streamlit interface connected to a locally hosted LLM via Ollama")
st.divider()

# ─────────────────────────────────────────────
#  TEXT INPUT BOX — User Query
# ─────────────────────────────────────────────
st.markdown("### ✏️ Type Your Query Below")

with st.form(key="query_form", clear_on_submit=True):
    user_query = st.text_area(
        label="Your Question",
        placeholder="Type your question here...\n\nExample: What is machine learning?\nExample: Explain Python in simple words.",
        height=150,
        key="query_box",
        label_visibility="collapsed"
    )

    col_send, col_clear, col_space = st.columns([1, 1, 4])
    with col_send:
        submitted = st.form_submit_button("🚀 Send Query", use_container_width=True)
    with col_clear:
        pass

st.divider()

# ─────────────────────────────────────────────
#  RESPONSE AREA — Model Generated Answer
# ─────────────────────────────────────────────
st.markdown("### 💬 Model Response")

if submitted and user_query.strip():
    if not is_connected:
        st.error("⚠️ Ollama server is not running! Start it first using: `ollama serve`")
    else:
        # Save user message
        st.session_state.messages.append({"role": "user", "content": user_query.strip()})

        # Build API messages
        api_messages = [{"role": "system", "content": "You are a helpful, friendly, and concise AI assistant."}]
        for msg in st.session_state.messages:
            api_messages.append({"role": msg["role"], "content": msg["content"]})

        # Stream response from Ollama
        response_container = st.empty()
        full_response = ""

        try:
            with st.spinner("🔄 Generating response..."):
                stream = client.chat(
                    model=selected_model,
                    messages=api_messages,
                    options={"temperature": temperature},
                    stream=True
                )
                for chunk in stream:
                    token = chunk.get("message", {}).get("content", "")
                    full_response += token
                    response_container.markdown(
                        f'<div class="response-box">{full_response}▌</div>',
                        unsafe_allow_html=True
                    )

            # Final response
            response_container.markdown(
                f'<div class="response-box">{full_response}</div>',
                unsafe_allow_html=True
            )

            # Save to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.last_response = full_response

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

elif st.session_state.last_response:
    # Show last response when page reloads
    st.markdown(
        f'<div class="response-box">{st.session_state.last_response}</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div class="response-box" style="color: #6e7681; font-style: italic;">Your AI response will appear here after you type a query and click Send...</div>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
#  Full Chat History Display
# ─────────────────────────────────────────────
if st.session_state.messages:
    st.divider()
    st.markdown("### 📋 Full Conversation")
    for msg in st.session_state.messages:
        icon = "🤖" if msg["role"] == "assistant" else "🧑"
        with st.chat_message(msg["role"], avatar=icon):
            st.markdown(msg["content"])

# Footer
st.divider()
st.caption("🔒 100% Local & Private — No data leaves your machine | Built with Streamlit + Ollama")
