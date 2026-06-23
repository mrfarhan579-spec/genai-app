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

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
        border-right: 1px solid #30363d;
    }

    /* Headings glow */
    h1 { color: #58a6ff !important; text-shadow: 0 0 30px rgba(88,166,255,0.3); }
    h2, h3, h4 { color: #79c0ff !important; }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: rgba(22, 27, 34, 0.8) !important;
        border: 1px solid #30363d !important;
        border-radius: 12px !important;
        padding: 12px !important;
        margin: 6px 0 !important;
        backdrop-filter: blur(10px);
    }

    /* Text input box styling */
    .stTextInput input {
        background: rgba(22, 27, 34, 0.9) !important;
        border: 2px solid #388bfd !important;
        border-radius: 10px !important;
        color: #c9d1d9 !important;
        font-size: 16px !important;
        padding: 12px !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .stTextInput input:focus {
        border-color: #58a6ff !important;
        box-shadow: 0 0 0 3px rgba(88,166,255,0.2) !important;
    }
    .stTextInput input::placeholder {
        color: #6e7681 !important;
    }

    /* Response area styling */
    .response-area {
        background: rgba(22, 27, 34, 0.85);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        min-height: 100px;
        color: #c9d1d9;
        font-size: 15px;
        line-height: 1.7;
    }
    .response-label {
        color: #58a6ff;
        font-weight: 600;
        font-size: 14px;
        margin-bottom: 8px;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        transition: all 0.3s ease !important;
        font-weight: 500 !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 500;
    }
    .status-connected {
        background: rgba(46,160,67,0.15);
        border: 1px solid #2ea043;
        color: #3fb950;
    }
    .status-disconnected {
        background: rgba(218,54,51,0.15);
        border: 1px solid #da3633;
        color: #f85149;
    }

    /* History panel */
    .history-container {
        background: rgba(22, 27, 34, 0.6);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 12px;
        max-height: 400px;
        overflow-y: auto;
    }
    .history-msg {
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 8px;
        font-size: 13px;
    }
    .history-user {
        background: rgba(56,139,253,0.1);
        border-left: 3px solid #388bfd;
    }
    .history-assistant {
        background: rgba(46,160,67,0.1);
        border-left: 3px solid #2ea043;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #58a6ff; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Session State Initialization
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_response" not in st.session_state:
    st.session_state.last_response = ""

# ─────────────────────────────────────────────
#  Ollama Client & Connection Check
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
#  SIDEBAR — Conversation History Panel
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📜 Conversation History")
    st.caption("All your queries and AI responses in this session")
    st.divider()

    # ── Conversation History Panel ──
    if st.session_state.messages:
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="history-msg history-user">🧑 <b>You:</b> {msg["content"][:100]}{"..." if len(msg["content"]) > 100 else ""}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="history-msg history-assistant">🤖 <b>AI:</b> {msg["content"][:100]}{"..." if len(msg["content"]) > 100 else ""}</div>',
                    unsafe_allow_html=True
                )
    else:
        st.info("No conversation yet. Ask your first question!", icon="💬")

    st.divider()

    # ── Reset Button ──
    st.markdown("### 🔧 Actions")
    if st.button("🗑️ Reset Conversation", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.last_response = ""
        st.toast("✅ Conversation history cleared!", icon="🧹")
        st.rerun()

    st.divider()

    # ── Connection Status ──
    st.markdown("### 🔌 Connection Status")
    if is_connected:
        st.markdown(
            '<span class="status-badge status-connected">🟢 Connected to Ollama</span>',
            unsafe_allow_html=True
        )
        st.caption(f"Endpoint: `{OLLAMA_HOST}`")
        if available_models:
            st.caption(f"Models: `{'`, `'.join(available_models)}`")
    else:
        st.markdown(
            '<span class="status-badge status-disconnected">🔴 Ollama Offline</span>',
            unsafe_allow_html=True
        )
        st.warning("Start Ollama server to begin chatting.")
        if st.button("🔄 Retry Connection", use_container_width=True):
            st.rerun()

    st.divider()

    # ── Model Settings ──
    if is_connected and available_models:
        st.markdown("### ⚙️ Model Settings")
        selected_model = st.selectbox(
            "Select Model",
            options=available_models,
            index=0,
            help="Choose which locally hosted model to use."
        )
        temperature = st.slider(
            "Temperature", 0.0, 1.5, 0.7, 0.05,
            help="Higher = more creative. Lower = more focused."
        )
    else:
        selected_model = "qwen2.5:0.5b"
        temperature = 0.7

# ─────────────────────────────────────────────
#  MAIN AREA
# ─────────────────────────────────────────────

# ── Title ──
st.markdown("# 🤖 Local LLM Chat Interface")
st.caption("A simple and interactive Streamlit interface connected to a locally hosted LLM via Ollama")
st.divider()

if is_connected:

    # ─────────────────────────────────────────
    #  TEXT INPUT BOX for User Queries
    # ─────────────────────────────────────────
    st.markdown("#### ✏️ Enter Your Query")
    user_query = st.text_input(
        label="Type your question below and press Enter",
        placeholder="Ask me anything... (e.g., What is machine learning?)",
        key="user_input_box",
        label_visibility="collapsed"
    )
    st.caption("💡 Type your question above and press **Enter** to get a response from the local LLM.")

    # ─────────────────────────────────────────
    #  Process Query & Generate Response
    # ─────────────────────────────────────────
    if user_query:
        # Save user message to history
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Build message context for API call
        api_messages = [
            {"role": "system", "content": "You are a helpful, friendly, and concise AI assistant."}
        ]
        for msg in st.session_state.messages:
            api_messages.append({"role": msg["role"], "content": msg["content"]})

        # ── Response Area ──
        st.markdown("#### 💬 Model Response")

        # Stream the response from Ollama API
        response_placeholder = st.empty()
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
                    # Update the response area in real-time (streaming effect)
                    response_placeholder.markdown(
                        f'<div class="response-area">{full_response}▌</div>',
                        unsafe_allow_html=True
                    )

            # Final response without cursor
            response_placeholder.markdown(
                f'<div class="response-area">{full_response}</div>',
                unsafe_allow_html=True
            )

            # Save assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.last_response = full_response

        except Exception as e:
            st.error(f"⚠️ Error communicating with Ollama: {e}")

    # ─────────────────────────────────────────
    #  Display Previous Response (if no new query)
    # ─────────────────────────────────────────
    elif st.session_state.last_response:
        st.markdown("#### 💬 Last Response")
        st.markdown(
            f'<div class="response-area">{st.session_state.last_response}</div>',
            unsafe_allow_html=True
        )

    # ─────────────────────────────────────────
    #  Full Chat Display Area
    # ─────────────────────────────────────────
    if st.session_state.messages:
        st.divider()
        st.markdown("#### 📋 Full Conversation")
        for msg in st.session_state.messages:
            icon = "🤖" if msg["role"] == "assistant" else "🧑"
            with st.chat_message(msg["role"], avatar=icon):
                st.markdown(msg["content"])

else:
    # ── Offline Message ──
    st.error("⚠️ Cannot connect to Ollama server.")
    st.markdown("""
    **Start the Ollama server** by running this command in your terminal:

    ```
    ollama serve
    ```

    Then click **Retry Connection** in the sidebar.
    """)

# ─────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────
st.divider()
st.caption("🔒 100% Local & Private — No data leaves your machine | Built with Streamlit + Ollama")
