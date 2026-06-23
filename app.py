import streamlit as st
import ollama
from ollama import Client

# ─────────────────────────────────────────────
#  Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Local LLM Playground | Ollama",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  Custom Premium CSS Styling
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Outfit', 'Segoe UI', sans-serif !important; }

    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #0f1923 50%, #0d1117 100%);
        color: #c9d1d9;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
        border-right: 1px solid #30363d;
    }

    /* Main title glow effect */
    h1 { color: #58a6ff !important; text-shadow: 0 0 30px rgba(88,166,255,0.3); }
    h2, h3 { color: #79c0ff !important; }

    /* Chat message styling */
    [data-testid="stChatMessage"] {
        background: rgba(22, 27, 34, 0.8) !important;
        border: 1px solid #30363d !important;
        border-radius: 12px !important;
        padding: 12px !important;
        margin: 6px 0 !important;
        backdrop-filter: blur(10px);
    }

    /* Input Text Area Styling */
    .stTextArea textarea {
        background: rgba(22, 27, 34, 0.9) !important;
        border: 1px solid #388bfd !important;
        border-radius: 10px !important;
        color: #c9d1d9 !important;
        font-size: 15px !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: #58a6ff !important;
        box-shadow: 0 0 0 3px rgba(88,166,255,0.15) !important;
    }

    /* Chat input box */
    [data-testid="stChatInputTextArea"] {
        background: rgba(22, 27, 34, 0.9) !important;
        border: 1px solid #388bfd !important;
        border-radius: 10px !important;
        color: #c9d1d9 !important;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        background: linear-gradient(135deg, #238636, #2ea043) !important;
        color: white !important;
        border: none !important;
        transition: all 0.3s ease !important;
        font-weight: 500 !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(46,160,67,0.4) !important;
    }

    /* Reset button */
    .stButton>button[kind="secondary"] {
        background: linear-gradient(135deg, #b91c1c, #dc2626) !important;
    }
    .stButton>button[kind="secondary"]:hover {
        box-shadow: 0 4px 15px rgba(220,38,38,0.4) !important;
    }

    /* Status badges */
    .status-connected {
        display: inline-block;
        background: rgba(46,160,67,0.15);
        border: 1px solid #2ea043;
        color: #3fb950;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 500;
    }
    .status-disconnected {
        display: inline-block;
        background: rgba(218,54,51,0.15);
        border: 1px solid #da3633;
        color: #f85149;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 500;
    }

    /* Info boxes */
    .info-box {
        background: rgba(56,139,253,0.1);
        border: 1px solid #388bfd;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 14px;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(22,27,34,0.8);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 10px;
    }

    /* Sliders */
    [data-testid="stSlider"] > div > div {
        color: #58a6ff !important;
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
WELCOME_MSG = "👋 **Hello!** I am your locally hosted AI assistant powered by **Ollama**. Ask me anything — I'm running entirely on your machine, no internet required!"

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": WELCOME_MSG}]
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "chat"         # "chat" or "textarea"
if "textarea_input" not in st.session_state:
    st.session_state.textarea_input = ""
if "submit_textarea" not in st.session_state:
    st.session_state.submit_textarea = False

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
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 LLM Console")
    st.caption("Your local AI command center")
    st.divider()

    # Connection Status
    st.markdown("**Connection Status**")
    if is_connected:
        st.markdown('<span class="status-connected">🟢 Connected to Ollama</span>', unsafe_allow_html=True)
        st.caption(f"Endpoint: `{OLLAMA_HOST}`")
    else:
        st.markdown('<span class="status-disconnected">🔴 Ollama Offline</span>', unsafe_allow_html=True)
        st.warning("Start Ollama server and retry.")
        if st.button("🔄 Retry Connection", use_container_width=True):
            st.rerun()
    st.divider()

    # Model Selection
    st.markdown("**⚙️ Model Settings**")
    if is_connected and available_models:
        selected_model = st.selectbox(
            "Active Model",
            options=available_models,
            index=0,
            help="Select which locally installed model to chat with."
        )
    else:
        selected_model = "qwen2.5:0.5b"
        st.info("No models found. Pull a model below.", icon="ℹ️")

    # Parameters
    temperature = st.slider(
        "🌡️ Temperature",
        min_value=0.0, max_value=1.5, value=0.7, step=0.05,
        help="Higher = more creative. Lower = more focused & precise."
    )
    top_p = st.slider(
        "🎯 Top P",
        min_value=0.0, max_value=1.0, value=0.9, step=0.05,
        help="Nucleus sampling threshold. Controls vocabulary diversity."
    )

    system_prompt = st.text_area(
        "📋 System Prompt",
        value="You are a helpful, friendly, and concise AI assistant.",
        height=100,
        help="Instructions that define the AI's behavior and personality."
    )

    # Input Mode Toggle
    st.divider()
    st.markdown("**✏️ Input Mode**")
    input_mode = st.radio(
        "Choose how to type your query:",
        options=["💬 Chat Input (Quick)", "📝 Text Box (Multi-line)"],
        index=0,
        help="Chat Input is great for quick messages. Text Box is better for long or structured queries."
    )
    st.session_state.input_mode = "chat" if "Chat" in input_mode else "textarea"

    # Pull Model
    st.divider()
    if is_connected:
        with st.expander("📥 Download a Model", expanded=False):
            st.caption("Recommended for 4GB RAM:")
            st.markdown("- `qwen2.5:0.5b` — Fastest (~350MB)")
            st.markdown("- `llama3.2:1b` — Smarter (~1.3GB)")
            model_to_pull = st.text_input("Model identifier", value="qwen2.5:0.5b", key="pull_model_input")
            if st.button("⬇️ Download Model", use_container_width=True):
                if model_to_pull.strip():
                    prog = st.progress(0, text="Connecting…")
                    try:
                        for part in client.pull(model=model_to_pull, stream=True):
                            status = part.get('status', '')
                            completed = part.get('completed', 0)
                            total = part.get('total', 1)
                            pct = min(int((completed / total) * 100), 100) if total > 0 else 0
                            prog.progress(pct, text=f"{status} — {pct}%")
                        st.success(f"✅ `{model_to_pull}` downloaded!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.warning("Enter a valid model name.")

    # Stats
    st.divider()
    st.markdown("**📊 Session Stats**")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Messages", len(st.session_state.messages))
    with col2:
        st.metric("Queries", st.session_state.total_queries)

    # Reset Button
    st.divider()
    if st.button("🗑️ Reset Conversation", use_container_width=True, type="secondary"):
        st.session_state.messages = [{"role": "assistant", "content": WELCOME_MSG}]
        st.session_state.total_queries = 0
        st.toast("✅ Conversation cleared!", icon="🧹")
        st.rerun()

# ─────────────────────────────────────────────
#  MAIN CHAT AREA
# ─────────────────────────────────────────────
# Header
col_title, col_model = st.columns([3, 1])
with col_title:
    st.markdown("# 💬 Local AI Chat Playground")
    st.caption(f"Powered by **Ollama** • Model: `{selected_model}` • Fully local & private")
with col_model:
    if is_connected:
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-box">🔒 <b>100% Local</b><br>No data leaves your machine</div>', unsafe_allow_html=True)

st.divider()

# Conversation History Panel
with st.container():
    for message in st.session_state.messages:
        icon = "🤖" if message["role"] == "assistant" else "🧑"
        with st.chat_message(message["role"], avatar=icon):
            st.markdown(message["content"])

st.divider()

# ─────────────────────────────────────────────
#  INPUT SECTION
# ─────────────────────────────────────────────
def generate_and_display(user_prompt: str):
    """Handles sending a message, streaming the response and saving history."""
    if not user_prompt.strip():
        return

    # Add and show user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_prompt)

    # Build API context
    api_messages = []
    if system_prompt.strip():
        api_messages.append({"role": "system", "content": system_prompt})
    for msg in st.session_state.messages:
        if msg["content"] == WELCOME_MSG:
            continue
        api_messages.append({"role": msg["role"], "content": msg["content"]})

    # Stream response
    with st.chat_message("assistant", avatar="🤖"):
        def response_stream():
            try:
                stream = client.chat(
                    model=selected_model,
                    messages=api_messages,
                    options={"temperature": temperature, "top_p": top_p},
                    stream=True
                )
                for chunk in stream:
                    yield chunk.get("message", {}).get("content", "")
            except Exception as err:
                yield f"\n\n⚠️ **Error:** {err}"

        full_response = st.write_stream(response_stream)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.total_queries += 1


if is_connected:
    # ── MODE 1: Chat Input (inline, quick) ──────────────────
    if st.session_state.input_mode == "chat":
        st.markdown("#### 💬 Quick Chat Input")
        if prompt := st.chat_input("Type your message and press Enter…", key="chat_input_box"):
            generate_and_display(prompt)
            st.rerun()

    # ── MODE 2: Text Box (multi-line, structured) ────────────
    else:
        st.markdown("#### 📝 Text Box Input")
        st.caption("Type your full query below. Supports multi-line input for detailed questions.")

        user_text = st.text_area(
            label="Your Query",
            placeholder="Type your question or prompt here...\n\nYou can write multiple lines, paste code, or ask complex questions.",
            height=160,
            key="main_text_area",
            label_visibility="collapsed"
        )

        btn_col1, btn_col2, _ = st.columns([1, 1, 4])
        with btn_col1:
            send_clicked = st.button("🚀 Send Query", use_container_width=True, key="send_btn")
        with btn_col2:
            clear_input = st.button("🧹 Clear Input", use_container_width=True, key="clear_input_btn")

        if send_clicked and user_text.strip():
            generate_and_display(user_text)
            st.rerun()
        elif send_clicked and not user_text.strip():
            st.warning("⚠️ Please enter a query before sending.")

        if clear_input:
            st.rerun()

else:
    st.markdown("""
    <div class="info-box">
    ⚠️ <b>Ollama server is not running.</b><br>
    Please start the Ollama server using the command below, then click <b>Retry Connection</b> in the sidebar:
    <br><br>
    <code>E:\\ollama\\ollama.exe serve</code>
    </div>
    """, unsafe_allow_html=True)
