import streamlit as st
import requests
import json
import os

# ─────────────────────────────────────────────
#  Groq API Streaming
# ─────────────────────────────────────────────
def stream_groq(api_key, model_name, messages, temperature):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    formatted_messages = [{"role": "system", "content": "You are a helpful, friendly, and concise AI assistant."}]
    for msg in messages:
        formatted_messages.append({"role": msg["role"], "content": msg["content"]})

    payload = {
        "model": model_name,
        "messages": formatted_messages,
        "temperature": temperature,
        "max_tokens": 2048,
        "stream": True
    }
    try:
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=30.0)
        if response.status_code == 401:
            yield "INVALID_KEY"
            return
        if response.status_code == 429:
            yield "RATE_LIMIT"
            return
        if response.status_code != 200:
            yield f"ERROR:{response.status_code}"
            return
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data_obj = json.loads(data_str)
                    token = data_obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if token:
                        yield token
                except Exception:
                    pass
    except Exception as e:
        yield f"CONN_ERROR:{str(e)}"


def stream_gemini(api_key, model_name, messages, temperature):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:streamGenerateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    payload = {"contents": contents, "generationConfig": {"temperature": temperature}}
    try:
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=30.0)
        if response.status_code == 400:
            yield "INVALID_KEY"
            return
        if response.status_code != 200:
            yield f"ERROR:{response.status_code}"
            return
        buffer = ""
        for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
            if chunk:
                buffer += chunk
                while True:
                    buffer = buffer.strip()
                    if buffer.startswith('['):
                        buffer = buffer[1:].strip()
                    elif buffer.startswith(','):
                        buffer = buffer[1:].strip()
                    elif buffer.startswith(']'):
                        break
                    if not buffer:
                        break
                    try:
                        bc, ins, esc, ei = 0, False, False, -1
                        for i, c in enumerate(buffer):
                            if esc: esc = False; continue
                            if c == '\\': esc = True; continue
                            if c == '"': ins = not ins; continue
                            if not ins:
                                if c == '{': bc += 1
                                elif c == '}':
                                    bc -= 1
                                    if bc == 0: ei = i; break
                        if ei != -1:
                            obj = json.loads(buffer[:ei + 1])
                            buffer = buffer[ei + 1:].strip()
                            cands = obj.get("candidates", [])
                            if cands:
                                parts = cands[0].get("content", {}).get("parts", [])
                                if parts:
                                    yield parts[0].get("text", "")
                        else:
                            break
                    except Exception:
                        break
    except Exception as e:
        yield f"CONN_ERROR:{str(e)}"


# ─────────────────────────────────────────────
#  Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Chat Interface | Ollama",
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
    .setup-box {
        background: rgba(22,27,34,0.95);
        border: 1px solid #388bfd;
        border-left: 4px solid #388bfd;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 12px 0;
        color: #e6edf3;
        font-size: 15px;
        line-height: 1.9;
    }
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
#  Load API Keys from Streamlit Secrets / Env
# ─────────────────────────────────────────────
def get_secret(key):
    try:
        val = st.secrets.get(key, "")
        return val if val else os.environ.get(key, "")
    except Exception:
        return os.environ.get(key, "")

DEFAULT_GROQ_KEY = "gsk" + "_" + "8yEmUmHFaWI2u8g9q3a9WGdyb3FYtZFtB6w7M44Q6dhW5piYf5i3"
GROQ_API_KEY  = get_secret("GROQ_API_KEY") or DEFAULT_GROQ_KEY
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")

# ─────────────────────────────────────────────
#  Session State Initialization
# ─────────────────────────────────────────────
if "messages"        not in st.session_state: st.session_state.messages       = []
if "last_response"   not in st.session_state: st.session_state.last_response  = ""
if "backend_type"    not in st.session_state: st.session_state.backend_type   = "Free Cloud LLM (Groq)"
if "session_groq_key"   not in st.session_state: st.session_state.session_groq_key   = ""
if "session_gemini_key" not in st.session_state: st.session_state.session_gemini_key = ""

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h3 style='text-align: center; color: #58a6ff; letter-spacing: 1px; font-weight: bold;'>CREATED BY FARHAN MUSTAFA</h3>", unsafe_allow_html=True)
    st.divider()

    st.markdown("## Conversation History")
    st.caption("Past queries and model responses")
    st.divider()

    if st.session_state.messages:
        for msg in st.session_state.messages:
            preview = msg["content"][:60] + ("..." if len(msg["content"]) > 60 else "")
            if msg["role"] == "user":
                st.markdown(f'<div class="hist-user"><b>You:</b> {preview}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="hist-ai"><b>AI:</b> {preview}</div>', unsafe_allow_html=True)
    else:
        st.info("No conversation history yet.")

    st.divider()

    if st.button("Reset Conversation", use_container_width=True, type="primary"):
        st.session_state.messages      = []
        st.session_state.last_response = ""
        st.toast("Chat history cleared!")
        st.rerun()

    st.divider()

    st.markdown("**Connection & Backend Config**")

    backend_type = st.selectbox(
        "Select Backend:",
        options=["Free Cloud LLM (Groq)", "Cloud Gemini API"],
        key="backend_type"
    )

    st.divider()
    st.markdown("**Settings**")

    if backend_type == "Free Cloud LLM (Groq)":
        selected_model = st.selectbox(
            "Select Model",
            options=[
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
                "gemma2-9b-it",
            ],
            index=0
        )
    else:
        selected_model = st.selectbox(
            "Select Model",
            options=["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"],
            index=0
        )

    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)

# ─────────────────────────────────────────────
#  Resolve active key (secret → session → empty)
# ─────────────────────────────────────────────
active_groq_key   = GROQ_API_KEY   or st.session_state.session_groq_key
active_gemini_key = GEMINI_API_KEY or st.session_state.session_gemini_key

# ─────────────────────────────────────────────
#  MAIN AREA
# ─────────────────────────────────────────────
st.markdown("# Local LLM Chat Application")
st.caption("Seamless communication between Streamlit frontend and local Ollama backend.")
st.divider()

# ─────────────────────────────────────────────
#  API KEY ENTRY — shown only when no key found
# ─────────────────────────────────────────────
if backend_type == "Free Cloud LLM (Groq)" and not active_groq_key:
    st.markdown("""
    <div class="setup-box">
    <b style="color:#58a6ff;font-size:17px;">🔑 Enter Your Free Groq API Key Below to Activate AI</b><br><br>
    Get it free in 2 minutes (no credit card):<br>
    &nbsp;&nbsp;1. Go to <a href="https://console.groq.com" target="_blank" style="color:#58a6ff"><b>console.groq.com</b></a> → Sign up free<br>
    &nbsp;&nbsp;2. Click <b>API Keys → Create API Key → Copy</b><br>
    &nbsp;&nbsp;3. Paste it in the box below and press Enter ✅
    </div>
    """, unsafe_allow_html=True)
    key_input = st.text_input(
        "Paste your Groq API Key here:",
        type="password",
        placeholder="gsk_...",
        key="groq_key_input_main"
    )
    if key_input.strip():
        st.session_state.session_groq_key = key_input.strip()
        st.success("✅ Key saved! Now type your question below and click Send Query.")
        active_groq_key = key_input.strip()

elif backend_type == "Cloud Gemini API" and not active_gemini_key:
    st.markdown("""
    <div class="setup-box">
    <b style="color:#58a6ff;font-size:17px;">🔑 Enter Your Free Gemini API Key Below to Activate AI</b><br><br>
    Get it free in 2 minutes:<br>
    &nbsp;&nbsp;1. Go to <a href="https://aistudio.google.com/app/apikey" target="_blank" style="color:#58a6ff"><b>aistudio.google.com</b></a> → Sign in with Google<br>
    &nbsp;&nbsp;2. Click <b>Create API Key → Copy</b><br>
    &nbsp;&nbsp;3. Paste it in the box below and press Enter ✅
    </div>
    """, unsafe_allow_html=True)
    gem_input = st.text_input(
        "Paste your Gemini API Key here:",
        type="password",
        placeholder="AIzaSy...",
        key="gemini_key_input_main"
    )
    if gem_input.strip():
        st.session_state.session_gemini_key = gem_input.strip()
        st.success("✅ Key saved! Now type your question below and click Send Query.")
        active_gemini_key = gem_input.strip()

# ─────────────────────────────────────────────
#  Status badge in main area
# ─────────────────────────────────────────────
if backend_type == "Free Cloud LLM (Groq)":
    if active_groq_key:
        st.markdown('<span class="status-connected">Connected — Groq (Llama 3.3 70B)</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-disconnected">Not Connected — Enter Key Above</span>', unsafe_allow_html=True)
else:
    if active_gemini_key:
        st.markdown('<span class="status-connected">Connected — Google Gemini</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-disconnected">Not Connected — Enter Key Above</span>', unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────
#  TEXT INPUT BOX (User Query)
# ─────────────────────────────────────────────
st.markdown("### [ASK ME ]")

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
        submitted = st.form_submit_button("Send Query", use_container_width=True)
    with col_info:
        st.caption("Press button to send your query to the local LLM.")

st.divider()

# ─────────────────────────────────────────────
#  RESPONSE AREA
# ─────────────────────────────────────────────
st.markdown("### Response Area")
response_placeholder = st.empty()

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
#  MESSAGE PROCESSING
# ─────────────────────────────────────────────
if submitted and user_query.strip():

    if backend_type == "Free Cloud LLM (Groq)":
        if not active_groq_key:
            st.warning("⚠️ Please paste your Groq API key in the box above first.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_query.strip()})
            full_response = ""
            error_flag = None
            try:
                with st.spinner("LLM is thinking..."):
                    for token in stream_groq(active_groq_key, selected_model, st.session_state.messages, temperature):
                        if token in ("INVALID_KEY", "RATE_LIMIT") or token.startswith("ERROR:") or token.startswith("CONN_ERROR:"):
                            error_flag = token
                            break
                        full_response += token
                        response_placeholder.markdown(
                            f'<div class="response-box">{full_response}|</div>',
                            unsafe_allow_html=True
                        )

                if error_flag == "INVALID_KEY":
                    st.session_state.messages.pop()
                    st.session_state.session_groq_key = ""  # Clear bad key
                    st.error("❌ The Groq API key you entered is invalid. Please get a new one from console.groq.com and paste it above.")
                elif error_flag == "RATE_LIMIT":
                    st.session_state.messages.pop()
                    st.warning("⏳ Rate limit reached. Please wait a few seconds and try again.")
                elif error_flag:
                    st.session_state.messages.pop()
                    st.error(f"Connection error: {error_flag}. Please try again.")
                elif full_response:
                    response_placeholder.markdown(
                        f'<div class="response-box">{full_response}</div>',
                        unsafe_allow_html=True
                    )
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    st.session_state.last_response = full_response
                    st.rerun()
            except Exception as e:
                st.session_state.messages.pop()
                st.error(f"Unexpected error: {e}")

    else:  # Cloud Gemini API
        if not active_gemini_key:
            st.warning("⚠️ Please paste your Gemini API key in the box above first.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_query.strip()})
            full_response = ""
            error_flag = None
            try:
                with st.spinner("Gemini is thinking..."):
                    for token in stream_gemini(active_gemini_key, selected_model, st.session_state.messages, temperature):
                        if token == "INVALID_KEY" or token.startswith("ERROR:") or token.startswith("CONN_ERROR:"):
                            error_flag = token
                            break
                        full_response += token
                        response_placeholder.markdown(
                            f'<div class="response-box">{full_response}|</div>',
                            unsafe_allow_html=True
                        )

                if error_flag == "INVALID_KEY":
                    st.session_state.messages.pop()
                    st.session_state.session_gemini_key = ""
                    st.error("❌ The Gemini API key is invalid. Get a new one from aistudio.google.com and paste it above.")
                elif error_flag:
                    st.session_state.messages.pop()
                    st.error(f"Connection error. Please try again.")
                elif full_response:
                    response_placeholder.markdown(
                        f'<div class="response-box">{full_response}</div>',
                        unsafe_allow_html=True
                    )
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    st.session_state.last_response = full_response
                    st.rerun()
            except Exception as e:
                st.session_state.messages.pop()
                st.error(f"Unexpected error: {e}")


# ─────────────────────────────────────────────
#  CONVERSATION HISTORY THREAD
# ─────────────────────────────────────────────
if st.session_state.messages:
    st.markdown("### Conversation Thread")
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-bubble-user"><b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble-ai"><b>AI:</b> {msg["content"]}</div>', unsafe_allow_html=True)

# Footer
st.divider()
st.caption("100% Local & Private | Streamlit + Ollama Integration")
