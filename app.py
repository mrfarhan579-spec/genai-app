import streamlit as st
import requests
import json
import os

# ─────────────────────────────────────────────
#  Helper Functions for Cloud Integrations
# ─────────────────────────────────────────────
def stream_groq(api_key, model_name, messages, temperature):
    """
    Streams response from Groq's free API (OpenAI-compatible).
    Groq is free, fast, no credit card needed.
    """
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
            yield "Error: Invalid Groq API key."
            return
        if response.status_code == 429:
            yield "Error: Rate limit reached. Please wait a moment and try again."
            return
        if response.status_code != 200:
            yield f"Error: Groq API returned status {response.status_code}."
            return

        for line in response.iter_lines(decode_unicode=True):
            if line:
                if line.startswith("data: "):
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
        yield f"Connection Error: Failed to communicate with Groq API. {str(e)}"


def stream_gemini(api_key, model_name, messages, temperature):
    """
    Streams response from Gemini API using requests and a custom bracket-matching JSON parser.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:streamGenerateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    # Map roles from Streamlit/Ollama ("user", "assistant") to Gemini ("user", "model")
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({
            "role": role,
            "parts": [{"text": msg["content"]}]
        })

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=10.0)
        if response.status_code != 200:
            yield f"Error: Gemini API returned status code {response.status_code}. Detail: {response.text}"
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
                        buffer = buffer[1:].strip()
                        break

                    if not buffer:
                        break

                    try:
                        bracket_count = 0
                        in_string = False
                        escape = False
                        end_idx = -1
                        for idx, char in enumerate(buffer):
                            if escape:
                                escape = False
                                continue
                            if char == '\\':
                                escape = True
                                continue
                            if char == '"':
                                in_string = not in_string
                                continue
                            if not in_string:
                                if char == '{':
                                    bracket_count += 1
                                elif char == '}':
                                    bracket_count -= 1
                                    if bracket_count == 0:
                                        end_idx = idx
                                        break

                        if end_idx != -1:
                            obj_str = buffer[:end_idx + 1]
                            buffer = buffer[end_idx + 1:].strip()
                            obj = json.loads(obj_str)

                            candidates = obj.get("candidates", [])
                            if candidates:
                                content = candidates[0].get("content", {})
                                parts = content.get("parts", [])
                                if parts:
                                    text = parts[0].get("text", "")
                                    yield text
                        else:
                            break
                    except Exception:
                        break
    except Exception as e:
        yield f"Connection Error: Failed to communicate with Gemini API. {str(e)}"


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
#  Load API Keys from Streamlit Secrets / Env
# ─────────────────────────────────────────────
def get_secret(key, default=""):
    try:
        return st.secrets.get(key, "") or os.environ.get(key, default)
    except Exception:
        return os.environ.get(key, default)

GROQ_API_KEY = get_secret("GROQ_API_KEY")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")

# ─────────────────────────────────────────────
#  Session State Initialization
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_response" not in st.session_state:
    st.session_state.last_response = ""
if "backend_type" not in st.session_state:
    st.session_state.backend_type = "Free Cloud LLM (Groq)"
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = GEMINI_API_KEY

# ─────────────────────────────────────────────
#  SIDEBAR — Author Title + History Panel + Reset
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h3 style='text-align: center; color: #58a6ff; letter-spacing: 1px; font-weight: bold;'>CREATED BY FARHAN MUSTAFA</h3>", unsafe_allow_html=True)
    st.divider()

    st.markdown("## Conversation History")
    st.caption("Past queries and model responses")
    st.divider()

    # History Display
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

    # Reset Button
    if st.button("Reset Conversation", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.last_response = ""
        st.toast("Chat history cleared!")
        st.rerun()

    st.divider()

    # Connection Status & Configuration
    st.markdown("**Connection & Backend Config**")

    # Backend Type Selector
    backend_type = st.selectbox(
        "Select Backend:",
        options=["Free Cloud LLM (Groq)", "Cloud Gemini API"],
        key="backend_type"
    )

    if backend_type == "Free Cloud LLM (Groq)":
        # Use key from secrets by default; allow manual override
        custom_key = st.text_input(
            "Groq API Key:",
            type="password",
            placeholder="Auto-loaded from secrets...",
            value="",
            help="Leave blank — it is pre-configured automatically."
        )
        active_groq_key = custom_key.strip() if custom_key.strip() else GROQ_API_KEY

        if active_groq_key:
            st.markdown('<span class="status-connected">Connected</span>', unsafe_allow_html=True)
            st.caption("Connected to Groq Free API (Llama 3.3 70B)")
        else:
            st.markdown('<span class="status-disconnected">Key Missing</span>', unsafe_allow_html=True)
            st.warning("Add GROQ_API_KEY to your Streamlit secrets to activate.")

        # Model Settings
        st.divider()
        st.markdown("**Settings**")
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
        temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)

    else:  # Cloud Gemini API
        gemini_key_input = st.text_input(
            "Gemini API Key:",
            type="password",
            key="gemini_api_key",
            placeholder="AIzaSy..."
        )

        active_gemini_key = gemini_key_input.strip() if gemini_key_input.strip() else GEMINI_API_KEY

        if active_gemini_key:
            st.markdown('<span class="status-connected">Key Provided</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-disconnected">Key Missing</span>', unsafe_allow_html=True)
            st.info("Get a free Gemini API Key from Google AI Studio and paste it above, or set it as a `GEMINI_API_KEY` secret in your Streamlit Cloud dashboard.")

        # Model Settings
        st.divider()
        st.markdown("**Settings**")
        selected_model = st.selectbox(
            "Select Model",
            options=["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"],
            index=0
        )
        temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)

# ─────────────────────────────────────────────
#  MAIN AREA
# ─────────────────────────────────────────────
st.markdown("# Local LLM Chat Application")
st.caption("Seamless communication between Streamlit frontend and local Ollama backend.")
st.divider()

# ─────────────────────────────────────────────
#  TEXT INPUT BOX (User Query)
# ─────────────────────────────────────────────
st.markdown("### [ASK ME ]")

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
        submitted = st.form_submit_button("Send Query", use_container_width=True)
    with col_info:
        st.caption("Press button to send your query to the local LLM.")

st.divider()

# ─────────────────────────────────────────────
#  RESPONSE AREA (Shows current or last output)
# ─────────────────────────────────────────────
st.markdown("### Response Area")

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
#  API Connection and Message Processing
# ─────────────────────────────────────────────
if submitted and user_query.strip():

    if backend_type == "Free Cloud LLM (Groq)":
        if not active_groq_key:
            st.error("Cannot communicate with the LLM backend. Please add GROQ_API_KEY to your Streamlit secrets.")
        else:
            # Add user query to conversation history
            st.session_state.messages.append({"role": "user", "content": user_query.strip()})

            # Stream the response from Groq
            full_response = ""
            try:
                with st.spinner("LLM is thinking..."):
                    stream = stream_groq(
                        api_key=active_groq_key,
                        model_name=selected_model,
                        messages=st.session_state.messages,
                        temperature=temperature
                    )

                    has_error = False
                    for token in stream:
                        if token.startswith("Error:") or token.startswith("Connection Error:"):
                            has_error = True
                            st.error(f"Error communicating with LLM backend: {token}")
                            break
                        full_response += token
                        response_placeholder.markdown(
                            f'<div class="response-box">{full_response}|</div>',
                            unsafe_allow_html=True
                        )

                if not has_error and full_response:
                    # Final output formatting
                    response_placeholder.markdown(
                        f'<div class="response-box">{full_response}</div>',
                        unsafe_allow_html=True
                    )
                    # Save assistant response
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    st.session_state.last_response = full_response
                    st.rerun()
                elif has_error:
                    st.session_state.messages.pop()

            except Exception as e:
                st.error(f"Error communicating with LLM backend: {e}")
                st.session_state.messages.pop()

    else:  # Cloud Gemini API
        if not active_gemini_key:
            st.error("Please enter a valid Gemini API Key in the sidebar to send queries.")
        else:
            # Add user query to conversation history
            st.session_state.messages.append({"role": "user", "content": user_query.strip()})

            # Stream the response from Gemini
            full_response = ""
            try:
                with st.spinner("Gemini is thinking..."):
                    stream = stream_gemini(
                        api_key=active_gemini_key,
                        model_name=selected_model,
                        messages=st.session_state.messages,
                        temperature=temperature
                    )

                    has_error = False
                    for token in stream:
                        if token.startswith("Error:") or token.startswith("Connection Error:"):
                            has_error = True
                            st.error(f"Error communicating with LLM backend: {token}")
                            break
                        full_response += token
                        response_placeholder.markdown(
                            f'<div class="response-box">{full_response}|</div>',
                            unsafe_allow_html=True
                        )

                if not has_error and full_response:
                    # Final output formatting
                    response_placeholder.markdown(
                        f'<div class="response-box">{full_response}</div>',
                        unsafe_allow_html=True
                    )
                    # Save assistant response
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    st.session_state.last_response = full_response
                    st.rerun()
                elif has_error:
                    st.session_state.messages.pop()

            except Exception as e:
                st.error(f"Error communicating with LLM backend: {e}")
                st.session_state.messages.pop()


# ─────────────────────────────────────────────
#  CONVERSATION HISTORY THREAD (Full back-and-forth)
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
