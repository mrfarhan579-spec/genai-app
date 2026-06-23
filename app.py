import streamlit as st
import requests
import json
import os
from ollama import Client

# ─────────────────────────────────────────────
#  Helper Functions for Gemini Integration
# ─────────────────────────────────────────────
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
#  Session State Initialization
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_response" not in st.session_state:
    st.session_state.last_response = ""
if "ollama_host_config" not in st.session_state:
    st.session_state.ollama_host_config = "http://127.0.0.1:11434"
if "backend_type" not in st.session_state:
    st.session_state.backend_type = "Cloud Gemini API"

# Retrieve default Gemini API key from environment/secrets
default_gemini_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = default_gemini_key

# ─────────────────────────────────────────────
#  Ollama Connection Check (Only run if Ollama backend is selected)
# ─────────────────────────────────────────────
OLLAMA_HOST = st.session_state.ollama_host_config.strip()
client = Client(host=OLLAMA_HOST)

is_connected = False
available_models = []

if st.session_state.backend_type == "Local Ollama":
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=1.5)
        if r.status_code == 200:
            is_connected = True
            available_models = [m['name'] for m in r.json().get('models', [])]
    except Exception:
        is_connected = False

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
        options=["Cloud Gemini API", "Local Ollama"],
        key="backend_type"
    )
    
    if backend_type == "Local Ollama":
        # Text input to dynamically modify host URL
        new_host = st.text_input(
            "Ollama Host URL:",
            key="ollama_host_config",
            placeholder="e.g. http://127.0.0.1:11434"
        )
        
        if is_connected:
            st.markdown('<span class="status-connected">Connected</span>', unsafe_allow_html=True)
            st.caption(f"Connected to backend: `{OLLAMA_HOST}`")
        else:
            st.markdown('<span class="status-disconnected">Disconnected</span>', unsafe_allow_html=True)
            st.warning("Cannot communicate with the LLM backend. Please make sure Ollama server is running locally on port 11434.")
            st.info("**Streamlit Cloud Note:** If you are running this app on Streamlit Cloud, you must expose your local Ollama port (11434) using a tunnel (e.g. `ngrok http 11434`) and paste the ngrok URL above.")
            
            if st.button("Retry Connection", use_container_width=True):
                st.rerun()
                
        # Model Settings
        st.divider()
        st.markdown("**Settings**")
        if is_connected and available_models:
            selected_model = st.selectbox("Select Model", options=available_models, index=0)
        else:
            selected_model = st.selectbox("Select Model", options=["qwen2.5:0.5b"], index=0)
        temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
        
    else:  # Cloud Gemini API
        gemini_key_input = st.text_input(
            "Gemini API Key:",
            type="password",
            key="gemini_api_key",
            placeholder="AIzaSy..."
        )
        
        if gemini_key_input:
            st.markdown('<span class="status-connected">Key Provided</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-disconnected">Key Missing</span>', unsafe_allow_html=True)
            st.info("Get a free Gemini API Key from Google AI Studio and paste it above, or set it as a `GEMINI_API_KEY` secret in your Streamlit Cloud dashboard.")
            
        # Model Settings
        st.divider()
        st.markdown("**Settings**")
        selected_gemini_model = st.selectbox(
            "Select Model",
            options=["gemini-1.5-flash", "gemini-1.5-pro"],
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
    if backend_type == "Local Ollama":
        if not is_connected:
            st.error("Cannot communicate with the LLM backend. Please make sure Ollama server is running locally on port 11434.")
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
                with st.spinner("LLM is thinking..."):
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
                            f'<div class="response-box">{full_response}|</div>',
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
                st.rerun()

            except Exception as e:
                st.error(f"Error communicating with LLM backend: {e}")

    else:  # Cloud Gemini API
        if not st.session_state.gemini_api_key.strip():
            st.error("Please enter a valid Gemini API Key in the sidebar to send queries.")
        else:
            # Add user query to conversation history
            st.session_state.messages.append({"role": "user", "content": user_query.strip()})

            # Stream the response from Gemini
            full_response = ""
            try:
                with st.spinner("Gemini is thinking..."):
                    stream = stream_gemini(
                        api_key=st.session_state.gemini_api_key.strip(),
                        model_name=selected_gemini_model,
                        messages=st.session_state.messages,
                        temperature=temperature
                    )
                    for token in stream:
                        full_response += token
                        response_placeholder.markdown(
                            f'<div class="response-box">{full_response}|</div>',
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
                st.rerun()

            except Exception as e:
                st.error(f"Error communicating with Gemini API: {e}")


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
