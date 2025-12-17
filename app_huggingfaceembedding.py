import os
import tempfile
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
import time
import base64
import speech_recognition as sr
from gtts import gTTS
import pygame
import io
import threading
from audio_recorder_streamlit import audio_recorder

# Load .env variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Page configuration
st.set_page_config(
    page_title="Premium Car Manual AI",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize pygame mixer for audio playback
try:
    pygame.mixer.init()
except:
    st.warning("Audio playback may not work properly. Please ensure pygame is installed.")


# ✅ Function to set background image
def set_background(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{encoded_image}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
                color: #fff;
            }}
            .block-container {{
                background-color: rgba(0, 0, 0, 0.6);
                padding: 2rem;
                border-radius: 1rem;
                box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.5);
            }}
            .stTextArea textarea {{
                background-color: #fefefe;
                color: #333;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.warning(f"Background image not loaded: {e}")

# ✅ Call the function to apply background
set_background(r"C:\Users\BIT\OneDrive\Desktop\car-dummy\tata.jpg")

# Premium automotive-themed CSS with voice controls
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Orbitron:wght@400;700;900&display=swap');
    
    /* Main styling */
    .main {
        padding-top: 0rem;
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%);
        color: #ffffff;
    }
    
    /* Voice Control Styling */
    .voice-control-panel {
        background: linear-gradient(135deg, rgba(255,107,53,0.15) 0%, rgba(247,147,30,0.15) 100%);
        border: 2px solid rgba(255,107,53,0.3);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
        backdrop-filter: blur(15px);
        box-shadow: 0 10px 30px rgba(255,107,53,0.2);
    }
    
    .voice-button {
        background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%);
        border: none;
        border-radius: 50%;
        width: 80px;
        height: 80px;
        font-size: 2rem;
        color: white;
        cursor: pointer;
        margin: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(255, 107, 53, 0.4);
        animation: pulse 2s infinite;
    }
    
    .voice-button:hover {
        transform: scale(1.1);
        box-shadow: 0 12px 35px rgba(255, 107, 53, 0.6);
    }
    
    .voice-button.recording {
        background: linear-gradient(135deg, #ff0000 0%, #cc0000 100%);
        animation: recording-pulse 1s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 8px 25px rgba(255, 107, 53, 0.4); }
        50% { box-shadow: 0 8px 25px rgba(255, 107, 53, 0.8); }
        100% { box-shadow: 0 8px 25px rgba(255, 107, 53, 0.4); }
    }
    
    .voice-button.stop {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        animation: stop-pulse 1s infinite;
    }
    
    @keyframes stop-pulse {
        0% { transform: scale(1); box-shadow: 0 8px 25px rgba(220, 38, 38, 0.4); }
        50% { transform: scale(1.05); box-shadow: 0 12px 35px rgba(220, 38, 38, 0.6); }
        100% { transform: scale(1); box-shadow: 0 8px 25px rgba(220, 38, 38, 0.4); }
    }
    
    .speaking-indicator {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(22, 163, 74, 0.15) 100%);
        border: 2px solid rgba(34, 197, 94, 0.3);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
        backdrop-filter: blur(15px);
        animation: speaking-glow 2s infinite;
    }
    
    @keyframes speaking-glow {
        0% { border-color: rgba(34, 197, 94, 0.3); }
        50% { border-color: rgba(34, 197, 94, 0.8); }
        100% { border-color: rgba(34, 197, 94, 0.3); }
    }
    
    .voice-status {
        font-size: 1.2rem;
        font-weight: 600;
        margin: 1rem 0;
        color: #ff6b35;
    }
    
    .audio-visualizer {
        width: 100%;
        height: 60px;
        background: rgba(255,107,53,0.1);
        border-radius: 10px;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
    }
    
    .audio-bar {
        width: 4px;
        background: linear-gradient(to top, #ff6b35, #f7931e);
        margin: 0 2px;
        border-radius: 2px;
        animation: audio-wave 1s infinite ease-in-out;
    }
    
    @keyframes audio-wave {
        0%, 100% { height: 10px; }
        50% { height: 40px; }
    }
    
    /* Header styling - Premium automotive theme */
    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.08) 100%);
        pointer-events: none;
    }
    
    .header-title {
        font-size: 3.5rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 20px rgba(255,255,255,0.3);
        background: linear-gradient(135deg, #ffffff 0%, #cccccc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: 2px;
    }
    
    .header-subtitle {
        font-size: 1.3rem;
        font-weight: 300;
        opacity: 0.8;
        letter-spacing: 1px;
        margin-top: 1rem;
    }
    
    .luxury-accent {
        width: 100px;
        height: 3px;
        background: linear-gradient(90deg, #ff6b35 0%, #f7931e 100%);
        margin: 1rem auto;
        border-radius: 2px;
        box-shadow: 0 0 15px rgba(255, 107, 53, 0.5);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a1a 0%, #0f0f0f 100%);
    }
    
    /* Premium card styling */
    .feature-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 
            0 8px 32px rgba(0,0,0,0.3),
            inset 0 1px 0 rgba(255,255,255,0.1);
        margin: 1.5rem 0;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: left 0.6s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 
            0 15px 50px rgba(0,0,0,0.4),
            inset 0 1px 0 rgba(255,255,255,0.2);
        border-color: rgba(255, 107, 53, 0.3);
    }
    
    .feature-card:hover::before {
        left: 100%;
    }
    
    .card-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 15px;
       
        letter-spacing: 1px;
    }
    
    .card-subtitle {
        color: #cccccc;
        font-size: 1rem;
        margin-bottom: 1rem;
        line-height: 1.6;
    }
    
    /* Premium status messages */
    .success-message {
        background: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%);
        color: #000000;
        padding: 1.5rem;
        border-radius: 15px;
        font-weight: 600;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(0, 255, 136, 0.3);
        border: 1px solid rgba(0, 255, 136, 0.4);
    }
    
    .info-message {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        font-weight: 500;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.3);
    }
    
    .warning-message {
        background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        font-weight: 600;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(255, 107, 53, 0.3);
    }
    
    /* Premium chat styling */
    .chat-message {
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 20px;
      
        backdrop-filter: blur(10px);
        position: relative;
    }
    
    .user-message {
        background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%);
        color: white;
        margin-left: 15%;
        text-align: right;
        box-shadow: 0 8px 25px rgba(255, 107, 53, 0.3);
        border: 1px solid rgba(255, 107, 53, 0.4);
    }
    
    .assistant-message {
        background: linear-gradient(135deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.04) 100%);
        border: 1px solid rgba(255,255,255,0.15);
        margin-right: 15%;
        color: #ffffff;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    
    /* Premium image styling */
    .uploaded-image {
        border-radius: 20px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.4);
        margin: 2rem 0;
        border: 2px solid rgba(255,255,255,0.1);
        overflow: hidden;
    }
    
    .caption-box {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.1) 0%, rgba(0, 204, 106, 0.1) 100%);
        border: 1px solid rgba(0, 255, 136, 0.3);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        font-style: italic;
        color: #00ff88;
        backdrop-filter: blur(10px);
    }
    
    /* Premium metrics styling */
    .metric-container {
        background: linear-gradient(135deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.04) 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        text-align: center;
        border: 1px solid rgba(255,255,255,0.15);
        backdrop-filter: blur(10px);
    }
    
    .metric-value {
        font-family: 'Orbitron', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        color: #ff6b35;
        text-shadow: 0 0 15px rgba(255, 107, 53, 0.5);
    }
    
    .metric-label {
        color: #cccccc;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Premium button styling */
    .stButton > button {
        background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
        letter-spacing: 1px;
        transition: all 0.4s ease;
        box-shadow: 
            0 8px 25px rgba(255, 107, 53, 0.3),
            inset 0 1px 0 rgba(255,255,255,0.2);
        text-transform: uppercase;
        font-size: 0.9rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 
            0 15px 40px rgba(255, 107, 53, 0.4),
            inset 0 1px 0 rgba(255,255,255,0.3);
        background: linear-gradient(135deg, #f7931e 0%, #ff6b35 100%);
    }
    
    /* Premium progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #ff6b35 0%, #f7931e 100%);
        box-shadow: 0 0 15px rgba(255, 107, 53, 0.5);
    }
    
    /* Car brand logos styling */
    .brand-logos {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 30px;
        padding: 2rem 0;
        opacity: 0.6;
        filter: grayscale(100%);
        transition: all 0.3s ease;
    }
    
    .brand-logos:hover {
        opacity: 0.9;
        filter: grayscale(0%);
    }
    
    .brand-logo {
        width: 40px;
        height: 40px;
        background: rgba(255,255,255,0.1);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.3s ease;
    }
    
    .brand-logo:hover {
        background: rgba(255,255,255,0.2);
        transform: scale(1.1);
    }
    
    /* Luxury divider */
    .luxury-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.2) 50%, transparent 100%);
        margin: 3rem 0;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #f7931e 0%, #ff6b35 100%);
    }
</style>
""", unsafe_allow_html=True)

# Voice Assistant Functions
def text_to_speech(text, lang='en'):
    """Convert text to speech and play it with stop mechanism"""
    try:
        st.session_state.is_speaking = True
        
        tts = gTTS(text=text, lang=lang, slow=False)
        
        # Save to BytesIO object
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        # Save temporarily and play
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tmp_file.write(audio_buffer.read())
            tmp_file_path = tmp_file.name
        
        # Play audio using pygame
        pygame.mixer.music.load(tmp_file_path)
        pygame.mixer.music.play()
        
        # Check for stop signal during playback
        while pygame.mixer.music.get_busy() and not st.session_state.stop_speaking:
            time.sleep(0.1)
        
        # Stop if requested
        if st.session_state.stop_speaking:
            pygame.mixer.music.stop()
            st.session_state.stop_speaking = False
        
        # Clean up
        os.unlink(tmp_file_path)
        st.session_state.is_speaking = False
        
        return True
    except Exception as e:
        st.session_state.is_speaking = False
        st.error(f"Text-to-speech error: {str(e)}")
        return False

def stop_voice():
    """Stop current voice playback"""
    st.session_state.stop_speaking = True
    try:
        pygame.mixer.music.stop()
    except:
        pass
    st.session_state.is_speaking = False

def speech_to_text(audio_data):
    """Convert speech to text"""
    try:
        r = sr.Recognizer()
        
        # Save audio data to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_data)
            tmp_file_path = tmp_file.name
        
        # Load audio file
        with sr.AudioFile(tmp_file_path) as source:
            audio = r.record(source)
        
        # Recognize speech
        text = r.recognize_google(audio)
        
        # Clean up
        os.unlink(tmp_file_path)
        
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Speech recognition error: {e}"
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize models with caching
@st.cache_resource
def load_models():
    """Load and cache the AI models"""
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return llm, embedding, processor, blip_model

# Premium Header with Voice Indicator
st.markdown("""
<div class="header-container">
    <div class="header-title">🎙️ AutoAssist AI</div>
    <div class="luxury-accent"></div>
    <div class="header-subtitle">Advanced Automotive Intelligence & Voice Support System</div>
</div>
""", unsafe_allow_html=True)

# Load models
with st.spinner("⚡ Initializing Premium AI Systems..."):
    llm, embedding, processor, blip_model = load_models()

# Premium Sidebar
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <h2 style="color: #ff6b35;  letter-spacing: 2px;">🏎️ CONTROL HUB</h2>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown('<div class="luxury-divider"></div>', unsafe_allow_html=True)

# Initialize session state
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "kb_built" not in st.session_state:
    st.session_state.kb_built = False
if "voice_enabled" not in st.session_state:
    st.session_state.voice_enabled = True
if "auto_speak" not in st.session_state:
    st.session_state.auto_speak = True
if "is_speaking" not in st.session_state:
    st.session_state.is_speaking = False
if "stop_speaking" not in st.session_state:
    st.session_state.stop_speaking = False

# Voice Settings in Sidebar
st.sidebar.markdown("### 🎙️ VOICE CONTROLS")
st.session_state.voice_enabled = st.sidebar.toggle("🔊 Voice Input", value=st.session_state.voice_enabled)
st.session_state.auto_speak = st.sidebar.toggle("🗣️ Auto Response Speech", value=st.session_state.auto_speak)

# Voice Stop Button
if st.session_state.is_speaking:
    if st.sidebar.button("⏹️ STOP VOICE", type="primary", use_container_width=True):
        stop_voice()
        st.rerun()

voice_language = st.sidebar.selectbox(
    "🌐 Voice Language",
    options=['en', 'hi', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko'],
    index=0,
    format_func=lambda x: {
        'en': '🇺🇸 English',
        'hi': '🇮🇳 Hindi',
        'es': '🇪🇸 Spanish',
        'fr': '🇫🇷 French',
        'de': '🇩🇪 German',
        'it': '🇮🇹 Italian',
        'pt': '🇵🇹 Portuguese',
        'ru': '🇷🇺 Russian',
        'ja': '🇯🇵 Japanese',
        'ko': '🇰🇷 Korean'
    }[x]
)

st.sidebar.markdown('<div class="luxury-divider"></div>', unsafe_allow_html=True)

# Premium Sidebar metrics
st.sidebar.markdown("### 📊 SYSTEM STATUS")

col1, col2 = st.sidebar.columns(2)
with col1:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">{len(st.session_state.messages)}</div>
        <div class="metric-label">MESSAGES</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    kb_status = "🟢" if st.session_state.kb_built else "🔴"
    kb_text = "READY" if st.session_state.kb_built else "OFFLINE"
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">{kb_status}</div>
        <div class="metric-label">{kb_text}</div>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown('<div class="luxury-divider"></div>', unsafe_allow_html=True)

# Premium Knowledge Base Section
st.sidebar.markdown("### 🧠 AI KNOWLEDGE CORE")
if st.sidebar.button("⚡ INITIALIZE SYSTEM", use_container_width=True):
    with st.spinner("🚀 Deploying AI Neural Networks..."):
        progress_bar = st.progress(0)
        
        try:
            # Load PDFs
            progress_bar.progress(25)
            st.text("📁 Scanning Manual Database...")
            loader = PyPDFDirectoryLoader("research_papers")
            docs = loader.load()
            
            if not docs:
                st.error("❌ ALERT: No manuals detected in database!")
                # Create dummy knowledge base for demo
                st.warning("🔧 Creating demo knowledge base...")
                from langchain.schema import Document
                demo_docs = [
                    Document(page_content="Car maintenance includes regular oil changes, tire rotations, brake inspections, and fluid checks. Premium vehicles require synthetic oil and high-quality parts.", metadata={"source": "demo"}),
                    Document(page_content="Safety protocols include wearing seatbelts, checking mirrors, maintaining safe following distance, and regular vehicle inspections.", metadata={"source": "demo"}),
                    Document(page_content="Performance optimization involves proper tire pressure, clean air filters, quality fuel, and regular engine tuning.", metadata={"source": "demo"})
                ]
                docs = demo_docs
            
            # Split documents
            progress_bar.progress(50)
            st.text("🔄 Processing Knowledge Fragments...")
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = splitter.split_documents(docs)
            
            # Create vector database
            progress_bar.progress(75)
            st.text("🧠 Building Neural Pathways...")
            vectordb = FAISS.from_documents(texts, embedding)
            
            # Store in session
            progress_bar.progress(100)
            st.text("✅ System Online!")
            st.session_state.vectordb = vectordb
            st.session_state.chat_memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True, output_key="answer"
            )
            st.session_state.kb_built = True
            
            st.success("🎯 PREMIUM AI SYSTEM ACTIVATED!")
            
            # Voice announcement
            if st.session_state.voice_enabled and st.session_state.auto_speak:
                threading.Thread(
                    target=text_to_speech,
                    args=("Premium AI system activated and ready for voice commands!", voice_language)
                ).start()
            
            time.sleep(2)
            st.rerun()
                
        except Exception as e:
            st.error(f"⚠️ SYSTEM ERROR: {str(e)}")
        finally:
            progress_bar.empty()

# Premium clear chat button
if st.sidebar.button("🗑️ RESET MEMORY", use_container_width=True):
    st.session_state.messages = []
    if st.session_state.chat_memory:
        st.session_state.chat_memory.clear()
    st.rerun()

# Voice Control Panel
if st.session_state.voice_enabled:
    st.markdown("""
    <div class="voice-control-panel">
        <div class="card-title">🎙️ VOICE COMMAND CENTER</div>
        <div class="card-subtitle">Speak naturally to interact with your AI assistant</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show speaking indicator
    if st.session_state.is_speaking:
        st.markdown("""
        <div class="speaking-indicator">
            <div style="font-size: 1.2rem; font-weight: 600; color: #22c55e;">
                🔊 AI IS SPEAKING...
            </div>
            <div style="margin-top: 0.5rem; color: #86efac;">
                Click "STOP VOICE" button to interrupt
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Stop button in main area
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("⏹️ STOP VOICE NOW", type="primary", use_container_width=True):
                stop_voice()
                st.rerun()
    
    # Audio recorder (only show when not speaking)
    if not st.session_state.is_speaking:
        audio_bytes = audio_recorder(
            text="🎙️ SPEAK NOW",
            recording_color="#ff6b35",
            neutral_color="#ffffff",
            icon_name="microphone",
            icon_size="2x",
            pause_threshold=2.0,
            sample_rate=16000
        )
        
        if audio_bytes:
            st.markdown("""
            <div class="voice-status">🎵 Processing voice input...</div>
            <div class="audio-visualizer">
                <div class="audio-bar" style="animation-delay: 0s;"></div>
                <div class="audio-bar" style="animation-delay: 0.1s;"></div>
                <div class="audio-bar" style="animation-delay: 0.2s;"></div>
                <div class="audio-bar" style="animation-delay: 0.3s;"></div>
                <div class="audio-bar" style="animation-delay: 0.4s;"></div>
                <div class="audio-bar" style="animation-delay: 0.5s;"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Convert speech to text
            with st.spinner("🧠 Processing your voice command..."):
                transcribed_text = speech_to_text(audio_bytes)
            
            if transcribed_text and "Could not understand" not in transcribed_text and "Error" not in transcribed_text:
                st.success(f"🎙️ **Voice Command Received:** {transcribed_text}")
                
                # Add to messages and process
                if st.session_state.kb_built:
                    st.session_state.messages.append({"role": "user", "content": transcribed_text})
                    
                    # Get AI response
                    qa = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=st.session_state.vectordb.as_retriever(),
                        memory=st.session_state.chat_memory,
                        return_source_documents=True,
                        output_key="answer"
                    )
                    
                    with st.spinner("🧠 AI Processing Voice Query..."):
                        result = qa.invoke({"question": transcribed_text})
                        answer = result["answer"]
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # Speak the response
                    if st.session_state.auto_speak:
                        threading.Thread(
                            target=text_to_speech,
                            args=(answer, voice_language)
                        ).start()
                    
                    st.rerun()
                else:
                    st.warning("⚠️ Please initialize the AI system first!")
            else:
                st.error(f"❌ Voice Recognition Error: {transcribed_text}")
    else:
        st.info("🔊 Voice output is active. Please wait or use the stop button to interrupt.")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Premium Image Upload Section
    st.markdown("""
    <div class="feature-card">
        <div class="card-title">📸 VISUAL ANALYSIS MODULE</div>
        <div class="card-subtitle">Upload vehicle images for advanced AI-powered diagnostic analysis and component identification</div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_image = st.file_uploader(
        "DEPLOY IMAGE FOR ANALYSIS", 
        type=["png", "jpg", "jpeg"],
        help="Upload high-quality images of vehicle components, dashboard displays, or diagnostic areas"
    )
    
    if uploaded_image:
        img = Image.open(uploaded_image).convert("RGB")
        
        # Display image with premium styling
        st.markdown('<div class="uploaded-image">', unsafe_allow_html=True)
        st.image(img, caption="🔍 ANALYZING AUTOMOTIVE COMPONENT", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        with st.spinner("🧠 Processing Visual Data..."):
            inputs = processor(images=img, return_tensors="pt")
            output = blip_model.generate(**inputs)
            caption = processor.decode(output[0], skip_special_tokens=True)
        
        # Display premium analysis results
        st.markdown(f"""
        <div class="caption-box">
            <strong>🤖 AI VISUAL ANALYSIS:</strong> {caption}
        </div>
        """, unsafe_allow_html=True)
        
        # Add to premium chat history
        st.session_state.messages.append({
            "role": "user", 
            "content": f"📸 Visual Analysis Complete: {caption}",
            "type": "image"
        })
        
        # Voice announcement of image analysis
        if st.session_state.voice_enabled and st.session_state.auto_speak:
            threading.Thread(
                target=text_to_speech,
                args=(f"Image analysis complete. I can see {caption}", voice_language)
            ).start()

with col2:
    # Premium Quick Actions
    st.markdown("""
    <div class="feature-card">
        <div class="card-title">⚡ RAPID ACCESS COMMANDS</div>
        <div class="card-subtitle">Instant access to critical automotive information</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔧 MAINTENANCE PROTOCOL", use_container_width=True):
        if st.session_state.kb_built:
            query = "What are the essential maintenance procedures for premium vehicles?"
            st.session_state.messages.append({"role": "user", "content": query})
            
            # Voice feedback
            if st.session_state.voice_enabled and st.session_state.auto_speak:
                threading.Thread(
                    target=text_to_speech,
                    args=("Loading maintenance protocols", voice_language)
                ).start()
            st.rerun()
    
    if st.button("🛡️ SAFETY PROTOCOLS", use_container_width=True):
        if st.session_state.kb_built:
            query = "Provide comprehensive safety protocols and emergency procedures for vehicle operation."
            st.session_state.messages.append({"role": "user", "content": query})
            
            # Voice feedback
            if st.session_state.voice_enabled and st.session_state.auto_speak:
                threading.Thread(
                    target=text_to_speech,
                    args=("Loading safety protocols", voice_language)
                ).start()
            st.rerun()
    
    if st.button("⚡ PERFORMANCE OPTIMIZATION", use_container_width=True):
        if st.session_state.kb_built:
            query = "How can I optimize my vehicle's performance and efficiency?"
            st.session_state.messages.append({"role": "user", "content": query})
            
            # Voice feedback
            if st.session_state.voice_enabled and st.session_state.auto_speak:
                threading.Thread(
                    target=text_to_speech,
                    args=("Loading performance optimization tips", voice_language)
                ).start()
            st.rerun()
    
    # GPS System Button with Voice Support
    st.markdown("""
    <a href="https://www.google.com/maps" target="_blank">
        <button style="
            padding: 20px 40px;
            font-size: 20px;
            font-weight: bold;
            border: none;
            border-radius: 12px;
            background-color: #FFA500;
            color: white;
            cursor: pointer;
            width: 100%;
            transition: all 0.3s ease;
        " onmouseover="this.style.backgroundColor='#FF8C00'" onmouseout="this.style.backgroundColor='#FFA500'">
           🗺️ GPS NAVIGATION SYSTEM
        </button>
    </a>
    """, unsafe_allow_html=True)
    
    # Emergency Voice Commands
    st.markdown("### 🚨 EMERGENCY VOICE COMMANDS")
    st.markdown("""
    <div style="background: rgba(255,0,0,0.1); padding: 1rem; border-radius: 10px; border: 1px solid rgba(255,0,0,0.3);">
        <small>
        <strong>Say:</strong><br/>
        • "Emergency help"<br/>
        • "Breakdown assistance"<br/>
        • "Safety protocols"<br/>
        • "Call roadside assistance"
        </small>
    </div>
    """, unsafe_allow_html=True)

# Premium Luxury Divider
st.markdown('<div class="luxury-divider"></div>', unsafe_allow_html=True)

# Premium Chat Interface
st.markdown("""
<div class="feature-card">
    <div class="card-title">💬 PREMIUM AI CONSULTATION INTERFACE</div>
    <div class="card-subtitle">Engage with advanced automotive AI for expert-level guidance and technical support</div>
</div>
""", unsafe_allow_html=True)

# Chat container
if st.session_state.kb_built:
    # Create premium QA chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.vectordb.as_retriever(),
        memory=st.session_state.chat_memory,
        return_source_documents=True,
        output_key="answer"
    )
    
    # Display premium chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                # Add voice icon for voice messages
                voice_icon = "🎙️" if message.get("type") != "image" else "📸"
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>{voice_icon} YOU:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 PREMIUM AI:</strong> {message["content"]}
                    <div style="margin-top: 10px; display: flex; justify-content: space-between; align-items: center;">
                        <small style="opacity: 0.7;">
                            🔊 {"Voice response enabled" if st.session_state.auto_speak else "Text only"}
                        </small>
                        {"<small style='color: #22c55e;'>🔊 Currently Speaking</small>" if st.session_state.is_speaking else ""}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Premium chat input with voice support indicator
    chat_placeholder = "🗣️ SPEAK OR TYPE YOUR AUTOMOTIVE INQUIRY..." if st.session_state.voice_enabled else "⌨️ TYPE YOUR AUTOMOTIVE INQUIRY..."
    user_query = st.chat_input(chat_placeholder)
    
    if user_query:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Get AI response with premium processing
        with st.spinner("🧠 PROCESSING THROUGH AI NEURAL NETWORK..."):
            result = qa.invoke({"question": user_query})
            answer = result["answer"]
        
        # Add premium assistant response
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        # Auto-speak response if enabled
        if st.session_state.auto_speak and st.session_state.voice_enabled:
            threading.Thread(
                target=text_to_speech,
                args=(answer, voice_language)
            ).start()
        
        st.rerun()

else:
    # Premium setup instructions
    st.markdown("""
    <div class="info-message">
        🚀 <strong>PREMIUM SYSTEM INITIALIZATION REQUIRED:</strong> 
        Activate the AI Knowledge Core by clicking "INITIALIZE SYSTEM" in the Control Hub. 
        The system will create a demo knowledge base if no manuals are found.
    </div>
    """, unsafe_allow_html=True)
    
    # Premium features showcase with voice capabilities
    st.markdown("### ✨ PREMIUM VOICE-ENABLED FEATURES SUITE")
    
    features_col1, features_col2, features_col3 = st.columns(3)
    
    with features_col1:
        st.markdown("""
        <div class="feature-card">
            <div class="card-title">🎙️ VOICE INTELLIGENCE</div>
            <div class="card-subtitle">Advanced speech recognition with multi-language support for hands-free automotive assistance</div>
        </div>
        """, unsafe_allow_html=True)
    
    with features_col2:
        st.markdown("""
        <div class="feature-card">
            <div class="card-title">🔍 VISUAL INTELLIGENCE</div>
            <div class="card-subtitle">State-of-the-art computer vision for automotive component analysis with voice-announced results</div>
        </div>
        """, unsafe_allow_html=True)
    
    with features_col3:
        st.markdown("""
        <div class="feature-card">
            <div class="card-title">🗺️ INTEGRATED NAVIGATION</div>
            <div class="card-subtitle">Voice-activated GPS integration with conversational interface and expert automotive knowledge</div>
        </div>
        """, unsafe_allow_html=True)

# Voice Commands Help Section
if st.session_state.voice_enabled:
    with st.expander("🎙️ VOICE COMMANDS REFERENCE"):
        st.markdown("""
        ### 🗣️ Available Voice Commands:
        
        **General Queries:**
        - "How do I change my car's oil?"
        - "What are the maintenance schedules?"
        - "Tell me about brake safety"
        - "How to improve fuel efficiency?"
        
        **Emergency Commands:**
        - "Emergency help"
        - "Breakdown assistance"
        - "Safety protocols"
        - "Roadside assistance"
        
        **System Commands:**
        - "Clear chat history"
        - "Show maintenance protocols"
        - "Open GPS navigation"
        - "Analyze uploaded image"
        
        **Multi-Language Support:**
        - English, Hindi, Spanish, French, German, Italian, Portuguese, Russian, Japanese, Korean
        
        **Tips for Better Recognition:**
        - Speak clearly and at moderate pace
        - Minimize background noise
        - Hold the microphone button while speaking
        - Wait for the processing indicator before speaking again
        """)

# Premium Footer with Voice Credits
st.markdown('<div class="luxury-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #cccccc; padding: 3rem;">
    <div style="font-family: 'Orbitron', monospace; font-size: 1.5rem; color: #ff6b35; margin-bottom: 1rem;">
        🎙️ PREMIUM AI VOICE ASSISTANT
    </div>
    <p style="font-size: 1rem; margin-bottom: 0.5rem;">
        <strong>Advanced Automotive Intelligence Platform with Voice Control</strong>
    </p>
    <p style="font-size: 0.9rem; opacity: 0.7;">
        Powered by Neural Networks • Computer Vision • Speech Recognition • Natural Language Processing
    </p>
    <div style="margin: 1rem 0; font-size: 0.8rem; opacity: 0.6;">
        🔊 Voice Features: Speech-to-Text • Text-to-Speech • Multi-Language • Hands-Free Operation
    </div>
    <div class="luxury-accent" style="margin: 2rem auto; width: 150px;"></div>
</div>
""", unsafe_allow_html=True)