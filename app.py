import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="DetectAI | Neon Engine", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- RESPONSIVE CSS STYLES ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    
    /* Global Background */
    .stApp { 
        background: radial-gradient(circle at top left, #1e2a4a, #0b0e14);
        font-family: 'JetBrains Mono', monospace !important; 
        color: #ffffff !important;
    }

    /* === DESKTOP STYLES (Base) === */
    .glow-text { 
        font-size: 3rem !important;
        font-weight: 800;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px !important;
        padding-bottom: 0px !important;
        filter: drop-shadow(0 0 10px rgba(79, 172, 254, 0.5));
    }
    .sub-text { 
        color: #00d2ff !important; 
        margin-top: -15px !important; 
        margin-bottom: 5px !important;
        letter-spacing: 2px;
        font-size: 0.9rem;
        font-weight: 600;
    }

    /* Seamless Inputs (Desktop) */
    div[data-testid="stRadio"] { 
        background: rgba(255, 255, 255, 0.05); 
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px 15px 0px 0px; 
        padding: 15px 20px 5px 20px !important; 
        margin-top: -10px !important; 
    }
    div[data-testid="stFileUploader"], div[data-testid="stCameraInput"] { 
        background: rgba(255, 255, 255, 0.02); 
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-top: none; 
        border-radius: 0px 0px 15px 15px; 
        padding: 0px 20px 20px 20px !important; 
        margin-top: -1rem !important; 
    }

    /* Gradient Buttons */
    .stButton { margin-top: -5px !important; }
    .stButton>button { 
        width: 100%; 
        background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%) !important; 
        border: none !important; 
        color: white !important; 
        border-radius: 12px !important; 
        font-weight: bold !important; 
        height: 3.5rem;
        font-size: 1.1rem;
        transition: 0.3s !important;
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 0 20px rgba(79, 172, 254, 0.8); }

    /* Footer Metrics */
    .tech-footer { display: flex; justify-content: space-between; gap: 20px; margin-top: 40px; }
    .tech-box { flex: 1; padding: 20px; border-radius: 15px; text-align: center; background: rgba(255, 255, 255, 0.03); }
    .neon-cnn { border-bottom: 3px solid #00ff88; box-shadow: 0 4px 15px -5px #00ff88; }
    .neon-tfl { border-bottom: 3px solid #bc13fe; box-shadow: 0 4px 15px -5px #bc13fe; }
    .neon-lat { border-bottom: 3px solid #00d2ff; box-shadow: 0 4px 15px -5px #00d2ff; }
    .tech-label { font-size: 10px; font-weight: bold; margin-bottom: 5px; text-transform: uppercase; }
    .tech-value { font-size: 18px; font-weight: 700; color: white; }

    /* Hide Clutter */
    footer {visibility: hidden;}
    [data-testid="stHeader"] {background: rgba(0,0,0,0);}


    /* === MOBILE RESPONSIVE FIXES (Triggered below 768px width) === */
    @media (max-width: 768px) {
        /* Resize Headers for Mobile */
        .glow-text { font-size: 2.2rem !important; }
        .sub-text { font-size: 0.7rem !important; letter-spacing: 1px; margin-top: -5px !important; }
        
        /* Un-stick the Radio and Uploader inputs so they don't break */
        div[data-testid="stRadio"] { 
            border-radius: 15px !important; 
            padding: 15px !important; 
            margin-top: 5px !important; 
        }
        div[data-testid="stFileUploader"], div[data-testid="stCameraInput"] { 
            border-radius: 15px !important; 
            border-top: 1px solid rgba(255, 255, 255, 0.1) !important; /* Restore border */
            padding: 15px !important; 
            margin-top: 10px !important; /* Push down so it doesn't overlap */
        }
        
        /* Stack the Footer Boxes Vertically */
        .tech-footer { 
            flex-direction: column !important; 
            gap: 15px !important; 
        }
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Model Loader
@st.cache_resource
def load_neon_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_neon_model('model_quantized.tflite')
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception:
    st.error("SYSTEM ERROR: Neural weights 'model_quantized.tflite' not detected.")
    st.stop()

# 3. Header Layer
st.markdown('<h1 class="glow-text">DetectAI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">NEURAL CLASSIFICATION INFRASTRUCTURE</p>', unsafe_allow_html=True)

# 4. Input Layer
choice = st.radio("SOURCE", ["📂 Upload from Gallery", "📷 Use Live Camera"], horizontal=True, label_visibility="collapsed")

if "Gallery" in choice:
    source = st.file_uploader("UPLOAD", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
else:
    source = st.camera_input("CAMERA", label_visibility="collapsed")

# 5. Analysis Layer
if source is not None:
    raw_img = Image.open(source).convert("RGB")
    raw_img = ImageOps.exif_transpose(raw_img)
    
    col_l, col_r = st.columns(2, gap="large")
    
    with col_l:
        st.markdown('<p style="color: #00d2ff; font-size: 12px; font-weight: bold; margin-bottom: 5px;">SOURCE TENSOR FEED</p>', unsafe_allow_html=True)
        st.image(raw_img, use_container_width=True)

    with col_r:
        st.markdown('<p style="color: #bc13fe; font-size: 12px; font-weight: bold; margin-bottom: 5px;">INFERENCE CONTROL</p>', unsafe_allow_html=True)
        
        processed = raw_img.resize((32, 32))
        tensor = tf.keras.preprocessing.image.img_to_array(processed)
        tensor = np.expand_dims(tensor, axis=0).astype(np.float32)

        if st.button("INITIATE SCAN"):
            with st.spinner('Scanning manifolds...'):
                interpreter.set_tensor(input_details[0]['index'], tensor)
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
                
                is_ai = prediction < 0.5
                res_col, shadow = ("#ff0055", "rgba(255, 0, 85, 0.5)") if is_ai else ("#00ff88", "rgba(0, 255, 136, 0.5)")
                res_lbl = "AI GENERATED" if is_ai else "HUMAN AUTHENTIC"
                conf = (1 - prediction if is_ai else prediction) * 100

                st.markdown(f"""
                    <div style="padding: 20px; text-align: center; margin-top: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); background: rgba(255,255,255,0.05); backdrop-filter: blur(10px);">
                        <h2 style="color: {res_col}; text-shadow: 0 0 10px {shadow}; margin: 0;">{res_lbl}</h2>
                        <h1 style="color: white; margin: 10px 0; font-size: 3.5rem;">{conf:.1f}%</h1>
                        <p style="color: #8b949e; font-size: 12px; margin: 0; text-transform: uppercase;">Confidence Matrix</p>
                    </div>
                """, unsafe_allow_html=True)
                st.progress(float(conf/100))

# 6. Technical Footer Layer
st.markdown(f"""
    <div class="tech-footer">
        <div class="tech-box neon-cnn">
            <div class="tech-label" style="color: #00ff88;">ML CORE</div>
            <div class="tech-value">CIFAKE</div>
        </div>
        <div class="tech-box neon-tfl">
            <div class="tech-label" style="color: #bc13fe;">DL ENGINE</div>
            <div class="tech-value">TFLite</div>
        </div>
        <div class="tech-box neon-lat">
            <div class="tech-label" style="color: #00d2ff;">SPEED</div>
            <div class="tech-value">~0.42s</div>
        </div>
    </div>
    <div style="text-align: center; margin-top: 30px; color: #777; font-size: 11px;">
    BCA Final Year Project 2026
    </div>
""", unsafe_allow_html=True)
