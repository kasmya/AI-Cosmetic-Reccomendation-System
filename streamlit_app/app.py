import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import time

# Session state for theme toggle
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# Toggle theme function
def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

# Page config
st.set_page_config(page_title="SkynCare.AI", page_icon="🎀", layout="wide")

# Theme colors
bg_color = "#000000" if st.session_state.dark_mode else "#fff0f6"
text_color = "#ffffff" if st.session_state.dark_mode else "#000000"
header_color = "#e91e63"

# Custom styles
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {bg_color};
        font-family: 'Segoe UI', sans-serif;
        color: {text_color};
    }}
    .navbar {{
        position: sticky;
        top: 0;
        z-index: 999;
        background-color: #000;
        color: {header_color};
        padding: 1rem 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 2px solid #f48fb1;
        width: 100%;
    }}
    .navbar h2 {{
        margin: 0;
        font-size: 1.8rem;
        font-weight: bold;
        color: {header_color};
    }}
    .toggle-btn {{
        background-color: #f8bbd0;
        color: #000;
        font-size: 1.2rem;
        border: none;
        border-radius: 50%;
        padding: 0.4rem 0.6rem;
        cursor: pointer;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    }}
    .pink-title {{
        color: {header_color};
        font-size: 2.2rem;
        font-weight: 700;
    }}
    .pink-paragraph {{
        color: #d81b60;
        font-size: 1.05rem;
    }}
    .footer {{
        background-color: #fce4ec;
        border-top: 2px solid #f48fb1;
        padding: 1rem;
        text-align: center;
        font-size: 0.9rem;
        color: #880e4f;
        border-radius: 0.5rem;
        margin-top: 2rem;
        width: 100%;
    }}
    .testimonial-card {{
        border-radius: 12px;
        background-color: #fce4ec;
        padding: 1rem;
        font-style: italic;
        color: #880e4f;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }}
    .stButton>button {{
        background-color: #1e1e2f !important;
        color: white !important;
    }}
    .stRadio > label, .stSelectbox > label, .stSlider > label {{
        color: black !important;
        font-weight: bold;
    }}

    

    </style>
""", unsafe_allow_html=True)

# Navbar 
navbar_col1, navbar_col2 = st.columns([10, 1])
with navbar_col1:
    st.markdown(f"""
        <div class="navbar">
            <h2> 💜SkynCare.AI</h2>
        </div>
    """, unsafe_allow_html=True)

with navbar_col2:
    if st.button("🌙/☀️", key="theme_toggle_button"):
        toggle_theme()

# Navigation
selected = option_menu(
    menu_title=None,
    options=["Home", "Get Recommendation", "Skin Care 101"],
    icons=["house", "stars", "book"],
    orientation="horizontal"
)

# Home Page
if selected == "Home":
    st.markdown('<h1 class="pink-title">Welcome to SkynCare.AI</h1>', unsafe_allow_html=True)
    st.markdown("""<p class="pink-paragraph">💄 <b>Your personalized AI-powered Skin Care Recommendation System.</b><br><br></p>""", unsafe_allow_html=True)

    st.markdown("<h2 style='color: black;'>🔥 Trending Skin Care Products</h2>", unsafe_allow_html=True)
    st.markdown("""
    <style>
    .product-card {
        border: 1px solid #ff91a4;
        border-radius: 12px;
        padding: 1rem;
        background-color: #ffeef4;
        margin-bottom: 1rem;
        transition: all 0.3s ease-in-out;
        box-shadow: 1px 2px 8px rgba(0,0,0,0.1);
    }
    .product-card:hover {
        background-color: #ffe1ec;
        transform: scale(1.01);
        box-shadow: 0 0 15px rgba(255,105,180,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

    products = [
        {"name": "Glow Serum", "brand": "DewDrop", "price": "₹699", "tag": "Hydrating", "rating": 4.7, "emoji": "💧"},
        {"name": "Peptide Lip Oil", "brand": "ColorPop", "price": "₹499", "tag": "Long-lasting", "rating": 4.5, "emoji": "💋"},
        {"name": "Sunscreen SPF 50", "brand": "SunShield", "price": "₹849", "tag": "Oil-Free", "rating": 4.8, "emoji": "☀️"}
    ]
    cols = st.columns(3)
    for i, product in enumerate(products):
        with cols[i]:
            st.markdown(f"""
                <div class="product-card">
                    <h4>{product['emoji']} <b>{product['name']}</b></h4>
                    <p><b>Brand:</b> <i>{product['brand']}</i></p>
                    <p>💰 <b>{product['price']}</b></p>
                    <p>🍿️ {product['tag']}</p>
                    <p>⭐ {product['rating']} / 5</p>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<h2>🗣️ What Users Say</h2>", unsafe_allow_html=True)
    testimonials = [
        "\"SkynCare.AI completely changed my skincare routine.\" 🌸 - Priya",
        "\"Guided me so well as a beginner.\" 💄 - Rhea",
        "\"My acne cleared up in 3 weeks!\" ✨ - Anjali"
    ]
    placeholder = st.empty()
    for i in range(10):
        with placeholder.container():
            st.markdown(f"<div class='testimonial-card'>{testimonials[i % len(testimonials)]}</div>", unsafe_allow_html=True)
        time.sleep(2)

# Recommendation Page
elif selected == "Get Recommendation":
    st.title("Get Product Recommendations")
    st.info("This is a UI demo only. Backend logic not connected.")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_img:
            st.image(uploaded_img, caption="Preview")
    with col2:
        class FaceDetector(VideoTransformerBase):
            def transform(self, frame):
                img = frame.to_ndarray(format="bgr24")
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
                return img
        webrtc_streamer(key="face", video_processor_factory=FaceDetector, media_stream_constraints={"video": True})

    if uploaded_img:
        st.success("Facial data processed! Recommending product...")
        st.write("💄 **Recommended:** Rose Mist Toner")
        st.button("Buy Now")

    st.markdown("<h2>🏋️️ Skincare Quiz</h2>", unsafe_allow_html=True)
    with st.form("quiz_form"):
        skin_type = st.radio("1. What's your skin type?", ["Oily", "Dry", "Combination", "Normal"])
        concern = st.selectbox("2. What's your top concern?", ["Acne", "Dark spots", "Dryness", "Wrinkles"])
        goal = st.select_slider("3. What's your skincare goal?", ["Clean", "Smooth", "Glowing", "Youthful"])
        submit = st.form_submit_button("Get Recommendation")
        if submit:
            st.info(f"For {skin_type} skin targeting {concern}, we recommend: GlowFix Serum for a {goal} look! 🌟")

# Skincare Tips Page
elif selected == "Skin Care 101":
    st.title("Skin Care 101")
    st.markdown("""### 💡 Tips
- Wash your face twice daily.
- Moisturize after cleansing.
- Use sunscreen daily.
- Stay hydrated.
- Avoid over-exfoliation.
""")

# Footer
st.markdown("""
<div class="footer">
👩‍💼 Created with ❤️ by <strong>Team SkynCare.AI</strong><br>
📩 Contact us at <a href="mailto:contact@SkynCare.ai">contact@SkynCare.ai</a><br>
💻 Follow us on <a href="https://github.com/kasmya/AI-Cosmetic-Reccomendation-System">GitHub</a>
</div>
""", unsafe_allow_html=True)
