import streamlit as st
import joblib

# =========================
# LOAD MODEL
# =========================
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💜",
    layout="centered"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

/* Google Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]{
    font-family: 'Poppins', sans-serif;
}

/* Background */
.stApp{
    background:
    radial-gradient(circle at top left, #6d28d9 0%, transparent 25%),
    radial-gradient(circle at bottom right, #2563eb 0%, transparent 25%),
    #050816;
    color: white;
}

/* Remove Header */
header{
    visibility:hidden;
}

/* Main Container */
.main-container{
    max-width: 900px;
    margin: auto;
    margin-top: 50px;
    padding: 45px;
    border-radius: 30px;

    background: rgba(255,255,255,0.05);

    backdrop-filter: blur(20px);

    border: 1px solid rgba(255,255,255,0.1);

    box-shadow:
    0 0 40px rgba(111,66,193,0.25),
    0 0 80px rgba(37,99,235,0.15);
}

/* Hero Section */
.hero{
    text-align:center;
    margin-bottom:40px;
}

/* Title */
.title{
    font-size:68px;
    font-weight:700;
    line-height:1.1;

    background: linear-gradient(to right,#ffffff,#8b5cf6,#3b82f6);

    -webkit-background-clip:text;

    -webkit-text-fill-color:transparent;

    margin-bottom:15px;
}

/* Subtitle */
.subtitle{
    color:#cbd5e1;
    font-size:22px;
    font-weight:300;
}

/* Input Card */
.input-card{

    background: rgba(255,255,255,0.04);

    border:1px solid rgba(255,255,255,0.08);

    padding:30px;

    border-radius:25px;

    margin-top:40px;
}

/* Label */
.stTextArea label{
    color:white !important;
    font-size:24px !important;
    font-weight:600 !important;
}

/* Text Area */
.stTextArea textarea{

    background: rgba(10,15,35,0.9) !important;

    color:white !important;

    border:2px solid rgba(139,92,246,0.35) !important;

    border-radius:20px !important;

    padding:20px !important;

    font-size:18px !important;

    min-height:220px !important;

    transition:0.3s;
}

/* Focus */
.stTextArea textarea:focus{

    border:2px solid #8b5cf6 !important;

    box-shadow: 0 0 25px rgba(139,92,246,0.5) !important;
}

/* Button */
.stButton > button{

    width:100%;

    height:65px;

    border:none;

    border-radius:18px;

    background: linear-gradient(90deg,#9333ea,#2563eb);

    color:white;

    font-size:24px;

    font-weight:600;

    margin-top:25px;

    transition:0.3s ease;

    box-shadow:0 0 25px rgba(147,51,234,0.35);
}

/* Hover */
.stButton > button:hover{

    transform:translateY(-3px) scale(1.01);

    box-shadow:
    0 0 30px rgba(147,51,234,0.55),
    0 0 60px rgba(37,99,235,0.35);
}

/* Result Box */
.result-box{

    margin-top:35px;

    padding:25px;

    border-radius:22px;

    text-align:center;

    font-size:34px;

    font-weight:700;

    color:white;

    animation:fadeIn 0.5s ease;
}

/* Positive */
.positive{
    background:linear-gradient(135deg,#16a34a,#22c55e);
}

/* Negative */
.negative{
    background:linear-gradient(135deg,#dc2626,#ef4444);
}

/* Neutral */
.neutral{
    background:linear-gradient(135deg,#475569,#64748b);
}

/* Animation */
@keyframes fadeIn{

    from{
        opacity:0;
        transform:translateY(10px);
    }

    to{
        opacity:1;
        transform:translateY(0px);
    }
}

</style>
""", unsafe_allow_html=True)

# =========================
# MAIN UI
# =========================

st.markdown('<div class="main-container">', unsafe_allow_html=True)

# HERO
st.markdown("""

<div class="hero">

<div class="title">
💬 Sentiment Analyzer
</div>

<div class="subtitle">
Analyze your text sentiment using Machine Learning
</div>

</div>

""", unsafe_allow_html=True)

# INPUT CARD
st.markdown('<div class="input-card">', unsafe_allow_html=True)

user_text = st.text_area(
    "Enter your text",
    placeholder="Type something here to analyze sentiment..."
)

# BUTTON
if st.button("🚀 Analyze Sentiment"):

    if user_text.strip() != "":

        transformed_text = vectorizer.transform([user_text])

        prediction = str(
            model.predict(transformed_text)[0]
        ).lower()

        # POSITIVE
        if prediction == "positive":

            st.markdown("""
            <div class="result-box positive">
            😊 Positive Sentiment
            </div>
            """, unsafe_allow_html=True)

        # NEGATIVE
        elif prediction == "negative":

            st.markdown("""
            <div class="result-box negative">
            😠 Negative Sentiment
            </div>
            """, unsafe_allow_html=True)

        # OTHER
        else:

            st.markdown(f"""
            <div class="result-box neutral">
            📝 {prediction.title()}
            </div>
            """, unsafe_allow_html=True)

    else:
        st.warning("⚠ Please enter text first!")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)