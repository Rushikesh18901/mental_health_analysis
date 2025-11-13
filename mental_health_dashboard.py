import streamlit as st
from textblob import TextBlob
import random

# -------------------------------
# 1. Function to Analyze Sentiment
# -------------------------------
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # Range: -1 (Negative) to +1 (Positive)
    return polarity

# -------------------------------
# 2. Function to Suggest Relaxation Tips
# -------------------------------
def suggest_relaxation(polarity):
    if polarity < -0.2:
        return random.choice([
            "🌱 Try 5 minutes of deep breathing exercises.",
            "🎵 Listen to calming music or meditation sounds.",
            "🧘 Practice a short guided meditation.",
            "📖 Take a 10-minute break and read something positive.",
            "💬 Talk to a trusted friend or HR representative."
        ])
    elif polarity <= 0.2:
        return random.choice([
            "☕ Take a short break with your favorite beverage.",
            "🚶 Go for a 5-minute walk to refresh your mind.",
            "💡 Write down your thoughts to clear your mind.",
            "🌞 Step outside and get some fresh air."
        ])
    else:
        return random.choice([
            "🎉 Keep up the positive vibes! Spread happiness to others.",
            "👏 Great job managing your stress!",
            "🌟 Celebrate small wins to stay motivated."
        ])

# -------------------------------
# 3. Streamlit Dashboard Layout
# -------------------------------
st.set_page_config(page_title="Mental Health Support Dashboard", page_icon="🧠", layout="centered")

st.title("🧠 Mental Health Support Dashboard")
st.write("Welcome! Share how you're feeling today, and we'll provide some personalized relaxation tips.")

# -------------------------------
# 4. User Input
# -------------------------------
user_input = st.text_area("How are you feeling today?", placeholder="e.g., I'm feeling stressed due to heavy workload.")

if st.button("Analyze My Feelings"):
    if user_input.strip():
        polarity = analyze_sentiment(user_input)

        # Determine stress level
        if polarity < -0.2:
            stress_level = "High Stress 😟"
            color = "red"
        elif polarity <= 0.2:
            stress_level = "Moderate Stress 😐"
            color = "orange"
        else:
            stress_level = "Low Stress 🙂"
            color = "green"

        # Show Results
        st.subheader("📊 Sentiment Analysis Result")
        st.markdown(f"**Stress Level:** <span style='color:{color}'>{stress_level}</span>", unsafe_allow_html=True)
        st.progress((polarity + 1) / 2)  # Normalize polarity (-1 to 1 → 0 to 1)

        # Suggest Relaxation Tip
        suggestion = suggest_relaxation(polarity)
        st.subheader("💡 Relaxation Tip for You:")
        st.success(suggestion)
    else:
        st.warning("⚠️ Please type something about your feelings before analyzing.")

# -------------------------------
# 5. Footer
# -------------------------------
st.markdown("---")
st.caption("Developed by [Rushikesh Bhavar] | BSc Data Science Project")