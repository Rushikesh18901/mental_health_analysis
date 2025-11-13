import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from textblob import TextBlob
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, accuracy_score

st.set_page_config(page_title="Mental Health Dashboard", page_icon="🧠", layout="wide")

# Custom CSS for cleaner, modern styling
st.markdown("""
    <style>
        body, .main, .css-uf99v8 {
            background: linear-gradient(135deg, #e0edfa 0%, #fdfbfb 100%) no-repeat !important;
            font-family: 'Segoe UI', 'Open Sans', 'Arial', sans-serif;
        }
        .header-container {
            text-align: center;
            margin-bottom: 24px;
        }
        .header-container img {
            width: 64px;
            margin: 0 auto 8px auto;
            display: block;
        }
        .header-title {
            font-size: 2.7em;
            font-weight: 800;
            color: #264653;
        }
        .header-sub {
            color: #5397f6;
            font-size: 1.4em;
            margin-bottom: 10px;
            font-weight:600;
        }
        .stTabs [data-baseweb="tab-list"] {
            font-size:1.1em;
        }
        .card {
            background-color: #fff;
            border-radius: 15px;
            box-shadow: 0 2px 16px #dbe5ea;
            padding: 36px 24px;
            max-width: 650px;
            margin: 24px auto 36px auto;
        }
        .footer {
            text-align: center;
            color: #888;
            font-size: 0.98em;
            margin-top: 28px;
            opacity: 0.9;
        }
        label, .stTextInput>div>input, .stTextArea>div>textarea {
            font-size: 1.13em;
        }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
    <div class="header-container">
        <div class="header-title" style="font-family: 'Arial Black', Arial, sans-serif; font-weight: 900; font-size: 3em; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); color: #1e40af;">Mental Health Support Dashboard</div>
        <div class="header-sub" style="font-family: 'Georgia', serif; font-style: italic; color: #34495e;">Tech Industry Employee Insights</div>
    </div>
    """, unsafe_allow_html=True)

# Load data
df = pd.read_csv('Data/processed/clustered_data.csv')
df_processed = pd.read_csv('Data/processed/processed_data.csv')

# Initialize session state for user reports
if 'user_reports' not in st.session_state:
    st.session_state.user_reports = []

# Functions for sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    return polarity

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
            "👏 Great job managing your stress! Stay motivated.",
            "🌟 Celebrate small wins to stay motivated."
        ])

# --- TABS ---
tabs = st.tabs([
    "🏠 Home",
    "📈 Visual Trends",
    "📥 Download Report",
    "📊 Model Performance"
])

# --- HOME TAB ---
with tabs[0]:
    st.markdown(
        "<div class='card' style='text-align: center;'>"
        "<b>Welcome!</b><br>Share how you're feeling today, and we'll provide some personalized relaxation tips.<br><br>",
        unsafe_allow_html=True
    )
    with st.form(key="feeling_form"):
        feeling = st.text_area(
            "Share how you're feeling for quick tips:",
            placeholder="e.g., I'm feeling stressed due to heavy workload.",
            key="feeling_input"
        )
        submitted = st.form_submit_button("Analyze My Feelings")
        if submitted:
            if feeling.strip():
                polarity = analyze_sentiment(feeling)

                if polarity < -0.2:
                    stress_level = "High Stress 😟"
                    color = "red"
                elif polarity <= 0.2:
                    stress_level = "Moderate Stress 😐"
                    color = "orange"
                else:
                    stress_level = "Low Stress 🙂"
                    color = "green"

                st.subheader("📊 Sentiment Analysis Result")
                st.markdown(f"**Stress Level:** <span style='color:{color}'>{stress_level}</span>", unsafe_allow_html=True)
                st.progress((polarity + 1) / 2)

                suggestion = suggest_relaxation(polarity)
                st.subheader("💡 Relaxation Tip for You:")
                st.success(suggestion)

                # Store user report
                report = {
                    'timestamp': pd.Timestamp.now(),
                    'user_input': feeling,
                    'polarity': polarity,
                    'stress_level': stress_level,
                    'suggestion': suggestion
                }
                st.session_state.user_reports.append(report)
            else:
                st.warning("⚠️ Please type something about your feelings before analyzing.")
    st.markdown("</div>", unsafe_allow_html=True)

# --- VISUAL TRENDS TAB ---
with tabs[1]:
    # Mood Distribution (using work_interfere as proxy for mood)
    st.subheader("Mood Distribution")
    mood_counts = df['work_interfere_Often'].value_counts()
    fig_pie = px.pie(values=mood_counts.values, names=mood_counts.index, title="Work Interference Distribution (Mood Proxy)")
    st.plotly_chart(fig_pie)

    fig_bar = px.bar(x=mood_counts.index, y=mood_counts.values, labels={'x':'Work Interference', 'y':'Count'}, title="Work Interference Count")
    st.plotly_chart(fig_bar)

    # Moods Over Time
    st.subheader("Moods Over Time")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True)
    mood_time = df.groupby([df['Timestamp'].dt.date, 'work_interfere_Often']).size().unstack().fillna(0)
    fig_line = px.line(mood_time, title="Work Interference Trends Over Time")
    st.plotly_chart(fig_line)

    # Cluster Visualization
    st.subheader("Employee Cluster Visualization")
    features = df[['Age', 'work_interfere_Often', 'work_interfere_Rarely']]  # Sample features
    pca = PCA(n_components=2)
    cluster_labels = df['cluster']

    pca_result = pca.fit_transform(features)
    clustered_df = pd.DataFrame({'PCA1': pca_result[:,0], 'PCA2': pca_result[:,1], 'Cluster': cluster_labels})

    fig_clusters = px.scatter(clustered_df, x='PCA1', y='PCA2', color='Cluster', title="Employee Groups by Mental Health Patterns")
    st.plotly_chart(fig_clusters)

# --- DOWNLOAD REPORT TAB ---
with tabs[2]:
    if st.session_state.user_reports:
        user_df = pd.DataFrame(st.session_state.user_reports)
        csv = user_df.to_csv(index=False).encode('utf-8')
        st.success("Your personalized report is ready!")
        st.download_button(
            label=" 📥 Download Your Report (CSV) ",
            data=csv,
            file_name="mental_health_report.csv",
            mime="text/csv"
        )
        st.dataframe(user_df)
    else:
        st.markdown("""
            <div style="background-color:#f1f6fa; border-radius:10px; padding:20px; margin-bottom:20px; box-shadow: 2px 2px 8px #dbe5ea;">
                <p style="color:#214365; font-size:1.1em;">
                    <span style="font-size:1.5em;">📥</span>
                    No user reports available yet.<br> Please submit your feelings in the Home tab first.
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div style="margin-top:10px;">
            <b>How to get your report?</b><br>
            1. Go to the Home tab.<br>
            2. Share your current feelings.<br>
            3. Once submitted, your report will appear here!
        </div>
    """, unsafe_allow_html=True)

# --- MODEL PERFORMANCE TAB ---
with tabs[3]:
    # Load and preprocess data
    df_model = pd.read_csv('Data/raw/survey.csv')
    df_model = df_model.dropna(subset=['treatment', 'Age', 'family_history', 'work_interfere'])

    df_model['treatment_bin'] = df_model['treatment'].replace({'Yes': 1, 'No': 0})
    df_model['family_history_bin'] = df_model['family_history'].replace({'Yes': 1, 'No': 0})
    df_model['work_interfere_bin'] = df_model['work_interfere'].replace({
        'Often': 3, 'Sometimes': 2, 'Rarely': 1, 'Never': 0
    })

    X = df_model[['Age', 'family_history_bin', 'work_interfere_bin']]
    y = df_model['treatment_bin']

    # Train/test split and model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics calculation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Display in Streamlit (inside your Model Performance tab)
    st.markdown("""
    <div style="background:rgba(255,255,255,0.95);padding:32px;border-radius:18px;box-shadow:0 2px 16px #cfd8dc37;text-align:center;width:95%;max-width:600px;margin:auto;">
        <h2 style="color:#23395d;margin-bottom:18px;">Model Performance Metrics</h2>
        <div style="display:flex;justify-content:space-around;flex-wrap:wrap;margin-bottom:25px;">
            <div style="background:#e0eafc;border-radius:8px;padding:18px 32px;margin:7px;">
                <span style="font-weight:600;color:#264653;font-size:1.12em;">Accuracy</span><br>
                <span style="font-size:1.5em;color:#4B8BBE;font-weight:700;">{:.2f}</span>
            </div>
            <div style="background:#e0eafc;border-radius:8px;padding:18px 32px;margin:7px;">
                <span style="font-weight:600;color:#264653;font-size:1.12em;">Precision</span><br>
                <span style="font-size:1.5em;color:#4B8BBE;font-weight:700;">{:.2f}</span>
            </div>
            <div style="background:#e0eafc;border-radius:8px;padding:18px 32px;margin:7px;">
                <span style="font-weight:600;color:#264653;font-size:1.12em;">F1 Score</span><br>
                <span style="font-size:1.5em;color:#4B8BBE;font-weight:700;">{:.2f}</span>
            </div>
        </div>
    </div>
    """.format(accuracy, precision, f1), unsafe_allow_html=True)

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', fontsize=15, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    st.pyplot(fig)

# Professional Footer
st.markdown("""
    <div class='footer'>
        Developed by Rushikesh Bhavar | BSc Data Science Project
    </div>
    """, unsafe_allow_html=True)