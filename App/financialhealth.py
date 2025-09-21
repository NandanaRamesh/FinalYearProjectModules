import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Configure your Gemini API key
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

def financial_health_check_page():
    st.header("🩺 AI-Powered Financial Health Check")

    questions = [
        ("Do you have anyone who depends on you financially?", ["Yes", "No"]),
        ("Do you have life insurance cover for at least 15-20 times your annual income?", ["Yes", "No"]),
        ("Do you have health insurance for yourself and dependents?", ["Yes", "No"]),
        ("How well can you handle sudden, unexpected expenses?", ["Very well", "Somewhat", "Poorly"]),
        ("Do you pay your credit card bills in full every month?", ["Yes", "No"]),
        ("Do you have any personal or unsecured loans?", ["Yes", "No"]),
        ("Is your total EMI more than 40% of your monthly income?", ["Yes", "No"]),
        ("Do you create a budget for your income?", ["Yes", "No"]),
        ("Have you calculated the corpus needed for a comfortable retirement?", ["Yes", "No"]),
        ("Are you investing enough for retirement?", ["Yes", "No"]),
        ("Have you shared investment details with your dependents?", ["Yes", "No"]),
        ("Have you added nomination details to all investments?", ["Yes", "No"]),
        ("Have you made a will?", ["Yes", "No"]),
    ]

    # Store responses in session state
    if "health_answers" not in st.session_state.user_data:
        st.session_state.user_data["health_answers"] = {}

    with st.form("health_form"):
        for q, opts in questions:
            st.session_state.user_data["health_answers"][q] = st.radio(q, opts, key=q)

        submitted = st.form_submit_button("🔍 Get Assessment")

    if submitted:
        responses = st.session_state.user_data["health_answers"]

        # Build AI prompt
        prompt = "The user answered the following financial health questions:\n"
        for q, a in responses.items():
            prompt += f"- {q}: {a}\n"
        prompt += "\nPlease provide a detailed, personalized financial health assessment and actionable recommendations."

        # Call Gemini AI
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            ai_reply = response.text

            st.subheader("✅ AI-Powered Financial Health Recommendation")
            st.write(ai_reply)

        except Exception as e:
            st.error(f"Error calling AI: {e}")
