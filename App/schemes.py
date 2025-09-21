import streamlit as st
import pandas as pd
import google.generativeai as genai
import os

# Configure Gemini API key (ensure this is in your secrets.toml)
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Load the government schemes from your CSV file
# NOTE: Make sure 'govt_schemes.csv' is in the same directory
try:
    schemes_df = pd.read_csv(os.path.join("assets", "govt_schemes.csv"))
    schemes_text = schemes_df.to_string(index=False)
except FileNotFoundError:
    st.error("Error: 'govt_schemes.csv' not found. Please place it in the assets directory.")

def generate_schemes_prompt(user_data, schemes_text):
    """
    Generates a detailed prompt for the LLM to suggest government schemes.
    """
    profile = user_data["profile"]
    fin_info = user_data["financial_info"]
    
    prompt = f"""
    Based on the following user information, please recommend a few relevant government schemes from the provided list.
    Explain why each scheme is a good fit for the user and provide a brief summary of the scheme's benefits.
    
    User Financial Information:
    - Monthly Income: {fin_info.get("monthly_income", "Not provided")}
    - Annual Income: {fin_info.get("annual_income", "Not provided")}
    - Name: {profile.get("full_name", "Not provided")}
    
    Available Schemes (in CSV format):
    {schemes_text}
    """
    return prompt

def schemes_page():
    st.header("Government Schemes Recommendation")
    
    user_data = st.session_state.user_data
    
    if not user_data["financial_info"]:
        st.warning("Please go to the 'Document Processing' page first to extract and store your financial information.")
        return
        
    if st.button("Get Government Scheme Recommendations"):
        with st.spinner("Fetching personalized schemes..."):
            try:
                prompt = generate_schemes_prompt(user_data, schemes_text)
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                
                st.subheader("Your Recommended Schemes")
                st.markdown(response.text)
                
            except Exception as e:
                st.error(f"Failed to get a response from Gemini: {e}")