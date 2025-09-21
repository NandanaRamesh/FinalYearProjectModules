import streamlit as st
import os
import json
from datetime import datetime

# Import functions from your other files
from process_docs import process_documents_page
from financialhealth import financial_health_check_page
from recommender import get_recommendation_for_user
from loan_eligibility import loan_eligibility_page
from chatbot import chatbot_page
from schemes import schemes_page  # New import

# Initialize session state for user data
if "user_data" not in st.session_state:
    st.session_state.user_data = {
        "profile": {},
        "financial_info": {},
        "health_answers": {},
        "loan_eligibility": None
    }
    
st.set_page_config(layout="wide")
st.title("Data-Driven AI Financial Advisor")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Document Processing", "Financial Health Check", "Government Schemes", "Chatbot", "Loan Eligibility"])

# Display the user's current data
st.sidebar.subheader("Your Data (In Session)")
st.sidebar.json(st.session_state.user_data)

# Call the appropriate function based on the selected page
if page == "Dashboard":
    st.header("Personalized Dashboard")
    
    profile = st.session_state.user_data["profile"]
    fin_info = st.session_state.user_data["financial_info"]
    
    if profile or fin_info:
        st.subheader("Your Stored Information")
        st.json({"Profile": profile, "Financials": fin_info})
        
        if "risk_profile" in profile:
            st.subheader("Financial Recommendation")
            get_recommendation_for_user(profile, fin_info)
    else:
        st.info("No data found. Please visit other pages to populate your information.")
    
elif page == "Document Processing":
    process_documents_page()
    
elif page == "Financial Health Check":
    financial_health_check_page()
    
elif page == "Government Schemes":
    schemes_page()  # Call the new function
    
elif page == "Chatbot":
    chatbot_page()

elif page == "Loan Eligibility":
    loan_eligibility_page()