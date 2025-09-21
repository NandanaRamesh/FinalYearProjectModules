import streamlit as st

def get_recommendation_for_user(profile: dict, fin_info: dict):
    # Extract values safely
    age = profile.get("age", None)
    occupation = profile.get("occupation", "Unknown")
    risk = profile.get("risk_profile", "Moderate")
    goals = profile.get("goals", [])
    
    income = fin_info.get("income", 0)
    expenses = fin_info.get("expenses", 0)
    savings = income - expenses if income and expenses else 0
    
    # Show basic info
    st.write("### 📊 Profile Summary")
    st.json({
        "Age": age,
        "Occupation": occupation,
        "Risk Profile": risk,
        "Goals": goals,
        "Income": income,
        "Expenses": expenses,
        "Estimated Savings": savings
    })

    st.write("### 💡 Personalized Recommendation")
    
    # Recommendation logic
    if savings <= 0:
        st.warning("Your expenses exceed or equal your income. Focus on budgeting before investing.")
        return
    
    if risk == "Conservative":
        st.success("Recommendation: Fixed Deposits, Recurring Deposits, Government Bonds, and Insurance Plans.")
    elif risk == "Moderate":
        st.success("Recommendation: Balanced Mutual Funds, Index Funds, NPS, and Health Insurance.")
    elif risk == "Aggressive":
        st.success("Recommendation: Equity Mutual Funds, Direct Equity, ETFs, and Long-Term SIPs.")
    else:
        st.info("Recommendation: Default mix of debt and equity instruments.")
    
    # Goal-based tips
    if "Retirement" in goals:
        st.info("Tip: Start an NPS or retirement-focused mutual fund for long-term wealth creation.")
    if "Home Purchase" in goals:
        st.info("Tip: Consider a mix of Fixed Deposits + Debt Funds to build short-to-mid term savings.")
    if "Child Education" in goals:
        st.info("Tip: Invest in Child Education Plans, PPF, and long-term SIPs in equity funds.")
    
    # Emergency Fund advice
    if savings > 0:
        emergency_fund = expenses * 6
        st.info(f"Maintain at least ₹{emergency_fund} as an emergency fund in liquid instruments.")
