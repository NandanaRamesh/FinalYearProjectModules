import streamlit as st
import os
import pytesseract
import pdfplumber
import fitz
import re
import json
from PIL import Image
from datetime import datetime
from openai import OpenAI as OpenAIChatClient

client = OpenAIChatClient(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

def extract_raw_text(path):
    """Extract text from PDF or image files."""
    text = ""
    if path.lower().endswith(".pdf"):
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        if not text.strip():
            doc = fitz.open(path)
            for page_num in range(len(doc)):
                pix = doc[page_num].get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text += pytesseract.image_to_string(img) + "\n"
        return text.strip()
    else:
        img = Image.open(path)
        return pytesseract.image_to_string(img)

def extract_identity_info_with_llm(raw_text):
    prompt = f"""
    You are an OCR post-processor. Given Aadhaar/PAN/VoterID text, 
    extract JSON with:
    - id_type
    - name
    - gender
    - dob
    - state
    - age_years
    Respond with JSON only.
    Text:
    {raw_text}
    """
    resp = client.chat.completions.create(
        model="qwen3:1.7b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    content = resp.choices[0].message.content.strip()
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return {"error": "No JSON found", "raw": content}
    data = json.loads(match.group(0))

    # Compute age
    if "dob" in data and data["dob"]:
        try:
            dob = datetime.strptime(data["dob"], "%d/%m/%Y")
            today = datetime.today()
            data["age_years"] = today.year - dob.year - (
                (today.month, today.day) < (dob.month, dob.day)
            )
        except:
            pass
    return data

def extract_financial_info_with_llm(raw_text):
    prompt = f"""
    You are a financial document parser.
    Extract JSON with:
    - document_type
    - name
    - employer_or_bank
    - profession
    - monthly_income
    - annual_income
    - deductions
    - net_salary
    Text:
    {raw_text}
    """
    resp = client.chat.completions.create(
        model="qwen3:1.7b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    content = resp.choices[0].message.content.strip()
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return {"error": "No JSON found", "raw": content}

    return json.loads(match.group(0))

def process_documents_page():
    st.header("Upload Documents")
    id_file = st.file_uploader("Upload Aadhaar/PAN/Voter ID", type=["pdf", "png", "jpg", "jpeg"])
    fin_file = st.file_uploader("Upload Bank Statement / Payslip", type=["pdf", "png", "jpg", "jpeg"])

    if st.button("Process Documents"):
        if not id_file or not fin_file:
            st.error("Please upload both ID and financial documents.")
            return

        # Save temporary files
        id_path = f"temp_id.{id_file.name.split('.')[-1]}"
        fin_path = f"temp_fin.{fin_file.name.split('.')[-1]}"
        with open(id_path, "wb") as f:
            f.write(id_file.read())
        with open(fin_path, "wb") as f:
            f.write(fin_file.read())

        raw_id = extract_raw_text(id_path)
        identity_info = extract_identity_info_with_llm(raw_id)

        raw_fin = extract_raw_text(fin_path)
        financial_info = extract_financial_info_with_llm(raw_fin)

        st.session_state.user_data["profile"] = identity_info
        st.session_state.user_data["financial_info"] = financial_info

        st.success("Documents processed and stored in session.")
        st.json({"Profile": identity_info, "Financial Info": financial_info})
