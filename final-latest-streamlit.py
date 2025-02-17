import streamlit as st
import json
import csv
import re
from io import StringIO

# Load patient data from JSON file
def load_patient_data():
    with open("ner_test_flat.json", "r") as f:
        patient_raw_data = json.load(f)

    # Extract the header row (assuming it's the first key)
    header_key = list(patient_raw_data.keys())[0]  
    headers = header_key.split(",")  # Extract headers: ["note_ID", "text", "ner_text", "ner_type"]

    patients = {}

    # Parse the data inside the key
    for _, entry in patient_raw_data[header_key].items():
        csv_reader = csv.reader(StringIO(entry))  # Read CSV-formatted string
        row = next(csv_reader)  # Extract first row
        
        # Ensure the row has enough columns
        if len(row) < 4:
            continue  # Skip malformed data
        
        note_id, text, ner_text, ner_type = row[:4]  # Extract relevant columns
        
        # Initialize patient entry
        if note_id not in patients:
            patients[note_id] = {
                "medical_record": text,
                "medications": [],
                "symptoms": []
            }
        
        # Categorize the extracted NER text
        if "medication" in ner_type.lower():
            patients[note_id]["medications"].append(ner_text)
        elif "symptom" in ner_type.lower():
            patients[note_id]["symptoms"].append(ner_text)

    # Remove duplicate medications/symptoms
    for patient in patients.values():
        patient["medications"] = list(set(patient["medications"]))
        patient["symptoms"] = list(set(patient["symptoms"]))

    return patients

# Load literature data
with open("rag-data-url.json", "r") as f:
    LITERATURE = json.load(f)

# Initialize Streamlit session state
if "show_ner" not in st.session_state:
    st.session_state.show_ner = False
if "show_details" not in st.session_state:
    st.session_state.show_details = False

# Function to highlight medications and symptoms
def highlight_entities(text, medications, symptoms):
    for med in medications:
        text = re.sub(rf"\b{med}\b", f'<span style="background-color: yellow">{med}</span>', text, flags=re.IGNORECASE)
    for sym in symptoms:
        text = re.sub(rf"\b{sym}\b", f'<span style="background-color: lightgreen">{sym}</span>', text, flags=re.IGNORECASE)
    return text

# Streamlit UI
st.title("📚 Medication Adverse Effect Checker")

# Load patient data
PATIENTS = load_patient_data()

if not PATIENTS:
    st.warning("No patient data available.")
else:
    # Section 1: Select Patient
    st.header("1️⃣ Select Patient")
    patient = st.selectbox("Choose a patient", list(PATIENTS.keys()))

    # Section 2: Show Medical Record
    st.header("2️⃣ Patient Medical Record")
    medical_record = PATIENTS[patient]["medical_record"]
    st.text_area("Medical Record", medical_record, height=150, disabled=True)

    # Section 3: Perform NER
    st.header("3️⃣ Named Entity Recognition (NER)")
    if st.button("🔍 Perform NER"):
        st.session_state.show_ner = True

    if st.session_state.show_ner:
        highlighted_text = highlight_entities(medical_record, PATIENTS[patient]["medications"], PATIENTS[patient]["symptoms"])
        st.markdown(f"> {highlighted_text}", unsafe_allow_html=True)

    # Section 4: Select Medication & Symptom
    st.header("4️⃣ Select Medication & Symptom")
    medication = st.selectbox("Select Medication", PATIENTS[patient]["medications"])
    symptom = st.selectbox("Select Symptom", PATIENTS[patient]["symptoms"])

    # Section 5: Search Literature
    st.header("5️⃣ Search in Literature")
    if st.button("🔎 Search Literature"):
        with st.spinner("Searching..."):
            literature_data = LITERATURE.get(medication.lower(), {}).get(symptom.lower(), {})

            if literature_data:
                pubmed_ids = list(literature_data.keys())
                st.success(f"✅ Found {len(pubmed_ids)} relevant literature(s):")
                for pubmed_id in pubmed_ids:
                    st.write(f"{pubmed_id}: {literature_data[pubmed_id]}")

                # Store search results in session state
                st.session_state.literature_data = literature_data
                st.session_state.show_details = False  # Reset details visibility
            else:
                st.error("❌ No relevant literature found.")

    # Section 6: Show Details Button
    if "literature_data" in st.session_state and st.session_state.literature_data:
        if st.button("📄 Show Details"):
            st.session_state.show_details = True

        if st.session_state.show_details:
            st.header("6️⃣ Literature Details")
            for pubmed_id, text in st.session_state.literature_data.items():
                # Highlight medication and symptom in yellow
                highlighted_text = text.replace(
                    medication, f'<span style="background-color: yellow">{medication}</span>'
                ).replace(
                    symptom, f'<span style="background-color: lightgreen">{symptom}</span>'
                )
                st.write(f"📄 {pubmed_id}")
                st.markdown(f"> {highlighted_text}", unsafe_allow_html=True)