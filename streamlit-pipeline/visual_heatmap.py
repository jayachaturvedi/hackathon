import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import random

import openai

# Load API key from a file
with open("/Users/fuqi/TOKEN/gpt_token", "r") as f:
    API_KEY = f.read().strip()

def query_chatgpt(prompt):
    client = openai.OpenAI(api_key=API_KEY)  # Initialize the OpenAI client
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300
    )
    return response.choices[0].message.content

# Generate synthetic data
def generate_realistic_data():
    patients = ["Patient_A", "Patient_B"]

    # Generating EHR records
    ehr_data = {
        "Patient_A": "EHR: Hypertension, Type 2 Diabetes, High Cholesterol. Medications: Lisinopril, Metformin, Atorvastatin. Symptoms: Fatigue, Dizziness, Headache, Muscle Pain. Treatment history includes regular follow-ups and medication adjustments to manage blood pressure and glucose levels. Patient reports occasional side effects including mild dizziness and fatigue.",
        "Patient_B": "EHR: Migraine, Acid Reflux, Arthritis. Medications: Ibuprofen, Omeprazole, Aspirin. Symptoms: Stomach Pain, Nausea, Headache, Fatigue. History of episodic migraines treated with analgesics and lifestyle modifications. Reflux symptoms managed with proton pump inhibitors. Reports of occasional nausea related to medication use."
    }

    # Extract medications and symptoms per patient
    patient_medications = {
        "Patient_A": ["Lisinopril", "Metformin", "Atorvastatin"],
        "Patient_B": ["Ibuprofen", "Omeprazole", "Aspirin"]
    }

    patient_symptoms = {
        "Patient_A": ["Fatigue", "Dizziness", "Headache", "Muscle Pain"],
        "Patient_B": ["Stomach Pain", "Nausea", "Headache", "Fatigue"]
    }

    # Create evidence heatmap based on patient-specific data
    def generate_evidence_matrix(medications, symptoms,seed):
        matrix_size = (len(medications), len(symptoms))
        np.random.seed(seed)
        evidence_matrix = np.random.choice([-7, -5, -3, -2, 0, 0, 0, 0, 2, 3, 5, 7], size=matrix_size)

        return pd.DataFrame(evidence_matrix, index=medications, columns=symptoms)

    evidence_matrix = {
        "Patient_A": generate_evidence_matrix(patient_medications["Patient_A"], patient_symptoms["Patient_A"],32),
        "Patient_B": generate_evidence_matrix(patient_medications["Patient_B"], patient_symptoms["Patient_B"],34)
    }

    # Generate literature with 20 papers related to these diseases
    literature = {}
    pmc_ids = [f"PMC{random.randint(100000, 999999)}" for _ in range(20)]

    for patient in patients:
        for med in patient_medications[patient]:
            for sym in patient_symptoms[patient]:
                literature[(med, sym)] = [f"{pmc_id}: There is a link between {med} and {sym} because of the actual concentration of ACE inhibitors and cognition and if there was a detectable difference between the two types of ACE inhibitors. " for pmc_id in random.sample(pmc_ids, 2)]

    return patients, ehr_data, patient_medications, patient_symptoms, evidence_matrix, literature


patients, ehr_data, patient_medications, patient_symptoms, evidence_matrix, literature = generate_realistic_data()

# Streamlit UI
st.set_page_config(layout="wide")  # Use full width
st.title("Real-time Medication-Symptom Evidence Checker")

# Layout for Patient Selection and EHR Display
col1, col2 = st.columns([2, 3])  # 40% for patient selection, 60% for heatmap

with col1:
    selected_patient = st.selectbox("Select a patient", patients)
    st.text_area("Patient EHR Record", ehr_data[selected_patient], height=120)

    st.text_area("Identified Symptoms", patient_symptoms[selected_patient], height=80)
    st.text_area("Identified Medications", patient_medications[selected_patient], height=80)


# Heatmap
with col2:
    st.header("Could the symptom an adverse effect caused by the medication?")
    st.text("Based on evidence from the literature.")
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.xticks(fontsize=14, wrap=True)
    plt.yticks(fontsize=14, wrap=True)

    cmap = sns.diverging_palette(10, 240, as_cmap=True)
    evidence_display = evidence_matrix[selected_patient].copy()
    mask = evidence_display == 0
    sns.heatmap(evidence_display, annot=False, cmap=cmap, center=0, cbar=False, ax=ax, fmt="g", mask=mask,
                linewidths=0.5, linecolor='grey')

    ax.set_xticklabels(patient_symptoms[selected_patient], rotation=45, ha='right')
    ax.set_yticklabels(patient_medications[selected_patient], rotation=0, va='center')
    ax.grid(True, which='minor', color='black', linewidth=0.5)
    ax.tick_params(which='minor', size=0)
    plt.xlabel("Symptoms/Diseases")
    plt.ylabel("Medications")

    red_patch = mpatches.Patch(color=cmap(-0.9), label='Adverse Effects', alpha=0.7)
    blue_patch = mpatches.Patch(color=cmap(0.9), label='Co-occurence only')
    # add another legant to explain the shades

    #plt.legend(handles=[red_patch, blue_patch], loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=2, fontsize=14)
    plt.legend(handles=[red_patch, blue_patch], loc='lower center',bbox_to_anchor=(0.5, -0.5), ncol=2, fontsize=14)

    st.pyplot(fig)

# Streamlit UI # Use full width
st.header("Related information from PubMed")

# Clickable heatmap functionality
col1, col2 = st.columns(2)

with col1:
    selected_med = st.selectbox("Select a medication", patient_medications[selected_patient], key="medication")

    st.markdown("""
    <div style="background-color: #ADD8E6; padding: 15px; border-radius: 5px;">
        <h4>Ask LLM questions based on the literature</h4>
        <div>
    """, unsafe_allow_html=True)



    example_questions = [
        f"Based on the literature, why does {selected_med} cause the adverse effect?",
        f"What does the literature say about medications that should be avoided to use together with {selected_med}?"
    ]
    selected_question = st.selectbox("Example Questions", example_questions)
    st.markdown("""
        <style>
            div[data-baseweb="input"] input {
                color: darkblue !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # Text input field
    user_question = st.text_input("Ask your own question")
    if st.button("Submit Question"):
        query = user_question if user_question else selected_question
        response = query_chatgpt(query)
        st.markdown(
            f"<div style='background-color: #ADD8E6; padding: 10px; border-radius: 5px;'>LLM response: {response}</div>",
            unsafe_allow_html=True)

    st.markdown("""</div>""", unsafe_allow_html=True)

with col2:
    selected_sym = st.selectbox("Select a symptom", patient_symptoms[selected_patient], key="symptom")

    if (selected_med, selected_sym) in literature:
        st.write("Related publications")
        for paper in literature[(selected_med, selected_sym)]:
            pmc_id = paper.split(":")[0]
            st.write(f"- {pmc_id}")

        st.write("Sections in related publications")
        for paper in literature[(selected_med, selected_sym)]:
            highlighted_excerpt = paper.replace(selected_med,
                                                f"<span style='background-color: yellow'>{selected_med}</span>")
            highlighted_excerpt = highlighted_excerpt.replace(selected_sym,
                                                              f"<span style='background-color: yellow'>{selected_sym}</span>")
            st.markdown(f"**Excerpt from {highlighted_excerpt}:**", unsafe_allow_html=True)
            #st.text_area("",
            #             f"{paper} discusses the relationship between {selected_med} and {selected_sym} in detail...",
            #             height=100)
    else:
        st.write("No supporting literature found for this pair.")
