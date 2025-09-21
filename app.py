import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ SET INTERACTIVE BACKGROUND -------------------
def set_background():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://www.transparenttextures.com/patterns/pw-maze-white.png");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background()

# ------------------ LOAD DATA -------------------
@st.cache_data
def load_data():
    df = pd.read_csv('medicines.csv')
    df['features'] = df['symptom'] + ' ' + df['disease']
    return df

df = load_data()
symptoms_list = sorted(df['symptom'].unique())

# ------------------ TRAIN MODEL -------------------
@st.cache_resource
def train_model(df):
    X = df['features']
    y = df[['disease', 'medicine']].apply(lambda x: x['disease'] + '||' + x['medicine'], axis=1)
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(X, y)
    return model

model = train_model(df)

# ------------------ PROFILE FORM -------------------
with st.sidebar:
    st.header("👤 User Profile")
    age = st.number_input("Age", min_value=0)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    allergies = st.text_input("Allergies (comma separated)")
    medical_history = st.text_area("Medical History")
    st.markdown("---")
    if st.checkbox("Show raw data"):
        st.dataframe(df)

# ------------------ AI CHATBOT FOR SYMPTOM CHECK -------------------
st.title("💊 AI-Powered Medicine Recommendation System")

st.subheader("🧠 AI Symptom Checker")
user_query = st.text_input("Describe your symptoms (e.g., headache, fever):")
suggested_symptoms = difflib.get_close_matches(user_query, symptoms_list, n=5, cutoff=0.4)

if suggested_symptoms:
    st.markdown("Did you mean:")
    for s in suggested_symptoms:
        if st.button(s):
            user_query = s

# ------------------ MULTI SELECT -------------------
selected_symptoms = st.multiselect("Or select from list:", symptoms_list,
                                   default=[user_query] if user_query in symptoms_list else [])

# ------------------ MEDICINE RECOMMENDATION -------------------
def recommend(selected_symptoms, top_n=3):
    if not selected_symptoms:
        return []
    input_text = ' '.join(selected_symptoms)
    probs = model.predict_proba([input_text])[0]
    classes = model.classes_

    top_indices = probs.argsort()[-top_n:][::-1]
    results = []
    for idx in top_indices:
        disease, medicine = classes[idx].split('||')
        prob = probs[idx]
        results.append((disease, medicine, prob))
    return results

if st.button("🔎 Get Recommendations"):
    results = recommend(selected_symptoms)
    if results:
        st.subheader("✅ Recommendations")
        meds = []
        for i, (disease, medicine, prob) in enumerate(results, 1):
            st.markdown(f"**{i}. Disease:** {disease}  \n**Medicine:** {medicine}  \n**Confidence:** {prob:.2f}")
            meds.append(medicine)
            st.markdown("---")

        # ------------------ DRUG INTERACTION CHECKER -------------------
        st.subheader("⚠️ Drug Interaction Checker")
        drug_interactions = {
            ("Paracetamol", "Ibuprofen"): "Risk of liver damage",
            ("Aspirin", "Ibuprofen"): "Increased bleeding risk"
        }
        warnings = []
        for i in range(len(meds)):
            for j in range(i+1, len(meds)):
                pair = tuple(sorted([meds[i], meds[j]]))
                if pair in drug_interactions:
                    warnings.append(f"{pair[0]} + {pair[1]}: {drug_interactions[pair]}")
        if warnings:
            for w in warnings:
                st.error(w)
        else:
            st.success("No harmful interactions found.")

        # ------------------ LOCAL PHARMACY AVAILABILITY -------------------
        st.subheader("🏪 Nearby Pharmacy Availability (Mock)")
        pharmacy_stock = {
            "Paracetamol": ["CityMed", "PharmaOne"],
            "Ibuprofen": ["Wellness Pharmacy"],
        }
        for med in meds:
            stock = pharmacy_stock.get(med, [])
            st.markdown(f"**{med}:** {' | '.join(stock) if stock else 'Not available nearby'}")

    else:
        st.warning("Please enter or select symptoms to get recommendations.")

# ------------------ REMINDER SYSTEM -------------------
st.subheader("⏰ Smart Reminder System (Demo)")
if st.button("Set Daily Reminder"):
    st.success("You will receive a reminder at 8 AM and 8 PM!")

# ------------------ MEDICINE IMAGE RECOGNITION -------------------
st.subheader("📸 Upload Pill or Medicine Image")
img = st.file_uploader("Upload an image of a pill or box", type=["png", "jpg", "jpeg"])
if img:
    st.image(img, width=150)
    st.info("Recognizing...")
    st.success("This looks like: **Paracetamol** (Mock AI Detection)")

# ------------------ AI DOCTOR CHATBOT -------------------
st.subheader("🤖 AI Doctor Assistant")
chat_q = st.text_input("Ask a health question (e.g., What to take for a cold?)")
if chat_q:
    st.markdown("**AI Response:** Based on your symptoms, you may take an over-the-counter cold medicine such as Paracetamol or consult a doctor if symptoms persist.")

# ------------------ FEEDBACK -------------------
st.subheader("⭐ Share Your Experience")
rating = st.slider("How helpful was the recommendation?", 1, 5)
feedback = st.text_area("Write your feedback here")
if st.button("Submit Feedback"):
    st.success("Thank you for your feedback!")

