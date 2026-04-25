#main
import streamlit as st
import pickle

# Load models + vectorizer
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

models = {
    "Logistic Regression": pickle.load(open("lr.pkl", "rb")),
    "Naive Bayes": pickle.load(open("nb.pkl", "rb")),
    "Support Vector Classifier": pickle.load(open("svc.pkl", "rb")),
    "Decision Tree Classification": pickle.load(open("dtc.pkl", "rb")),
    "Random Forest Classifier": pickle.load(open("rfc.pkl", "rb"))
}

st.set_page_config(page_title="Spam Detection", layout="centered")

st.markdown("""
<style>
.box {
    padding: 12px;
    border-radius: 6px;
    margin-bottom: 10px;
    font-weight: bold;
    border: 2px solid;
}
.ham {
    background-color: #d4edda;
    border-color: #28a745;
    color: #155724;
}
.spam {
    background-color: #f8d7da;
    border-color: #dc3545;
    color: #721c24;
}
</style>
""", unsafe_allow_html=True)

st.title("📩 Spam Detection")

col1, col2 = st.columns([4,1])

with col1:
    user_input = st.text_area("Input:", height=100)

with col2:
    st.write("")
    st.write("")
    process = st.button("Process")

if process:
    if user_input.strip() == "":
        st.warning("Please enter a message")
    else:
        input_vector = vectorizer.transform([user_input.lower()])

        for name, model in models.items():
            pred = model.predict(input_vector)[0]

            if pred == 1:
                result = "Ham ✅"
                css_class = "ham"
            else:
                result = "Spam ❌"
                css_class = "spam"

            st.markdown(f"""
                <div class="box {css_class}">
                    {name}: {result}
                </div>
            """, unsafe_allow_html=True)