import streamlit as st
from joblib import load

st.set_page_config(page_title="Prediksi SMS Penipuan", page_icon="📩")

# Memuat TfidfVectorizer dan model yang disimpan dari file .joblib
vectorizer = load('tfidf_vectorizer.joblib')
model = load('knn_neighbors_1.joblib')

# Mendefinisikan kamus label
label_dict = {0: 'Bukan Penipuan', 1: 'Penipuan'}

# Mendefinisikan warna label untuk menampilkan hasil prediksi
label_colors = {
    0: 'green',  # Bukan Penipuan
    1: 'red',    # Penipuan
}

# Halaman web Streamlit
st.title('Prediksi SMS Penipuan')
st.write('Tool ini memprediksi apakah pesan SMS yang diberikan adalah Penipuan atau Bukan Penipuan.')

# Kotak teks untuk input pengguna
input_sms = st.text_area("Masukkan teks SMS yang ingin dianalisis:", "")

# Proses input pengguna dan buat prediksi
def classify_message(model, vectorizer, message):
    processed_message = vectorizer.transform([message])
    prediction = model.predict(processed_message)
    proba = model.predict_proba(processed_message)
    return prediction, proba

if st.button('Prediksi'):
    if input_sms:
        prediction, proba = classify_message(model, vectorizer, input_sms)
        result_label = label_dict[prediction[0]]
        result_color = label_colors[prediction[0]]
        
        st.markdown(f"""
            <div style='color: white; font-size: 30px;'><strong>SMS diklasifikasikan sebagai:</strong></div>
            <div style='color: {result_color}; font-size: 36px;'><strong>{result_label}</strong></div>
            <br>
            """, unsafe_allow_html=True)

        # # Menampilkan probabilitas prediksi sebagai persentase
        # st.subheader('Probabilitas Prediksi:')
        # for index, label in enumerate(label_dict.values()):
        #     st.write(f"{label}: {proba[0][index]*100:.1f}%")
    else:
        st.error("Silakan masukkan teks SMS untuk diklasifikasikan.")
