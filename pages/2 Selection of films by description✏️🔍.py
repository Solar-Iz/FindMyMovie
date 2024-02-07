import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import faiss
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

# Загрузка стоп-слов для английского языка
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    return df

@st.cache_data
def embedding_and_index():
    embeddings_array = np.load('data/embeddings_eng.npy')
    index = faiss.read_index('data/desc_faiss_index_eng.index')
    return embeddings_array, index

@st.cache_data
def load_model():
    model = BertModel.from_pretrained('bert-base-uncased')
    return model

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    
    return text

st.header("Selection of films by description✏️🔍")

# Загрузка данных
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
df = load_data('data/eng_data.csv')
embeddings_array, index = embedding_and_index()
model = load_model()

# Пользовательский ввод
user_input = st.text_input("Enter a movie description:", value="", help="The more detailed your description is, the more accurately we can choose a film for you 🤗'")

if st.button("Search🔍🎦"):
    if user_input:
        def encode_description(description, tokenizer, model):
            tokens = tokenizer(description, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**tokens)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.cpu().numpy().astype('float32')
        
        # Применяем очистку текста к пользовательскому вводу
        cleaned_input = clean_text(user_input)
        
        # Векторизация очищенного запроса
        input_embedding = encode_description(cleaned_input, tokenizer, model)

        # Поиск с использованием Faiss
        _, sorted_indices = index.search(input_embedding.reshape(1, -1), 5)

        # Используйте индексы для извлечения строк из DataFrame
        recs = df.iloc[sorted_indices[0]].reset_index(drop=True)
        recs.index = recs.index + 1

        # Вывод рекомендованных фильмов с изображениями
        st.subheader("Recommended movies 🎉:")
        for i in range(5):
            st.markdown(f"<span style='font-size:{20}px; color:purple'>{recs['movie_title'].iloc[i]}</span>", unsafe_allow_html=True)
        # Создаем две колонки: одну для текста, другую для изображения
            col1, col2 = st.columns([2, 1])
    
        # В  колонке отображаем название фильма, описание, роли и ссылку
            col1.info(recs['description'].iloc[i])
            col1.markdown(f"**You can watch the film [here]({recs['page_url'].iloc[i]})**")
    
        # В  колонке отображаем изображение
            col2.image(recs['image_url'].iloc[i], caption=recs['movie_title'].iloc[i], width=200)
        with st.sidebar:
            st.info("""
                 #### Were we able to help you with the choice? 
                """)
            feedback = st.text_input('Share with us')
        
            feedback_button = st.button("Send feedback", key="feedback_button")
        
        if feedback_button and feedback:
            feedback_container.success("Thank you, every day we try to be better for you 💟")
        elif feedback_button:
            feedback_container.warning("Please enter a review before submitting")