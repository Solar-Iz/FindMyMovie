import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np

@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    return df

@st.cache_data
def embedding_and_index():
    embeddings_array = np.load('data/embeddings_final.npy')
    index = faiss.read_index('data/desc_faiss_index_final.index')
    return embeddings_array, index

@st.cache_data
def load_model():
    model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
    return model


st.header("Подбор фильмов по описанию ✏️🔍")

# Загрузка данных
tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
df = load_data('data/final_data.csv')
embeddings_array, index = embedding_and_index()
model = load_model()

# Пользовательский ввод
user_input = st.text_input("Введите описание фильма:", value="", help="Чем подробнее будет ваше описание, тем точнее мы сможем подобрать для вас фильм 🤗'")
genre_list = ['анимация', 'аниме', 'балет', 'биография', 'боевик', 'вестерн', 'военный', 'детектив', 'детский', 'документальный', 'драма', 'исторический', 'катастрофа', 'комедия', 'концерт', 'короткометражный', 'криминал', 'мелодрама', 'мистика', 'музыка', 'мюзикл', 'нуар', 'приключения', 'сборник', 'семейный', 'сказка', 'спорт', 'триллер', 'ужасы', 'фантастика', 'фэнтези', 'эротика']

user_select_genre = st.multiselect('Выберите жанр', genre_list)

if st.button("Искать🔍🎦"):
    if user_input:
        def encode_description(description, tokenizer, model):
            tokens = tokenizer(description, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**tokens)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.cpu().numpy().astype('float32')

        # Векторизация введенного запроса с использованием переданных tokenizer и model
        input_embedding = encode_description(user_input, tokenizer, model)

        # Поиск с использованием Faiss
        _, sorted_indices = index.search(input_embedding.reshape(1, -1), 5)

        # Используйте индексы для извлечения строк из DataFrame
        recs = df.iloc[sorted_indices[0]].reset_index(drop=True)
        recs.index = recs.index + 1

        if user_select_genre: 
            genres_selected = pd.Series(user_select_genre)
            genre_mask = df['genre'].str.contains('')
            for i in range(len(genres_selected)):
                genre_mask_i = df['genre'].str.contains(genres_selected.iloc[i])
                genre_mask = genre_mask & genre_mask_i
            recs = recs[genre_mask]

        if not recs.empty:
            # Вывод рекомендованных фильмов с изображениями
            st.subheader("Рекомендованные фильмы 🎉:")
            for i in range(min(5, len(recs))):
                st.markdown(f"<span style='font-size:{20}px; color:purple'>{recs['movie_title'].iloc[i]}</span>", unsafe_allow_html=True)
                # Создаем две колонки: одну для текста, другую для изображения
                col1, col2 = st.columns([2, 1])

                # В  колонке отображаем название фильма, описание, роли и ссылку
                col1.info(recs['description'].iloc[i])
                col1.markdown(f"**В ролях:** {recs['actors'].iloc[i]}")
                col1.markdown(f"**Фильм можно посмотреть [здесь]({recs['page_url'].iloc[i]})**")

                # В  колонке отображаем изображение
                col2.image(recs['image_url'].iloc[i], caption=recs['movie_title'].iloc[i], width=200)
            
            with st.sidebar:
                st.info("""
                    #### Мы смогли помочь вам с выбором? 
                """)
                feedback = st.text_input('Поделитесь с нами вашим мнением')
            
                feedback_button = st.button("Отправить отзыв", key="feedback_button")
            
            if feedback_button and feedback:
                st.success("Спасибо, каждый день мы стараемся быть лучше для вас 💟")
            elif feedback_button:
                st.warning("Пожалуйста, введите отзыв перед отправкой.")
        else:
            st.subheader("Подходящих фильмов не найдено, ослабьте фильтры 😔:")
