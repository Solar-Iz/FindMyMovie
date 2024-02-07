import streamlit as st

st.header('Инструменты для создания проекта: ')

st.subheader('Для русской версии: ')

list_text = """
<div style='color: violet; border: 2px solid purple; padding: 10px;'>
    <ul>
        <li>Используемые языковые модели: rubert-base-cased-sentence, rubert-tiny2</li>
        <li>Библиотека Sentence Transformers</li>
        <li>Faiss (для уменьшения времени генерации подборки фильмов)</li>
        <li>Сайт-жертва для парсинга - <a href="https://www.kinoafisha.info/" style='color: purple;'>Киноафиша</a></li>
    </ul>
</div>
"""

# Отображение HTML-разметки в Streamlit
st.markdown(list_text, unsafe_allow_html=True)

st.subheader('Для английской версии: ')

list_text2 = """
<div style='color: pink; border: 2px solid violet; padding: 10px;'>
    <ul>
        <li>Используемые языковые модели: bert-base-uncased</li>
        <li>Очистка текста: приведение к нижнему регистру, очистка от знаков препинания, стоп-слова</li>
        <li>Библиотека Sentence Transformers</li>
        <li>Faiss (для уменьшения времени генерации подборки фильмов)</li>
        <li>Сайт-жертва для парсинга - <a href="https://www.themoviedb.org/" style='color: violet;'>TMDB</a></li>
    </ul>
</div>
"""

st.markdown(list_text2, unsafe_allow_html=True)


st.markdown("<p style='color: pink; font-size: 28px; text-align: center;'>"
            "А теперь, когда фильм успешно выбран, вооружайтесь теплым пледом и глинтвейном и бегите смотреть 🎄🍿"
            "</p>", unsafe_allow_html=True)

st.image("apps/2.jpg",  use_column_width=True)