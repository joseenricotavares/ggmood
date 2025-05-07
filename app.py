import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
import joblib
import pandas as pd
import numpy as np
import json
import nltk
from nltk.corpus import stopwords
#from nltk.corpus import brown
#from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import os
import zipfile

max_cache_time = 3600

st.set_page_config( page_title = 'GGMood',
                    page_icon = './images/logo_clean.PNG',
                    layout = 'wide',
                    initial_sidebar_state = 'expanded')

@st.cache_resource(show_spinner=False, ttl=max_cache_time)
def load_models():
    model_dir = "./models/sbert_models/gte-small"

    if not os.path.exists(model_dir):
        with zipfile.ZipFile(f"{model_dir}.zip", 'r') as zip_ref:
            zip_ref.extractall("./models/sbert_models/")

    embedding_model = SentenceTransformer(model_dir)
    sentiment_model = joblib.load("./mlruns/294574624479352871/76bceb6b85914aa59da16f77472c86aa/artifacts/model/model.pkl")
    return embedding_model, sentiment_model

@st.cache_data(show_spinner=False, ttl=max_cache_time)
def get_df_embeddings(_embedding_model, text):
    embeddings = _embedding_model.encode(text,batch_size=8)
    df_embeddings = pd.DataFrame(embeddings, columns=[f"CLS{i}" for i in range(embeddings.shape[1])])
    return df_embeddings

@st.cache_data(show_spinner=False, ttl=max_cache_time)
def predict_model(_model, df_embeddings):
    probs = _model.predict_proba(df_embeddings)
    df_out = df_embeddings.copy()
    df_out[['prediction_score_0', 'prediction_score_1']] = probs
    return df_out

@st.cache_data(show_spinner=False, ttl=max_cache_time)
def search_games(name):
    url = f"https://store.steampowered.com/api/storesearch/?term={name}&cc=us&l=en"
    res = requests.get(url)
    if res.ok:
        return res.json().get("items", [])[:4]
    return []

@st.cache_data(show_spinner=False, ttl=max_cache_time)
def get_game_details(appid):
    url = f"https://store.steampowered.com/api/appdetails?appids={appid}&l=en"
    res = requests.get(url)
    if res.ok and res.json().get(str(appid), {}).get("success"):
        return res.json()[str(appid)]["data"]
    return {}

def get_reviews(appid, max_reviews=600):
    reviews = []
    cursor = "*"
    count = 0
    seen_review_ids = set()  
    while count < max_reviews:
        url = f"https://store.steampowered.com/appreviews/{appid}?json=1&language=english&cursor={cursor}&num_per_page=100"
        res = requests.get(url)
        if not res.ok:
            break
        data = res.json()
        chunk = data.get("reviews", [])
        new_reviews = []
        for review in chunk:
            review_id = review["recommendationid"]
            if review_id not in seen_review_ids:
                seen_review_ids.add(review_id)
                new_reviews.append(review["review"])

        reviews.extend(new_reviews)

        if not data.get("cursor") or not new_reviews:
            break
        
        cursor = data["cursor"]
        count += len(new_reviews)

    return reviews[:max_reviews]

@st.cache_data(show_spinner=False, ttl=max_cache_time)
def get_stopwords():
    nltk.download('stopwords')
    #nltk.download('brown')
    stop_words = set(stopwords.words('english'))
    #adjectives = [word.lower() for word, tag in brown.tagged_words() if tag.startswith('JJ')]
    #commom_adj = [palavra for palavra, _ in Counter(adjectives).most_common(100)]
    custom_stopwords = {"game", "games", "play", "playing", "player", "steam", "review", "download","like", "love", "hate", "good", "bad", "great", "best", "worst", "fun", "awesome", "amazing", "excellent", "fantastic", "recommend", "recommendation", "would", "definitely", "highly", "enjoy", "enjoyed", "enjoying", "experience", "experiences", "time", "hours", "hours of playtime"}                            
    stop_words.update(custom_stopwords)
    #stop_words.update(commom_adj)
    return stop_words

@st.cache_resource(show_spinner=False, ttl=max_cache_time)
def generate_wordcloud(text, color, stopwords):
    wordcloud = WordCloud(stopwords=stopwords, width=400, height=200, background_color="white", colormap=color).generate(" ".join(text))
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    return fig

@st.cache_data(show_spinner=False, ttl=max_cache_time)
def extract_keywords_from_reviews(reviews_df, stopwords, n_clusters=3, top_n_words=7):

    embedding_cols = [col for col in reviews_df.columns if col.startswith("CLS")]
    embeddings_df = reviews_df[embedding_cols]

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings_df)
    reviews_df = reviews_df.copy()
    reviews_df['cluster'] = labels

    # Ordering clusters by size
    cluster_sizes = reviews_df['cluster'].value_counts().sort_values(ascending=False)
    sorted_clusters = cluster_sizes.index.tolist()

    keywords_by_cluster = []
    used_keywords = set()

    for cluster_num in sorted_clusters:
        cluster_reviews = reviews_df[reviews_df['cluster'] == cluster_num]['review'].tolist()
        vectorizer = CountVectorizer(stop_words=list(stopwords), max_features=1000)
        X = vectorizer.fit_transform(cluster_reviews)
        word_freq = np.asarray(X.sum(axis=0)).flatten()
        feature_names = vectorizer.get_feature_names_out()

        # Ordering words by frequency and excluding used keywords from next clusters
        sorted_indices = word_freq.argsort()[::-1]

        top_words = []
        for idx in sorted_indices:
            word = feature_names[idx]
            if word not in used_keywords:
                top_words.append(word)
                used_keywords.add(word)
            if len(top_words) >= top_n_words:
                break

        keywords_by_cluster.append(", ".join(top_words))

    return keywords_by_cluster

def display_topics_by_sentiment(reviews_df, sentiment_label, signal, stopwords):
    try:
        if len(reviews_df) >= 20:
            keywords_list = extract_keywords_from_reviews(reviews_df, stopwords=stopwords)
            
            st.markdown(f"<p style='text-align: center;'><strong>{signal} Top {len(keywords_list)} {sentiment_label} Topics</strong></p>", unsafe_allow_html=True)

            topics_df = pd.DataFrame({
                'Topic': [f"Topic {i+1}" for i in range(len(keywords_list))],
                'Keywords': keywords_list
            })

            st.dataframe(topics_df, hide_index=True)
        else:
            st.markdown(f"**{signal} Not enough {sentiment_label.lower()} reviews to identify topics.**")
    except Exception as e:
        st.markdown(f"**{signal} Vocabulary too small to extract {sentiment_label.lower()} topics.**")

with st.sidebar:
    col1, col2, col3 = st.columns([.1, 2, .1])
    with col2:
        st.image('./images/logo_name.PNG', width=250)
        time.sleep(0.01)
        st.markdown(
            """
            <p style='text-align: center; font-size:18px; margin-top:10px; margin-bottom:4px;'>
                "What gamers say and feel."
            </p>
            <p style='text-align: center; font-size:14px; color:gray; margin-top:0;'>
                by José Tavares
            </p>
            """,
            unsafe_allow_html=True
        )
        st.write("---")
    with open('threshold.json', 'r') as f:
        threshold_options = json.load(f)
    selected_label = st.selectbox("Sentiment Sensitivity:", list(threshold_options.keys()),index=2)
    selected_data = threshold_options[selected_label]
    threshold = selected_data["threshold"]
    description = selected_data["description"]
    st.markdown(f"**Boundary Description:**<br>{description}", unsafe_allow_html=True)

embedding_model, sentiment_model = load_models()


col1, col2 = st.columns([4,1])
with col1:
    game_query = st.text_input("Search for a game available on Steam:")#, placeholder="e.g. Dota 2, Counter-Strike: Global Offensive")

with col2:
    st.session_state.max_reviews = st.slider("Max reviews", min_value=100, max_value=600, step=10, value=300)

if 'selected_game' not in st.session_state:
    st.session_state.selected_game = None

if game_query: 
    results = search_games(game_query)
    if results:
        num_buttons = len(results)
        cols = st.columns(min(num_buttons, 4))
        for idx, game in enumerate(results):
            game_name = f"{game['name']}"
            with cols[idx % len(cols)]:  
                if st.button(game_name, key=game['id'], use_container_width=True):
                    st.session_state.selected_game = game


        if st.session_state.selected_game is not None:
            details = get_game_details(st.session_state.selected_game['id'])
            if details:
                ct1, ct2, ct3 = st.columns([.2,.2,.2])
                ct1.image(details.get("header_image", ""))
                ct2.subheader(details.get("name", ""))
                ct2.markdown(details.get("short_description", "No description available."))

                with st.spinner("Fetching and analyzing reviews..."):
                    reviews = get_reviews(st.session_state.selected_game['id'],max_reviews=st.session_state.max_reviews )
                    if reviews:
                        df_embeddings = get_df_embeddings(embedding_model,reviews)
                        predictions = predict_model(sentiment_model, df_embeddings)

                        predictions['review'] = reviews
                        predictions['sentiment'] = predictions['prediction_score_1'].apply(lambda score: "Positive" if score >= threshold else "Negative")
                        positive_reviews = predictions[predictions['sentiment'] == "Positive"]
                        negative_reviews = predictions[predictions['sentiment'] == "Negative"]
                        total_reviews = len(predictions)
                        positive_percentage = len(positive_reviews) / total_reviews * 100
                        negative_percentage = len(negative_reviews) / total_reviews * 100
                        positivity_percentage = positive_percentage  # Percentual de positividade
                        avg_positive_score = positive_reviews['prediction_score_1'].mean() if len(positive_reviews) > 0 else 0
                        avg_negative_score = negative_reviews['prediction_score_0'].mean() if len(negative_reviews) > 0 else 0

                        ct3.subheader("Overview")
                        ct3.markdown(f"**📝 Total Reviews Analyzed**: <span style='font-size:16px; color:#333;'>{total_reviews}</span>", unsafe_allow_html=True)
                        ct3.markdown(f"**🟢 Positive Sentiment Reviews**: <span style='font-size:16px; color:#2e8b57;'>{len(positive_reviews)} ({positive_percentage:.2f}%)</span>", unsafe_allow_html=True)
                        ct3.markdown(f"**🔴 Negative Sentiment Reviews**: <span style='font-size:16px; color:#a94442;'>{len(negative_reviews)} ({negative_percentage:.2f}%)</span>", unsafe_allow_html=True)
                        ct3.markdown(f"**🎯 Avg. Confidence Score (Positive)**: <span style='font-size:16px; color:#2e8b57;'>{avg_positive_score:.2f}</span>", unsafe_allow_html=True)
                        ct3.markdown(f"**🎯 Avg. Confidence Score (Negative)**: <span style='font-size:16px; color:#a94442;'>{avg_negative_score:.2f}</span>", unsafe_allow_html=True)     

                        col_reviews, col_analysis = st.columns([6, 4])

                        with col_reviews:
                            c1,c2 = st.columns([1,2])
                            
                            with c1:
                                st.write("")
                                st.write("")
                                show_scores = st.toggle("Show scores", value=False)
                            with c2:
                                filter_option = st.radio("", ("Show All", "Only Positives", "Only Negatives"),horizontal=True)

                            with st.container(height=1000):
                                if filter_option == "Only Positives":
                                    display = positive_reviews
                                elif filter_option == "Only Negatives":
                                    display = negative_reviews
                                else:
                                    display = predictions
                                
                                for _, row in display.iterrows():
                                    sentiment_icon = "🟢" if row['sentiment'] == "Positive" else "🔴"
                                    if show_scores:
                                        score = row['prediction_score_1'] if row['sentiment'] == "Positive" else row['prediction_score_0']
                                        st.markdown(f"**{sentiment_icon} {row['sentiment']} ({score:.2f})** - {row['review']}")
                                    else:
                                        st.markdown(f"**{sentiment_icon} {row['sentiment']}** - {row['review']}")

                        with col_analysis:

                            st.write("")
                            st.write("")
                            st.markdown("<h4 style='text-align: center;'>Sentiment Wordclouds</h4>", unsafe_allow_html=True)

                            game_title = details.get("name", "")
                            title_words = set(word.lower() for word in game_title.split())
                            stop_words = get_stopwords()
                            stop_words.update(title_words)

                            if not positive_reviews.empty:
                                fig_pos = generate_wordcloud(positive_reviews['review'].tolist(), "Greens", stop_words)
                                st.pyplot(fig_pos)
                                display_topics_by_sentiment(positive_reviews, "Positive", "🟢", stopwords=stop_words)

                            if not negative_reviews.empty:
                                fig_neg = generate_wordcloud(negative_reviews['review'].tolist(), "Reds", stop_words)
                                st.pyplot(fig_neg)
                                display_topics_by_sentiment(negative_reviews, "Negative", "🔴", stopwords=stop_words)
                    else:
                        st.info("No reviews found. 🙁")
    else:
        st.warning("We couldn’t find any games matching your search. 😕")