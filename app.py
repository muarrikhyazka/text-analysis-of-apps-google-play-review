import pandas as pd
import streamlit as st
from PIL import Image
from bokeh.models.widgets import Div
import plotly.express as px
import nltk

# Layout
st.set_page_config(page_title='Muarrikh Yazka', page_icon='🖖', layout='wide')





st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)


padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)

file_name='style.css'
with open(file_name) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)






# Content
@st.cache
def load_data():
    df_raw = pd.read_csv(r'data/data-for-text-analysis-streamlit.csv', sep=';')
    df = df_raw.copy()
    return df_raw, df

df_raw, df = load_data()

with st.sidebar:
    if st.button('🏠 HOME'):
        js = "window.location.href = 'https://muarrikhyazka.github.io'"  # Current tab
        html = '<img src onerror="{}">'.format(js)
        div = Div(text=html)
        st.bokeh_chart(div)





st.title('Text Analysis of Apps Google Play Review')


st.subheader('Business Understanding')
st.write(
    """
    Every apps in play store should get any comments, critics, suggestion from users. 
    From there, we can know how our users think about the apps, so it can be a great way to evaluate the apps performance and what we can improve from the apps.
    """
)

st.write(
    """
    I choose Netflix as the case because I have already known how is the bussiness and I am a user of Netflix.
    """
)

st.subheader('Data Understanding')
st.write(
    """
    **Source : Scrapping from Google Play.** You can see on my jupyter notebook in github to know how to get the data.
    """
)

st.write(
    """
    **Below is sample of the data.** 
    """
)

st.dataframe(df[['reviewId', 'userName', 'userImage', 'content', 'score',
       'thumbsUpCount', 'reviewCreatedVersion', 'at', 'replyContent',
       'repliedAt']].head())

st.subheader('Method')
st.write(
    """
    Sentiment Classification : Using pre-trained model from ...
    Category Classification : Using pre-trained model from ...
    """
)

st.subheader('Insights')
st.write(
    """
    I calculate word frequency and see on top 10.
    """
)

## convert to corpus
top=10
corpus = df["content_clean"]
lst_tokens = nltk.tokenize.word_tokenize(corpus.str.cat(sep=" "))

    
## calculate words unigrams
dic_words_freq = nltk.FreqDist(lst_tokens)
dtf_uni = pd.DataFrame(dic_words_freq.most_common(), 
                       columns=["Word","Freq"])
fig_uni = px.bar(dtf_uni.iloc[:top,:].sort_values(by="Freq"), x="Freq", y="Word", orientation='h',
             hover_data=["Word", "Freq"],
             height=400,
             title='Unigram')
st.plotly_chart(fig_uni, use_container_width=True)

st.write(
    """
    In unigram, we can see some device or operating system were mentioned, such as tv, android and nexus. From here, we can know
    """
)
    
## calculate words bigrams
dic_words_freq = nltk.FreqDist(nltk.ngrams(lst_tokens, 2))
dtf_bi = pd.DataFrame(dic_words_freq.most_common(), 
                      columns=["Word","Freq"])
dtf_bi["Word"] = dtf_bi["Word"].apply(lambda x: " ".join(
                   string for string in x) )
fig_bi = px.bar(dtf_bi.iloc[:top,:].sort_values(by="Freq"), x="Freq", y="Word", orientation='h',
             hover_data=["Word", "Freq"],
             height=400,
             title='Bigrams')
st.plotly_chart(fig_bi, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.info('**Website: [Web](https://muarrikhyazka.github.io)**', icon="🍣")
with c2:
    st.info('**GitHub: [muarrikhyazka](https://github.com/muarrikhyazka)**', icon="🍱") 
