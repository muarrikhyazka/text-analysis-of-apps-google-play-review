import pandas as pd
import streamlit as st
from PIL import Image
from bokeh.models.widgets import Div
import plotly.express as px
import nltk
nltk.download('punkt')
import graphviz
import base64
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


title = 'Text Analysis of Apps Google Play Review'




# Layout
img = Image.open('assets/icon_pink-01.png')
st.set_page_config(page_title=title, page_icon=img, layout='wide')






st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.center {
  display: block;
  margin-left: auto;
  margin-right: auto;
#   width: 50%;
}
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

def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s" class="center" width="100" height="100"/>' % b64
    st.write(html, unsafe_allow_html=True)

with st.sidebar:
    f = open("assets/icon-01.svg","r")
    lines = f.readlines()
    line_string=''.join(lines)

    render_svg(line_string)

    st.write('\n')
    st.write('\n')
    st.write('\n')

    if st.button('🏠 HOME'):
        # js = "window.location.href = 'http://www.muarrikhyazka.com'"  # Current tab
        js = "window.open('http://www.muarrikhyazka.com')"
        html = '<img src onerror="{}">'.format(js)
        div = Div(text=html)
        st.bokeh_chart(div)

    if st.button('🍱 GITHUB'):
        # js = "window.location.href = 'https://www.github.com/muarrikhyazka'"  # Current tab
        js = "window.open('https://www.github.com/muarrikhyazka')"
        html = '<img src onerror="{}">'.format(js)
        div = Div(text=html)
        st.bokeh_chart(div)

    






st.title(title)

st.write(
    """
    \n
    \n
    \n
    """
)

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

st.write(
    """
    \n
    \n
    \n
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

st.write(
    """
    \n
    \n
    \n
    """
)

st.subheader('Method')
st.write("""
    **Flowchart**
""")

graph = graphviz.Digraph()
graph.edge('Data Scrapping', 'Text Preprocessing')
graph.edge('Text Preprocessing', 'Category Prediction')
graph.edge('Text Preprocessing', 'Sentiment Prediction')
graph.edge('Category Prediction', 'Analysis')
graph.edge('Sentiment Prediction', 'Analysis')


st.graphviz_chart(graph)

st.write(
    """
    \n
    \n
    \n
    """
)

st.subheader('Text Preprocessing')
st.write(
    """
    As usual, we need only the main meaning from the text, thats why we need to summarize by doing text preprocessing. Below the steps :
    1. Convert to lowercase and clean punctuations, characters, and whitespaces
    2. Tokenization : Split the text by word
    3. Remove Stopwords : Stopword is meaningless word and not importance word such as 'and', 'or', 'which', etc. Thats why we dont need it and remove it. Here used stopwords from nltk library
    4. Stemming : Remove -ing, -ly, etc. 
    5. Lemmatisation : Convert the word into root word.
    """
)

st.write(
    """
    \n
    \n
    \n
    """
)

st.subheader('Sentiment and Category Prediction')
st.write(
    """
    Sentiment Classification Prediction : Using pre-trained model from huggingface.co ([cardiffnlp/twitter-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)) \n
    Category Classification Prediction : Using pre-trained model from huggingface.co ([alperiox/autonlp-user-review-classification-536415182](https://huggingface.co/alperiox/autonlp-user-review-classification-536415182)) \n
    """
)

def show_word_freq(df, text_column):
    ## convert to corpus
    top=10
    corpus = df[text_column]
    lst_tokens = nltk.tokenize.word_tokenize(corpus.str.cat(sep=" "))


    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.suptitle("Most frequent words", fontsize=15)
    fig.set_size_inches(18.5, 10.5)
        
    ## calculate words unigrams
    dic_words_freq = nltk.FreqDist(lst_tokens)
    dtf_uni = pd.DataFrame(dic_words_freq.most_common(), 
                        columns=["Word","Freq"])
    dtf_uni.set_index("Word").iloc[:top,:].sort_values(by="Freq").plot(
                    kind="barh", title="Unigrams", ax=ax[0], 
                    legend=False).grid(axis='x')
    ax[0].set(ylabel=None)
        
    ## calculate words bigrams
    dic_words_freq = nltk.FreqDist(nltk.ngrams(lst_tokens, 2))
    dtf_bi = pd.DataFrame(dic_words_freq.most_common(), 
                        columns=["Word","Freq"])
    dtf_bi["Word"] = dtf_bi["Word"].apply(lambda x: " ".join(
                    string for string in x) )
    dtf_bi.set_index("Word").iloc[:top,:].sort_values(by="Freq").plot(
                    kind="barh", title="Bigrams", ax=ax[1],
                    legend=False).grid(axis='x')
    ax[1].set(ylabel=None)
    return fig

st.write(
    """
    \n
    \n
    \n
    """
)

st.subheader('Analysis')
## define label
idx_to_label_sentiments = {
    0: 'NEGATIVE',
    1: 'NEUTRAL',
    2: 'POSITIVE'
}

idx_to_label = {
    0: 'CONTENT',
    1: 'INTERFACE',
    2: 'SUBSCRIPTION',
    3: 'USER_EXPERIENCE'}

## group by 'predicted_category', 'sentiment'
df_grouped = df.groupby(['predicted_category', 'sentiment'])['reviewId'].count().reset_index()
df_grouped.columns = ['predicted_category', 'sentiment', 'count']

st.write(
    """
    1. Cross category and sentiment
    """
)

plot_1 = df_grouped.pivot('predicted_category', 'sentiment', 'count').plot(kind='bar')
st.pyplot(plot_1.figure)

st.write(
    """
    2. Count word unigram and bigram by sentiment
    """
)
## show all chart each sentiment
fig, ax = plt.subplots()
for i in idx_to_label_sentiments.values():
    print(i)
    plot_2 = show_word_freq(df[df['sentiment']==i], 'content_clean')
st.pyplot(plot_2.figure)

st.write(
    """
    3. Count word unigram and bigram by category
    """
)
## show all chart each category
fig, ax = plt.subplots()
for j in idx_to_label.values():
    print(j)
    plot_3 = show_word_freq(df[df['predicted_category']==j], 'content_clean')
st.pyplot(plot_3.figure)

st.write(
    """
    4. Count word unigram and bigram by category and sentiment
    """
)
## show all chart combination between sentiment and category
fig, ax = plt.subplots()
for i in idx_to_label_sentiments.values():
    for j in idx_to_label.values():
        print(i, 'and', j)
        plot_4 = show_word_freq(df[(df['sentiment']==i)&(df['predicted_category']==j)], 'content_clean')
st.pyplot(plot_4.figure)
    
st.write(
    """
    \n
    \n
    \n
    """
)

st.subheader('Insight')
st.write(
    """
    I calculated word frequency and see on top 10 in unigram and bigram. Try to see all chart combination between sentiment and category and will show you which has insight.
    """
)

## convert to corpus
top=10
corpus = df["content_clean"][(df['sentiment']=='NEGATIVE') & (df['predicted_category']=='INTERFACE')]
lst_tokens = nltk.tokenize.word_tokenize(corpus.str.cat(sep=" "))

    
## calculate words bigrams
dic_words_freq = nltk.FreqDist(nltk.ngrams(lst_tokens, 2))
dtf_bi = pd.DataFrame(dic_words_freq.most_common(), 
                      columns=["Word","Freq"])
dtf_bi["Word"] = dtf_bi["Word"].apply(lambda x: " ".join(
                   string for string in x) )
fig_bi = px.bar(dtf_bi.iloc[:top,:].sort_values(by="Freq"), x="Freq", y="Word", orientation='h',
             hover_data=["Word", "Freq"],
             height=400,
             title='Top 10 Bigrams Text')
st.plotly_chart(fig_bi, use_container_width=True)

st.write("""
    We can see here, its combination between negative sentiment and interface category. 
    It shows us that interface in TV is needed to be improved because android tv was mentioned sometimes. 
    Many other words which is related to TV such as mi Box (Xiaomi set top box for TV), dolby digital (sound in smart tv), and nvidia shield (android tv-based digital media player).
    It indicates that Netflix should prioritize to improve their app in TV. 
    Furthermore, in detail many complaints for voice search feature, so It should be attention to start.
""")

st.write(
    """
    \n
    \n
    \n
    """
)

c1, c2 = st.columns(2)
with c1:
    st.info('**[Github Repo](https://github.com/muarrikhyazka/text-analysis-of-apps-google-play-review)**', icon="🍣")

