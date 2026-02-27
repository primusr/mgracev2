import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def plot_sentiment_distribution(df):
    fig, ax = plt.subplots()
    sns.countplot(x="Sentiment_Label", data=df, ax=ax)
    ax.set_title("Sentiment Distribution")
    return fig

def generate_wordcloud(words):
    wc_dict = dict(words)
    wc = WordCloud(background_color="white").fit_words(wc_dict)

    fig, ax = plt.subplots()
    ax.imshow(wc)
    ax.axis("off")
    return fig