import dash
import dash_core_components as dcc
import dash_html_components as html
from sqlalchemy import create_engine
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from dash.dependencies import Input, Output
from textblob import TextBlob
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
nltk.download('averaged_perceptron_tagger')


class DrawPlot:
    def __init__(self,df,nlp_process):
        self.df = df
        self.nlp_process = nlp_process
    def draw_category_distribution(self):
        cols = self.df.columns
        cat_cols = cols[4:]
        self.df[cat_cols].sum()
        x=cat_cols
        y=self.df[cat_cols].sum().sort_values(ascending=True)
        trace1 = go.Bar(x=y,y=x,marker=dict(color='#ffdc51'),orientation="h")
        layout1 = go.Layout(title="Data distribution across different label categories", legend=dict(x=0.1, y=1.1, orientation='h')
                  ,xaxis=dict(title="Number of data points for each label"),
                   yaxis=dict(title="Label Name"))
        fig1 = go.Figure(data = [trace1], layout = layout1)
        return fig1
    
    def draw_genre_distribution(self):
        temp_df=self.df.groupby(['genre']).count()
        temp_df = temp_df.reset_index()
        y_genre = temp_df['id']
        x_genre = temp_df['genre']
        trace2 = go.Bar(x=x_genre,y=y_genre,marker=dict(color='#ffdc51'))
        layout2 = go.Layout(title="Data distribution across different genres", legend=dict(x=0.1, y=1.1, orientation='h')
                          ,xaxis=dict(title="Number of data points for each genre"),
                           yaxis=dict(title="Genre Name"))
        fig2 = go.Figure(data = [trace2], layout = layout2)
        return fig2
    
    def draw_message_length(self):
        x_length = self.nlp_process.get_message_length()
        trace5 = go.Histogram(x=x_length, marker=dict(color='#ffdc51'),nbinsx=100)
        layout5 = go.Layout(title="Number of messages", legend=dict(x=0.1, y=1.1, orientation='h')
                          ,xaxis=dict(title="Message length"))
        fig5 = go.Figure(data = [trace5], layout = layout5)
        return fig5
    
    def draw_word_count(self):
        word_length = self.nlp_process.get_message_word_count()
        trace6 = go.Histogram(x=word_length, marker=dict(color='#ffdc51'),nbinsx=100)
        layout6 = go.Layout(title="Number of messages", legend=dict(x=0.1, y=1.1, orientation='h')
                          ,xaxis=dict(title="word count length"))
        fig6 = go.Figure(data = [trace6], layout = layout6)
        return fig6
    
    def draw_pos_distribution(self):
        pos_df = self.nlp_process.get_pos_tag()
        x_pos = pos_df['pos']
        y_pos = pos_df['count']
        trace7 = go.Bar(x=x_pos,y=y_pos,marker=dict(color='#ffdc51'))
        layout7 = go.Layout(title="Data distribution across different genres", legend=dict(x=0.1, y=1.1, orientation='h')
                          ,xaxis=dict(title="Number of data points for each part of speech"),
                           yaxis=dict(title="Part of Speech Name"))
        fig7 = go.Figure(data = [trace7], layout = layout7)

        
    def draw_most_frequent_words(self,input1):
        df_top_word = self.nlp_process.get_top_n_words(n=input1)
        x_word = df_top_word['word']
        y_word = df_top_word['count'] 
        trace3 = go.Bar(x=x_word,y=y_word, marker=dict(color='#ffdc51'))
        layout3 = go.Layout(title="Top most frequently used "+str(input1) +" words", legend=dict(x=0.1, y=1.1, orientation='h')
                      ,xaxis=dict(title="Words"),
                       yaxis=dict(title="Word Frequency"))
        fig3 = go.Figure(data = [trace3], layout = layout3)
        return fig3  
    
    def draw_most_frequent_bigrams(self,input2):
        df_top_word = self.nlp_process.get_top_n_bigrams(n=input2)
        x_word = df_top_word['bigram']
        y_word = df_top_word['count'] 
        trace4 = go.Bar(x=x_word,y=y_word, marker=dict(color='#ffdc51'))
        layout4 = go.Layout(title="Top most frequently used " +str(input2)+" words", legend=dict(x=0.1, y=1.1, orientation='h')
                      ,xaxis=dict(title="bigrams"),
                       yaxis=dict(title="Bigram Frequency"))
        fig4 = go.Figure(data = [trace4], layout = layout4)
        return fig4 


class NLP_process:
    def __init__(self,df):
        self.df = df
        self.words_freq = pd.DataFrame()
        self.df_word = pd.DataFrame()
        self.bigram_words_freq = pd.DataFrame()
        self.bigram_df_word = pd.DataFrame()
        self.pos_df = pd.DataFrame()
        
    def tokenize(self,text):
        text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
        stop_words = stopwords.words("english")
        #tokenize
        words = word_tokenize (text)
        words_lemmed = [WordNetLemmatizer().lemmatize(w) for w in words if w not in stop_words]
        return words_lemmed
    
    def calc_word_frequency(self,corpus):
        vec = CountVectorizer(tokenizer=self.tokenize).fit(corpus)
        bag_of_words = vec.fit_transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        self.words_freq = words_freq
        self.df_word = pd.DataFrame(words_freq, columns = ['word' , 'count'])
    
    def get_top_n_words(self,n=10):
        if self.df_word.empty:
            self.calc_word_frequency(self.df['message'])
        return self.df_word[:n]
    
    def calc_bigram_frequency(self,corpus):
        vec = CountVectorizer(ngram_range=(2, 2), tokenizer = self.tokenize)
        bag_of_words = vec.fit_transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        bigram_words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        bigram_words_freq =sorted(bigram_words_freq, key = lambda x: x[1], reverse=True)
        self.bigram_words_freq = bigram_words_freq
        self.bigram_df_word = pd.DataFrame(bigram_words_freq, columns = ['bigram' , 'count'])
    
    def calc_pos_tag(self):
        blob = TextBlob(self.df['message'].to_string(index=False))
        pos_df = pd.DataFrame(blob.tags, columns = ['word' , 'pos'])
        pos_df = pos_df.groupby('pos').count()
        pos_df = pos_df.reset_index()
        pos_df = pos_df.rename(columns={'word':'count'})
        self.pos_df = pos_df
    
    def get_top_n_bigrams(self,n=10):
        if self.bigram_df_word.empty:
            self.calc_bigram_frequency(self.df['message'])
        return self.bigram_df_word[:n]
    
    def get_message_length(self):
        return self.df['message'].str.len()
    
    def get_message_word_count(self):
        word_length = self.df['message'].apply(lambda x: len(x.split()))
        return word_length
    
    def get_pos_tag(self):
        if self.pos_df.empty:
            self.calc_pos_tag()
        return self.pos_df 