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

# This class is used to create figures in plotly for visualization
class DrawPlot:
    def __init__(self,df,nlp_process):
        self.df = df
        self.nlp_process = nlp_process
    def draw_category_distribution(self):
        '''
        This function returns a Figure object to plot data distribution across
        different labels
        '''
        cols = self.df.columns
        cat_cols = cols[4:]
        self.df[cat_cols].sum()
        x=cat_cols
        y=self.df[cat_cols].sum().sort_values(ascending=True)
        trace1 = go.Bar(x=y,y=x,marker=dict(color='#ffdc51'),orientation="h")
        layout1 = go.Layout(width=1200,height=800,title="Data distribution across different label categories", legend=dict(x=0.1, y=1.1, orientation='h')
                  ,xaxis=dict(title="Number of data points for each label"),
                   yaxis=dict(title="Label Name"))
        fig1 = go.Figure(data = [trace1], layout = layout1)
        return fig1
    
    def draw_genre_distribution(self):
        '''
        This function returns a Figure object to plot data distribution across
        different genre
        '''
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
        '''
        This function returns a Figure object that gives a histogram of message length
        across the data
        '''
        x_length = self.nlp_process.get_message_length()
        trace5 = go.Histogram(x=x_length, marker=dict(color='#ffdc51'),nbinsx=100)
        layout5 = go.Layout(title="Message length distribution in the data", legend=dict(x=0.1, y=1.1, orientation='h')
                          ,xaxis=dict(title="Message length"),yaxis=dict(title="Number of Messages"))
        fig5 = go.Figure(data = [trace5], layout = layout5)
        return fig5
    
    def draw_word_count(self):
        '''
        This function returns a Figure object that gives a histogram of word count of message
        across the data
        '''
        word_length = self.nlp_process.get_message_word_count()
        trace6 = go.Histogram(x=word_length, marker=dict(color='#ffdc51'),nbinsx=100)
        layout6 = go.Layout(title="Message word count distribution in the data", legend=dict(x=0.1, y=1.1, orientation='h')
                          ,xaxis=dict(title="word count length"),yaxis=dict(title="Number of messages"))
        fig6 = go.Figure(data = [trace6], layout = layout6)
        return fig6
    
    def draw_pos_distribution(self):
        '''
        This function returns Figure object to plot part of speech tags distribution
        in the data
        '''
        pos_df = self.nlp_process.get_pos_tag()
        x_pos = pos_df['pos']
        y_pos = pos_df['count']
        trace7 = go.Bar(x=x_pos,y=y_pos,marker=dict(color='#ffdc51'))
        layout7 = go.Layout(title="Data distribution across different genres", legend=dict(x=0.1, y=1.1, orientation='h')
                          ,xaxis=dict(title="Number of part of speech tags"),
                           yaxis=dict(title="Part of Speech Name"))
        fig7 = go.Figure(data = [trace7], layout = layout7)
        return fig7

        
    def draw_most_frequent_words(self,top_k_words):
        '''
        Input
        top_k_words: return Figure object for top k most frequent words 
        along with their frequency
        '''
        df_top_word = self.nlp_process.get_top_n_words(n=top_k_words)
        x_word = df_top_word['word']
        y_word = df_top_word['count'] 
        trace3 = go.Bar(x=x_word,y=y_word, marker=dict(color='#ffdc51'))
        layout3 = go.Layout(title="Top most frequently used "+str(top_k_words) +" words", legend=dict(x=0.1, y=1.1, orientation='h')
                      ,xaxis=dict(title="Words"),
                       yaxis=dict(title="Word Frequency"))
        fig3 = go.Figure(data = [trace3], layout = layout3)
        return fig3  
    
    def draw_most_frequent_bigrams(self,top_k_words):
        '''
        Input
        top_k_words: return Figure object for top k most frequent bigrams
        along with their frequency
        '''
        df_top_word = self.nlp_process.get_top_n_bigrams(n=top_k_words)
        x_word = df_top_word['bigram']
        y_word = df_top_word['count'] 
        trace4 = go.Bar(x=x_word,y=y_word, marker=dict(color='#ffdc51'))
        layout4 = go.Layout(title="Top most frequently used " +str(top_k_words)+" words", legend=dict(x=0.1, y=1.1, orientation='h')
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
        '''
        Input
        text: textual data
        Output: This function returns textual list after lemmatization and
        removing stopwords
        '''
        text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
        stop_words = stopwords.words("english")
        #tokenize
        words = word_tokenize (text)
        words_lemmed = [WordNetLemmatizer().lemmatize(w) for w in words if w not in stop_words]
        return words_lemmed
    
    def calc_word_frequency(self,corpus):
        '''
        Input
        corpus: It takes the whole text data
        Output: It creates a data frame with each word and its frequency
        '''
        vec = CountVectorizer(tokenizer=self.tokenize).fit(corpus)
        bag_of_words = vec.fit_transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        self.words_freq = words_freq
        self.df_word = pd.DataFrame(words_freq, columns = ['word' , 'count'])
    
    def get_top_n_words(self,n=10):
        '''
        Input
        n: number of words
        Output: It returns top n most frequent words 
        '''
        if self.df_word.empty:
            self.calc_word_frequency(self.df['message'])
        return self.df_word[:n]
    
    def calc_bigram_frequency(self,corpus):
        '''
        Input
        corpus: It takes the whole text data
        Output: It creates a data frame with each bigram and its frequency
        '''
        vec = CountVectorizer(ngram_range=(2, 2), tokenizer = self.tokenize)
        bag_of_words = vec.fit_transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        bigram_words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        bigram_words_freq =sorted(bigram_words_freq, key = lambda x: x[1], reverse=True)
        self.bigram_words_freq = bigram_words_freq
        self.bigram_df_word = pd.DataFrame(bigram_words_freq, columns = ['bigram' , 'count'])
    
    def calc_pos_tag(self):
        '''
        This function calculates part of speech frequency
        '''
        blob = TextBlob(self.df['message'].to_string(index=False))
        pos_df = pd.DataFrame(blob.tags, columns = ['word' , 'pos'])
        pos_df = pos_df.groupby('pos').count()
        pos_df = pos_df.reset_index()
        pos_df = pos_df.rename(columns={'word':'count'})
        self.pos_df = pos_df
    
    def get_top_n_bigrams(self,n=10):
        '''
        Input
        n: number of words
        Output: It returns top n most frequent bigrams 
        '''
        if self.bigram_df_word.empty:
            self.calc_bigram_frequency(self.df['message'])
        return self.bigram_df_word[:n]
    
    def get_message_length(self):
        '''
        This function returns message length
        '''
        return self.df['message'].str.len()
    
    def get_message_word_count(self):
        '''
        This function returns word count of message
        '''
        word_length = self.df['message'].apply(lambda x: len(x.split()))
        return word_length
    
    def get_pos_tag(self):
        '''
        This function returns part of speech frequency
        '''
        if self.pos_df.empty:
            self.calc_pos_tag()
        return self.pos_df 
