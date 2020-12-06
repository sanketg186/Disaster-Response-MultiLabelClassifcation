import json
import plotly
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import joblib
from sqlalchemy import create_engine
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from utils import NLP_process,DrawPlot
import dash
import dash_core_components as dcc
import dash_html_components as html
from sqlalchemy import create_engine
import pandas as pd
import plotly.graph_objects as go
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from dash.dependencies import Input, Output
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    stop_words = stopwords.words("english")
    
    #tokenize
    words = word_tokenize (text)
    words_lemmed = [WordNetLemmatizer().lemmatize(w) for w in words if w not in stop_words]
    return words_lemmed

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('message_and_category', engine)

# load model
model = joblib.load("../models/pipeline.sav")

nlp_process = NLP_process(df)
draw_obj = DrawPlot(df,nlp_process)

fig1 = draw_obj.draw_category_distribution()
fig2 = draw_obj.draw_genre_distribution()
fig5 = draw_obj.draw_message_length()
fig7 = draw_obj.draw_word_count()
fig8 = draw_obj.draw_pos_distribution()
fig3 = draw_obj.draw_most_frequent_words(10)
fig4 = draw_obj.draw_most_frequent_bigrams(10)

visualization_layout = html.Div(children = [
    html.Div([html.H1('Disaster Response Message Visualization')],style={'text-align':'center','color':'purple'})
    ,
    html.Div([
    dcc.Graph( 
        id = 'graph1',
        figure= fig1
    )])
    ,
    html.Div([
        html.Div([
        dcc.Graph( 
            id = 'graph2',
            figure= fig2
        )
        ],className="six columns"),
        html.Div([
        dcc.Graph( 
            id = 'graph5',
            figure= fig5
        )
        ],className="six columns")],className="row")
        ,
    html.Div([
        html.Div([
        dcc.Graph( 
            id = 'graph7',
            figure= fig7
        )
        ],className="six columns"),
        html.Div([
        dcc.Graph( 
            id = 'graph8',
            figure = fig8
        )
        ],className="six columns")],className="row")
        ,
        html.Div([
            dcc.Graph( 
            id = 'graph3',
            figure=fig3
        )

        ])
    ,
    html.Div([
        dcc.Slider(
               id='slider-1',
               min=1,
               max=100,
               step=1,
               value=10,
               marks={i: str(i) for i in range(0, 100,5)})
    ]),
    html.Div([
        dcc.Graph( 
        id = 'graph4',
        figure=fig4
    )
        
    ]),
    html.Div([
        dcc.Slider(
               id='slider-2',
               min=1,
               max=100,
               step=1,
               value=10,
               marks={i: str(i) for i in range(0, 100,5)})
    ])
])

prediction_layout = html.Div(children = [
    html.Div([html.H1('Disaster Response Message Prediction')],style={'text-align':'center','color':'purple'})
    ,
    html.Div([
        dcc.Input(
            id="input_text",
            type="text",
            placeholder="Enter message to classify"
        ),
        html.Div(id='output_graph')],style={'text-align':'center'})
])


index_layout = html.Div([
    html.Div(html.H1("Figure-8-Disaster Response Message Visualization and Prediction"),style={'text-align':'center','color':'yellow'}),
    dbc.Row([
    dbc.Col(html.Div([dbc.Button('Data Visualization',color="success", id='visualize-button',className="mr-1", n_clicks=0)],className="six-columns")),
    dbc.Col(html.Div([dbc.Button('Prediction', id='predict-button',color="warning",className="mr-1", n_clicks=0)],className="six-columns"))
    ])
],style={'text-align':'center'})
        
main_layout = html.Div(children=[
    index_layout,
    html.Div(id='page-content'),
    dcc.Location(id='url', refresh=False)
])

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=[external_stylesheets,dbc.themes.BOOTSTRAP])

app.layout = main_layout

@app.callback(Output('page-content', 'children'),
              Input('visualize-button', 'n_clicks'),
              Input('predict-button', 'n_clicks')
              )

def displayClick(btn1,btn2):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'visualize-button' in changed_id:
        return visualization_layout
    elif 'predict-button' in changed_id:
        return prediction_layout

@app.callback(Output('output_graph', 'children'),
             [Input('input_text', 'value')]
             )
def update_figure(input1):
    if input1 is not None:
        y_predict = model.predict_proba([input1])[0]
        cols = draw_obj.df.columns
        cat_cols = cols[4:]
        x=cat_cols
        y=y_predict
        print(y)
        trace1 = go.Bar(x=y,y=x,marker=dict(color='#ffdc51'),orientation="h")
        layout1 = go.Layout(width=1200,height=800,title="Data distribution across different label categories", legend=dict(x=0.1, y=1.1, orientation='h')
                  ,xaxis=dict(title="Number of data points for each label"),
                   yaxis=dict(title="Label Name"))
        fig1 = go.Figure(data = [trace1], layout = layout1)
        return dcc.Graph(
            id='example_graph',
            figure= fig1
        )
    else:
        return ""

@app.callback(Output('graph3', 'figure'),
             [Input('slider-1', 'value')]
             )
def update_figure(input1):
    fig3 = draw_obj.draw_most_frequent_words(input1)
    return fig3

@app.callback(Output('graph4', 'figure'),
             [Input('slider-2', 'value')])
def update_figure2(input2):
    fig4 = draw_obj.draw_most_frequent_bigrams(input2)
    return fig4

def main():
    app.run_server(port = 3001)

if __name__ == '__main__':
    main()