import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import re
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import multioutput
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
#ML models
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings("ignore")
def evaulation_metric(y_true,y_pred):
    '''
    Input 
    y_true: ground truth dataframe
    y_pred: predicted dataframe
    
    Output
    report: dataframe that contains mean f1-score,precision and recall value for each class
    '''
    report = pd.DataFrame ()
    for col in y_true.columns:
        class_dict = classification_report (output_dict = True, y_true = y_true.loc [:,col], y_pred = y_pred.loc [:,col])
    
        metric_df = pd.DataFrame (pd.DataFrame.from_dict (class_dict))
        
        metric_df.drop(['macro avg', 'weighted avg'], axis =1, inplace = True)
        
        metric_df.drop(index = 'support', inplace = True)
        
        metric_df = pd.DataFrame (metric_df.transpose ().mean ())
         
        metric_df = metric_df.transpose ()
    
        report = report.append (metric_df, ignore_index = True)    
    
    report.index = y_true.columns
    
    return report
    


def load_data(database_filepath):
    '''
    Input
    database_filepath: filepath of database
    Output: X,y and category names from the database
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('message_and_category', engine)
    X = df['message']
    y = df.iloc[:, 4:]
    category_names = list(df.columns[4:])
    return X, y, category_names

def tokenize(text):
    '''
    Input
    text: textual data
    Output: returns a lemmatized and stopwords removed list of words
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    stop_words = stopwords.words("english")
    
    #tokenize
    words = word_tokenize (text)
    words_lemmed = [WordNetLemmatizer().lemmatize(w) for w in words if w not in stop_words]
    return words_lemmed


def build_model():
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('scale',StandardScaler(with_mean=False)),
                     ('clf', OneVsRestClassifier(LinearSVC()))])

    search_space = [{'clf':[OneVsRestClassifier(LinearSVC())],
                 'clf__estimator__C': [1, 10, 100]},
                
                {'clf': [OneVsRestClassifier(LogisticRegression(solver='sag'))], 
                 'clf__estimator__C': [1, 10, 100]},
                
                {'clf': [OneVsRestClassifier(MultinomialNB())],
                 'clf__estimator__alpha': [0.1, 0.5, 1]},
                {'clf':[multioutput.MultiOutputClassifier(RandomForestClassifier())]}]

    cv_pipeline = GridSearchCV(pipeline, search_space)

    return cv_pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model performance using test data
    Input: 
        model: Model to be evaluated
        X_test: Test data (features)
        Y_test: True lables for Test data
        category_names: Labels for 36 categories
    Output:
        Print accuracy and classfication report for each category
    '''
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame (Y_pred, columns = Y_test.columns)
    #Calculate the accuracy for each of them.
    report = evaulation_metric(Y_test,Y_pred)
    print(report)

def save_model(model, model_filepath):
    '''
    Save model as a pickle file 
    Input: 
        model: Model to be saved
        model_filepath: path of the output pick file
    Output:
        A pickle file of saved model
    '''
    pickle.dump(model, open(model_filepath, "wb"))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()