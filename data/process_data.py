import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

class DataProcessing:
    def __init__(self,df):
        self.df = df
    
    def convert_to_category(self,column):
        categories = (self.df[column]).str.split (pat = ';', expand = True)
        cat_cols = categories.iloc[0].apply(lambda x: x.rstrip('- 0 1'))
        categories.columns = cat_cols

        for col in cat_cols:
            categories[col] = categories[col].apply(lambda x: int(x[-1]))
        self.df.drop ([column], axis = 1, inplace = True)
        self.df = pd.concat ([self.df, categories], axis = 1, sort = False)
    
    def remove_duplicates(self,col):
        self.df.drop_duplicates(subset=[col],inplace=True)
    
    def merge_df(self,df2,col):
        self.df = self.df.merge (df2, left_on = col, right_on = col, how = 'inner', validate = 'many_to_many')

        
    def drop_column(self,col):
        self.df.drop(columns=[col],inplace=True)

    def drop_rows(self,col,value):
    	self.df.drop(self.df[self.df[col]==value].index,inplace=True)



def load_data(messages_filepath, categories_filepath):
    df_message = pd.read_csv(messages_filepath)
    df_category = pd.read_csv(categories_filepath)
    data_process  = DataProcessing(df_category)
    data_process.merge_df(df_message,'id')
    return data_process.df


def clean_data(df):
    data_process  = DataProcessing(df)
    data_process.convert_to_category('categories')
    data_process.drop_column('child_alone')
    data_process.remove_duplicates('message')
    data_process.drop_rows('related',2)
    return data_process.df

def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+ str (database_filename))
    df.to_sql('message_and_category', engine, index=False, if_exists = 'replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()