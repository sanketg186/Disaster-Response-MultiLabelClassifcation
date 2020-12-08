import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

# This class is used to clean the data 
class DataProcessing:
    def __init__(self,df):
        '''
        Input
        df: datafame
        Initialize the class instance object variable df with data frame
        '''
        self.df = df
    
    def convert_to_category(self,column):
        '''
        Input: column whose values needs to be converted to binary category columns
        Example: This column value "related-1;request-0;offer-0;aid_related-0"
        will be converted to 4 different columns having column names 
        ('related','request','offer','aid_related') with values (1,0,0,0)
        '''
        categories = (self.df[column]).str.split (pat = ';', expand = True)
        cat_cols = categories.iloc[0].apply(lambda x: x.rstrip('- 0 1'))
        categories.columns = cat_cols

        for col in cat_cols:
            categories[col] = categories[col].apply(lambda x: int(x[-1]))
        self.df.drop ([column], axis = 1, inplace = True)
        self.df = pd.concat ([self.df, categories], axis = 1, sort = False)
    
    def remove_duplicates(self,col):
        '''
        Input 
        col: column name
        All the rows having duplicate values for that column are dropped
        '''
        self.df.drop_duplicates(subset=[col],inplace=True)
    
    def merge_df(self,df2,col):
        '''
        Input
        df2: Second data frame with which self.df needs to be merged
        col: column name on which the merge has to happen
        '''
        self.df = self.df.merge (df2, left_on = col, right_on = col, how = 'inner', validate = 'many_to_many')

        
    def drop_column(self,col):
        '''
        Input
        col: column name that has to be dropped
        '''
        self.df.drop(columns=[col],inplace=True)

    def drop_row_column_value(self,col,val):
        '''
        Input
        col: column name
        val: value that needs to be checked and the corresponding rows has to be removed
        '''
        self.df.drop(self.df[self.df['related']==2].index,inplace=True)



def load_data(messages_filepath, categories_filepath):
    '''
    Input
    messages_filepath: message csv filepath
    categories_filepath: category csv filepath
    Output
    This function loads the data from both csv into dataframe and return it
    '''
    df_message = pd.read_csv(messages_filepath)
    df_category = pd.read_csv(categories_filepath)
    data_process  = DataProcessing(df_category)
    data_process.merge_df(df_message,'id')
    return data_process.df


def clean_data(df):
    '''
    This function cleans the data using DataProcessing class
    '''
    data_process  = DataProcessing(df)
    data_process.convert_to_category('categories')
    data_process.drop_column('child_alone')
    data_process.remove_duplicates('message')
    data_process.drop_row_column_value('related',2)
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