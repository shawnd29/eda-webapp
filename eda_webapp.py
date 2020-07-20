import io
import os
import numpy as np
import pandas as pd
import seaborn as sns 

import matplotlib.pyplot as plt
import scipy.stats as ss

import streamlit as st 

import pydeck as pdk 
import altair as alt 
import missingno as msno 

from datetime import date, time
from datetime import datetime
from dateutil.parser import parse

from math import log
from math import ceil


import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_column',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.max_seq_items',None)
pd.set_option('display.max_colwidth', 500)
pd.set_option('expand_frame_repr', True)

plt.style.use('seaborn-colorblind')

#Color palette for Box plot
c_palette = ['tab:blue', 'tab:orange']
c_palette3 = ['tab:blue', 'tab:orange', 'tab:green']

import logging
logging.basicConfig(level=logging.INFO)
logging.info("Packages imported successfully")

documentation_string= "\n"
documentation_substring=""
df=pd.DataFrame()
df_categorical=pd.DataFrame()
df_numeric=pd.DataFrame()
df_date=pd.DataFrame()
target=pd.Series()
# If you want to add your own dataset 
files1={'file_name':["bank-additional-full.csv","shelter_cat_outcome_eng.csv","diabetes data.csv","googleplaystore.csv","<Experimental Reading data>"],
        'name':["Bank information","Cat Shelter information","Diableies information","Google Playstore","<Experimental Reading data>"],
        'target':["y","outcome_type","Diabetes","","Find your target variable"],
        'description':["This is a relatively cleaned dataset with balanced categorical and numeric values.","This dataset focuses on missing values.","This dataset contains numeric-heavy features.",
        "This dataset contains categorical- heavy features with no target variable.","This dataset shows how you could locally add your own data to explore"]   }

files=pd.DataFrame(files1)


def read_file(): 
#File location
    global documentation_string
    global documentation_substring
    global df
    st.write("### It's really great that you are curious! :smile: This is where you can add your own files if you download the source code")
    reading_data_choice= st.selectbox("Choose a way to read a file",["By manually writing the commands","By uploading the file"])
    if reading_data_choice=="By manually writing the commands":

        DATA_FOLDER = st.text_area("Enter Folder path", '')
        DATA_FILE = st.text_input("Enter File path", 'bank-additional-full.csv')
        sepretaion = st.text_input("Enter Seperation", ',')
        # DATA_FILE = 'bank-additional-full.csv'
        df=read_data(DATA_FOLDER,DATA_FILE,sepretaion)
        #df= pd.read_csv(os.path.join(DATA_FOLDER,DATA_FILE), sep=';')
        documentation_substring= f"File {DATA_FILE} successfully read from {DATA_FOLDER}\n"
        logging.info(documentation_substring)
        documentation_string+=documentation_substring+'\n'
    if reading_data_choice=="By uploading the file":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write(df)
    pass 

#@st.cache(allow_output_mutation=True)
def read_data(DATA_FOLDER,DATA_FILE,sepretaion):
    df= pd.read_csv(os.path.join(DATA_FOLDER,DATA_FILE), sep=sepretaion)
    return df


def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        st.write ("Your selected dataframe has " + str(df.shape[1]) + " columns and a total of "+str(df.shape[0])+" values\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
                      
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

#@st.cache(allow_output_mutation=True)
def find_target(target_name):
        global df
        target=df[target_name]
        return target

#@st.cache()
def seperate_features():
    global df
    global df_numeric
    global df_categorical
    global df_date
    df_numeric=df.select_dtypes(include=['float64', 'int64'])
    df_date=df.select_dtypes(include=['datetime64'])
    df_categorical=df.select_dtypes(exclude=['float64', 'int64','datetime64'])
    pass



def categorical_summarized(dataframe, x=None, y=None, hue=None, palette='Set1', ax=None, order=None, verbose=True):
    '''
    Helper function that gives a quick summary of a given column of categorical data
    Arguments
    =========
    dataframe: pandas dataframe
    x: str. horizontal axis to plot the labels of categorical data, y would be the count
    y: str. vertical axis to plot the labels of categorical data, x would be the count
    hue: str. if you want to compare it another variable (usually the target variable)
    palette: array-like. Colour of the plot
    Returns
    =======
    Quick Stats of the data and also the count plot
    '''
    
    if x == None:
        column_interested = y
    else:
        column_interested = x
    series = dataframe[column_interested]
    st.write('\t', column_interested)
    st.write(series.describe())
    st.write('mode: ', series.mode())
    if verbose:
        
        st.write(series.value_counts())
    if (series.value_counts()).count()>50:
        st.write("Too many values to populate")
    else:
        sns.countplot(x=x, y=y, hue=hue, data=dataframe, palette=palette)
        plt.show()
        st.pyplot() 
    st.write('\n'*2)
    st.write('='*80)

def quantitative_summarized(dataframe, x=None, y=None, hue=None, palette='Set1', ax=None, order=None, verbose=True):
    '''
    Helper function that gives a quick summary of quantattive data
    Arguments
    =========
    dataframe: pandas dataframe
    x: str. horizontal axis to plot the labels of categorical data (usually the target variable)
    y: str. vertical axis to plot the quantitative data
    hue: str. if you want to compare it another categorical variable (usually the target variable if x is another variable)
    palette: array-like. Colour of the plot
    Returns
    =======
    Quick Stats of the data and also the violin plot of the distribution
    '''
    series = dataframe[y]
    st.write ('\t', y)
    st.write(series.describe())
    st.write('mode: ', series.mode())
    st.write('Unique values: ', series.unique().size)
    if verbose:
        st.write('\n'*2)
        st.write(series.value_counts())
    
    sns.violinplot(x=x, y=y, hue=hue, data=dataframe,
                palette=palette, order=order, ax=ax)
    

    plt.show()
    st.pyplot()
    st.write('\n'+'#'*80+'\n')


def eda_analysis():

    global documentation_string
    global documentation_substring
    global df
    global df_categorical
    global df_numeric
    global df_date
    # Utilizing a documentation platform to see all the changes we would be using (Useful for pipelining)
   


    st.subheader('Data Input')
    #read_file()
    option = st.selectbox(
        'Choose which type of data',files.name)
    st.write("You have chosen "+option)
    option_index=files.index[files['name']==option]
    # st.write(files.loc[option_index,'file_name'].item())
    option_name=files.loc[option_index,'file_name'].item()
    st.write(files.loc[option_index,'description'].item())
    if (option_name=='<Experimental Reading data>'):
        read_file()
    else: 
        df= read_data("",option_name,",")




    if st.button('Initial features'):
        st.subheader('First 5 Features')
        st.write(df.head())
        st.subheader('Columns Present')
        st.write(df.columns)
        st.subheader('Info')
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        
    if st.button('Check for duplicated values'):
        if len(df[df.duplicated()]) > 0:
            st.write("No. of duplicated entries: ", len(df[df.duplicated()]))
            st.write("### Duplicated values")
            st.write(df[df.duplicated(keep=False)].sort_values(by=list(df.columns)))
        else:
            st.write("No duplicated entries found")

    # if st.checkbox('Drop Duplicates?'):
    #     #Drop the duplicate
    #     documentation_substring= f"Dropped {len(df[df.duplicated()])} values\n"
    #     df.drop_duplicates(inplace=True)    
    #     logging.info(documentation_substring)
    #     documentation_string+=documentation_substring+'\n'
    #     st.write(documentation_substring)


    # if st.button('Check for duplicated values 2'):
    #     if len(df[df.duplicated()]) > 0:
    #         st.write("No. of duplicated entries: ", len(df[df.duplicated()]))
    #         st.write(df[df.duplicated(keep=False)].sort_values(by=list(df.columns)))
    #     else:
    #         st.write("No duplicated entries found")
    # Function to calculate missing values by column# Funct 


    if st.button('In-depth analysis on missing values'):
        missing_values = missing_values_table(df)
        st.write("### Missing value rows:")
        st.write(missing_values)

    if st.button('Visualize missing values'):
        # Visualize missing values as a matrix 
        # Checks if the missing values are localized
        st.write("### Where are the missing values located")
        st.write(msno.matrix(df)) 
        st.pyplot() 
        # Visualize missing values as a heatmap 
        st.write("### Heatmap of the missing values")
        st.write(msno.heatmap(df))
        st.pyplot() 
    # if st.button("(1) Drop Missing Rows"):
    #     st.write(1)

    # if st.button("(1) Drop Missing Rows"):
    #     st.write(1)

    # if st.button("(1) Drop Missing Rows"):
    #     st.write(1)


    st.info('Make sure you define the target variable for bivariate classification')
    if st.checkbox('Find the target variable'):
        if (files.loc[option_index,'name'].item() == "<Experimental Reading data>") and (files.loc[option_index,'target'].item()=="Find your target variable"):
            st.write(df.head())
            st.write("Search for the target variable from your dataset")
        else:
            st.write("For this dataset, it is {0}".format(files.loc[option_index,'target'].item()) )
        target_name = st.text_input("Enter the target name",files.loc[option_index,'target'].item())
        target=find_target(target_name)
        st.write("Target: ",target_name)
        st.write("Target type: ",type(target))
        st.write("### Overview")
        st.write(target.head())

    if st.button("Check the data type of each column with an example"):
        interesting= pd.DataFrame(df.dtypes,columns=["Data_Type"])
        interesting["First_value"]=df.iloc[0,:]
        unique_values= df.nunique()
        interesting["Unique_values"]=unique_values
        st.write(interesting)

    if st.button('Column-wise analysis'):
        st.write("### Unique Values")
        unique_values= df.nunique()
        st.write(unique_values)
        num_rows = len(df.index)
        low_information_cols = [] #
        st.write("### Individual column analysis")
        st.write("If there isn't a data frame for a column, it implies that the values in said column is either too custered or too sparse (i.e. It has low information)")
        for col in df.columns:

            cnts = df[col].value_counts(dropna=False)
            top_pct = (cnts/num_rows).iloc[0]
            
            if top_pct< .10:
                st.write("Column {0} has low information".format(col))
                low_information_cols.append(col)
                continue

            if top_pct < 0.85 and top_pct > 0.10:
                low_information_cols.append(col)
                st.write('Column: {0}       Most populated value: {1}  which covers {2:.5f}% of the column'.format(col,cnts.index[0], top_pct*100))
                st.write(cnts)
                st.write('\n')
        

            plt.figure()
            plt.title(f'{col} - {unique_values[col]} unique values')
            plt.ylabel('Count');
            values=pd.value_counts(df[col]).plot.bar()
            plt.xticks(rotation = 75);
            st.pyplot() 
        st.write("Columns with low information are:")
        st.write(np.setdiff1d(df.columns,low_information_cols))



    # if st.button("Convert numeric to categorical feature <Pending>"):
    #     pass

    # if st.button("Convert string to datetime feature <Pending>"):
    #     pass


    # if st.button("Overview of summary based on the target variable <Pending>"):
    #     pass


    # if st.button("Rename columns if needed <Pending>"):
    #     pass

    # if st.button("Drop the target variable from the dataframe <Pending>"):
    #     pass

    df_numeric=df.select_dtypes(include=['float64', 'int64'])
    df_date=df.select_dtypes(include=['datetime64'])
    df_categorical=df.select_dtypes(exclude=['float64', 'int64','datetime64'])
    
    

    
    if st.button("Get numeric, categorical and datetime features"):
        st.write("### Numierc Features")
        st.write(df_numeric.head())
        st.write("### Categorical Features")
        st.write(df_categorical.head())
        st.write("### Date-time features")
        st.write(df_date.head())
        st.text(f"The following columns were categorized as: \n numeric: {df_numeric.columns}\n categroical: {df_categorical.columns}\n date-time: {df_date.columns}\n")
        
    # if st.button("Remove extra white space in text columns <Pending>"):
    #     pass
    st.markdown("## Categorical columns")

    if st.button("Information on categorical columns"):
        st.write("### Categorical Column names")
        st.write(df_categorical.columns)
        st.write("### Categorical Info")
        buffer = io.StringIO()
        df_categorical.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        #st.write(df_numeric)

    categorical_selector= st.radio("Choose what type of categorical analysis to conduct:",["Select one of the two", "Univariate analysis of categorical feature","Bivariate analysis of categorical feature"])
    categorical_names=df_categorical.columns.tolist()
    categorical_names.append("All columns")
    if (categorical_selector=="Univariate analysis of categorical feature"):
        categorical_option=st.selectbox("Choose which column",categorical_names)
        if (categorical_option=="All columns"):
            for col in df_categorical.columns:
                categorical_summarized(df_categorical,y=col)
        else:
            categorical_summarized(df_categorical,y=categorical_option)


    if (categorical_selector=="Bivariate analysis of categorical feature"):
        st.write("**Make sure that you have defined the target variable**")
        categorical_option=st.selectbox("Choose which column",categorical_names)
        if (categorical_option=="All columns"):
            for col in df_categorical.columns:
                categorical_summarized(df_categorical,y=col,hue=target)
        else:
            categorical_summarized(df_categorical,y=categorical_option,hue=target)

    # if st.button("Categorical Data Imputation <Pending>"):
    #     pass

    # if st.button("Chi square analysis <Pending>"):
    #     pass

    # if st.button("Encoding categorical data <Pending>"):
    #     pass



    if st.button(" View Finalized Categorical columns"):
        st.write(df_categorical.head(15))

        

    st.markdown("## Date-time columns")
    st.write(df_date.head())
    if st.button("All functions <Pending>"):
        st.write("Still working on extracting dates {since they are not always a date64 data type}. Would love some advice on this")


    st.markdown("## Numeric columns")

    if st.button("Initial numeric features"):
        st.write("### Numeric Overviews")
        st.write(df_numeric.head())
        df_numeric.hist(figsize=(20, 20), bins=10, xlabelsize=8, ylabelsize=8);
        st.pyplot() 
    
    numeric_selector= st.radio("Choose what type of numeric analysis to conduct:",["Select one of the three", "Univariate analysis of numeric feature","Bivariate analysis of numeric feature","Multivariate variate analysis of numeric feature"])
    numeric_names=df_numeric.columns.tolist()
    numeric_names.append("All columns")
    if (numeric_selector=="Univariate analysis of numeric feature"):
        numeric_option=st.selectbox("Choose which column",numeric_names)
        if (numeric_option=="All columns"):
            for col in df_numeric.columns:
                quantitative_summarized(df_numeric,y=col)
        else:
            quantitative_summarized(df_numeric,y=numeric_option)


    if (numeric_selector=="Bivariate analysis of numeric feature"):
        st.write("**Make sure that you have defined the target variable**")
        numeric_option=st.selectbox("Choose which column",numeric_names)
        if (numeric_option=="All columns"):
            for col in df_numeric.columns:
                quantitative_summarized(dataframe= df_numeric, y = col, palette=c_palette, x = target, verbose=False)
        else:
            quantitative_summarized(dataframe= df_numeric, y = numeric_option, palette=c_palette, x = target, verbose=False)
        
    if (numeric_selector=="Multivariate variate analysis of numeric feature"):
        var1 = st.text_input("Enter the first variable")
        var2 = st.text_input("Enter the second variable")
        quantitative_summarized(dataframe= df_numeric, y = var1, x = var2, hue = target, palette=c_palette3, verbose=False)

    if st.button("Target details"):
        st.write(target.name)
        st.write(type(target))
        st.write(target.value_counts())
    st.write()
    st.write()
    
    if st.button("You're done!! Click here to celebrate"):    
        st.balloons()


#### This is an expereimental feature that I was trying to implement. It focuses on manually seperating the categorical, numeric and time-series data. 
#### Any advice on this would be greatly beneficial

def choose_data_types():
    global df
    global df_categorical,df_numeric,df_date
    categorical_name_options = st.multiselect(
    'Choose your categorical variables',
        df.columns.tolist(),
        df_categorical.columns.tolist())
    st.write('You selected:', df[categorical_name_options].head())

    numeric_name_options = st.multiselect(
    'Choose your  Numeric variables',
        df.columns.tolist(),
        df_numeric.columns.tolist())
    st.write('You selected:', df[numeric_name_options].head())

    datetime_name_options = st.multiselect(
    'Choose your  Datetime variables',
        df.columns.tolist(),
        df_date.columns.tolist())
    st.write('You selected:',  df[datetime_name_options].head())

    st.write("")
    st.write("")
    st.warning("Make sure that you confirm the changes")
    if st.button("Confirm options?"):
        df_categorical,df_numeric,df_date=confirm_options(df_categorical,df_numeric,df_date,categorical_name_options,numeric_name_options,datetime_name_options)
        pass
    else:
        raise st.ScriptRunner.StopException


#@st.cache(allow_output_mutation=True)
def confirm_options(df_categorical,df_numeric,df_date,categorical_name_options,numeric_name_options,datetime_name_options):
    df_categorical=df[categorical_name_options]
    df_numeric=df[numeric_name_options]
    df_date=df[datetime_name_options]
    return df_categorical,df_numeric,df_date

#     return(df_categorical,df_numeric,df_date)
if __name__ == "__main__":
    eda_analysis()