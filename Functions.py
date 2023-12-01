import pandas as pd
import numpy as np
import streamlit as st


# importData: Input(Data File)
# Open file and convert to data frame
def importData(filePath):
    try:
        DF = pd.read_csv(filePath)  # read csv into dataframe
        return DF
    except Exception as e:
        print(f"Error: {e}")
        return None


# removeNAN: Input(DataFrame)
# remove NAN values from data frame
def removeNAN(df):
    quantitativeDF = df.select_dtypes(include='number').copy()  # select only columns including numbers
    return quantitativeDF  # return the copy


# removeZero: Input(DataFrame)
# remove zero values from data frame
def removeZero(df):
    df = df.loc[:, (df != 0).any(axis=0)]
    return df


# corrNoNAN: Input(Correlation matrix)
# remove NAN values from correlation matrix
def corrNoNAN(corr):
    corrClean = ~corr.isna().all()  # get all values in our matrix that are not Nan
    corr = corr.loc[corrClean, corrClean]
    return corr


# findStrongest: Input(dataFrame, number of results)
# find the n strongest relationships in the data (10 is default)
def findStrongest(dataFrame, n=10, asc=False):
    correlation = dataFrame.corr().abs()
    upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
    highest = upper.stack().sort_values(ascending=asc)
    return highest.head(n)

# TEMPORARILY NOT IN USE
# MADE FOR CACHING AND EXCHANGING DATA BETWEEN STREAMLIT PAGES
@st.cache_data
def loadData(filePath):
    data = importData(filePath)
    numericData = removeZero(removeNAN(data))
    cleanData = corrNoNAN(numericData.corr())
    return data, numericData, cleanData
