import pandas as pd
import numpy as np
import streamlit as st

import secrets  # for filenames
import string


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


# uploadFile:
# prompts the user to upload a csv file to look at the data
def uploadFile():
    st.subheader("Please upload a CSV to continue")
    # Upload CSV
    uploadedFile = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploadedFile is not None:
        # Load the uploaded data
        st.session_state.data = pd.read_csv(uploadedFile)

        # Process the data
        st.session_state.numericData = removeZero(removeNAN(st.session_state.data))
        st.session_state.cleanData = corrNoNAN(st.session_state.numericData.corr())
        st.experimental_rerun()


# generateFileName : Input(type of graph)
# use secrets library to generate characters for unique filenames
def generateFileName(type):
    fileSuffix = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(6))
    fileOut = f"{type}_{fileSuffix}.pdf"
    return fileOut


def generateTextReport(type):
    fileSuffix = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(6))
    fileOut = f"{type}_{fileSuffix}.csv"
    return fileOut

# exportPlot : Input(plot, type of plot)
# export a generated seaborn plot
def exportPlot(currentGraph, plotType):
    fileName = generateFileName(plotType)
    currentGraph.savefig(fileName, format="pdf")
    with open(fileName, 'rb') as file:
        pdf_data = file.read()
    return pdf_data


# detect outliers : Input(type of graph)
# function for finding outliers of a feature
def detectOutliers(selectedColumn):
    Q1 = selectedColumn.quantile(0.25)
    Q3 = selectedColumn.quantile(0.75)
    IQR = Q3 - Q1
    lowerBound = Q1 - 1.5 * IQR
    upperBound = Q3 + 1.5 * IQR
    outliers = selectedColumn[(selectedColumn < lowerBound) | (selectedColumn > upperBound)]
    return outliers
