import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import statsmodels.api as sm
import streamlit as st

import Functions as tools

data = None
numericData = None
cleanData = None

st.set_page_config(layout="wide")
st.sidebar.title('Tools')

selectedPage = st.sidebar.radio('Available: ',
                                ['Home', 'Heatmap and Correlation Matrix', 'Raw Data', 'Scatterplot (Two Var.)',
                                 'Histogram (Single Var.)', 'Data Quality Report', 'Regression Analysis'])

if selectedPage == 'Home':
    if data is not None: #this is the problem endless loop, data is set to none on refresh
        # Display Current Date
        currentDate = datetime.datetime.now().strftime("%B %d, %Y")  # Format: Month day, Year
        st.subheader(currentDate)

        col1, col2 = st.columns(2)
        col1.header('Observation Count:')
        col1.subheader(data.shape[0])
        col2.header('Last Observation: ')
        lastTimestamp = data['Timestamp'].iloc[-1]
        col2.subheader(lastTimestamp)

        st.write('Most Recent Samples: ')
        recentData = data.tail(5)  # get the last 5 items from the dataframe
        st.table(recentData)

        st.subheader('Strongest Relationships: ')
        with st.expander("Set Number of Rows"):
            cutoff = st.slider("Select Number Cutoff:", 1, cleanData.shape[0], 5, 1)
            strongestRelations = tools.findStrongest(cleanData, cutoff)
        st.write("Top Correlations with Cutoff of:", cutoff)
        st.table(strongestRelations)
    else:
        st.title("Welcome")
        st.subheader("Select a CSV file for upload")
        st.stop()
        # Upload CSV
        uploadedFile = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploadedFile is not None:
            # Load the uploaded data
            data = pd.read_csv(uploadedFile)

            # Process the data
            numericData = tools.removeZero(tools.removeNAN(data))
            cleanData = tools.corrNoNAN(numericData.corr())

            st.experimental_rerun()

elif selectedPage == 'Heatmap and Correlation Matrix':
    st.title('Matrix Heatmap and Correlation Matrix')
    st.subheader('Heatmap of Correlation Matrix')

    # Dropdown for selecting heatmap color theme
    with st.expander("Customize Heatmap"):
        colorTheme = st.selectbox(
            'Select Heatmap Color Theme',
            ['coolwarm', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Spectral', 'RdYlGn']
        )

    # Larger figure size for the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cleanData, linewidths=1, linecolor='black', center=1, cmap=colorTheme)
    st.pyplot(plt)

    st.subheader('Raw Correlation Matrix')
    st.write(cleanData)

elif selectedPage == 'Raw Data':
    st.title('Raw Data')
    st.write('View and edit the original data file: ')
    st.data_editor(data)

elif selectedPage == 'Scatterplot (Two Var.)':
    st.title('Scatterplot of Data')

    xAxisColumn = st.selectbox('Select X-axis:', cleanData.columns)
    yAxisColumn = st.selectbox('Select Y-axis:', cleanData.columns)

    fig, ax = plt.subplots(figsize=(17, 7))
    sns.scatterplot(x=xAxisColumn, y=yAxisColumn, data=cleanData)
    plt.xlabel(xAxisColumn)
    plt.ylabel(yAxisColumn)
    st.pyplot(plt)

    # Display summary statistics
    st.subheader('Summary Stats')

    # Calculate and display relevant summary stats for X
    xMean = np.mean(cleanData[xAxisColumn])
    xMedian = np.median(cleanData[xAxisColumn])
    xStdDev = np.std(cleanData[xAxisColumn])

    # Calculate and display relevant summary stats for Y
    yMean = np.mean(cleanData[yAxisColumn])
    yMedian = np.median(cleanData[yAxisColumn])
    yStdDev = np.std(cleanData[yAxisColumn])

    # Calculate and display correlation coefficient
    correlationCoefficient = cleanData[xAxisColumn].corr(cleanData[yAxisColumn])

    # Create data frames for summary stats
    xSummaryStatsDf = pd.DataFrame({
        'X Mean': [xMean],
        'X Median': [xMedian],
        'X Standard Deviation': [xStdDev]
    })

    ySummaryStatsDf = pd.DataFrame({
        'Y Mean': [yMean],
        'Y Median': [yMedian],
        'Y Standard Deviation': [yStdDev]
    })

    correlationDf = pd.DataFrame({
        'Correlation Coefficient': [correlationCoefficient]
    })

    # Display X and Y summary stats, and correlation coefficient side by side
    col1, col2, col3 = st.columns(3)
    col1.table(xSummaryStatsDf)
    col2.table(ySummaryStatsDf)
    col3.table(correlationDf)

elif selectedPage == 'Histogram (Single Var.)':
    st.title('Histogram of Data')

    xAxisColumn = st.selectbox('Select X-axis:', cleanData.columns)
    kdeCheckbox = st.checkbox('Show KDE (Kernel Density Estimation)')

    fig, ax = plt.subplots(figsize=(17, 7))
    sns.histplot(x=xAxisColumn, data=cleanData, kde=kdeCheckbox)
    plt.xlabel(xAxisColumn)
    st.pyplot(plt)
    st.subheader('Summary Stats')

    # Calculate and display relevant summary stats
    meanVal = np.mean(cleanData[xAxisColumn])
    medianVal = np.median(cleanData[xAxisColumn])
    stdDev = np.std(cleanData[xAxisColumn])

    summaryStatsDf = pd.DataFrame({
        'Mean': [meanVal],
        'Median': [medianVal],
        'Standard Deviation': [stdDev]
    })

    st.table(summaryStatsDf)

elif selectedPage == 'Data Quality Report':
    st.title('Data Quality Report')
    st.write('Useful visual and numerical summaries all in one place.')

    # Missing Values Analysis
    with st.expander('Missing Values'):
        st.subheader('Missing Values Analysis')
        st.write('Null values represented as percentages')
        missingValues = data.isnull().sum()
        missingPercentage = (missingValues / len(data)) * 100
        missingDF = pd.DataFrame({'Count': missingValues, 'Percentage': missingPercentage})
        st.table(missingDF)

    # Unique Values Count
    with st.expander('Unique Values'):
        st.subheader('Unique Values Count')
        st.write('Total number of unique values for each feature')
        uniqueValues = data.nunique()
        st.table(uniqueValues)

    # Outliers Detection - Example using Box Plot
    with st.expander('Outlier Detection'):
        st.subheader('Outliers Detection (Box Plot)')
        st.write('Visual outlier analysis')
        numericalColumns = data.select_dtypes(include=[np.number]).columns
        selectedNumCols = st.selectbox('Select Column for Outlier Analysis', numericalColumns)
        fig, ax = plt.subplots(figsize=(17, 7))
        sns.boxplot(data[selectedNumCols])
        st.pyplot(plt)

    # Data Distribution Analysis
    with st.expander('Distribution Analysis'):
        st.subheader('Data Distribution Analysis')
        st.write('Visual distribution analysis')
        selectedDistCol = st.selectbox('Select Column for Distribution Analysis', numericalColumns)
        fig, ax = plt.subplots(figsize=(17, 7))
        sns.histplot(data[selectedDistCol])
        st.pyplot(plt)

    # Data Type Summary
    with st.expander('Data Type Summary'):
        st.subheader('Data Type Summary')
        st.table(data.dtypes)

elif selectedPage == 'Regression Analysis':
    st.title('Regression Analysis')

    # Dropdown to select the type of regression
    regressionType = st.selectbox('Select Regression Type:', ['Linear Regression', 'Logistic Regression'])

    # Dropdown to select the target variable
    targetVar = st.selectbox('Select Target Variable:', cleanData.columns)

    # Multiselect for predictor variables
    predictorVars = st.multiselect('Select Predictor Variables:', cleanData.columns, default=cleanData.columns[0])

    # Button to perform regression
    if st.button('Perform Regression'):
        # Perform the regression based on the type
        if regressionType == 'Linear Regression':
            # Linear regression code
            X = cleanData[predictorVars]
            y = cleanData[targetVar]

            # Adding constant for statsmodels
            X = sm.add_constant(X)

            # fit the model
            model = sm.OLS(y, X).fit()

            # plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.regplot(x=X.iloc[:, 1], y=y, data=data, ax=ax)

            # plot and write
            for predictor in predictorVars:
                fig = px.scatter(cleanData, x=predictor, y=targetVar, trendline="ols")
                st.plotly_chart(fig)

            st.write(model.summary())
        else:
            # Logistic regression code
            X = cleanData[predictorVars]
            y = cleanData[targetVar]

            # Adding constant for statsmodels
            X = sm.add_constant(X)

            model = sm.Logit(y, X).fit()

            # plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=X.iloc[:, 1], y=y, ax=ax)  # Actual data points
            sns.lineplot(x=X.iloc[:, 1], y=model.predict(sm.add_constant(X)), color='red', ax=ax)  # Predicted probabilities

            # plot and write
            st.write(model.summary())
            st.pyplot(fig)
