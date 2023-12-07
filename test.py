import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import statsmodels.api as sm
import streamlit as st

import Functions as tools

# Initialize data as None in session state
if 'data' not in st.session_state:
    st.session_state.data = None

# Configure layout
st.set_page_config(page_title="Wayfinder",
                   page_icon=":bar_chart:",
                   layout="wide"
                   )
st.sidebar.title("Wayfinder")
st.sidebar.subheader("Tools: ")

# Available pages
selectedPage = st.sidebar.radio('Available: ',
                                ['Home', 'Heatmap and Correlation Matrix', 'Scatterplot (Two Var.)',
                                 'Histogram (Single Var.)', 'Regression Analysis', 'Data Quality Report', 'Raw Data'])

# Prepare file upload
if st.session_state.data is not None:
    forUpload = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if forUpload is not None:
        # Load the uploaded data
        st.session_state.data = pd.read_csv(forUpload)

        # Process the data
        st.session_state.numericData = tools.removeZero(tools.removeNAN(st.session_state.data))
        st.session_state.cleanData = tools.corrNoNAN(st.session_state.numericData.corr())

if selectedPage == 'Home':
    if st.session_state.data is None:
        st.title("Welcome")
        st.subheader("Select a CSV file for upload")
        st.write("")
        # Upload CSV
        uploadedFile = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploadedFile is not None:
            # Load the uploaded data
            st.session_state.data = pd.read_csv(uploadedFile)

            # Process the data
            st.session_state.numericData = tools.removeZero(tools.removeNAN(st.session_state.data))
            st.session_state.cleanData = tools.corrNoNAN(st.session_state.numericData.corr())
            st.rerun()

    if st.session_state.data is not None:
        with st.spinner("Setting up..."):
            # Display Current Date
            currentDate = datetime.datetime.now().strftime("%B %d, %Y")  # Format: Month day, Year
            st.subheader(currentDate)

            # Configure columns
            col1, col2 = st.columns(2)
            col1.header('Observation Count:')
            col1.subheader(st.session_state.data.shape[0])

            st.write('Most Recent Samples: ')
            recentData = st.session_state.data.tail(5)  # get the last 5 items from the dataframe
            st.table(recentData)

            st.subheader('Strongest Relationships: ')
            with st.expander("Set Number of Rows"):
                cutoff = st.slider("Select Number Cutoff:", 1, st.session_state.cleanData.shape[0], 5, 1)
                strongestRelations = tools.findStrongest(st.session_state.cleanData, cutoff)
            st.write("Top Correlations with Cutoff of:", cutoff)
            st.table(strongestRelations)

elif selectedPage == 'Heatmap and Correlation Matrix':
    if st.session_state.data is not None:
        st.title('Matrix Heatmap and Correlation Matrix')
        st.write("Get an overview of your data, extract broad trends and relationships at a glance")
        st.subheader('Heatmap of Correlation Matrix')

        # Dropdown for selecting heatmap color theme
        with st.expander("Customize Heatmap"):
            colorTheme = st.selectbox(
                'Select Heatmap Color Theme',
                ['coolwarm', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Spectral', 'RdYlGn']
            )

        with st.spinner("Generating Heatmap..."):
            # Larger figure size for the heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(st.session_state.cleanData, linewidths=1, linecolor='black', center=1, cmap=colorTheme)
            st.pyplot(plt)

        st.subheader('Raw Correlation Matrix')
        st.write("Explore the raw data points, hone in on specific values")
        st.write(st.session_state.cleanData)
    else:
        tools.uploadFile()

elif selectedPage == 'Raw Data':
    if st.session_state.data is not None:
        st.title('Raw Data')
        st.write('View and edit the original data file: ')
        currentData = st.data_editor(st.session_state.data)

        download = st.button("Export modified data")
        if download:
            fileName = tools.generateTextReport('rawData')
            currentData.to_csv(fileName, index=False)
    else:
        tools.uploadFile()

elif selectedPage == 'Scatterplot (Two Var.)':
    if st.session_state.data is not None:
        st.title('Scatterplot of Data')
        st.write('Explore two variable relationships in your data.')

        # Extract x and y from our columns
        xAxisColumn = st.selectbox('Select X-axis:', st.session_state.cleanData.columns)
        yAxisColumn = st.selectbox('Select Y-axis:', st.session_state.cleanData.columns)

        # Plot it
        with st.spinner("Plotting..."):
            with st.expander("Customize Plot"):
                # Checkboxes and dropdowns for settings
                color = st.checkbox('Color mapping')
                if color:
                    colorTheme = st.selectbox('Hue Variable', st.session_state.cleanData.columns)
                else:
                    colorTheme = None

                size = st.checkbox('Size mapping')
                if size:
                    sizeVar = st.selectbox('Size Variable', st.session_state.cleanData.columns)
                else:
                    sizeVar = None

            fig, ax = plt.subplots(figsize=(17, 7))
            sns.scatterplot(x=xAxisColumn, y=yAxisColumn, data=st.session_state.cleanData, hue=colorTheme, size=sizeVar)
            plt.xlabel(xAxisColumn)
            plt.ylabel(yAxisColumn)
            st.pyplot(plt)
            currentGraph = plt

            download = st.button("Save plot")
            if download:
                fileName = tools.generateFileName('scatterplot')
                currentGraph.savefig(fileName, format="pdf")

        # Display summary statistics
        st.subheader('Summary Stats')

        # Calculate and display relevant summary stats for X
        xMean = np.mean(st.session_state.cleanData[xAxisColumn])
        xMedian = np.median(st.session_state.cleanData[xAxisColumn])
        xStdDev = np.std(st.session_state.cleanData[xAxisColumn])

        # Calculate and display relevant summary stats for Y
        yMean = np.mean(st.session_state.cleanData[yAxisColumn])
        yMedian = np.median(st.session_state.cleanData[yAxisColumn])
        yStdDev = np.std(st.session_state.cleanData[yAxisColumn])

        # Calculate and display correlation coefficient
        correlationCoefficient = st.session_state.cleanData[xAxisColumn].corr(st.session_state.cleanData[yAxisColumn])

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
    else:
        tools.uploadFile()

elif selectedPage == 'Histogram (Single Var.)':
    if st.session_state.data is not None:
        st.title('Histogram of Data')
        st.write('Explore the distribution of single variables.')

        # Extract x and y from our columns
        xAxisColumn = st.selectbox('Select X-axis:', st.session_state.cleanData.columns)
        kdeCheckbox = st.checkbox('Show KDE (Kernel Density Estimation)')

        # Plot
        with st.spinner("Plotting..."):
            fig, ax = plt.subplots(figsize=(17, 7))
            sns.histplot(x=xAxisColumn, data=st.session_state.cleanData, kde=kdeCheckbox)
            plt.xlabel(xAxisColumn)
            st.pyplot(plt)
            currentGraph = plt

            download = st.button("Save plot")
            if download:
                fileName = tools.generateFileName('histogram')
                currentGraph.savefig(fileName, format="pdf")

        # Calculate and display relevant summary stats
        st.subheader('Summary Stats')
        meanVal = np.mean(st.session_state.cleanData[xAxisColumn])
        medianVal = np.median(st.session_state.cleanData[xAxisColumn])
        stdDev = np.std(st.session_state.cleanData[xAxisColumn])

        summaryStatsDf = pd.DataFrame({
            'Mean': [meanVal],
            'Median': [medianVal],
            'Standard Deviation': [stdDev]
        })

        st.table(summaryStatsDf)
    else:
        tools.uploadFile()

elif selectedPage == 'Data Quality Report':
    if st.session_state.data is not None:
        st.title('Data Quality Report')
        st.write('Useful visual and numerical summaries all in one place.')

        # Missing Values Analysis
        with st.expander('Missing Values'):
            st.subheader('Missing Values Analysis')
            st.write('Null values represented as percentages')
            missingValues = st.session_state.data.isnull().sum()
            missingPercentage = (missingValues / len(st.session_state.data)) * 100
            missingDF = pd.DataFrame({'Count': missingValues, 'Percentage': missingPercentage})
            st.table(missingDF)

        # Unique Values Count
        with st.expander('Unique Values'):
            st.subheader('Unique Values Count')
            st.write('Total number of unique values for each feature')
            uniqueValues = st.session_state.data.nunique()  # Get unique items
            st.table(uniqueValues)

        # Outliers Detection using boxplot
        with st.expander('Outlier Detection'):
            st.subheader('Outliers Detection (Box Plot)')
            st.write('Visual outlier analysis')
            numericalColumns = st.session_state.data.select_dtypes(include=[np.number]).columns  # Numerical columns
            selectedNumCols = st.selectbox('Select Column for Outlier Analysis', numericalColumns)
            fig, ax = plt.subplots(figsize=(17, 7))
            sns.boxplot(st.session_state.data[selectedNumCols])
            st.pyplot(plt)

        # Data Distribution Analysis
        with st.expander('Distribution Analysis'):
            st.subheader('Data Distribution Analysis')
            st.write('Visual distribution analysis')
            selectedDistCol = st.selectbox('Select Column for Distribution Analysis', numericalColumns)
            fig, ax = plt.subplots(figsize=(17, 7))
            sns.histplot(st.session_state.data[selectedDistCol])
            st.pyplot(plt)

        # Data Type Summary
        with st.expander('Data Type Summary'):
            st.subheader('Data Type Summary')
            st.table(st.session_state.data.dtypes)
    else:
        tools.uploadFile()

elif selectedPage == 'Regression Analysis':
    if st.session_state.data is not None:
        st.title('Regression Analysis')
        st.write('Plot and explore predictions in your data')

        # Select the type of regression
        regressionType = st.selectbox('Select Regression Type:', ['Linear Regression', 'Logistic Regression'])
        # Select the target variable
        targetVar = st.selectbox('Select Target Variable:', st.session_state.cleanData.columns)

        # Predictor variables
        predictorVars = st.multiselect('Select Predictor Variables:', st.session_state.cleanData.columns,
                                       default=st.session_state.cleanData.columns[0])

        if st.button('Perform Regression'):
            if regressionType == 'Linear Regression':
                # Linear regression
                X = st.session_state.cleanData[predictorVars]
                y = st.session_state.cleanData[targetVar]

                # Adding constant for statsmodels
                X = sm.add_constant(X)

                # fit the model
                model = sm.OLS(y, X).fit()

                # plot
                with st.spinner("Plotting..."):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.regplot(x=X.iloc[:, 1], y=y, data=st.session_state.data, ax=ax)

                # plot and write
                for predictor in predictorVars:
                    fig = px.scatter(st.session_state.cleanData, x=predictor, y=targetVar, trendline="ols")
                    st.plotly_chart(fig)

                st.write(model.summary())
            else:
                # Logistic regression
                X = st.session_state.cleanData[predictorVars]
                y = st.session_state.cleanData[targetVar]

                # Adding constant for statsmodels
                X = sm.add_constant(X)

                model = sm.Logit(y, X).fit()

                # plot
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(x=X.iloc[:, 1], y=y, ax=ax)  # Actual data points
                sns.lineplot(x=X.iloc[:, 1], y=model.predict(sm.add_constant(X)), color='red',
                             ax=ax)  # Predicted probabilities

                # plot and write
                st.write(model.summary())
                st.pyplot(fig)
            # More here if I add more models
    else:
        tools.uploadFile()
