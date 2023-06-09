# -*- coding: utf-8 -*-
"""
Created on Thu Jun 8 22:23:32 2023

@author: domingosdeeulariadumba
"""


""" Libraries for displaying imported images"""

from IPython.display import display
from PIL import Image


""" SUMMARY """


# Scope

display(Image.open(""C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/AngolaHDIvsEducation&HealthSpendingAnalysis/idh.scope.png""))


# Business Problem

display(Image.open(""C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/AngolaHDIvsEducation&HealthSpendingAnalysis/idh.problem.png"))


# Goals

display(Image.open("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/AngolaHDIvsEducation&HealthSpendingAnalysis/idh.goals"))


# Approach

display(Image.open("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/AngolaHDIvsEducation&HealthSpendingAnalysis/idh.approach.png"))

# Tools

display(Image.open("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/AngolaHDIvsEducation&HealthSpendingAnalysis/idh.tools.png"))



""" Importing the required libraries """


# For EDA and Plotting

import os
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as mpl
mpl.style.use('ggplot')

# To extract table from image 

from img2table.document import Image
from img2table.ocr import TesseractOCR as tsr


# To analyse possible relationship between the variables

from sklearn.linear_model import LinearRegression as lreg
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split as tts, cross_validate as cv


# To check the statistical significance

import statsmodels.api as sm


# To connect with the DB

import pyodbc


# To ignore warnings

import warnings
warnings.filterwarnings('ignore')




"""" EXPLORATORY DATA ANALYSIS """


# Reading the printed screen containing the HDI table (in portuguese)

ocr=tsr(lang="por")

idh_table=Image("C:/Users/domingosdeeularia/Desktop/OGE/idh_PrtScr_edited.JPG")

idh_table.to_xlsx('C:/Users/domingosdeeularia/Desktop/OGE/IDH.xlsx', ocr=ocr)

df_idh=pd.read_excel('C:/Users/domingosdeeularia/Desktop/OGE/IDH.xlsx')


# Removing the xlsx file from directory, since it's not needed any longer

os.remove(r'C:/Users/domingosdeeularia/Desktop/OGE/IDH.xlsx')


# Extracting and cleaning the HDI column

        """
        reordering the HDI values
        """
df_idh=df_idh.sort_values(by='Data', ascending=True)


        """
        Taking the HDI column from the original dataframe
        """
df_HDI=df_idh['IDH']


        """
        changing the  decimal separator ',' to '.'
        and correcting OCR misreading
        """
df_HDI=df_HDI.replace(',','.', regex=True)

df_HDI=df_HDI.replace('A','4', regex=True)


# Connecting the DB to import the State Budget table

connection=pyodbc.connect('Driver={SQL Server};'
                          'Server=DOMINGOSDEEULAR\SQLEXPRESS;'
                          'Database=HealthandEducationStateBudget;'
                          'Trusted_Connection=yes;')

df_budget=pd.read_sql_query('SELECT * FROM budget', connection)


# Concataneting the HDI and BUDGET dataframes, and renaming the former column

DF_HDI=pd.concat([df_budget,df_HDI], axis=1)

DF_HDI=DF_HDI.rename({'IDH':'HDI'}, axis=1)


# Checkink the data type

DF_HDI.dtypes


# Converting HDI column to float

DF_HDI['HDI'] = DF_HDI['HDI'].astype(float)


# Checking again the data type

DF_HDI.dtypes


# Limiting the number of decimal places to two

DF_HDI=DF_HDI.round(decimals=2)


# Statistical summary of the dataframe

DF_HDI.describe()


# Showing the main plotting combination of the dataframe

sb.pairplot(DF_HDI)
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/AngolaHDIvsEducation&HealthSpendingAnalysis/pairplot.png")
mpl.close()

# Displaying the Displot

columns1 = ['GovBudg_Health','GovBudg_Education', 'HDI']
for i in columns1:
    sb.displot(DF_HDI[i])
    mpl.xticks(rotation = 25)
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/AngolaHDIvsEducation&HealthSpendingAnalysis/Displot.png")
mpl.close()

# KDE plot

columns2 = ['GovBudg_Health','GovBudg_Education', 'HDI']
for i in columns2:
    mpl.figure()
    sb.kdeplot(x=DF_HDI[i],shade=True)
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/AngolaHDIvsEducation&HealthSpendingAnalysis/kdeplot.png")
mpl.close()

# Correlation heatmap

sb.heatmap(DF_HDI[['GovBudg_Health','GovBudg_Education', 'HDI']].corr(), annot=True, cmap='inferno')
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/AngolaHDIvsEducation&HealthSpendingAnalysis/correlationmap.png")
mpl.close()

# Saving the dataframe as a csv file to later be used on Power BI

DF_HDI.to_excel('C:/Users/domingosdeeularia/Desktop/OGE/IDH_govbudgDataset.xlsx')

        """
        this file was further used on Power BI to proceed the analysis
        """


# Creating a new dataframe

    """
    we'll now make a combined analysis of Education and Health spending, as 
    independent variable, and HDI as the target, since we want to study any
    possible (linear) relationship between these two measures.
    
    """

DF_final=DF_HDI.copy()

DF_final['Health & Education Spendig']=DF_HDI['GovBudg_Health']+DF_HDI['GovBudg_Health']

DF_final=DF_final.drop(['GovBudg_Health','GovBudg_Education','Year'], axis=1)
DF_final

# Plotting the data points of this new data frame 

sb.scatterplot(data=DF_final,x='Health & Education Spendig',y='HDI', palette='tab10')
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/AngolaHDIvsEducation&HealthSpendingAnalysis/FinalScatterPlot.png")
mpl.close()

# Setting the independent and target variables

x=DF_final['Health & Education Spendig'].to_numpy()

y=DF_final['HDI']


# Splitting the data into train and test sets

X=x.reshape(-1,1)

X_train, X_test, y_train, y_test = tts(X,y, test_size=0.2, random_state=1)


# Printing the shape of the train and test sets

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# Training the regression model

LINREG=lreg()
LINREG.fit(X_train, y_train)


# Evaluating the regression model on test set

y_pred=LINREG.predict(X_test)

print('Test set (RMSE):', mean_squared_error(y_test, y_pred, squared=False))

print('Coefficient of determination (R^2):', r2_score(y_test, y_pred))
    
      """
    From r2_score we can infer that a linear regresssion model is not appropriate
    to describe a relationship between HDI and Spending of Angolan government
    on Health and Educatin combined.
    
    In the end we'll see how this line fits the data points, but firstly, let
    us analyse the statistical significance of the chosen model.
    """

# Statistical summary with statsmodels

X_train_=X_train
y_train_=y_train
X_train_=sm.add_constant(X_train_)
Reg=sm.OLS(y_train_, X_train_).fit()
Reg.summary()


    """
    From this summary, as we can see, the   p-value is greater than 0.05 (0.07), 
    which means the dependent variable does not affect the target, with 95% of
    confidence.
    """

# Plotting the regression line

Scores= cv(LINREG, X_train, y_train, scoring='neg_root_mean_squared_error', return_estimator=(True))
sb.scatterplot(data=DF_final, x='Health & Education Spendig', y='HDI')
mpl.plot(X, Scores['estimator'][0].predict(X))
mpl.ylabel('HDI')
mpl.xlabel('Health & Education Spendig')
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/AngolaHDIvsEducation&HealthSpendingAnalysis/Regression_Line.png")
mpl.close()
______________________________________end___________________________________