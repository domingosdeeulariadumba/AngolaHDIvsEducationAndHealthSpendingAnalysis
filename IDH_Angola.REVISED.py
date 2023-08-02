# -*- coding: utf-8 -*-
"""
Created on Thu Jun 8 22:23:32 2023

@author: domingosdeeulariadumba
"""


""" Importing the required libraries """


# For EDA and Plotting

import pandas as pd
import seaborn as sb
import matplotlib.pyplot as mpl
mpl.style.use('ggplot')


# To analyse possible relationship between the variables

from sklearn.linear_model import LinearRegression as lreg
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split as tts
from xgboost import XGBRegressor as xgbreg


# To connect with the DB

import pyodbc


# To ignore warnings

import warnings
warnings.filterwarnings('ignore')




"""" EXPLORATORY DATA ANALYSIS """


# Reading the printed screen containing the HDI table (in portuguese)

df_idh = pd.read_excel('C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/AngolaHDIvsEducation&HealthSpendingAnalysis/IDH.xlsx')


# Extracting and cleaning the HDI column


        """
        Taking the HDI column from the original dataframe
        """
df_HDI=df_idh['IDH']



# Connecting the DB to import the State Budget table

connection=pyodbc.connect('Driver={SQL Server};'
                          'Server=DOMINGOSDEEULAR\SQLEXPRESS;'
                          'Database=HealthandEducationStateBudget;'
                          'Trusted_Connection=yes;')

df_budget=pd.read_sql_query('SELECT * FROM budget', connection)


# Concatenating the HDI and BUDGET dataframes, and renaming the former column

DF_HDI=pd.concat([df_budget,df_HDI], axis=1)

DF_HDI=DF_HDI.rename({'IDH':'HDI'}, axis=1)


# Checkink the data type

DF_HDI.dtypes


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
    mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/AngolaHDIvsEducation&HealthSpendingAnalysis/{0}_Displot.png".format(i))
    mpl.close()

# KDE plot

columns2 = ['GovBudg_Health','GovBudg_Education', 'HDI']
for i in columns2:
    mpl.figure()
    sb.kdeplot(x=DF_HDI[i],shade=True)
    mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/AngolaHDIvsEducation&HealthSpendingAnalysis/{0}_kdeplot.png".format(i))
    mpl.close()

# Correlation heatmap

sb.heatmap(DF_HDI[['GovBudg_Health','GovBudg_Education', 'HDI']].corr(), annot=True, cmap='inferno')
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/AngolaHDIvsEducation&HealthSpendingAnalysis/correlationmap.png")
mpl.close()

# Saving the dataframe as a csv file to later be used on Power BI

DF_HDI.to_excel('C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/AngolaHDIvsEducation&HealthSpendingAnalysis/IDH_govbudgDataset.xlsx')

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
    From r2_score we can infer that a linear regresssion model is not 
    appropriate to describe a relationship between HDI and Spending of Angolan
    government on Health and Educatin combined.
    
    Additionally we'll use the XGBoost library to check any possible linear
    relationship between the two attributes.
    """
XGBreg = xgbreg()
XGBreg.fit(X_train, y_train)
y_pred1 = XGBreg.predict(y_test)
 
print('Test set (RMSE):', mean_squared_error(y_test, y_pred1, squared=False))
print('NRMSE:', (mean_squared_error(y_test, y_pred1, squared=False)/(y.max()-y.min())))
print('Coefficient of determination (R^2):', r2_score(y_test, y_pred1))


    """
   Running the lines above, it is noticed that the XGBoost regressor, also, 
   fails to find a linear relationship between the variables due to the
   the data points dispersion
    """
______________________________________end___________________________________