# Back-testing script code documentation

## Import Libraries

```python 
# Import libraries
import pandas as pd
from pandas.tseries.offsets import BMonthEnd, BDay
import numpy as np
import glob, os

import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display

import mplfinance as fplt
```

This section imports the required libraries for data processing, visualization, and interactive widgets. Interactive widgets are used only in **.ipynb** version of code.


## Load all ETFs Data

```python 
files = glob.glob('InputData/*.csv')
df = pd.concat([pd.read_csv(fp, parse_dates=True).assign(Title=os.path.basename(fp).split('.')[0]) 
       for fp in files])
```

This code loads data from multiple CSV files located in the **"InputData"** directory into a Pandas DataFrame. Each file represents a different ETF (Exchange-Traded Fund) and is assigned a title based on the file name.


Data Preprocessing

```python 
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.rename(columns={'Adj Close':'AdjClose'}, inplace=True)
```

These lines preprocess the data by converting the **'Date'** column to datetime format, setting it as the DataFrame index, and renaming a column.


## Calculation of Daily Returns

```python 
df['RDaily'] = df.groupby('Title')['AdjClose'].pct_change(1)
```
This line of code calculates the daily returns for each ETF in the DataFrame df. The **pct_change(1)** function is applied to the **'AdjClose'** column after grouping the DataFrame by the **'Title'** column. The resulting daily returns are stored in a new column called **'RDaily'**.


## Monthly Aggregation of Standard Deviation

```python 
tmp = []

grouped_df = df.groupby(['Title', df.index.year, df.index.month])

for (title, year, month), group in grouped_df:
    std_value = group['RDaily'].std()
    inv_std_value = 1/group['RDaily'].std()
    tmp.append([title, year, month, std_value, inv_std_value])

df_mthly = pd.DataFrame(tmp, columns=['Title', 'Year', 'Month', 'StdValue', 'InvStdVal'])
```

This section of code aggregates the standard deviation of daily returns (**'RDaily'**) for each ETF on a monthly basis.

- A list tmp is initialized to store the calculated values.
- The DataFrame df is grouped by the combination of **'Title'**, **index.year**, and **index.month** using the **groupby()* function.
- For each group, the standard deviation of daily returns is calculated using the **std()** function applied to the 'RDaily' column of the group. The result is stored in the variable **std_value**.
- The inverse of the standard deviation value is calculated as **1/std_value** and stored in the variable **inv_std_value**.
- The calculated values, along with the corresponding ETF title, year, and month, are appended as a list to **tmp**.
- Finally, a new DataFrame **df_mthly** is created from **tmp**, with columns labeled as **'Title'**, **'Year'**, **'Month'**, **'StdValue'**, and **'InvStdVal'**.

This code allows for the calculation and aggregation of monthly standard deviation and inverse standard deviation values for each ETF.


## Pivot and Weight Calculation

```python 
df_mthly_pivot = df_mthly.pivot(index=['Year','Month'], columns="Title", values="InvStdVal")
df_mthly_pivot['Sum_InvStdVal'] = df_mthly_pivot['EEM'] + df_mthly_pivot['GLD'] + df_mthly_pivot['SPY'] + df_mthly_pivot['TLT'] + df_mthly_pivot['VGK']

df_mthly_pivot['W_EEM'] = df_mthly_pivot['EEM'] / df_mthly_pivot['Sum_InvStdVal']
df_mthly_pivot['W_GLD'] = df_mthly_pivot['GLD'] / df_mthly_pivot['Sum_InvStdVal']
df_mthly_pivot['W_SPY'] = df_mthly_pivot['SPY'] / df_mthly_pivot['Sum_InvStdVal']
df_mthly_pivot['W_TLT'] = df_mthly_pivot['TLT'] / df_mthly_pivot['Sum_InvStdVal']
df_mthly_pivot['W_VGK'] = df_mthly_pivot['VGK'] / df_mthly_pivot['Sum_InvStdVal']
```

This section of code performs pivot operations on the **df_mthly** DataFrame and calculates the weights for different ETFs based on their inverse standard deviation values.

- The **pivot()** function is applied to **df_mthly** with the index set as **['Year', 'Month']**, the columns set as **'Title'**, and the values set as **'InvStdVal'**. This operation reshapes the DataFrame, creating a new DataFrame **df_mthly_pivot** where each ETF title becomes a column, and the values are the corresponding inverse standard deviation values.
- A new column **'Sum_InvStdVal'** is created in **df_mthly_pivot** by summing the inverse standard deviation values of all the ETFs for each month.
- The weights for each ETF are calculated by dividing the individual inverse standard deviation values by the **'Sum_InvStdVal'** for the corresponding month. The resulting weights are stored in columns **'*W_EEM'**, **'W_GLD'**, **'W_SPY'**, **'W_TLT'**, and **'W_VGK'**.







```python 

```













```python 

```







```python 

```







```python 

```










```python 

```









```python 

```
















