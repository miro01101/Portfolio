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


## Weight Summation Check

```python 
df_mthly_pivot['W_CHECK_SUM'] = df_mthly_pivot['W_EEM'] + df_mthly_pivot['W_GLD'] + df_mthly_pivot['W_SPY'] + df_mthly_pivot['W_TLT'] + df_mthly_pivot['W_VGK']
```

This line of code calculates the sum of the weights for each ETF in **df_mthly_pivot** and stores the result in a new column **'W_CHECK_SUM'**.

This code segment allows for the creation of a pivot table from the **df_mthly** DataFrame and calculates the weights for each ETF based on their inverse standard deviation values. Additionally, it performs a check by summing the weights to ensure they add up to 1.


## DataFrame Unpivoting and Filtering

```python 
df_mthly_unpivot = df_mthly_pivot.stack(level=(0)).reset_index()
df_mthly_unpivot = df_mthly_unpivot[df_mthly_unpivot["Title"].isin(["W_EEM", "W_GLD", 'W_SPY', 'W_TLT', 'W_VGK'])].copy()
df_mthly_unpivot.columns.values[-1] = "W"
df_mthly_unpivot['Title'] = df_mthly_unpivot['Title'].str.replace('W_', '')
```

This section of code performs the unpivoting operation on the df_mthly_pivot DataFrame and applies some additional data filtering and column transformations.

- The **stack()** function is called on **df_mthly_pivot** with **level=(0)**. This operation reshapes the DataFrame by stacking the columns, resulting in a multi-level index. The stacking is performed on the first level, which corresponds to the combination of **'Year'** and **'Month'**. The resulting DataFrame is stored in **df_mthly_unpivot**.
- The **reset_index()** function is then called on df_mthly_unpivot to convert the multi-level index into separate columns.
- The DataFrame **df_mthly_unpivot** is further filtered to include only rows where the **"Title"** column has values matching **'W_EEM'**, **'W_GLD'**, **'W_SPY'**, **'W_TLT'**, or **'W_VGK'**. The **.isin()** function is used to perform this filtering.
- The **.copy()** function is called to create a copy of the filtered DataFrame, ensuring that subsequent modifications are performed on the new DataFrame.
- The last column name in **df_mthly_unpivot** is changed to **'W'** using the **columns.values** attribute.
- The values in the **'Title'** column of **df_mthly_unpivot** are transformed by removing the prefix **'W_'** using the **str.replace()** function.

This code segment transforms the **df_mthly_pivot** DataFrame into a more structured format by unpivoting it. It then filters the data to include only the specified ETFs and performs column name modifications. The resulting DataFrame, **df_mthly_unpivot**, provides a more concise representation of the ETF weights.


## DataFrame Merging

```python 
df['Title'] = df['Title'].astype(str)
df_mthly_unpivot['Title'] = df_mthly_unpivot['Title'].astype(str)

df_comp = pd.merge(df.reset_index(), df_mthly_unpivot.reset_index(), how='left', 
                   left_on=[df.reset_index()['Date'].dt.year, df.reset_index()['Date'].dt.month, 'Title'], 
                   right_on=['Year', 'Month', 'Title'])
```

This section of code performs a merge operation between two DataFrames, **df** and **df_mthly_unpivot**, based on specific columns. Before merging, the code also converts the **'Title'** columns of both DataFrames to the string data type.

- The **'Title'** column of df is converted to the string data type using the **astype()** function.
- Similarly, the **'Title'** column of **df_mthly_unpivot** is also converted to the string data type.

The **pd.merge()** function is then called to merge the two DataFrames based on specified columns and the specified merge type (how=**'left'**). Here's a breakdown of the merge arguments:

- **df.reset_index()** is used to reset the index of df and create a temporary DataFrame with the reset index values.
- **df.reset_index()['Date'].dt.year** extracts the year component from the **'Date'** column of the temporary DataFrame.
- **df.reset_index()['Date'].dt.month** extracts the month component from the **'Date'** column of the temporary DataFrame.
- **'Title'** is included as the third column in the left_on argument, representing the column from **df**.
- **'Year'** and **'Month'** are included as the first two columns in the right_on argument, representing the columns from **df_mthly_unpivot**.

The resulting merged DataFrame is stored in **df_comp**, which combines the data from df and **df_mthly_unpivot** based on matching values in the specified columns.

```python
df_comp['Date'] = pd.to_datetime(df_comp['Date'])
df_comp['Year'] = df_comp['Date'].dt.year
df_comp['Month'] = df_comp['Date'].dt.month

min_max_dates = df_comp.groupby(['Year', 'Month'])['Date'].agg([min, max]).reset_index()

df_comp = pd.merge(df_comp, min_max_dates, on=['Year', 'Month'], how='left')
```

This section of code performs various operations on the DataFrame df_comp to manipulate its columns and merge it with the min_max_dates DataFrame.

**df_comp['Date'] = pd.to_datetime(df_comp['Date'])** converts the **'Date'** column of **df_comp** to the datetime data type using the **pd.to_datetime()** function. This ensures that the column is interpreted as dates.

**df_comp['Year'] = df_comp['Date'].dt.year** extracts the year component from the **'Date'** column of df_comp using the **.dt.year** attribute of the datetime data type. The extracted year values are assigned to a new column **'Year'** in **df_comp**.

**df_comp['Month'] = df_comp['Date'].dt.month** extracts the month component from the **'Date'** column of df_comp using the **.dt.month** attribute of the datetime data type. The extracted month values are assigned to a new column **'Month'** in **df_comp**.

**min_max_dates = df_comp.groupby(['Year', 'Month'])['Date'].agg([min, max]).reset_index()** groups the rows of **df_comp** based on the **'Year'** and **'Month'** columns and applies the **.agg([min, max])** function to the **'Date'** column. This calculates the minimum and maximum dates within each group. The resulting DataFrame **min_max_dates** contains three columns: **'Year'**, **'Month'**, and two additional columns **'min'** and **'max'** representing the minimum and maximum dates, respectively.

**df_comp = pd.merge(df_comp, min_max_dates, on=['Year', 'Month'], how='left')** merges df_comp with **min_max_dates** based on the **'Year'** and **'Month'** columns. The merge is performed using a left join (**how='left'**), meaning that all rows from **df_comp** are retained even if there is no corresponding match in min_max_dates. The merged DataFrame is assigned back to **df_comp**.

The resulting **df_comp** DataFrame contains the original columns along with the added **'Year'**, **'Month'**, **'min'**, and **'max'** columns, which provide information about the minimum and maximum dates within each year and month.


```python 
result = df_comp.pivot(index=['Date', 'Year', 'Month'], columns='Title', values=['RDaily', 'W'])

result.columns = ['{}_{}'.format(col[1], col[0]) for col in result.columns]

result = result.reset_index()

new_df_comp = pd.concat([df_comp[['Date', 'Year','Month','min','max', ]], result], axis=1)

new_df_comp = new_df_comp[new_df_comp['EEM_W'].notna()]
```

This section of code performs operations on the DataFrame df_comp to pivot it, rename columns, concatenate it with another DataFrame, and filter rows based on a condition.

- **result = df_comp.pivot(index=['Date', 'Year', 'Month'], columns='Title', values=['RDaily', 'W'])** pivots the **df_comp** DataFrame using the **pivot()** method. The index is set as **['Date', 'Year', 'Month']**, the columns to pivot are **'Title'**, and the values to populate the pivoted columns are **['RDaily', 'W']**. The resulting DataFrame is assigned to result.

- **result.columns = ['{}_{}'.format(col[1], col[0]) for col in result.columns]** renames the columns of result using a list comprehension. Each column name is formatted as **'{value}_{column_name}'** where value corresponds to the second level of the original column multi-index and column_name corresponds to the first level. The modified column names are assigned back to result.columns.

- **result = result.reset_index()** resets the index of result to turn the pivot index levels into columns.

- **new_df_comp = pd.concat([df_comp[['Date', 'Year', 'Month', 'min', 'max']], result], axis=1)** concatenates the columns **'Date', 'Year', 'Month', 'min'**, and **'max'** from df_comp with the result DataFrame. The concatenation is performed along axis=1, meaning the columns are added side by side. The concatenated DataFrame is assigned to **new_df_comp**.

- **new_df_comp = new_df_comp[new_df_comp['EEM_W'].notna()]** filters the rows of **new_df_comp** based on the condition **new_df_comp['EEM_W'].notna()**. This condition checks for non-null values in the **'EEM_W'** column. The resulting DataFrame is assigned back to **new_df_comp**.

The resulting **new_df_comp** DataFrame contains the original columns from df_comp along with the pivoted columns from result. Rows are filtered to remove any rows where the **'EEM_W'** column has a null value.


```python 
def remove_dup_columns(frame):
     keep_names = set() # Set to store unique column names
     keep_icols = list() # List to store indices of columns to keep
     for icol, name in enumerate(frame.columns):
          if name not in keep_names: # If the column name is not already present in keep_names
               keep_names.add(name) # Add the column name to keep_names
               keep_icols.append(icol) # Add the index of the column to keep_icols
     return frame.iloc[:, keep_icols] # Return the DataFrame with only the columns to keep

new_df_comp = remove_dup_columns(new_df_comp)]
```

This code defines a function called **remove_dup_columns** that takes a DataFrame frame as input and removes duplicate columns, keeping only the unique ones.

The function initializes an empty set **keep_names** to store unique column names and an empty list **keep_icols** to store the indices of the columns to keep. It then iterates over the columns of the input DataFrame using **enumerate(frame.columns)**. For each column, it checks if the column name is already present in **keep_names** using the name not in **keep_names** condition. If the column name is not present, it adds the name to **keep_names** using **keep_names.add(name)** and appends the column index icol to **keep_icols** using **keep_icols.append(icol)**.

Finally, the function returns the subset of the input DataFrame frame using **.iloc[:, keep_icols]**, which selects all rows and only the columns with indices present in **keep_icols**.

The returned DataFrame is assigned to **new_df_comp**, effectively removing duplicate columns from it.


```python 
new_df_comp['CHECK'] = 'check'

new_df_comp['EEM_pos_value'] = 1
new_df_comp['GLD_pos_value'] = 1
new_df_comp['SPY_pos_value'] = 1
new_df_comp['TLT_pos_value'] = 1
new_df_comp['VGK_pos_value'] = 1

new_df_comp['SUM_all_pos_value'] = 1

initial_capital = 100
```

This code snippet adds new columns to the DataFrame new_df_comp and assigns specific values to each column.

The first line adds a new column called **'CHECK'** to the DataFrame and assigns the value **'check'** to all rows.

The subsequent lines add new columns **'EEM_pos_value'**, **'GLD_pos_value'**, **'SPY_pos_value'**, **'TLT_pos_value'**, and **'VGK_pos_value'** to the DataFrame. All these columns are assigned the value **1** for all rows.

Finally, another new column **'SUM_all_pos_value'** is added to the DataFrame, and it is assigned the value **1** for all rows.


```python 
# Loop through the rows of the DataFrame 'new_df_comp', starting from index 1
for i in range(1, len(new_df_comp)):
    # Check if we are at the beginning of a month in the first year
    if ((new_df_comp.loc[i, 'Date'] == new_df_comp.loc[i, 'min']) & (new_df_comp.loc[i, 'Date'].year == np.sort(new_df_comp['Date'].dt.year.unique())[0]) & (new_df_comp.loc[i, 'Date'].month ==  np.sort(new_df_comp.loc[(new_df_comp['Date'].dt.year == np.sort(new_df_comp['Date'].dt.year.unique())[0])]['Date'].dt.month.unique())[1])):
        # Update position values based on initial capital and daily returns
        new_df_comp.loc[i, 'EEM_pos_value'] = initial_capital*new_df_comp.loc[i, 'EEM_W']*(1+new_df_comp.loc[i, 'EEM_RDaily'])
        new_df_comp.loc[i, 'GLD_pos_value'] = initial_capital*new_df_comp.loc[i, 'GLD_W']*(1+new_df_comp.loc[i, 'GLD_RDaily'])
        new_df_comp.loc[i, 'SPY_pos_value'] = initial_capital*new_df_comp.loc[i, 'SPY_W']*(1+new_df_comp.loc[i, 'SPY_RDaily'])
        new_df_comp.loc[i, 'TLT_pos_value'] = initial_capital*new_df_comp.loc[i, 'TLT_W']*(1+new_df_comp.loc[i, 'TLT_RDaily'])
        new_df_comp.loc[i, 'VGK_pos_value'] = initial_capital*new_df_comp.loc[i, 'VGK_W']*(1+new_df_comp.loc[i, 'VGK_RDaily'])
        
        # Update the 'CHECK' column with a specific label
        new_df_comp.loc[i, 'CHECK']  = 'First year Second month start'

        # Calculate the sum of all position values
        new_df_comp.loc[i, 'SUM_all_pos_value'] = new_df_comp.loc[i, 'EEM_pos_value']+new_df_comp.loc[i, 'GLD_pos_value']+new_df_comp.loc[i, 'SPY_pos_value']+new_df_comp.loc[i, 'TLT_pos_value']+new_df_comp.loc[i, 'VGK_pos_value']

    # Check if we are at the beginning of a month
    elif ((new_df_comp.loc[i, 'Date'] == new_df_comp.loc[i, 'min'])):
        new_df_comp.loc[i, 'EEM_pos_value'] = new_df_comp.loc[i-1, 'SUM_all_pos_value']*new_df_comp.loc[i, 'EEM_W']*(1+new_df_comp.loc[i, 'EEM_RDaily'])
        new_df_comp.loc[i, 'GLD_pos_value'] = new_df_comp.loc[i-1, 'SUM_all_pos_value']*new_df_comp.loc[i, 'GLD_W']*(1+new_df_comp.loc[i, 'GLD_RDaily'])
        new_df_comp.loc[i, 'SPY_pos_value'] = new_df_comp.loc[i-1, 'SUM_all_pos_value']*new_df_comp.loc[i, 'SPY_W']*(1+new_df_comp.loc[i, 'SPY_RDaily'])
        new_df_comp.loc[i, 'TLT_pos_value'] = new_df_comp.loc[i-1, 'SUM_all_pos_value']*new_df_comp.loc[i, 'TLT_W']*(1+new_df_comp.loc[i, 'TLT_RDaily'])
        new_df_comp.loc[i, 'VGK_pos_value'] = new_df_comp.loc[i-1, 'SUM_all_pos_value']*new_df_comp.loc[i, 'VGK_W']*(1+new_df_comp.loc[i, 'VGK_RDaily'])

        new_df_comp.loc[i, 'CHECK']  = 'Start of month'

        new_df_comp.loc[i, 'SUM_all_pos_value'] = new_df_comp.loc[i, 'EEM_pos_value']+new_df_comp.loc[i, 'GLD_pos_value']+new_df_comp.loc[i, 'SPY_pos_value']+new_df_comp.loc[i, 'TLT_pos_value']+new_df_comp.loc[i, 'VGK_pos_value']

    else:
        new_df_comp.loc[i, 'EEM_pos_value'] = new_df_comp.loc[i-1, 'EEM_pos_value']*(1+new_df_comp.loc[i, 'EEM_RDaily'])
        new_df_comp.loc[i, 'GLD_pos_value'] = new_df_comp.loc[i-1, 'GLD_pos_value']*(1+new_df_comp.loc[i, 'GLD_RDaily'])
        new_df_comp.loc[i, 'SPY_pos_value'] = new_df_comp.loc[i-1, 'SPY_pos_value']*(1+new_df_comp.loc[i, 'SPY_RDaily'])
        new_df_comp.loc[i, 'TLT_pos_value'] = new_df_comp.loc[i-1, 'TLT_pos_value']*(1+new_df_comp.loc[i, 'TLT_RDaily'])
        new_df_comp.loc[i, 'VGK_pos_value'] = new_df_comp.loc[i-1, 'VGK_pos_value']*(1+new_df_comp.loc[i, 'VGK_RDaily'])

        new_df_comp.loc[i, 'CHECK']  = '*-*-*-*-*-*-*-*-*-*'

        new_df_comp.loc[i, 'SUM_all_pos_value'] = new_df_comp.loc[i, 'EEM_pos_value']+new_df_comp.loc[i, 'GLD_pos_value']+new_df_comp.loc[i, 'SPY_pos_value']+new_df_comp.loc[i, 'TLT_pos_value']+new_df_comp.loc[i, 'VGK_pos_value']
```

This code snippet implements a loop that iterates through the rows of the DataFrame **new_df_comp** and performs calculations to update position values based on specific conditions.

Loop Initialization:

The loop starts from index **1** and iterates through each row of **new_df_comp** using the range function.

First Year Second Month Start:

The code checks if the current row represents the beginning of a month in the first year. This is determined by comparing the Date column with the minimum date value and checking if it matches the first year's second month.
If the condition is satisfied, the position values (**EEM_pos_value**, **GLD_pos_value**, **SPY_pos_value**, **TLT_pos_value**, and **VGK_pos_value**) are updated based on the initial capital and daily returns.
The **CHECK** column is updated with the label **'First year Second month start'**.
The sum of all position values (**SUM_all_pos_value**) is calculated.

Start of Month:

If the previous condition is not met, the code checks if the current row represents the beginning of a month.
If true, the position values are updated based on the previous row's sum of all position values multiplied by the respective weights and daily returns.
The **CHECK** column is updated with the label **'Start of month'**.
The sum of all position values is calculated.

Regular Update:

If neither of the previous conditions is met, the code updates the position values by multiplying the previous row's position value with the respective daily return.
The **CHECK** column is updated with the label **'*-*-*-*-*-*-*-*-*-*'**.

The sum of all position values is calculated.
The loop iterates through the rows of the DataFrame and updates the position values and the **CHECK** column based on different conditions. The result is an updated DataFrame with the modified position values and corresponding labels in the **CHECK** column.



```python 
sns.lineplot(data=new_df_comp.loc[(new_df_comp['Date'].dt.year >= 2018) & (new_df_comp['Date'].dt.month >= 5)], x="Date", y="SUM_all_pos_value")
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Portfolio value over time")
plt.savefig('PortfolioValue.png')
```

This code generates a plot of the portfolio value over time, specifically after the first rebalancing, using the Seaborn library.

Plotting the Portfolio Value:

The code uses the lineplot function from the Seaborn library (**sns.lineplot**) to create a line plot.
The data for the plot is obtained from the DataFrame new_df_comp by filtering rows based on specific conditions. Only rows where the year is greater than or equal to 2018 and the month is greater than or equal to 5 are selected.
The x-axis of the plot is set to the **"Date"** column, and the y-axis represents the **"SUM_all_pos_value"** column, which denotes the portfolio value.
The xlabel function is used to set the label for the x-axis as **"Date"**, and the ylabel function sets the label for the y-axis as **"Value"**.
If desired, the title function can be uncommented and used to set the title of the plot as **"Portfolio value over time"**.
Finally, the savefig function is used to save the plot as an image file named **"PortfolioValue.png"**.
The code generates a line plot illustrating the portfolio value over time, specifically after the first rebalancing, using the specified data range. The resulting plot provides insights into the performance of the portfolio.


```python 
year_first = 2020
first_date = df[df.index.year == year_first].index[0]

year_last = 2022
last_date = df[df.index.year == year_last].index[-1]

CAGR = (new_df_comp.loc[(new_df_comp['Date'] == last_date)]['SUM_all_pos_value'].values[0] / new_df_comp.loc[(new_df_comp['Date'] == first_date)]['SUM_all_pos_value'].values[0]) ** (1/3) - 1

print('Portfolio had a CAGR of {:.2%} '.format(CAGR))
```

This code calculates the Compound Annual Growth Rate (CAGR) for a portfolio based on the data in the DataFrame **'new_df_comp'** and prints the result.

Setting the First and Last Year:

The variable **'year_first'** is set to 2020, representing the desired first year for calculating the CAGR.
The variable **'year_last'** is set to 2022, representing the desired last year for calculating the CAGR.
Finding the First and Last Dates:

The code filters the DataFrame **'df'** to select only rows where the year matches the **'year_first'** value.
The index of the first matching row is extracted using **'.index[0]'** to obtain the first date in the DataFrame for the specified year.
Similarly, the code filters **'df'** to select rows where the year matches the **'year_last'** value.
The index of the last matching row is extracted using **'.index[-1]'** to obtain the last date in the DataFrame for the specified year.

Calculating the CAGR:

The CAGR is calculated using the formula: CAGR = (Ending Value / Beginning Value)^(1/Number of Years) - 1.
The **'SUM_all_pos_value'** value for the last date is obtained by filtering **'new_df_comp'** based on the **'Date'** column matching the **'last_date'** value.
The **'SUM_all_pos_value'** value for the first date is obtained in a similar manner using the **'first_date'** value.
The CAGR is calculated by dividing the value for the last date by the value for the first date, raising it to the power of 1/3 (as we are considering a 3-year period), and then subtracting 1.

Printing the CAGR:

The calculated CAGR is printed as a percentage with two decimal places using the **'print'** function and the formatted string **'Portfolio had a CAGR of {:.2%} '**.
The code calculates the CAGR for the specified portfolio over the specified period and displays it as a percentage.


```python 
mask = (new_df_comp['Date'] >= first_date) & (new_df_comp['Date'] <= last_date)

SR = new_df_comp.loc[mask].SUM_all_pos_value.pct_change(1).mean() / new_df_comp.loc[mask].SUM_all_pos_value.pct_change(1).std()
print ('Portfolio had a Sharpe ratio of {:.2f} '.format(SR))

A_SR = len(new_df_comp)**(1/2)*SR
print ('Portfolio had a annualized Sharpe ratio of {:.2f} '.format(A_SR))
```

This code calculates the Sharpe ratio for a portfolio based on the data in the DataFrame **'new_df_comp'** and prints the result, both the standard Sharpe ratio and the annualized Sharpe ratio.

Creating a Boolean Mask:

A boolean mask named **'mask'** is created to filter the rows in **'new_df_comp'** based on the condition that the **'Date'** column should be between **'first_date'** and **'last_date'**.

Calculating the Sharpe Ratio:

The Sharpe ratio is calculated as the ratio of the mean returns to the standard deviation of returns.
The code calculates the mean returns by calling the **'pct_change(1)'** method on the **'SUM_all_pos_value'** column of **'new_df_comp'** for the rows that satisfy the **'mask'** condition, and then computes the mean using the **'mean()'** function.
Similarly, the standard deviation of returns is calculated by calling **'pct_change(1)'** on the **'SUM_all_pos_value'** column of **'new_df_comp'** for the filtered rows, followed by the **'std()'** function.
The Sharpe ratio is obtained by dividing the mean returns by the standard deviation.

Printing the Sharpe Ratio:

The calculated Sharpe ratio is printed using the **'print'** function and the formatted string **'Portfolio had a Sharpe ratio of {:.2f} '**.

Calculating the Annualized Sharpe Ratio:

The annualized Sharpe ratio is calculated by multiplying the Sharpe ratio with the square root of the number of periods.
The number of periods is obtained by taking the length of the **'new_df_comp'** DataFrame using **'len(new_df_comp)'** and raising it to the power of 1/2.
The result is stored in the variable **'A_SR'**.

Printing the Annualized Sharpe Ratio:

The calculated annualized Sharpe ratio is printed using the **'print'** function and the formatted string **'Portfolio had an annualized Sharpe ratio of {:.2f} '**.
The code calculates the Sharpe ratio for the specified portfolio over the specified period and displays both the standard Sharpe ratio and the annualized Sharpe ratio.


```python 
new_df_comp.query("Date == min").loc[:, ['Date', 'EEM_pos_value', 'GLD_pos_value', 'SPY_pos_value', 'TLT_pos_value', 'VGK_pos_value', 'SUM_all_pos_value']].to_csv('SummarAlocations.csv', index=False)
```

The resulting CSV file will contain the summarized rebalancing dates and the corresponding allocations for each asset in the portfolio.










