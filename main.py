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


# Load all ETFs data
files = glob.glob('InputData/*.csv')
df = pd.concat([pd.read_csv(fp, parse_dates=True).assign(Title=os.path.basename(fp).split('.')[0]) 
       for fp in files])

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])
# set the index
df.set_index('Date', inplace=True)

df.rename(columns={'Adj Close':'AdjClose'}, inplace=True)

# Calculate daily returns. Calculate from Adj. Close price (not form Close). Create separate column
# 1 for ONE DAY lookback
df['RDaily'] = df.groupby('Title')['AdjClose'].pct_change(1)


# Initialize an empty list to store the values
tmp = []

# Grouped data by year and month
grouped_df = df.groupby(['Title', df.index.year, df.index.month])

# Calculate SD - Volatility of returns, for each title
for (title, year, month), group in grouped_df:
    std_value = group['RDaily'].std()
    inv_std_value = 1/group['RDaily'].std()
    tmp.append([title, year, month, std_value, inv_std_value])

# Create a new DataFrame from the monthly grouped data
df_mthly = pd.DataFrame(tmp, columns=['Title', 'Year', 'Month', 'StdValue', 'InvStdVal'])


df_mthly_pivot = df_mthly.pivot(index=['Year','Month'], columns="Title", values="InvStdVal")
df_mthly_pivot['Sum_InvStdVal'] = df_mthly_pivot['EEM'] + df_mthly_pivot['GLD'] + df_mthly_pivot['SPY'] + df_mthly_pivot['TLT'] + df_mthly_pivot['VGK']

# Weight of position for each title
df_mthly_pivot['W_EEM'] = df_mthly_pivot['EEM'] / df_mthly_pivot['Sum_InvStdVal']
df_mthly_pivot['W_GLD'] = df_mthly_pivot['GLD'] / df_mthly_pivot['Sum_InvStdVal']
df_mthly_pivot['W_SPY'] = df_mthly_pivot['SPY'] / df_mthly_pivot['Sum_InvStdVal']
df_mthly_pivot['W_TLT'] = df_mthly_pivot['TLT'] / df_mthly_pivot['Sum_InvStdVal']
df_mthly_pivot['W_VGK'] = df_mthly_pivot['VGK'] / df_mthly_pivot['Sum_InvStdVal']

# Only for check, W_CHECK_SUM = 1
df_mthly_pivot['W_CHECK_SUM'] = df_mthly_pivot['W_EEM'] + df_mthly_pivot['W_GLD'] + df_mthly_pivot['W_SPY'] + df_mthly_pivot['W_TLT'] + df_mthly_pivot['W_VGK']

# Unpivot the DataFrame by stacking the specified level (0)
df_mthly_unpivot = df_mthly_pivot.stack(level=(0)).reset_index()
# Filter the unpivoted DataFrame to keep only rows where the "Title" column has specific values
df_mthly_unpivot = df_mthly_unpivot[df_mthly_unpivot["Title"].isin(["W_EEM", "W_GLD", 'W_SPY', 'W_TLT', 'W_VGK'])].copy()
# Rename the last column to "W" (as Weight)
df_mthly_unpivot.columns.values[-1] = "W"
# Remove the prefix 'W_' from the values in the "Title" column
df_mthly_unpivot['Title'] = df_mthly_unpivot['Title'].str.replace('W_', '')

# Convert the 'Title' column in the DataFrame 'df' to string type
df['Title'] = df['Title'].astype(str)
# Convert the 'Title' column in the DataFrame 'df_mthly_unpivot' to string type
df_mthly_unpivot['Title'] = df_mthly_unpivot['Title'].astype(str)

# Merge the 'df' DataFrame and 'df_mthly_unpivot' DataFrame based on specified columns,
# performing a left join to include all rows from 'df' and matching rows from 'df_mthly_unpivot'
df_comp = pd.merge(df.reset_index(), df_mthly_unpivot.reset_index(), how='left', 
                   left_on=[df.reset_index()['Date'].dt.year, df.reset_index()['Date'].dt.month, 'Title'], 
                   right_on=['Year', 'Month', 'Title'])


# Convert the "Date" column to datetime if it's not already
df_comp['Date'] = pd.to_datetime(df_comp['Date'])

# Extract year and month from the "Date" column
df_comp['Year'] = df_comp['Date'].dt.year
df_comp['Month'] = df_comp['Date'].dt.month

# Find the maximum and minimum date for each year and month
min_max_dates = df_comp.groupby(['Year', 'Month'])['Date'].agg([min, max]).reset_index()

# Merge the min and max dates back to the original DataFrame
df_comp = pd.merge(df_comp, min_max_dates, on=['Year', 'Month'], how='left')


# Perform the pivot operation
result = df_comp.pivot(index=['Date', 'Year', 'Month'], columns='Title', values=['RDaily', 'W'])

# Flatten the column names
result.columns = ['{}_{}'.format(col[1], col[0]) for col in result.columns]

# Reset the index
result = result.reset_index()

# Concatenate the 'Date', 'Year', 'Month' columns from the original dataframe with the 'result' dataframe
new_df_comp = pd.concat([df_comp[['Date', 'Year','Month','min','max', ]], result], axis=1)

#new_df_comp.drop(new_df_comp.iloc[:, 0:1],axis = 1, inplace=True)

new_df_comp = new_df_comp[new_df_comp['EEM_W'].notna()]


# Define a function to remove duplicate columns from a DataFrame
def remove_dup_columns(frame):
     keep_names = set() # Set to store unique column names
     keep_icols = list() # List to store indices of columns to keep
     for icol, name in enumerate(frame.columns):
          if name not in keep_names: # If the column name is not already present in keep_names
               keep_names.add(name) # Add the column name to keep_names
               keep_icols.append(icol) # Add the index of the column to keep_icols
     return frame.iloc[:, keep_icols] # Return the DataFrame with only the columns to keep

# Remove duplicate columns from the DataFrame new_df_comp using the remove_dup_columns function
new_df_comp = remove_dup_columns(new_df_comp)


# Add a new column 'CHECK' to the DataFrame new_df_comp and assign the value 'check' to all rows
new_df_comp['CHECK'] = 'check'

# Add new columns 'EEM_pos_value', 'GLD_pos_value', 'SPY_pos_value', 'TLT_pos_value', and 'VGK_pos_value'
# to the DataFrame new_df_comp and assign the value 1 to all rows
new_df_comp['EEM_pos_value'] = 1
new_df_comp['GLD_pos_value'] = 1
new_df_comp['SPY_pos_value'] = 1
new_df_comp['TLT_pos_value'] = 1
new_df_comp['VGK_pos_value'] = 1

# Add a new column 'SUM_all_pos_value' to the DataFrame new_df_comp and assign the value 1 to all rows
new_df_comp['SUM_all_pos_value'] = 1

initial_capital = 100 # Initial capital invested at start of first year, second month


# Loop through the rows of the DataFrame 'new_df_comp', starting from index 1
for i in range(1, len(new_df_comp)):
    # Check if we are at the beginning of a month in the first year
    if ((new_df_comp.loc[i, 'Date'] == new_df_comp.loc[i, 'min']) & (new_df_comp.loc[i, 'Date'].year == np.sort(new_df_comp['Date'].dt.year.unique())[0]) & (new_df_comp.loc[i, 'Date'].month ==  np.sort(new_df_comp.loc[(new_df_comp['Date'].dt.year == np.sort(new_df_comp['Date'].dt.year.unique())[0])]['Date'].dt.month.unique())[1])): # ak sme na zaciatku mesiaca >>> tuna chceme rebalancovat ak sme na zaciatku druheho mesiaca prveho roku >>>tu rebalancujeme startovaci kapital 100 usd
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
        #new_df_comp.loc[i, 'VGK_pos_value'] = 100010000

        new_df_comp.loc[i, 'CHECK']  = '*-*-*-*-*-*-*-*-*-*'

        new_df_comp.loc[i, 'SUM_all_pos_value'] = new_df_comp.loc[i, 'EEM_pos_value']+new_df_comp.loc[i, 'GLD_pos_value']+new_df_comp.loc[i, 'SPY_pos_value']+new_df_comp.loc[i, 'TLT_pos_value']+new_df_comp.loc[i, 'VGK_pos_value']




# Plot of the portfolio value, after first rebalancing, with initial value 100
sns.lineplot(data=new_df_comp.loc[(new_df_comp['Date'].dt.year >= 2018) & (new_df_comp['Date'].dt.month >= 5)], x="Date", y="SUM_all_pos_value")
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Portfolio value over time") # You can comment this line out if you don't need title
plt.savefig('PortfolioValue.png')


# Set the first year and find the first date in the DataFrame 'df' for that year
year_first = 2020
first_date = df[df.index.year == year_first].index[0]

# Set the last year and find the last date in the DataFrame 'df' for that year
year_last = 2022
last_date = df[df.index.year == year_last].index[-1]

# Calculate the Compound Annual Growth Rate (CAGR) based on the last and first dates' SUM_all_pos_value values
CAGR = (new_df_comp.loc[(new_df_comp['Date'] == last_date)]['SUM_all_pos_value'].values[0] / new_df_comp.loc[(new_df_comp['Date'] == first_date)]['SUM_all_pos_value'].values[0]) ** (1/3) - 1

# Print the calculated CAGR as a percentage
print('Portfolio had a CAGR of {:.2%} '.format(CAGR))


# Create a boolean mask to filter rows in 'new_df_comp' between 'first_date' and 'last_date'
mask = (new_df_comp['Date'] >= first_date) & (new_df_comp['Date'] <= last_date)

# Calculate the Sharpe ratio as the ratio of mean returns to standard deviation of returns
SR = new_df_comp.loc[mask].SUM_all_pos_value.pct_change(1).mean() / new_df_comp.loc[mask].SUM_all_pos_value.pct_change(1).std()
# Print the calculated Sharpe ratio
print ('Portfolio had a Sharpe ratio of {:.2f} '.format(SR))

# Calculate the annualized Sharpe ratio by multiplying the Sharpe ratio with the square root of the number of periods
A_SR = len(new_df_comp)**(1/2)*SR
# Print the calculated annualized Sharpe ratio
print ('Portfolio had a annualized Sharpe ratio of {:.2f} '.format(A_SR))


# CSV file that summarize the rebalancing dates and the corresponding allocations
new_df_comp.query("Date == min").loc[:, ['Date', 'EEM_pos_value', 'GLD_pos_value', 'SPY_pos_value', 'TLT_pos_value', 'VGK_pos_value', 'SUM_all_pos_value']].to_csv('SummarAlocations.csv', index=False)


