##### data preprocessing for  credit risk modeling analysis #####
#################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

ld = pd.read_csv("loan_data_2007_2014.csv")

#pd.options.display.max_rows = None
# print(ld.isnull().sum())

''' preprocessing features'''
# print(ld['term'].unique())
ld['term_int'] = ld['term'].str.replace('months', '')
ld['term_int'] = pd.to_numeric(ld['term_int'])

ld['emp_length_int'] = ld['emp_length'].str.replace('\+ years', '')
ld['emp_length_int'] = ld['emp_length_int'].str.replace('years', '')
ld['emp_length_int'] = ld['emp_length_int'].str.replace('<', '')
ld['emp_length_int'] = ld['emp_length_int'].str.replace('year', '')
ld['emp_length_int'] = ld['emp_length_int'].str.replace('nan', 'str(0)')
ld['emp_length_int'] = pd.to_numeric(ld['emp_length_int'])

ld['earliest_cr_line_date'] = pd.to_datetime(ld['earliest_cr_line'], format= '%b-%y')
ld['mths_since_earliest_cr_line'] = round(pd.to_numeric((pd.to_datetime('2017-12-01') - ld['earliest_cr_line_date'])/ np.timedelta64(1, 'M')))
# print(ld.loc[:, ['earliest_cr_line', 'earliest_cr_line_date' ,'mths_since_earliest_cr_line']][ld['mths_since_earliest_cr_line'] < 0])
# if < 0 replace them with max('mths_since_earliest_cr_line')
ld['mths_since_earliest_cr_line'][ld['mths_since_earliest_cr_line'] < 0] = ld['mths_since_earliest_cr_line'].max()

ld['issue_d'] = pd.to_datetime(ld['issue_d'], format = '%b-%y')
ld['mths_since_issue_d'] = round(pd.to_numeric((pd.to_datetime('2017-12-01') - ld['issue_d'])/ np.timedelta64(1, 'M')))

''' Preprocessing discrete variables '''
ld_dummies = [ pd.get_dummies(ld['grade'], prefix='grade', prefix_sep= ':'),
               pd.get_dummies(ld['sub_grade'], prefix='sub_grade', prefix_sep= ':'),
               pd.get_dummies(ld['home_ownership'], prefix='home_ownership', prefix_sep=':'),
               pd.get_dummies(ld['loan_status'], prefix='loan_status', prefix_sep= ':'),
               pd.get_dummies(ld['purpose'], prefix='purpose', prefix_sep= ':'),
               pd.get_dummies(ld['addr_state'], prefix='addr_state', prefix_sep= ':'),
               pd.get_dummies(ld['initial_list_status'], prefix='initial_list_status', prefix_sep= ':'),
               pd.get_dummies(ld['verification_status'], prefix='verification_status', prefix_sep= ':')
               ]
ld_dummies = pd.concat(ld_dummies, axis =1)

''' concat the data frame of dummy variables (ld_dummies) to main data frame (ld)'''
ld = pd.concat([ld, ld_dummies], axis = 1)
# print(ld.columns.values)

''' Checking missing values '''
# pd.options.display.max_rows = None
# print(ld.isnull().sum())
ld['total_rev_hi_lim'].fillna(ld['funded_amnt'], inplace = True)
ld['annual_inc'].fillna(ld['annual_inc'].mean(), inplace = True)
ld['mths_since_earliest_cr_line'].fillna(0, inplace = True)
ld['total_acc'].fillna(0, inplace = True)
ld['pub_rec'].fillna(0, inplace = True)
ld['open_acc'].fillna(0, inplace = True)
ld['inq_last_6mths'].fillna(0, inplace = True)
ld['delinq_2yrs'].fillna(0, inplace = True)
ld['emp_length_int'].fillna(0, inplace = True)


''' Dependent variable for PD model (Good/Bad defaults) '''
# print(ld['loan_status'].unique())
# print(ld['loan_status'].value_counts())
# print(ld['loan_status'].value_counts()/ ld['loan_status'].count())
ld['good_bad'] = np.where(ld['loan_status'].isin(['Default',
                                                  'Charged Off',
                                                  'Late (31-120 days)',
                                                  'Does not meet the credit policy. Status:Charged Off',]), 0, 1)

#ld.to_csv("loan_data_2007_2014_preprocessed.csv")


''' preprocessing independent variables ''' ''' Split the data into train and test sets '''
from sklearn.model_selection import train_test_split
ld_train_X, ld_test_X, ld_train_y, ld_test_y = train_test_split(ld.drop('good_bad', axis = 1), ld['good_bad'], test_size = 0.2, random_state = 1)

df_input  = ld_train_X
df_target = ld_train_y

# df_input  = ld_test_X
# df_target = ld_test_y

''' WoE_discrete for each independent feature '''
def woe_discrete(df, input_variable, target):
    df = pd.concat([df[input_variable], target], axis = 1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index=False) [df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index=False) [df.columns.values[1]].mean()], axis = 1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns        = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs']  = df['n_obs'] / df['n_obs'].sum()
    df['n_good']      = df['prop_good'] * df['n_obs']
    df['n_bad']       = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()    # % good = no. of good / total no of good in that category
    df['prop_n_bad']  = df['n_bad'] / df['n_bad'].sum()      # % bad = no. of bad / total no of bad in that category
    df['woe']         = np.log(df['prop_n_good'] / df['prop_n_bad'])  # ln(% good / % bad)
    df = df.sort_values(['woe'])
    df = df.reset_index(drop=True)
    df['IV']          = (df['prop_n_good'] - df['prop_n_bad']) * df['woe']
    df['IV']          = df['IV'].sum()
    return df

''' WoE visualization '''
def plot_by_woe(df_woe, rotation_x_labels = 0):
    x = np.array(df_woe.iloc[:, 0].apply(str))
    y = df_woe['woe']
    plt.figure(figsize = (15, 6))
    plt.plot(x, y, color = "red", marker = 'o', linestyle = '--')
    plt.title(str("Weight of Evidence__" + df_woe.columns[0]))
    plt.xlabel(df_woe.columns[0])
    plt.ylabel("Weight of Evidence")
    plt.xticks(rotation = rotation_x_labels)
    plt.savefig("woe_"+ df_woe.columns[0])
    plt.show()


''' call the function for all dummy variables and plot the WoE (Weight Of Evidence)'''
list1 = ['grade', 'home_ownership', 'addr_state', 'verification_status', 'purpose', 'initial_list_status']
def try_fun(df_in, my_list, df_out):
    for _ in my_list:
        df_temp = woe_discrete(df_in, _, df_out)
        print(df_temp)
        plot_by_woe(df_temp)

funtest = try_fun(df_input, list1, df_target)



