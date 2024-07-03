#!/usr/bin/env python
# coding: utf-8

# # Census Income

# In[ ]:





# In[4]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[5]:


df = pd.read_csv('Census Income.csv')
df


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


df.info()


# In[9]:


df.describe()


# In[15]:


df.columns = df.columns.str.strip()
df.columns


# In[16]:


df.isnull().sum()


# In[23]:


sns.countplot(x='Income', data=df)
plt.title('Distribution of Income')
plt.show()


# In[ ]:





# In[50]:


df. columns = df. columns. str. strip()

print(df.columns)

#Identify object-type columns

categorical_columns = df.select_dtypes(include=['object']).columns


for col in categorical_columns:
    print(f"Unique values in column '{col}': {df[col].unique()}")


target_column = 'Income'

#Determine X and y

X = df.drop(target_column, axis=1)
y = df[target_column]

print(X.head())
print(y.head())


# In[51]:


df


# In[ ]:





# # RAINFALL

# In[9]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[10]:


df = pd.read_csv('Rainfall.csv')
df


# In[11]:


df.head()


# In[12]:


df.tail()


# In[13]:


df.describe()


# In[14]:


df.isnull().sum()


# In[15]:


corr_matrix = df.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()


# In[ ]:





# In[16]:


from sklearn.model_selection import train_test_split


# Split the dataset into features and target variables
X_classification = df.drop(['RainTomorrow'], axis=1)
y_classification = df['RainTomorrow']

X_regression = df.drop(['Rainfall'], axis=1)
y_regression = df['Rainfall']

# Split for classification
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

# Split for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_regression, y_regression, test_size=0.2, random_state=42)


# In[19]:


# Drop the 'Date' column
df = df.drop(columns=['Date'])

# Ensure all other columns are properly preprocessed (e.g., handling missing values, encoding)
df = df.dropna(subset=['RainTomorrow'])  # Drop rows where the target variable is missing
df.fillna(df.mean(), inplace=True)  # Fill missing values with column mean for numerical columns

# Encode categorical variables
categorical_cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Split the dataset into features and target variables
X_classification = df.drop(['RainTomorrow'], axis=1)
y_classification = df['RainTomorrow']

X_regression = df.drop(['Rainfall'], axis=1)
y_regression = df['Rainfall']

# Split for classification
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_classification, y_classificatioC:\Users\dell\AppData\Local\Temp\ipykernel_16688\2323233132.py:6: FutureWarning: The default value of numeric_only in DataFrame.mean is deprecated. In a future version, it will default to False. In addition, specifying 'numeric_only=None' is deprecated. Select only valid columns or specify the value of numeric_only to silence this warning.
  df.fillna(df.mean(), inplace=True)  # Fill missing values with column mean for numerical columns
n, test_size=0.2, random_state=42)

# Split for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_regression, y_regression, test_size=0.2, random_state=42)


# In[ ]:





# In[ ]:





# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


df = pd.read_csv('')
df

Error C:\User\Dell\zomato.csv is not utf - 8 encoded saving disabled. see console for more details
    
    
    #this error showing i am not able to work on zomato links 


# In[ ]:




