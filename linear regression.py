# %% markdown
# # Predicting Games Rating Score
# %% markdown
# In this project we are applying machine learning algorithms to predicts VGChartz rating scores for games. To achinve the best result possible we need to pick the best model that fit the data. The algorithms that we are planning to use are Linear Rigresson, Lasso and Ridge.
# %% markdown
# ## Imporing Libraries
# %% codecell
from utilities.VGChartz import get_games_data
import seaborn as sns
sns.set_style("darkgrid")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from rich import print
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model


from sklearn.linear_model import Lasso, Ridge, ElasticNet,LassoCV,RidgeCV

from sklearn.model_selection import cross_val_score, train_test_split,KFold, GridSearchCV

# %% markdown
# ## Get VGChartz data
# by calling `get_games_data` which scrap the data from VGChartz if the `/data/VGChartz.csv` file is not exsist.
# %% codecell
df = get_games_data()
df_ratings = pd.read_csv('data/Video_Games_Sales_as_at_22_Dec_2016.csv')
# %% codecell
df['Game'] = [x.split('    Read the review')[0] for x in df['Game']]
# %% codecell
df_ratings.head()
# %% codecell
df.Game[0]
# %% codecell
df_ratings[df_ratings['Name'] == 'Super Mario Galaxy']
# %% codecell
df[df['Game'] == 'Super Mario Galaxy']
# %% codecell
x = df.merge(df_ratings[['Rating', 'Year_of_Release', 'User_Count',
                         'Critic_Count', 'Global_Sales', 'Name', 'Platform']],
         left_on=['Game', 'Console'],
         right_on=['Name', 'Platform'])
# %% codecell
new_df = x.dropna(subset=['Rating'])[['ID', 'Name', 'Genre', 'Console', 'Rating', 'Global_Sales', 'VGChartz Score', 'Critic Score', 'Critic_Count', 'User Score', 'User_Count', 'Release Date']]

# %% markdown
# ## Cleaning VGChartz Data
# %% codecell
# Creating new Feature called Score by taking the average of the three type of scores
temp = new_df[['VGChartz Score', 'Critic Score', 'User Score']]
temp=temp.T.fillna(temp.T.mean()).T
new_df[['VGChartz Score', 'Critic Score', 'User Score']] = temp
new_df['Score'] = (new_df['VGChartz Score'] +new_df['Critic Score'] + new_df['User Score'])/3
new_df =new_df.dropna(subset=['Score'])
new_df['User_Count'] = new_df['User_Count'].fillna(new_df['User_Count'].mean())
new_df['Critic_Count'] = new_df['Critic_Count'].fillna(new_df['Critic_Count'].mean())
# %% codecell
new_df['Release Date'] = [int(x.split(' ')[-1]) +2000 for x in new_df['Release Date']]
new_df['Global_Sales'] = new_df['Global_Sales'] * 10**3
# %% codecell
new_df['Release Date'] = new_df['Release Date'] -new_df['Release Date'].min()
# %% codecell
new_df['Global_Sales'].min()
# %% codecell
new_df = new_df.loc[:, (new_df.columns != 'Unnamed: 0')]
# %% codecell
new_df
# %% codecell
new_df = new_df.loc[:, (new_df.columns != 'User Score')]

# %% codecell
new_df = new_df.loc[:, (new_df.columns != 'Critic Score')]

# %% codecell
new_df = new_df.loc[:, (new_df.columns != 'VGChartz Score')]

# %% codecell
new_df.Rating.unique()
# %% codecell
new_df.Console.unique()
# %% codecell
new_df.Genre.unique()
# %% markdown
# ## Spliting Features for testing the corralation
# %% codecell
new_df = new_df.loc[:, (new_df.columns != 'ID')]
new_df = new_df.loc[:, (new_df.columns != 'Name')]
new_df.head(1)
# %% codecell
new_df.corr()
# %% codecell
sns.pairplot(new_df)
# %% codecell
new_df.head(1)
# %% codecell
big_new_df = new_df.copy()

big_new_df = big_new_df.loc[:, (big_new_df.columns != 'Genre')]
big_new_df = big_new_df.loc[:, (big_new_df.columns != 'Console')]
big_new_df = big_new_df.loc[:, (big_new_df.columns != 'Rating')]

big_new_df_Genre1 = pd.get_dummies(new_df['Genre'], columns=['Genre'])
big_new_df_Genre = pd.concat([big_new_df, big_new_df_Genre1], axis=1)

big_new_df_Rating1 = pd.get_dummies(new_df['Rating'], columns=['Rating'])
big_new_df_Rating = pd.concat([big_new_df, big_new_df_Rating1], axis=1)


big_new_df_Console1 = pd.get_dummies(new_df['Console'], columns=['Console'])
big_new_df_Console = pd.concat([big_new_df, big_new_df_Console1], axis=1)


big_new_df_Genre_Console1 = pd.get_dummies(new_df[['Genre','Console']], columns=['Genre','Console'])
big_new_df_Genre_Console = pd.concat([big_new_df, big_new_df_Genre_Console1], axis=1)


big_new_df_Genre_Rating1 = pd.get_dummies(new_df[['Genre','Rating']], columns=['Genre','Rating'])
big_new_df_Genre_Rating = pd.concat([big_new_df, big_new_df_Genre_Rating1], axis=1)


big_new_df_Console_Rating1 = pd.get_dummies(new_df[['Console','Rating']], columns=['Console','Rating'])
big_new_df_Console_Rating = pd.concat([big_new_df, big_new_df_Console_Rating1], axis=1)


big_new_df_Genre_Console_Rating = pd.get_dummies(new_df, columns=['Genre','Console','Rating'])

#big_new_df = new_df.copy()

# %% codecell
big_new_df.info()
# %% codecell
sns.boxplot(y=big_new_df['Critic_Count'])
# %% markdown
# ## Creating Models
# %% codecell
def linear_regression_score(X_train, y_train, X_test, y_test):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    r2 = lr.score(X_test, y_test)
    return r2

def lasso_score(X_train, y_train, X_test, y_test):
    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(X_train, y_train)
    r2 = clf.score(X_test, y_test)
    return r2

def riggid_score(X_train, y_train, X_test, y_test):
    clf = Ridge(alpha=1.0)
    clf.fit(X_train, y_train)
    r2 = clf.score(X_test, y_test)
    return r2

# %% markdown
# ### Find the $R^2$ score for all models
# %% codecell
def find_scores(df, target:str):
    df_X = df[df.columns.difference([target])]
    df_y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.33, random_state=1024 ,shuffle=True)

    linear_score = linear_regression_score(X_train, y_train, X_test, y_test)
    lassov_score = lasso_score(X_train, y_train, X_test, y_test)
    riggidv_score = riggid_score(X_train, y_train, X_test, y_test)
    return {'Linear Score':linear_score,'Lasso Score':lassov_score,'Ridge Score':riggidv_score}

# %% codecell
a = pd.DataFrame(
    {
        'No Categorical':find_scores(big_new_df, 'Global_Sales'),
        'Genre':find_scores(big_new_df_Genre, 'Global_Sales'),
        'Console':find_scores(big_new_df_Console, 'Global_Sales'),
        'Rating':find_scores(big_new_df_Rating, 'Global_Sales'),
        'Console & Rating':find_scores(big_new_df_Console_Rating, 'Global_Sales'),
        'Genre & Console':find_scores(big_new_df_Genre_Console, 'Global_Sales'),
        'Genre & Rating':find_scores(big_new_df_Genre_Rating, 'Global_Sales'),
        'All Categorical':find_scores(big_new_df_Genre_Console_Rating, 'Global_Sales')
    }
)
# %% codecell
a.T
# %% codecell
big_new_df.head(1)
# %% codecell
a.keys()
# %% codecell
