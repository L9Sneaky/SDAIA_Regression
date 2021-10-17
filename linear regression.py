# %% codecell
from utilities.IMDb import get_games_data
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from rich import print

# %% codecell
df = get_games_data()

# %% codecell
df.sort_values('Genre')
# %% codecell
def split_and_validate(X, y):
    # perform train/val split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.35, random_state=42)

    # fit linear regression to training data
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # score fit model on validation data
    val_score = lr_model.score(X_val, y_val)

    # report results
    print('\nValidation R^2 score was:', val_score)
    print('Feature coefficient results: \n')
    for feature, coef in zip(X.columns, lr_model.coef_):
        print(feature, ':', f'{coef:.2f}')

df.info()

df1=df.dropna()
df1.sort_values('Year', ascending=False).head(5)
sns.pairplot(df1)
df1['Year'].describe()
df1.shape

smaller_df= df1.loc[:,['Year','Rating','Votes','Genre']]
smaller_df.head()
smaller_df.corr()

X = smaller_df.loc[:,['Year','Votes']]
y = smaller_df['Rating']

split_and_validate(X, y)

























df_train, df_val = train_test_split(df1, test_size=0.3, random_state=5)
