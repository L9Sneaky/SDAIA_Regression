# %% codecell
from utilities.IMDb import get_games_data

# %% codecell
df = get_games_data()

# %% codecell
df.sort_values('Genre')
# %% codecell

df.info()

df1=df. dropna()
df1.sort_values('Rating')
