from IMDb import get_games_data


df = get_games_data()
df.to_csv('F:\Books\T10\sdaia t5\MTA Project\SDAIA_Regression\data\imbd_games.csv')
