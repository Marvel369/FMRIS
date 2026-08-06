import pandas as pd 
import numpy as np


def create_features():

    #load
    df = pd.read_csv("data/processed_market_data.csv")


    # feature 1 Moving average 30 days
    df['sp500_MA_30'] = (df['sp500']
    .rolling(window=30)
    .mean()
    )

    # feature 2 Moving average 90 days for longer trend
    df['sp500_MA_90'] = (df['sp500']
        .rolling(window=90)
        .mean()
    )

    # feature 3 All time high value until today
    df['peak_price'] = (df['sp500']
        .cummax()
    )


    # feature 4 how much down from peak (current - peak) / peak
    df['drawdown'] = (df['sp500'] - df['peak_price']) / df['peak_price']

    #feature 5 vix % change
    df['vix_change'] =(
        df['vix']
        .pct_change()
    )

    #dropping nan values
    df = df.dropna()

    #save
    df.to_csv(
    "data/features.csv",
    index=False
)

    return df

if __name__ == "__main__":

    data = create_features()

    print(data.head(30))
    print(data.tail())
    print(data.shape)








