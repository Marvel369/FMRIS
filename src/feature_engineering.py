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

    #feature 6: Yield spread daily change
    df["yield_spread_change"] = (
        df["yield_spread"]
        .diff()
    )
    #feature 7: vix 10 day moving avg
    df["vix_MA_10"] = (
        df["vix"]
        .rolling(window=10)
        .mean()
    )

    #feature 8: sp500 market momentum 10d
    df["sp500_Momentum_10d"] = (
        df["sp500"]
        .pct_change(periods=10)
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

    print(data.head())
    print(data.tail())
    print(data.shape)








