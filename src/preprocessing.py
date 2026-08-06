import pandas as pd
from data_loader import load_all_data

def preprocessing_data():
    # df = preprocess_data()

    df = load_all_data()
    df = df.sort_values("Date")
    df["treasury_10y"] = df["treasury_10y"].ffill() # fill in missing values
    df = df.dropna() # drop initial mssing values 
    df["market_return"] = df["sp500"].pct_change()

    # Small volatility = Low risk & large volatility = high risk.
    df['volatility_30day'] = (
        df['market_return']
        .rolling(window=30)
        .std()
    )
    # removing NAN value 
    df=df.dropna()

    #save
    df.to_csv(
    "data/processed_market_data.csv",
    index=False
    )
    return df

if __name__ == "__main__":
    df = preprocessing_data()

    print(df.head())
    print(df.shape)
