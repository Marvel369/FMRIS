import pandas as pd

def calculate_drawdown(df: pd.DataFrame):
    rolling_max = df['sp500'].cummax()
    drawdown = (df['sp500'] - rolling_max) / rolling_max
    return drawdown

# creating a MArket Risk Label (Target value)
def create_risk_label(data: pd.DataFrame):
    data['risk_label'] = 0

    #high volatility
    high_vix = data['vix'] > 25.0

    #large market decline
    large_drawdown = data['drawdown'] < -0.10

    # negative return
    negative_return = data['market_return'] < -0.02

    # any of these true then label red flag or warn.
    data.loc[
        high_vix | large_drawdown | negative_return,
        "risk_label"
    ] = 1

    return data

def preprocessing_data():

    # df = preprocess_data()

    df = pd.read_csv("data/raw_market_data.csv")
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
    df["drawdown"] = calculate_drawdown(df)
    
    # removing NAN value 
    df=df.dropna()
    df= create_risk_label(df)


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
