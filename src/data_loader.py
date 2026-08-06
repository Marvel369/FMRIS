"""
data_loader.py
Loads historical market data from local CSV files.
"""

import pandas as pd
from pathlib import Path

DATA_PATH = Path("data")

def load_sp500():

    df = pd.read_csv(
        DATA_PATH / "sp500.csv"
    )
    df["Date"] = pd.to_datetime(df["Date"])

    df = df[["Date", "Close"]]
    df.rename(
        columns={
            "Close": "sp500"
        },
        inplace=True
    )
    return df

def load_vix():

    df = pd.read_csv(
        DATA_PATH / "vix.csv"
    )

    df["DATE"] = pd.to_datetime(df["DATE"])

    df = df[["DATE", "CLOSE"]]

    df.rename(
        columns={
            "DATE":"Date",
            "CLOSE":"vix"
        },
        inplace=True
    )
    return df

def load_treasury():

    df = pd.read_csv(
        DATA_PATH / "treasury_10y.csv"
    )
    print("\n=============================================\n")

    # FRED format
    if "observation_date" in df.columns:
        df.rename(
            columns={
                "observation_date": "Date"
            },
            inplace=True
        )


    df["Date"] = pd.to_datetime(df["Date"])

    # Rename value column
    if "GS10" in df.columns:
        df.rename(
            columns={
                "GS10": "treasury_10y"
            },
            inplace=True
        )

    return df[["Date", "treasury_10y"]]

def load_all_data():

    #load
    sp500 = load_sp500()
    vix = load_vix()
    treasury = load_treasury()

    #merge market = sp500 + vix 
    market = sp500.merge(
        vix,
        on="Date",
        how="left"
    )
    #merge market = market + treasury
    market = market.merge(
        treasury,
        on="Date",
        how="left"
    )

    market=market[market["Date"] >= "1990-01-01"]

    market = market.sort_values(
        "Date"
    )
    market = market.reset_index(drop=True)
    market["Date"] =  market["Date"].dt.date

    return market

if __name__ == "__main__":

    df = load_all_data() # new dataframe combines all market
    

    print(df.head())

    print("\n===========\nShape:",df.shape,"\n===========")

    print("\n===========\nNull Values:")
    print(df.isnull().sum(),"\n===========")
    # print(df[df["vix"].isna()])

    print("\n=======================\nSummary:\n",df.describe().round(3),"\n\n=======================")