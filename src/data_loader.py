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
    print("Treasury columns:", df.columns.tolist())


    # FRED format
    if "observation_date" in df.columns:
        df.rename(
            columns={
                "observation_date": "Date"
            },
            inplace=True
        )

    elif "DATE" in df.columns:
        df.rename(
            columns={
                "DATE": "Date"
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

    elif "VALUE" in df.columns:
        df.rename(
            columns={
                "VALUE": "treasury_10y"
            },
            inplace=True
        )

    return df[["Date", "treasury_10y"]]

def load_all_data():

    sp500 = load_sp500()
    vix = load_vix()
    treasury = load_treasury()

    market = sp500.merge(
        vix,
        on="Date",
        how="left"
    )

    market = market.merge(
        treasury,
        on="Date",
        how="left"
    )

    market=market[market["Date"] >="1990-01-01"]

    market = market.sort_values(
        "Date"
    )

    return market

if __name__ == "__main__":

    df = load_all_data()

    print(df.head(50))

    print("\nShape:")
    print(df.shape,"\n")