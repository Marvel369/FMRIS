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

def load_treasury_2y():

    df =pd.read_csv(
        DATA_PATH / "treasury_2y.csv"
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
    if "GS2" in df.columns:
        df.rename(
            columns={
                "GS2": "treasury_2y"
                },
            inplace=True
            )
    
    return df[["Date", "treasury_2y"]]
    
def load_unemployment():
    df = pd.read_csv(
        DATA_PATH / "unemployment.csv"
    )

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
    if "UNRATE" in df.columns:
        df.rename(
            columns={
                "UNRATE" : "unemployment"
                },
            inplace=True
            )
    return df[["Date", "unemployment"]]

def load_fed_funds():
    df = pd.read_csv(
            DATA_PATH / "fed_funds.csv"
        )

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
    if "FEDFUNDS" in df.columns:
        df.rename(
            columns={
                "FEDFUNDS": "fed_funds"
                },
            inplace=True
            )
        
    return df[["Date", "fed_funds"]]










def load_all_data():

    #load
    sp500 = load_sp500()
    vix = load_vix()
    treasury = load_treasury()
    treasury_2y = load_treasury_2y()
    unemployment = load_unemployment()
    fed_funds = load_fed_funds()

    #merge market = sp500 + vix 
    market = sp500.merge(
        vix,
        on="Date",
        how="outer"
    )
    #merge market = market + treasury
    market = market.merge(
        treasury,
        on="Date",
        how="outer"
    )
    # merge market = market + treasury_2y
    market = market.merge(
            treasury_2y,
            on="Date",
            how="outer"
    )
    # merge market = market + unemployment
    market = market.merge(
                unemployment,
                on="Date",
                how="outer"
    )
    # merge market = market + fed_funds
    market = market.merge(
                    fed_funds,
                    on="Date",
                    how="outer"
                )

    market = market.sort_values(
        "Date"
    ).reset_index(drop=True)

    # convert to numeric
    market["treasury_10y"] = pd.to_numeric(market["treasury_10y"], errors='coerce')
    market["treasury_2y"] = pd.to_numeric(market["treasury_2y"], errors='coerce')
    market["vix"] = pd.to_numeric(market["vix"], errors='coerce')
    market["sp500"] = pd.to_numeric(market["sp500"], errors='coerce')
    market["unemployment"] = pd.to_numeric(market["unemployment"], errors='coerce')
    market["fed_funds"] = pd.to_numeric(market["fed_funds"], errors='coerce')

    # Forward fill missing dates
    market[["sp500", "vix", "treasury_10y", "treasury_2y","unemployment","fed_funds"]] = market[["sp500", "vix", "treasury_10y", "treasury_2y","unemployment","fed_funds"]].ffill()
    
    # Backward fill any remaining edge cases.
    market[["sp500", "vix", "treasury_10y", "treasury_2y","unemployment","fed_funds"]] = market[["sp500", "vix", "treasury_10y", "treasury_2y","unemployment","fed_funds"]].bfill()

    # Keep ALL data here. Do NOT filter out
    market = market[market["Date"] >= "1990-01-01"]

    # Cleanup and export
    market = market.sort_values("Date").reset_index(drop=True)


    market.to_csv(
        DATA_PATH / "raw_market_data.csv",
        index=False
    )

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