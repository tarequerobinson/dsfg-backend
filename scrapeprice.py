import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import date


#Fetch the closing price for `symbol` from the Jamaica Stock Exchange ordinary-shares table.
def get_jse_closing_price(symbol: str, date_str: str = None) -> float:
    """
    Inputs:
        symbol: The stock symbol to look up (e.g. "JMMBGL").
        date_str: Date in "YYYY-MM-DD" format. Defaults to today if None.
    """
    # default date to today
    if date_str is None:
        date_str = date.today().isoformat()
    
    # request the page
    url = f"https://www.jamstockex.com/trading/trade-quotes/?market=50&date={date_str}"
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/58.0.3029.110 Safari/537.3'
        )
    }
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    
    # parse the trade date (optional, verifies page loaded)
    soup = BeautifulSoup(resp.text, 'html.parser')
    date_el = soup.find('span', id='tradeDate')
    trade_date = date_el.get_text(strip=True) if date_el else date_str
    
    # load all tables and pick ordinary‑shares (index may vary—usually 2)
    tables = pd.read_html(resp.text)
    try:
        df = tables[2]
    except IndexError:
        raise ValueError(f"Couldn't find the JMD ordinary‑shares table on {trade_date}")
    
    # clean up unnamed cols
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors='ignore')
    
    # filter for symbol
    symbol_col = "Symbol"
    close_col = "Close"
    matches = df[df[symbol_col] == symbol]
    if matches.empty:
        raise ValueError(f"No data for symbol '{symbol}' on {trade_date}")
    
    return float(matches[close_col].iloc[0])

