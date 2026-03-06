import pandas as pd
import requests
import io
import os
from config.settings import BASE_DIR
from src.utils.logger import get_logger

logger = get_logger("FetchSymbols")

def get_all_nse_symbols(save_to_csv: bool = True) -> list[str]:
    """
    Fetches the master list of all listed equities from the National Stock Exchange (NSE).
    Returns Yahoo Finance compatible tickers (appends .NS).
    """
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    
    try:
        logger.info(f"Fetching SEBI/NSE master equity list from {url}...")
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        df = pd.read_csv(io.StringIO(response.text))
        
        if 'SYMBOL' not in df.columns:
            logger.error("Could not find SYMBOL column in NSE data.")
            return []
            
        # Exclude ETFs and non-equity, typically filtered by SERIES == 'EQ'
        if ' SERIES' in df.columns:
            df = df[df[' SERIES'] == 'EQ']
            
        symbols = [f"{str(sym).strip()}.NS" for sym in df['SYMBOL'].tolist()]
        logger.info(f"Successfully fetched {len(symbols)} NSE symbols.")
        
        if save_to_csv:
            out_path = BASE_DIR / "data" / "nse_all_symbols.csv"
            os.makedirs(out_path.parent, exist_ok=True)
            df.to_csv(out_path, index=False)
            logger.info(f"Saved master list to {out_path}")
            
        return symbols
        
    except Exception as e:
        logger.error(f"Failed to fetch NSE symbols: {e}")
        return []

if __name__ == "__main__":
    symbols = get_all_nse_symbols()
    if symbols:
        print(f"Sample of fetched symbols: {symbols[:10]} ...")
