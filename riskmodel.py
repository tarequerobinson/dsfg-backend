import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime, timedelta
import argparse
import sys

def preprocess_jse_data(data_path):
    """
    Preprocess the JSE data to prepare it for risk analysis.
    
    Parameters:
    data_path (str): Path to the CSV file containing JSE data
    
    Returns:
    DataFrame: Preprocessed data
    """
    # Load the data
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    # Convert date strings to datetime objects
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by symbol and date
    if 'Symbol' in df.columns and 'Date' in df.columns:
        df = df.sort_values(['Symbol', 'Date'])
    
    # Parse ranges into separate columns - FIXED to handle asterisks
    def parse_range(range_str):
        if pd.isna(range_str):
            return np.nan, np.nan
        try:
            # Remove any asterisks (*) from the range string
            clean_range = range_str.replace('*', '').strip()
            # Handle comma-separated thousands
            clean_range = clean_range.replace(',', '')
            
            # Split by hyphen and convert to float
            parts = clean_range.split('-')
            if len(parts) != 2:
                return np.nan, np.nan
                
            low = float(parts[0].strip())
            high = float(parts[1].strip())
            return low, high
        except Exception as e:
            return np.nan, np.nan
    
    # Apply the parsing function to extract today's range
    if "Today's Range ($)" in df.columns:
        df[['Today_Low', 'Today_High']] = df["Today's Range ($)"].apply(
            lambda x: pd.Series(parse_range(x))
        )
    
    # Apply the parsing function to extract 52-week range
    if "52 Week Range ($)" in df.columns:
        df[['52W_Low', '52W_High']] = df["52 Week Range ($)"].apply(
            lambda x: pd.Series(parse_range(x))
        )
    
    # Calculate additional features
    if all(col in df.columns for col in ['Today_Low', 'Today_High', 'Closing Price ($)']):
        df['Daily_Range_Pct'] = (df['Today_High'] - df['Today_Low']) / df['Closing Price ($)']
    
    if all(col in df.columns for col in ['52W_Low', '52W_High', 'Closing Price ($)']):
        df['52W_Range_Pct'] = (df['52W_High'] - df['52W_Low']) / df['Closing Price ($)']
    
    # Fill NaN values with appropriate defaults using a safer approach
    if 'Daily_Range_Pct' in df.columns:
        daily_range_median = df['Daily_Range_Pct'].median()
        df['Daily_Range_Pct'] = df['Daily_Range_Pct'].fillna(daily_range_median)
    
    if '52W_Range_Pct' in df.columns:
        week52_range_median = df['52W_Range_Pct'].median()
        df['52W_Range_Pct'] = df['52W_Range_Pct'].fillna(week52_range_median)
            
    return df

def find_alternative_symbol(df, requested_symbol):
    """
    Find an alternative symbol if the requested one doesn't exist
    
    Parameters:
    df (DataFrame): JSE data
    requested_symbol (str): The originally requested symbol
    
    Returns:
    str: An alternative symbol with sufficient data
    """
    if 'Symbol' not in df.columns:
        return None
        
    # Get symbols with their counts, sorted by most data points
    symbol_counts = df['Symbol'].value_counts()
    
    # Filter for symbols with at least 100 data points
    valid_symbols = symbol_counts[symbol_counts >= 100]
    
    if len(valid_symbols) == 0:
        return None
    
    # Return the symbol with the most data points
    return valid_symbols.index[0]

def calculate_forward_returns(df, symbol, horizon_days=21, drastic_threshold=0.20):
    """
    Calculate forward returns for a specific stock over the given horizon.
    
    Parameters:
    df (DataFrame): Preprocessed JSE data
    symbol (str): Stock symbol
    horizon_days (int): Forward return horizon in trading days
    drastic_threshold (float): Threshold for defining drastic movement (default: 0.20 or 20%)
    
    Returns:
    DataFrame: Data with forward returns
    """
    # Filter for the specific symbol
    if 'Symbol' not in df.columns:
        return None
        
    stock_df = df[df['Symbol'] == symbol].copy()
    
    if len(stock_df) < 30:  # Need minimum data points
        return None
    
    # Sort by date
    if 'Date' in stock_df.columns:
        stock_df = stock_df.sort_values('Date')
    else:
        return None
    
    # Calculate daily returns
    if 'Closing Price ($)' in stock_df.columns:
        stock_df['Daily_Return'] = stock_df['Closing Price ($)'].pct_change()
    else:
        return None
    
    # Calculate forward returns for the specified horizon
    stock_df['Forward_Return'] = stock_df['Closing Price ($)'].pct_change(periods=horizon_days).shift(-horizon_days)
    
    # Flag drastic movements based on the provided threshold
    stock_df['Drastic_Move'] = (stock_df['Forward_Return'].abs() >= drastic_threshold).astype(int)
    
    return stock_df

def jse_risk_model(df, symbol, forecast_days=21, drastic_threshold=0.20):
    """
    Calculate risk probability of drastic price movements for a JSE stock.
    
    Parameters:
    df (DataFrame): Preprocessed JSE data
    symbol (str): Stock symbol
    forecast_days (int): Prediction horizon in trading days
    drastic_threshold (float): Threshold for defining drastic movement (default: 0.20 or 20%)
    
    Returns:
    dict: Risk analysis results
    """
    try:
        # Calculate forward returns
        stock_df = calculate_forward_returns(df, symbol, horizon_days=forecast_days, drastic_threshold=drastic_threshold)
        
        if stock_df is None or len(stock_df) < 50:
            alt_symbol = find_alternative_symbol(df, symbol)
            if alt_symbol:
                return {"Symbol": symbol, "Error": f"Insufficient data points for {symbol}. Try {alt_symbol} instead."}
            else:
                return {"Symbol": symbol, "Error": f"Insufficient data points for {symbol}"}
        
        # Check if required columns exist
        required_columns = ['Date', 'Closing Price ($)', 'Daily_Return', 'Forward_Return', 'Volume']
        missing_columns = [col for col in required_columns if col not in stock_df.columns]
        if missing_columns:
            return {"Symbol": symbol, "Error": f"Missing required columns: {missing_columns}"}
        
        # Get today's data (most recent available)
        latest_date = stock_df['Date'].max()
        latest_data = stock_df[stock_df['Date'] == latest_date].iloc[0]
        
        # Calculate rolling statistics
        stock_df['Vol_10d'] = stock_df['Daily_Return'].rolling(window=10).std() * np.sqrt(10)
        stock_df['Vol_21d'] = stock_df['Daily_Return'].rolling(window=21).std() * np.sqrt(21)
        stock_df['Vol_63d'] = stock_df['Daily_Return'].rolling(window=63).std() * np.sqrt(63)
        stock_df['Volume_SMA10'] = stock_df['Volume'].rolling(window=10).mean()
        stock_df['Volume_Ratio'] = stock_df['Volume'] / stock_df['Volume_SMA10']
        
        # Drop NaN values for volatility calculations
        vol_df = stock_df.dropna(subset=['Vol_21d', 'Forward_Return'])
        
        # Check if we have enough valid data after calculating all metrics
        if len(vol_df) < 30:
            return {"Symbol": symbol, "Error": f"Insufficient valid data points after calculating metrics"}
        
        # Fill remaining NaNs with 0 for counting purposes
        stock_df = stock_df.fillna(0)
        
        # 1. Historical drastic movement frequency
        drastic_move_count = stock_df['Drastic_Move'].sum()
        total_valid_periods = stock_df['Drastic_Move'].count()
        historical_probability = drastic_move_count / total_valid_periods if total_valid_periods > 0 else 0
        
        # 2. Latest volatility
        latest_vol_10d = stock_df['Vol_10d'].iloc[-1]
        latest_vol_21d = stock_df['Vol_21d'].iloc[-1]
        latest_vol_63d = stock_df['Vol_63d'].iloc[-1]
        
        # Check for valid volatility values
        if np.isnan(latest_vol_21d) or latest_vol_21d == 0:
            return {"Symbol": symbol, "Error": f"Invalid volatility calculation"}
        
        # 3. Statistical probability calculation
        # Method 1: Normal distribution
        z_score = drastic_threshold / latest_vol_21d
        normal_probability = 2 * (1 - stats.norm.cdf(z_score))  # Two-tailed probability
        
        # Method 2: Student's t-distribution with df=5 (better for fat tails)
        t_probability = 2 * (1 - stats.t.cdf(z_score, 5))
        
        # Method 3: Empirical probability based on volatility comparison
        vol_ratio = latest_vol_21d / stock_df['Vol_21d'].median()
        empirical_mult = min(2.0, max(0.5, vol_ratio))  # Bound multiplier between 0.5 and 2.0
        empirical_probability = min(0.95, historical_probability * empirical_mult)
        
        # 4. Volume-based risk factors
        # Recent volume volatility
        recent_volume_volatility = stock_df['Volume_Ratio'].tail(10).std()
        volume_factor = 1 + (0.2 * min(1.0, recent_volume_volatility))
        
        # 5. Price position within 52-week range
        # Check if 52-week high/low data is available
        if all(x in stock_df.columns for x in ['52W_High', '52W_Low']):
            current_price = latest_data['Closing Price ($)']
            week52_high = latest_data['52W_High']
            week52_low = latest_data['52W_Low']
            
            # Check if 52-week values are valid
            if pd.isna(week52_high) or pd.isna(week52_low) or week52_high <= week52_low:
                # Use historical data as fallback
                week52_high = stock_df['Closing Price ($)'].max()
                week52_low = stock_df['Closing Price ($)'].min()
        else:
            # Use historical data if 52-week range columns aren't available
            current_price = latest_data['Closing Price ($)']
            week52_high = stock_df['Closing Price ($)'].max()
            week52_low = stock_df['Closing Price ($)'].min()
        
        # Calculate normalized position (0 = at 52w low, 1 = at 52w high)
        if week52_high > week52_low:
            price_position = (current_price - week52_low) / (week52_high - week52_low)
        else:
            price_position = 0.5  # Default to middle if range is invalid
        
        # Convert to risk factor (U-shaped: higher risk at extremes)
        position_factor = 1 + (0.15 * (1 - 4 * (price_position - 0.5)**2))
        
        # 6. Volatility trend factor
        if not np.isnan(latest_vol_10d) and not np.isnan(latest_vol_63d) and latest_vol_63d > 0:
            vol_trend = (latest_vol_10d / latest_vol_63d) - 1
            vol_trend_factor = 1 + (0.2 * min(0.5, max(-0.5, vol_trend)))
        else:
            vol_trend = 0
            vol_trend_factor = 1
        
        # 7. Combine probabilities with weights
        weights = {
            'historical': 0.35,
            'normal_dist': 0.15,
            't_dist': 0.25,
            'empirical': 0.25
        }
        
        base_probability = (
            weights['historical'] * historical_probability +
            weights['normal_dist'] * normal_probability +
            weights['t_dist'] * t_probability +
            weights['empirical'] * empirical_probability
        )
        
        # Apply risk factors
        adjusted_probability = min(0.95, base_probability * volume_factor * position_factor * vol_trend_factor)
        
        # 8. Determine risk category
        if adjusted_probability < 0.30:
            risk_category = "Low"
        elif adjusted_probability < 0.60:
            risk_category = "Moderate"
        else:
            risk_category = "High"
        
        # 9. Compile results
        results = {
            "Symbol": symbol,
            "Drastic_Threshold": drastic_threshold,
            "Risk_Probability": adjusted_probability,
            "Risk_Category": risk_category
        }
        
        return results
        
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        return {"Symbol": symbol, "Error": f"Exception during analysis: {str(e)}"}

def get_stock_risk(symbol, data_file=None, df=None, drastic_threshold=0.20):
    """
    Simplified function to get just the risk probability and category for a stock.
    
    Parameters:
    symbol (str): Stock symbol to analyze
    data_file (str, optional): Path to CSV data file (required if df is None)
    df (DataFrame, optional): Preprocessed dataframe (if already loaded)
    drastic_threshold (float): Threshold for defining drastic movement (default: 0.20 or 20%)
    
    Returns:
    dict: Dictionary containing only Symbol, Risk_Probability, and Risk_Category
    """
    # Load data if needed
    if df is None:
        if data_file is None:
            return {"Symbol": symbol, "Error": "No data provided. Need either data_file or df."}
            
        try:
            # Try both filename formats
            try:
                df = preprocess_jse_data(data_file)
            except FileNotFoundError:
                alt_file = "all_stock_data_20242025.csv" if "-" in data_file else "all_stock_data_2024-2025.csv"
                df = preprocess_jse_data(alt_file)
        except:
            return {"Symbol": symbol, "Error": "Could not load data file."}
    
    # Check if symbol exists
    if symbol not in df['Symbol'].unique():
        # Try to find alternative
        alt_symbol = find_alternative_symbol(df, symbol)
        if alt_symbol:
            symbol = alt_symbol
        else:
            return {"Symbol": symbol, "Error": "Symbol not found and no valid alternative."}
    
    # Get risk analysis
    result = jse_risk_model(df, symbol, drastic_threshold=drastic_threshold)
    
    # Return just the risk info or the error
    if 'Error' in result:
        return result
    else:
        return {
            "Symbol": result["Symbol"],
            "Risk_Probability": result["Risk_Probability"],
            "Risk_Category": result["Risk_Category"],
            "Drastic_Threshold": result["Drastic_Threshold"]
        }

# Command line interface
if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='JSE Stock Risk Model - Simplified')
    parser.add_argument('--threshold', type=float, default=0.20,
                        help='Threshold for defining drastic movement (default: 0.20 or 20%%)')
    parser.add_argument('--symbol', type=str, required=True,
                        help='Stock symbol to analyze')
    parser.add_argument('--datafile', type=str, default="all_stock_data_2024-2025.csv",
                        help='Path to data file (default: all_stock_data_2024-2025.csv)')
    
    args = parser.parse_args()
    
    # Get risk analysis
    result = get_stock_risk(args.symbol, args.datafile, drastic_threshold=args.threshold)
    
    # Output results
    if 'Error' in result:
        print(f"ERROR: {result['Error']}")
        sys.exit(1)
    else:
        print(f"Symbol: {result['Symbol']}")
        print(f"Risk Probability: {result['Risk_Probability']:.4f}")
        print(f"Risk Category: {result['Risk_Category']}")
        print(f"Drastic Threshold: {result['Drastic_Threshold']*100:.0f}%")