import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json


api_key = 'CG-QaXFmHaFHT7gA7iaAvivbV4t'

# Function to fetch data from the API
def fetch_data(coin_id, vs_currency, numdays):
    url = f"https://pro-api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency={vs_currency}&days={numdays}&x_cg_pro_api_key={api_key}"
    headers = {'x-cg-pro-api-key': api_key} 
    req = requests.get(url)
    #coin_data = req.json()
    coin_data = json.loads(req.text)

    #print(f"Fetching data for {coin_id}: {coin_data.keys()}")

    # Process data into a DataFrame
    dates = [datetime.fromtimestamp(item[0] / 1000) for i, item in enumerate(coin_data['prices']) if i % 4 == 0]
    prices = [item[1] for i, item in enumerate(coin_data['prices']) if i % 4 == 0]

    
    df = pd.DataFrame({'date': dates, 'price': prices})
    return df

# Array of all coins
coins = ['dogwifcoin', 'bonk', 'myro', 'popcat', 'wen-4', 'donald-tremp', 'maga', 'jeo-boden', 'zack-morris', 'habibi-sol', 'daddy-tate', 'american-coin', 'peng', 'gigachad-2', 'monkeyhaircut', 'fwog', 'mumu-the-bull-3', 'ponke', 'sillynubcat', 'aura-on-sol', 'lock-in'] 

def calculate_ema(df, column_name, span):
    # Calculate Exponential Moving Average
    return df[column_name].ewm(span=span, adjust=False).mean()

def calculate_rsi(df, column_name, period=14):
    # Calculate RSI
    delta = df[column_name].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate MACD
def calculate_macd(df, column_name, fast_span=12, slow_span=26, signal_span=9):
    # Calculate fast and slow EMAs
    fast_ema = df[column_name].ewm(span=fast_span, adjust=False).mean()
    slow_ema = df[column_name].ewm(span=slow_span, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Calculate Signal line
    signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()
    
    # Store the MACD and signal line in the DataFrame
    df['MACD'] = macd_line
    df['Signal'] = signal_line
    
    return df

trend_strength_dict = {}

#_____________________________________________________________________________________________________________________________________
# Function to perform ratio analysis for dogwifcoin
def analyze_dogwifcoin(coins):
    base_coin = 'dogwifcoin'
    df_base = fetch_data(base_coin, 'usd', 40)  # Fetch base coin data

    total_trend_strength = 0

    for comp_coin in coins:
        if comp_coin != base_coin:  # Skip if the comparison coin is the same
            df_comp = fetch_data(comp_coin, 'usd', 40)  # Fetch comparison coin data
            
            # Merge the data on 'date' to ensure alignment
            merged_df = pd.merge_asof(df_base, df_comp, on='date', suffixes=('_dogwif', f'_{comp_coin}'))

            # Calculate the ratio
            merged_df['ratio'] = merged_df['price_dogwif'] / merged_df[f'price_{comp_coin}']

             # Calculate 12 and 21 EMA
            merged_df['EMA12'] = calculate_ema(merged_df, 'ratio', span = 12)
            merged_df['EMA21'] = calculate_ema(merged_df, 'ratio', span = 21)

            #Calculate RSI
            merged_df['RSI'] = calculate_rsi(merged_df, 'ratio')

            # **Calculate MACD**
            merged_df = calculate_macd(merged_df, 'ratio')

            # Determine trend strength based on EMA and RSI
            ema_trend = merged_df['EMA12'].iloc[-1] > merged_df['EMA21'].iloc[-1]  # True if EMA12 > EMA21 (bullish)
            rsi_trend = merged_df['RSI'].iloc[-1] > 50  # True if RSI > 50 (bullish)
            macd_trend = merged_df['MACD'].iloc[-1] > merged_df['Signal'].iloc[-1]  # True if MACD > Signal (bullish)

              # Initialize trend strength
            trend_strength = 0
            
            # Only add to trend strength if both EMA and RSI are positive
            if ema_trend:
                trend_strength += 1

            if rsi_trend:
                trend_strength += 1

            if macd_trend:
                trend_strength += 1

            # Add the current pair's trend strength to the total
            total_trend_strength += trend_strength

            # Store total trend strength for dogwifcoin
            trend_strength_dict[base_coin] = total_trend_strength

            # Print or store the trend strength for this pair
            print(f"Trend strength for {base_coin} vs {comp_coin}: {trend_strength}")

            # # Plotting the data
            # fig, ax1 = plt.subplots()

            # # Plot Ratio and EMAs on the primary y-axis
            # ax1.plot(merged_df['date'], merged_df['ratio'], label='Ratio')
            # ax1.plot(merged_df['date'], merged_df['EMA12'], label='EMA 12', linestyle='--', color='orange')
            # ax1.plot(merged_df['date'], merged_df['EMA21'], label='EMA 21', linestyle='--', color='red')
            # ax1.set_xlabel('Date')
            # ax1.set_ylabel('Ratio')
            # ax1.legend(loc='upper left')

            # # Create secondary y-axis for MACD and Signal
            # ax2 = ax1.twinx()
            # ax2.plot(merged_df['date'], merged_df['MACD'], label='MACD', color='blue')
            # ax2.plot(merged_df['date'], merged_df['Signal'], label='Signal', color='green')
            # ax2.set_ylabel('MACD')
            # ax2.legend(loc='upper right')

            # plt.title(f"{base_coin} vs {comp_coin} Ratio")
            # plt.xticks(rotation=45)
            # plt.tight_layout()
            # plt.show()           
#_____________________________________________________________________________________________________________________________________

#_____________________________________________________________________________________________________________________________________
# Function to perform ratio analysis for bonk
def analyze_bonk(coins):
    base_coin = 'bonk'
    df_base = fetch_data(base_coin, 'usd', 40)  # Fetch base coin data

    total_trend_strength = 0

    # Calculate the EMA for the base coin
    df_base['EMA'] = df_base['price'].ewm(span=12, adjust=False).mean()  # 12-period EMA

    for comp_coin in coins:
        if comp_coin != base_coin:  # Skip if the comparison coin is the same
            df_comp = fetch_data(comp_coin, 'usd', 40)  # Fetch comparison coin data
            
            # Merge the data on 'date' to ensure alignment
            merged_df = pd.merge_asof(df_base, df_comp, on='date', suffixes=('_bonk', f'_{comp_coin}'))

            # Calculate the ratio
            merged_df['ratio'] = merged_df['price_bonk'] / merged_df[f'price_{comp_coin}']

            # Calculate 12 and 21 EMA
            merged_df['EMA12'] = calculate_ema(merged_df, 'ratio', span = 12)
            merged_df['EMA21'] = calculate_ema(merged_df, 'ratio', span = 21)

            #Calculate RSI
            merged_df['RSI'] = calculate_rsi(merged_df, 'ratio')

            # **Calculate MACD**
            merged_df = calculate_macd(merged_df, 'ratio')

            # Determine trend strength based on EMA and RSI
            ema_trend = merged_df['EMA12'].iloc[-1] > merged_df['EMA21'].iloc[-1]  # True if EMA12 > EMA21 (bullish)
            rsi_trend = merged_df['RSI'].iloc[-1] > 50  # True if RSI > 50 (bullish)
            macd_trend = merged_df['MACD'].iloc[-1] > merged_df['Signal'].iloc[-1]  # True if MACD > Signal (bullish)

              # Initialize trend strength
            trend_strength = 0
            
            # Only add to trend strength if both EMA and RSI are positive
            if ema_trend:
                trend_strength += 1

            if rsi_trend:
                trend_strength += 1

            if macd_trend:
                trend_strength += 1

            # Add the current pair's trend strength to the total
            total_trend_strength += trend_strength

            # Store total trend strength for dogwifcoin
            trend_strength_dict[base_coin] = total_trend_strength

            # Print or store the trend strength for this pair
            print(f"Trend strength for {base_coin} vs {comp_coin}: {trend_strength}")

            #  # Plotting the data
            # fig, ax1 = plt.subplots()

            # # Plot Ratio and EMAs on the primary y-axis
            # ax1.plot(merged_df['date'], merged_df['ratio'], label='Ratio')
            # ax1.plot(merged_df['date'], merged_df['EMA12'], label='EMA 12', linestyle='--', color='orange')
            # ax1.plot(merged_df['date'], merged_df['EMA21'], label='EMA 21', linestyle='--', color='red')
            # ax1.set_xlabel('Date')
            # ax1.set_ylabel('Ratio')
            # ax1.legend(loc='upper left')

            # # Create secondary y-axis for MACD and Signal
            # ax2 = ax1.twinx()
            # ax2.plot(merged_df['date'], merged_df['MACD'], label='MACD', color='blue')
            # ax2.plot(merged_df['date'], merged_df['Signal'], label='Signal', color='green')
            # ax2.set_ylabel('MACD')
            # ax2.legend(loc='upper right')

            # plt.title(f"{base_coin} vs {comp_coin} Ratio")
            # plt.xticks(rotation=45)
            # plt.tight_layout()
            # plt.show()
#_____________________________________________________________________________________________________________________________________

#_____________________________________________________________________________________________________________________________________
# Function to perform ratio analysis for daddy-tate
def analyze_myro(coins):
    base_coin = 'myro'
    df_base = fetch_data(base_coin, 'usd', 40)  # Fetch base coin data

    total_trend_strength = 0

    # Calculate the EMA for the base coin
    df_base['EMA'] = df_base['price'].ewm(span=12, adjust=False).mean()  # 12-period EMA

    for comp_coin in coins:
        if comp_coin != base_coin:  # Skip if the comparison coin is the same
            df_comp = fetch_data(comp_coin, 'usd', 40)  # Fetch comparison coin data
            
            # Merge the data on 'date' to ensure alignment
            merged_df = pd.merge_asof(df_base, df_comp, on='date', suffixes=('_myro', f'_{comp_coin}'))

            # Calculate the ratio
            merged_df['ratio'] = merged_df['price_myro'] / merged_df[f'price_{comp_coin}']

            # Calculate 12 and 21 EMA
            merged_df['EMA12'] = calculate_ema(merged_df, 'ratio', span = 12)
            merged_df['EMA21'] = calculate_ema(merged_df, 'ratio', span = 21)

            #Calculate RSI
            merged_df['RSI'] = calculate_rsi(merged_df, 'ratio')

            # **Calculate MACD**
            merged_df = calculate_macd(merged_df, 'ratio')

            # Determine trend strength based on EMA and RSI
            ema_trend = merged_df['EMA12'].iloc[-1] > merged_df['EMA21'].iloc[-1]  # True if EMA12 > EMA21 (bullish)
            rsi_trend = merged_df['RSI'].iloc[-1] > 50  # True if RSI > 50 (bullish)
            macd_trend = merged_df['MACD'].iloc[-1] > merged_df['Signal'].iloc[-1]  # True if MACD > Signal (bullish)

              # Initialize trend strength
            trend_strength = 0
            
            # Only add to trend strength if both EMA and RSI are positive
            if ema_trend:
                trend_strength += 1

            if rsi_trend:
                trend_strength += 1

            if macd_trend:
                trend_strength += 1

            # Add the current pair's trend strength to the total
            total_trend_strength += trend_strength

            # Store total trend strength for dogwifcoin
            trend_strength_dict[base_coin] = total_trend_strength

            # Print or store the trend strength for this pair
            print(f"Trend strength for {base_coin} vs {comp_coin}: {trend_strength}")

            # # Plotting the data
            # fig, ax1 = plt.subplots()

            # # Plot Ratio and EMAs on the primary y-axis
            # ax1.plot(merged_df['date'], merged_df['ratio'], label='Ratio')
            # ax1.plot(merged_df['date'], merged_df['EMA12'], label='EMA 12', linestyle='--', color='orange')
            # ax1.plot(merged_df['date'], merged_df['EMA21'], label='EMA 21', linestyle='--', color='red')
            # ax1.set_xlabel('Date')
            # ax1.set_ylabel('Ratio')
            # ax1.legend(loc='upper left')

            # # Create secondary y-axis for MACD and Signal
            # ax2 = ax1.twinx()
            # ax2.plot(merged_df['date'], merged_df['MACD'], label='MACD', color='blue')
            # ax2.plot(merged_df['date'], merged_df['Signal'], label='Signal', color='green')
            # ax2.set_ylabel('MACD')
            # ax2.legend(loc='upper right')

            # plt.title(f"{base_coin} vs {comp_coin} Ratio")
            # plt.xticks(rotation=45)
            # plt.tight_layout()
            # plt.show()
#_____________________________________________________________________________________________________________________________________

#_____________________________________________________________________________________________________________________________________
# Function to perform ratio analysis for popcat
def analyze_popcat(coins):
    base_coin = 'popcat'
    df_base = fetch_data(base_coin, 'usd', 40)  # Fetch base coin data

    total_trend_strength = 0

    # Calculate the EMA for the base coin
    df_base['EMA'] = df_base['price'].ewm(span=12, adjust=False).mean()  # 12-period EMA

    for comp_coin in coins:
        if comp_coin != base_coin:  # Skip if the comparison coin is the same
            df_comp = fetch_data(comp_coin, 'usd', 40)  # Fetch comparison coin data
            
            # Merge the data on 'date' to ensure alignment
            merged_df = pd.merge_asof(df_base, df_comp, on='date', suffixes=('_popcat', f'_{comp_coin}'))

            # Calculate the ratio
            merged_df['ratio'] = merged_df['price_popcat'] / merged_df[f'price_{comp_coin}']

            # Calculate 12 and 21 EMA
            merged_df['EMA12'] = calculate_ema(merged_df, 'ratio', span = 12)
            merged_df['EMA21'] = calculate_ema(merged_df, 'ratio', span = 21)

            #Calculate RSI
            merged_df['RSI'] = calculate_rsi(merged_df, 'ratio')

            # **Calculate MACD**
            merged_df = calculate_macd(merged_df, 'ratio')

            # Determine trend strength based on EMA and RSI
            ema_trend = merged_df['EMA12'].iloc[-1] > merged_df['EMA21'].iloc[-1]  # True if EMA12 > EMA21 (bullish)
            rsi_trend = merged_df['RSI'].iloc[-1] > 50  # True if RSI > 50 (bullish)
            macd_trend = merged_df['MACD'].iloc[-1] > merged_df['Signal'].iloc[-1]  # True if MACD > Signal (bullish)

              # Initialize trend strength
            trend_strength = 0
            
            # Only add to trend strength if both EMA and RSI are positive
            if ema_trend:
                trend_strength += 1

            if rsi_trend:
                trend_strength += 1

            if macd_trend:
                trend_strength +- 1

            # Add the current pair's trend strength to the total
            total_trend_strength += trend_strength

            # Store total trend strength for dogwifcoin
            trend_strength_dict[base_coin] = total_trend_strength

            # Print or store the trend strength for this pair
            print(f"Trend strength for {base_coin} vs {comp_coin}: {trend_strength}")

            # # Plotting the data
            # fig, ax1 = plt.subplots()

            # # Plot Ratio and EMAs on the primary y-axis
            # ax1.plot(merged_df['date'], merged_df['ratio'], label='Ratio')
            # ax1.plot(merged_df['date'], merged_df['EMA12'], label='EMA 12', linestyle='--', color='orange')
            # ax1.plot(merged_df['date'], merged_df['EMA21'], label='EMA 21', linestyle='--', color='red')
            # ax1.set_xlabel('Date')
            # ax1.set_ylabel('Ratio')
            # ax1.legend(loc='upper left')

            # # Create secondary y-axis for MACD and Signal
            # ax2 = ax1.twinx()
            # ax2.plot(merged_df['date'], merged_df['MACD'], label='MACD', color='blue')
            # ax2.plot(merged_df['date'], merged_df['Signal'], label='Signal', color='green')
            # ax2.set_ylabel('MACD')
            # ax2.legend(loc='upper right')

            # plt.title(f"{base_coin} vs {comp_coin} Ratio")
            # plt.xticks(rotation=45)
            # plt.tight_layout()
            # plt.show()
#_____________________________________________________________________________________________________________________________________

#_____________________________________________________________________________________________________________________________________
# Function to perform ratio analysis for popcat
def analyze_wen(coins):
    base_coin = 'wen-4'
    df_base = fetch_data(base_coin, 'usd', 40)  # Fetch base coin data

    total_trend_strength = 0

    # Calculate the EMA for the base coin
    df_base['EMA'] = df_base['price'].ewm(span=12, adjust=False).mean()  # 12-period EMA

    for comp_coin in coins:
        if comp_coin != base_coin:  # Skip if the comparison coin is the same
            df_comp = fetch_data(comp_coin, 'usd', 40)  # Fetch comparison coin data
            
            # Merge the data on 'date' to ensure alignment
            merged_df = pd.merge_asof(df_base, df_comp, on='date', suffixes=('_wen-4', f'_{comp_coin}'))

            # Calculate the ratio
            merged_df['ratio'] = merged_df['price_wen-4'] / merged_df[f'price_{comp_coin}']

            # Calculate 12 and 21 EMA
            merged_df['EMA12'] = calculate_ema(merged_df, 'ratio', span = 12)
            merged_df['EMA21'] = calculate_ema(merged_df, 'ratio', span = 21)

            #Calculate RSI
            merged_df['RSI'] = calculate_rsi(merged_df, 'ratio')

            # **Calculate MACD**
            merged_df = calculate_macd(merged_df, 'ratio')

            # Determine trend strength based on EMA and RSI
            ema_trend = merged_df['EMA12'].iloc[-1] > merged_df['EMA21'].iloc[-1]  # True if EMA12 > EMA21 (bullish)
            rsi_trend = merged_df['RSI'].iloc[-1] > 50  # True if RSI > 50 (bullish)
            macd_trend = merged_df['MACD'].iloc[-1] > merged_df['Signal'].iloc[-1]  # True if MACD > Signal (bullish)

              # Initialize trend strength
            trend_strength = 0
            
            # Only add to trend strength if both EMA and RSI are positive
            if ema_trend:
                trend_strength += 1

            if rsi_trend:
                trend_strength += 1

            if macd_trend:
                trend_strength += 1

            # Add the current pair's trend strength to the total
            total_trend_strength += trend_strength

            # Store total trend strength for dogwifcoin
            trend_strength_dict[base_coin] = total_trend_strength

            # Print or store the trend strength for this pair
            print(f"Trend strength for {base_coin} vs {comp_coin}: {trend_strength}")

            #  # Plotting the data
            # fig, ax1 = plt.subplots()

            # # Plot Ratio and EMAs on the primary y-axis
            # ax1.plot(merged_df['date'], merged_df['ratio'], label='Ratio')
            # ax1.plot(merged_df['date'], merged_df['EMA12'], label='EMA 12', linestyle='--', color='orange')
            # ax1.plot(merged_df['date'], merged_df['EMA21'], label='EMA 21', linestyle='--', color='red')
            # ax1.set_xlabel('Date')
            # ax1.set_ylabel('Ratio')
            # ax1.legend(loc='upper left')

            # # Create secondary y-axis for MACD and Signal
            # ax2 = ax1.twinx()
            # ax2.plot(merged_df['date'], merged_df['MACD'], label='MACD', color='blue')
            # ax2.plot(merged_df['date'], merged_df['Signal'], label='Signal', color='green')
            # ax2.set_ylabel('MACD')
            # ax2.legend(loc='upper right')

            # plt.title(f"{base_coin} vs {comp_coin} Ratio")
            # plt.xticks(rotation=45)
            # plt.tight_layout()
            # plt.show()
#_____________________________________________________________________________________________________________________________________

#_____________________________________________________________________________________________________________________________________
# Function to perform ratio analysis for popcat
def analyze_tremp(coins):
    base_coin = 'donald-tremp'
    df_base = fetch_data(base_coin, 'usd', 40)  # Fetch base coin data

    total_trend_strength = 0

    # Calculate the EMA for the base coin
    df_base['EMA'] = df_base['price'].ewm(span=12, adjust=False).mean()  # 12-period EMA

    for comp_coin in coins:
        if comp_coin != base_coin:  # Skip if the comparison coin is the same
            df_comp = fetch_data(comp_coin, 'usd', 40)  # Fetch comparison coin data
            
            # Merge the data on 'date' to ensure alignment
            merged_df = pd.merge_asof(df_base, df_comp, on='date', suffixes=('_donald-tremp', f'_{comp_coin}'))

            # Calculate the ratio
            merged_df['ratio'] = merged_df['price_donald-tremp'] / merged_df[f'price_{comp_coin}']

            # Calculate 12 and 21 EMA
            merged_df['EMA12'] = calculate_ema(merged_df, 'ratio', span = 12)
            merged_df['EMA21'] = calculate_ema(merged_df, 'ratio', span = 21)

            #Calculate RSI
            merged_df['RSI'] = calculate_rsi(merged_df, 'ratio')

            # **Calculate MACD**
            merged_df = calculate_macd(merged_df, 'ratio')

            # Determine trend strength based on EMA and RSI
            ema_trend = merged_df['EMA12'].iloc[-1] > merged_df['EMA21'].iloc[-1]  # True if EMA12 > EMA21 (bullish)
            rsi_trend = merged_df['RSI'].iloc[-1] > 50  # True if RSI > 50 (bullish)
            macd_trend = merged_df['MACD'].iloc[-1] > merged_df['Signal'].iloc[-1]  # True if MACD > Signal (bullish)

              # Initialize trend strength
            trend_strength = 0
            
            # Only add to trend strength if both EMA and RSI are positive
            if ema_trend:
                trend_strength += 1

            if rsi_trend:
                trend_strength += 1

            if macd_trend:
                trend_strength += 1

            # Add the current pair's trend strength to the total
            total_trend_strength += trend_strength

            # Store total trend strength for dogwifcoin
            trend_strength_dict[base_coin] = total_trend_strength

            # Print or store the trend strength for this pair
            print(f"Trend strength for {base_coin} vs {comp_coin}: {trend_strength}")

            # # Plotting the data
            # fig, ax1 = plt.subplots()

            # # Plot Ratio and EMAs on the primary y-axis
            # ax1.plot(merged_df['date'], merged_df['ratio'], label='Ratio')
            # ax1.plot(merged_df['date'], merged_df['EMA12'], label='EMA 12', linestyle='--', color='orange')
            # ax1.plot(merged_df['date'], merged_df['EMA21'], label='EMA 21', linestyle='--', color='red')
            # ax1.set_xlabel('Date')
            # ax1.set_ylabel('Ratio')
            # ax1.legend(loc='upper left')

            # # Create secondary y-axis for MACD and Signal
            # ax2 = ax1.twinx()
            # ax2.plot(merged_df['date'], merged_df['MACD'], label='MACD', color='blue')
            # ax2.plot(merged_df['date'], merged_df['Signal'], label='Signal', color='green')
            # ax2.set_ylabel('MACD')
            # ax2.legend(loc='upper right')

            # plt.title(f"{base_coin} vs {comp_coin} Ratio")
            # plt.xticks(rotation=45)
            # plt.tight_layout()
            # plt.show()
#_____________________________________________________________________________________________________________________________________

#_____________________________________________________________________________________________________________________________________
# Function to perform ratio analysis for popcat
def analyze_maga(coins):
    base_coin = 'maga'
    df_base = fetch_data(base_coin, 'usd', 40)  # Fetch base coin data

    total_trend_strength = 0

    # Calculate the EMA for the base coin
    df_base['EMA'] = df_base['price'].ewm(span=12, adjust=False).mean()  # 12-period EMA

    for comp_coin in coins:
        if comp_coin != base_coin:  # Skip if the comparison coin is the same
            df_comp = fetch_data(comp_coin, 'usd', 40)  # Fetch comparison coin data
            
            # Merge the data on 'date' to ensure alignment
            merged_df = pd.merge_asof(df_base, df_comp, on='date', suffixes=('_maga', f'_{comp_coin}'))

            # Calculate the ratio
            merged_df['ratio'] = merged_df['price_maga'] / merged_df[f'price_{comp_coin}']

            # Calculate 12 and 21 EMA
            merged_df['EMA12'] = calculate_ema(merged_df, 'ratio', span = 12)
            merged_df['EMA21'] = calculate_ema(merged_df, 'ratio', span = 21)

            #Calculate RSI
            merged_df['RSI'] = calculate_rsi(merged_df, 'ratio')

            # **Calculate MACD**
            merged_df = calculate_macd(merged_df, 'ratio')

            # Determine trend strength based on EMA and RSI
            ema_trend = merged_df['EMA12'].iloc[-1] > merged_df['EMA21'].iloc[-1]  # True if EMA12 > EMA21 (bullish)
            rsi_trend = merged_df['RSI'].iloc[-1] > 50  # True if RSI > 50 (bullish)
            macd_trend = merged_df['MACD'].iloc[-1] > merged_df['Signal'].iloc[-1]  # True if MACD > Signal (bullish)

              # Initialize trend strength
            trend_strength = 0
            
            # Only add to trend strength if both EMA and RSI are positive
            if ema_trend:
                trend_strength += 1

            if rsi_trend:
                trend_strength += 1

            if macd_trend:
                trend_strength += 1

            # Add the current pair's trend strength to the total
            total_trend_strength += trend_strength

            # Store total trend strength for dogwifcoin
            trend_strength_dict[base_coin] = total_trend_strength

            # Print or store the trend strength for this pair
            print(f"Trend strength for {base_coin} vs {comp_coin}: {trend_strength}")

            # # Plotting the data
            # fig, ax1 = plt.subplots()

            # # Plot Ratio and EMAs on the primary y-axis
            # ax1.plot(merged_df['date'], merged_df['ratio'], label='Ratio')
            # ax1.plot(merged_df['date'], merged_df['EMA12'], label='EMA 12', linestyle='--', color='orange')
            # ax1.plot(merged_df['date'], merged_df['EMA21'], label='EMA 21', linestyle='--', color='red')
            # ax1.set_xlabel('Date')
            # ax1.set_ylabel('Ratio')
            # ax1.legend(loc='upper left')

            # # Create secondary y-axis for MACD and Signal
            # ax2 = ax1.twinx()
            # ax2.plot(merged_df['date'], merged_df['MACD'], label='MACD', color='blue')
            # ax2.plot(merged_df['date'], merged_df['Signal'], label='Signal', color='green')
            # ax2.set_ylabel('MACD')
            # ax2.legend(loc='upper right')

            # plt.title(f"{base_coin} vs {comp_coin} Ratio")
            # plt.xticks(rotation=45)
            # plt.tight_layout()
            # plt.show()
#_____________________________________________________________________________________________________________________________________

#_____________________________________________________________________________________________________________________________________
# Function to perform ratio analysis for popcat
def analyze_boden(coins):
    base_coin = 'jeo-boden'
    df_base = fetch_data(base_coin, 'usd', 40)  # Fetch base coin data

    total_trend_strength = 0

    # Calculate the EMA for the base coin
    df_base['EMA'] = df_base['price'].ewm(span=12, adjust=False).mean()  # 12-period EMA

    for comp_coin in coins:
        if comp_coin != base_coin:  # Skip if the comparison coin is the same
            df_comp = fetch_data(comp_coin, 'usd', 40)  # Fetch comparison coin data
            
            # Merge the data on 'date' to ensure alignment
            merged_df = pd.merge_asof(df_base, df_comp, on='date', suffixes=('_jeo-boden', f'_{comp_coin}'))

            # Calculate the ratio
            merged_df['ratio'] = merged_df['price_jeo-boden'] / merged_df[f'price_{comp_coin}']

            # Calculate 12 and 21 EMA
            merged_df['EMA12'] = calculate_ema(merged_df, 'ratio', span = 12)
            merged_df['EMA21'] = calculate_ema(merged_df, 'ratio', span = 21)

            #Calculate RSI
            merged_df['RSI'] = calculate_rsi(merged_df, 'ratio')

            # **Calculate MACD**
            merged_df = calculate_macd(merged_df, 'ratio')

            # Determine trend strength based on EMA and RSI
            ema_trend = merged_df['EMA12'].iloc[-1] > merged_df['EMA21'].iloc[-1]  # True if EMA12 > EMA21 (bullish)
            rsi_trend = merged_df['RSI'].iloc[-1] > 50  # True if RSI > 50 (bullish)
            macd_trend = merged_df['MACD'].iloc[-1] > merged_df['Signal'].iloc[-1]  # True if MACD > Signal (bullish)

              # Initialize trend strength
            trend_strength = 0
            
            # Only add to trend strength if both EMA and RSI are positive
            if ema_trend:
                trend_strength += 1

            if rsi_trend:
                trend_strength += 1

            if macd_trend:
                trend_strength += 1

            # Add the current pair's trend strength to the total
            total_trend_strength += trend_strength

            # Store total trend strength for dogwifcoin
            trend_strength_dict[base_coin] = total_trend_strength

            # Print or store the trend strength for this pair
            print(f"Trend strength for {base_coin} vs {comp_coin}: {trend_strength}")

            # # Plotting the data
            # fig, ax1 = plt.subplots()

            # # Plot Ratio and EMAs on the primary y-axis
            # ax1.plot(merged_df['date'], merged_df['ratio'], label='Ratio')
            # ax1.plot(merged_df['date'], merged_df['EMA12'], label='EMA 12', linestyle='--', color='orange')
            # ax1.plot(merged_df['date'], merged_df['EMA21'], label='EMA 21', linestyle='--', color='red')
            # ax1.set_xlabel('Date')
            # ax1.set_ylabel('Ratio')
            # ax1.legend(loc='upper left')

            # # Create secondary y-axis for MACD and Signal
            # ax2 = ax1.twinx()
            # ax2.plot(merged_df['date'], merged_df['MACD'], label='MACD', color='blue')
            # ax2.plot(merged_df['date'], merged_df['Signal'], label='Signal', color='green')
            # ax2.set_ylabel('MACD')
            # ax2.legend(loc='upper right')

            # plt.title(f"{base_coin} vs {comp_coin} Ratio")
            # plt.xticks(rotation=45)
            # plt.tight_layout()
            # plt.show()
#_____________________________________________________________________________________________________________________________________

#_____________________________________________________________________________________________________________________________________
# Function to perform ratio analysis for popcat
def analyze_zack(coins):
    base_coin = 'zack-morris'
    df_base = fetch_data(base_coin, 'usd', 40)  # Fetch base coin data

    total_trend_strength = 0

    # Calculate the EMA for the base coin
    df_base['EMA'] = df_base['price'].ewm(span=12, adjust=False).mean()  # 12-period EMA

    for comp_coin in coins:
        if comp_coin != base_coin:  # Skip if the comparison coin is the same
            df_comp = fetch_data(comp_coin, 'usd', 40)  # Fetch comparison coin data
            
            # Merge the data on 'date' to ensure alignment
            merged_df = pd.merge_asof(df_base, df_comp, on='date', suffixes=('_zack-morris', f'_{comp_coin}'))

            # Calculate the ratio
            merged_df['ratio'] = merged_df['price_zack-morris'] / merged_df[f'price_{comp_coin}']

            # Calculate 12 and 21 EMA
            merged_df['EMA12'] = calculate_ema(merged_df, 'ratio', span = 12)
            merged_df['EMA21'] = calculate_ema(merged_df, 'ratio', span = 21)

            #Calculate RSI
            merged_df['RSI'] = calculate_rsi(merged_df, 'ratio')

            # **Calculate MACD**
            merged_df = calculate_macd(merged_df, 'ratio')

            # Determine trend strength based on EMA and RSI
            ema_trend = merged_df['EMA12'].iloc[-1] > merged_df['EMA21'].iloc[-1]  # True if EMA12 > EMA21 (bullish)
            rsi_trend = merged_df['RSI'].iloc[-1] > 50  # True if RSI > 50 (bullish)
            macd_trend = merged_df['MACD'].iloc[-1] > merged_df['Signal'].iloc[-1]  # True if MACD > Signal (bullish)

              # Initialize trend strength
            trend_strength = 0
            
            # Only add to trend strength if both EMA and RSI are positive
            if ema_trend:
                trend_strength += 1

            if rsi_trend:
                trend_strength += 1

            if macd_trend:
                trend_strength += 1

            # Add the current pair's trend strength to the total
            total_trend_strength += trend_strength

            # Store total trend strength for dogwifcoin
            trend_strength_dict[base_coin] = total_trend_strength

            # Print or store the trend strength for this pair
            print(f"Trend strength for {base_coin} vs {comp_coin}: {trend_strength}")

            # # Plotting the data
            # fig, ax1 = plt.subplots()

            # # Plot Ratio and EMAs on the primary y-axis
            # ax1.plot(merged_df['date'], merged_df['ratio'], label='Ratio')
            # ax1.plot(merged_df['date'], merged_df['EMA12'], label='EMA 12', linestyle='--', color='orange')
            # ax1.plot(merged_df['date'], merged_df['EMA21'], label='EMA 21', linestyle='--', color='red')
            # ax1.set_xlabel('Date')
            # ax1.set_ylabel('Ratio')
            # ax1.legend(loc='upper left')

            # # Create secondary y-axis for MACD and Signal
            # ax2 = ax1.twinx()
            # ax2.plot(merged_df['date'], merged_df['MACD'], label='MACD', color='blue')
            # ax2.plot(merged_df['date'], merged_df['Signal'], label='Signal', color='green')
            # ax2.set_ylabel('MACD')
            # ax2.legend(loc='upper right')

            # plt.title(f"{base_coin} vs {comp_coin} Ratio")
            # plt.xticks(rotation=45)
            # plt.tight_layout()
            # plt.show()
#_____________________________________________________________________________________________________________________________________

#_____________________________________________________________________________________________________________________________________
# Function to perform ratio analysis for popcat
def analyze_habibi(coins):
    base_coin = 'habibi-sol'
    df_base = fetch_data(base_coin, 'usd', 40)  # Fetch base coin data

    total_trend_strength = 0

    # Calculate the EMA for the base coin
    df_base['EMA'] = df_base['price'].ewm(span=12, adjust=False).mean()  # 12-period EMA

    for comp_coin in coins:
        if comp_coin != base_coin:  # Skip if the comparison coin is the same
            df_comp = fetch_data(comp_coin, 'usd', 40)  # Fetch comparison coin data
            
            # Merge the data on 'date' to ensure alignment
            merged_df = pd.merge_asof(df_base, df_comp, on='date', suffixes=('_habibi-sol', f'_{comp_coin}'))

            # Calculate the ratio
            merged_df['ratio'] = merged_df['price_habibi-sol'] / merged_df[f'price_{comp_coin}']

            # Calculate 12 and 21 EMA
            merged_df['EMA12'] = calculate_ema(merged_df, 'ratio', span = 12)
            merged_df['EMA21'] = calculate_ema(merged_df, 'ratio', span = 21)

            #Calculate RSI
            merged_df['RSI'] = calculate_rsi(merged_df, 'ratio')

            # **Calculate MACD**
            merged_df = calculate_macd(merged_df, 'ratio')

            # Determine trend strength based on EMA and RSI
            ema_trend = merged_df['EMA12'].iloc[-1] > merged_df['EMA21'].iloc[-1]  # True if EMA12 > EMA21 (bullish)
            rsi_trend = merged_df['RSI'].iloc[-1] > 50  # True if RSI > 50 (bullish)
            macd_trend = merged_df['MACD'].iloc[-1] > merged_df['Signal'].iloc[-1]  # True if MACD > Signal (bullish)

              # Initialize trend strength
            trend_strength = 0
            
            # Only add to trend strength if both EMA and RSI are positive
            if ema_trend:
                trend_strength += 1

            if rsi_trend:
                trend_strength += 1

            if macd_trend:
                trend_strength += 1

            # Add the current pair's trend strength to the total
            total_trend_strength += trend_strength

            # Store total trend strength for dogwifcoin
            trend_strength_dict[base_coin] = total_trend_strength

            # Print or store the trend strength for this pair
            print(f"Trend strength for {base_coin} vs {comp_coin}: {trend_strength}")

            # # Plotting the data
            # fig, ax1 = plt.subplots()

            # # Plot Ratio and EMAs on the primary y-axis
            # ax1.plot(merged_df['date'], merged_df['ratio'], label='Ratio')
            # ax1.plot(merged_df['date'], merged_df['EMA12'], label='EMA 12', linestyle='--', color='orange')
            # ax1.plot(merged_df['date'], merged_df['EMA21'], label='EMA 21', linestyle='--', color='red')
            # ax1.set_xlabel('Date')
            # ax1.set_ylabel('Ratio')
            # ax1.legend(loc='upper left')

            # # Create secondary y-axis for MACD and Signal
            # ax2 = ax1.twinx()
            # ax2.plot(merged_df['date'], merged_df['MACD'], label='MACD', color='blue')
            # ax2.plot(merged_df['date'], merged_df['Signal'], label='Signal', color='green')
            # ax2.set_ylabel('MACD')
            # ax2.legend(loc='upper right')

            # plt.title(f"{base_coin} vs {comp_coin} Ratio")
            # plt.xticks(rotation=45)
            # plt.tight_layout()
            # plt.show()
#_____________________________________________________________________________________________________________________________________

#_____________________________________________________________________________________________________________________________________
# Function to perform ratio analysis for popcat
def analyze_daddy(coins):
    base_coin = 'daddy-tate'
    df_base = fetch_data(base_coin, 'usd', 40)  # Fetch base coin data

    total_trend_strength = 0

    # Calculate the EMA for the base coin
    df_base['EMA'] = df_base['price'].ewm(span=12, adjust=False).mean()  # 12-period EMA

    for comp_coin in coins:
        if comp_coin != base_coin:  # Skip if the comparison coin is the same
            df_comp = fetch_data(comp_coin, 'usd', 40)  # Fetch comparison coin data
            
            # Merge the data on 'date' to ensure alignment
            merged_df = pd.merge_asof(df_base, df_comp, on='date', suffixes=('_daddy-tate', f'_{comp_coin}'))

            # Calculate the ratio
            merged_df['ratio'] = merged_df['price_daddy-tate'] / merged_df[f'price_{comp_coin}']

            # Calculate 12 and 21 EMA
            merged_df['EMA12'] = calculate_ema(merged_df, 'ratio', span = 12)
            merged_df['EMA21'] = calculate_ema(merged_df, 'ratio', span = 21)

            #Calculate RSI
            merged_df['RSI'] = calculate_rsi(merged_df, 'ratio')

            # **Calculate MACD**
            merged_df = calculate_macd(merged_df, 'ratio')

            # Determine trend strength based on EMA and RSI
            ema_trend = merged_df['EMA12'].iloc[-1] > merged_df['EMA21'].iloc[-1]  # True if EMA12 > EMA21 (bullish)
            rsi_trend = merged_df['RSI'].iloc[-1] > 50  # True if RSI > 50 (bullish)
            macd_trend = merged_df['MACD'].iloc[-1] > merged_df['Signal'].iloc[-1]  # True if MACD > Signal (bullish)

              # Initialize trend strength
            trend_strength = 0
            
            # Only add to trend strength if both EMA and RSI are positive
            if ema_trend:
                trend_strength += 1

            if rsi_trend:
                trend_strength += 1

            if macd_trend:
                trend_strength += 1

            # Add the current pair's trend strength to the total
            total_trend_strength += trend_strength

            # Store total trend strength for dogwifcoin
            trend_strength_dict[base_coin] = total_trend_strength

            # Print or store the trend strength for this pair
            print(f"Trend strength for {base_coin} vs {comp_coin}: {trend_strength}")

            # # Plotting the data
            # fig, ax1 = plt.subplots()

            # # Plot Ratio and EMAs on the primary y-axis
            # ax1.plot(merged_df['date'], merged_df['ratio'], label='Ratio')
            # ax1.plot(merged_df['date'], merged_df['EMA12'], label='EMA 12', linestyle='--', color='orange')
            # ax1.plot(merged_df['date'], merged_df['EMA21'], label='EMA 21', linestyle='--', color='red')
            # ax1.set_xlabel('Date')
            # ax1.set_ylabel('Ratio')
            # ax1.legend(loc='upper left')

            # # Create secondary y-axis for MACD and Signal
            # ax2 = ax1.twinx()
            # ax2.plot(merged_df['date'], merged_df['MACD'], label='MACD', color='blue')
            # ax2.plot(merged_df['date'], merged_df['Signal'], label='Signal', color='green')
            # ax2.set_ylabel('MACD')
            # ax2.legend(loc='upper right')

            # plt.title(f"{base_coin} vs {comp_coin} Ratio")
            # plt.xticks(rotation=45)
            # plt.tight_layout()
            # plt.show()
#_____________________________________________________________________________________________________________________________________

#_____________________________________________________________________________________________________________________________________
# Function to perform ratio analysis for popcat
def analyze_usacoin(coins):
    base_coin = 'american-coin'
    df_base = fetch_data(base_coin, 'usd', 40)  # Fetch base coin data

    total_trend_strength = 0

    # Calculate the EMA for the base coin
    df_base['EMA'] = df_base['price'].ewm(span=12, adjust=False).mean()  # 12-period EMA

    for comp_coin in coins:
        if comp_coin != base_coin:  # Skip if the comparison coin is the same
            df_comp = fetch_data(comp_coin, 'usd', 40)  # Fetch comparison coin data
            
            # Merge the data on 'date' to ensure alignment
            merged_df = pd.merge_asof(df_base, df_comp, on='date', suffixes=('_american-coin', f'_{comp_coin}'))

            # Calculate the ratio
            merged_df['ratio'] = merged_df['price_american-coin'] / merged_df[f'price_{comp_coin}']

            # Calculate 12 and 21 EMA
            merged_df['EMA12'] = calculate_ema(merged_df, 'ratio', span = 12)
            merged_df['EMA21'] = calculate_ema(merged_df, 'ratio', span = 21)

            #Calculate RSI
            merged_df['RSI'] = calculate_rsi(merged_df, 'ratio')

            # **Calculate MACD**
            merged_df = calculate_macd(merged_df, 'ratio')

            # Determine trend strength based on EMA and RSI
            ema_trend = merged_df['EMA12'].iloc[-1] > merged_df['EMA21'].iloc[-1]  # True if EMA12 > EMA21 (bullish)
            rsi_trend = merged_df['RSI'].iloc[-1] > 50  # True if RSI > 50 (bullish)
            macd_trend = merged_df['MACD'].iloc[-1] > merged_df['Signal'].iloc[-1]  # True if MACD > Signal (bullish)

              # Initialize trend strength
            trend_strength = 0
            
            # Only add to trend strength if both EMA and RSI are positive
            if ema_trend:
                trend_strength += 1

            if rsi_trend:
                trend_strength += 1

            if macd_trend:
                trend_strength += 1

            # Add the current pair's trend strength to the total
            total_trend_strength += trend_strength

            # Store total trend strength for dogwifcoin
            trend_strength_dict[base_coin] = total_trend_strength

            # Print or store the trend strength for this pair
            print(f"Trend strength for {base_coin} vs {comp_coin}: {trend_strength}")

            #  # Plotting the data
            # fig, ax1 = plt.subplots()

            # # Plot Ratio and EMAs on the primary y-axis
            # ax1.plot(merged_df['date'], merged_df['ratio'], label='Ratio')
            # ax1.plot(merged_df['date'], merged_df['EMA12'], label='EMA 12', linestyle='--', color='orange')
            # ax1.plot(merged_df['date'], merged_df['EMA21'], label='EMA 21', linestyle='--', color='red')
            # ax1.set_xlabel('Date')
            # ax1.set_ylabel('Ratio')
            # ax1.legend(loc='upper left')

            # # Create secondary y-axis for MACD and Signal
            # ax2 = ax1.twinx()
            # ax2.plot(merged_df['date'], merged_df['MACD'], label='MACD', color='blue')
            # ax2.plot(merged_df['date'], merged_df['Signal'], label='Signal', color='green')
            # ax2.set_ylabel('MACD')
            # ax2.legend(loc='upper right')

            # plt.title(f"{base_coin} vs {comp_coin} Ratio")
            # plt.xticks(rotation=45)
            # plt.tight_layout()
            # plt.show()
#_____________________________________________________________________________________________________________________________________

#_____________________________________________________________________________________________________________________________________
# Function to perform ratio analysis for popcat
def analyze_peng(coins):
    base_coin = 'peng'
    df_base = fetch_data(base_coin, 'usd', 40)  # Fetch base coin data

    total_trend_strength = 0

    # Calculate the EMA for the base coin
    df_base['EMA'] = df_base['price'].ewm(span=12, adjust=False).mean()  # 12-period EMA

    for comp_coin in coins:
        if comp_coin != base_coin:  # Skip if the comparison coin is the same
            df_comp = fetch_data(comp_coin, 'usd', 40)  # Fetch comparison coin data
            
            # Merge the data on 'date' to ensure alignment
            merged_df = pd.merge_asof(df_base, df_comp, on='date', suffixes=('_peng', f'_{comp_coin}'))

            # Calculate the ratio
            merged_df['ratio'] = merged_df['price_peng'] / merged_df[f'price_{comp_coin}']

            # Calculate 12 and 21 EMA
            merged_df['EMA12'] = calculate_ema(merged_df, 'ratio', span = 12)
            merged_df['EMA21'] = calculate_ema(merged_df, 'ratio', span = 21)

            #Calculate RSI
            merged_df['RSI'] = calculate_rsi(merged_df, 'ratio')

            # **Calculate MACD**
            merged_df = calculate_macd(merged_df, 'ratio')

            # Determine trend strength based on EMA and RSI
            ema_trend = merged_df['EMA12'].iloc[-1] > merged_df['EMA21'].iloc[-1]  # True if EMA12 > EMA21 (bullish)
            rsi_trend = merged_df['RSI'].iloc[-1] > 50  # True if RSI > 50 (bullish)
            macd_trend = merged_df['MACD'].iloc[-1] > merged_df['Signal'].iloc[-1]  # True if MACD > Signal (bullish)

              # Initialize trend strength
            trend_strength = 0
            
            # Only add to trend strength if both EMA and RSI are positive
            if ema_trend:
                trend_strength += 1

            if rsi_trend:
                trend_strength += 1

            if macd_trend:
                trend_strength += 1

            # Add the current pair's trend strength to the total
            total_trend_strength += trend_strength

            # Store total trend strength for dogwifcoin
            trend_strength_dict[base_coin] = total_trend_strength

            # Print or store the trend strength for this pair
            print(f"Trend strength for {base_coin} vs {comp_coin}: {trend_strength}")

            # # Plotting the data
            # fig, ax1 = plt.subplots()

            # # Plot Ratio and EMAs on the primary y-axis
            # ax1.plot(merged_df['date'], merged_df['ratio'], label='Ratio')
            # ax1.plot(merged_df['date'], merged_df['EMA12'], label='EMA 12', linestyle='--', color='orange')
            # ax1.plot(merged_df['date'], merged_df['EMA21'], label='EMA 21', linestyle='--', color='red')
            # ax1.set_xlabel('Date')
            # ax1.set_ylabel('Ratio')
            # ax1.legend(loc='upper left')

            # # Create secondary y-axis for MACD and Signal
            # ax2 = ax1.twinx()
            # ax2.plot(merged_df['date'], merged_df['MACD'], label='MACD', color='blue')
            # ax2.plot(merged_df['date'], merged_df['Signal'], label='Signal', color='green')
            # ax2.set_ylabel('MACD')
            # ax2.legend(loc='upper right')

            # plt.title(f"{base_coin} vs {comp_coin} Ratio")
            # plt.xticks(rotation=45)
            # plt.tight_layout()
            # plt.show()
#_____________________________________________________________________________________________________________________________________

#_____________________________________________________________________________________________________________________________________
# Function to perform ratio analysis for popcat
def analyze_giga(coins):
    base_coin = 'gigachad-2'
    df_base = fetch_data(base_coin, 'usd', 40)  # Fetch base coin data

    total_trend_strength = 0

    # Calculate the EMA for the base coin
    df_base['EMA'] = df_base['price'].ewm(span=12, adjust=False).mean()  # 12-period EMA

    for comp_coin in coins:
        if comp_coin != base_coin:  # Skip if the comparison coin is the same
            df_comp = fetch_data(comp_coin, 'usd', 40)  # Fetch comparison coin data
            
            # Merge the data on 'date' to ensure alignment
            merged_df = pd.merge_asof(df_base, df_comp, on='date', suffixes=('_gigachad-2', f'_{comp_coin}'))

            # Calculate the ratio
            merged_df['ratio'] = merged_df['price_gigachad-2'] / merged_df[f'price_{comp_coin}']

            # Calculate 12 and 21 EMA
            merged_df['EMA12'] = calculate_ema(merged_df, 'ratio', span = 12)
            merged_df['EMA21'] = calculate_ema(merged_df, 'ratio', span = 21)

            #Calculate RSI
            merged_df['RSI'] = calculate_rsi(merged_df, 'ratio')

            # **Calculate MACD**
            merged_df = calculate_macd(merged_df, 'ratio')

            # Determine trend strength based on EMA and RSI
            ema_trend = merged_df['EMA12'].iloc[-1] > merged_df['EMA21'].iloc[-1]  # True if EMA12 > EMA21 (bullish)
            rsi_trend = merged_df['RSI'].iloc[-1] > 50  # True if RSI > 50 (bullish)
            macd_trend = merged_df['MACD'].iloc[-1] > merged_df['Signal'].iloc[-1]  # True if MACD > Signal (bullish)

              # Initialize trend strength
            trend_strength = 0
            
            # Only add to trend strength if both EMA and RSI are positive
            if ema_trend:
                trend_strength += 1

            if rsi_trend:
                trend_strength += 1

            if macd_trend:
                trend_strength += 1

            # Add the current pair's trend strength to the total
            total_trend_strength += trend_strength

            # Store total trend strength for dogwifcoin
            trend_strength_dict[base_coin] = total_trend_strength

            # Print or store the trend strength for this pair
            print(f"Trend strength for {base_coin} vs {comp_coin}: {trend_strength}")

            # # Plotting the data
            # fig, ax1 = plt.subplots()

            # # Plot Ratio and EMAs on the primary y-axis
            # ax1.plot(merged_df['date'], merged_df['ratio'], label='Ratio')
            # ax1.plot(merged_df['date'], merged_df['EMA12'], label='EMA 12', linestyle='--', color='orange')
            # ax1.plot(merged_df['date'], merged_df['EMA21'], label='EMA 21', linestyle='--', color='red')
            # ax1.set_xlabel('Date')
            # ax1.set_ylabel('Ratio')
            # ax1.legend(loc='upper left')

            # # Create secondary y-axis for MACD and Signal
            # ax2 = ax1.twinx()
            # ax2.plot(merged_df['date'], merged_df['MACD'], label='MACD', color='blue')
            # ax2.plot(merged_df['date'], merged_df['Signal'], label='Signal', color='green')
            # ax2.set_ylabel('MACD')
            # ax2.legend(loc='upper right')

            # plt.title(f"{base_coin} vs {comp_coin} Ratio")
            # plt.xticks(rotation=45)
            # plt.tight_layout()
            # plt.show()
#_____________________________________________________________________________________________________________________________________

#_____________________________________________________________________________________________________________________________________
# Function to perform ratio analysis for popcat
def analyze_monk(coins):
    base_coin = 'monkeyhaircut'
    df_base = fetch_data(base_coin, 'usd', 40)  # Fetch base coin data

    total_trend_strength = 0

    # Calculate the EMA for the base coin
    df_base['EMA'] = df_base['price'].ewm(span=12, adjust=False).mean()  # 12-period EMA

    for comp_coin in coins:
        if comp_coin != base_coin:  # Skip if the comparison coin is the same
            df_comp = fetch_data(comp_coin, 'usd', 40)  # Fetch comparison coin data
            
            # Merge the data on 'date' to ensure alignment
            merged_df = pd.merge_asof(df_base, df_comp, on='date', suffixes=('_monkeyhaircut', f'_{comp_coin}'))

            # Calculate the ratio
            merged_df['ratio'] = merged_df['price_monkeyhaircut'] / merged_df[f'price_{comp_coin}']

            # Calculate 12 and 21 EMA
            merged_df['EMA12'] = calculate_ema(merged_df, 'ratio', span = 12)
            merged_df['EMA21'] = calculate_ema(merged_df, 'ratio', span = 21)

            #Calculate RSI
            merged_df['RSI'] = calculate_rsi(merged_df, 'ratio')

            # **Calculate MACD**
            merged_df = calculate_macd(merged_df, 'ratio')

            # Determine trend strength based on EMA and RSI
            ema_trend = merged_df['EMA12'].iloc[-1] > merged_df['EMA21'].iloc[-1]  # True if EMA12 > EMA21 (bullish)
            rsi_trend = merged_df['RSI'].iloc[-1] > 50  # True if RSI > 50 (bullish)
            macd_trend = merged_df['MACD'].iloc[-1] > merged_df['Signal'].iloc[-1]  # True if MACD > Signal (bullish)

              # Initialize trend strength
            trend_strength = 0
            
            # Only add to trend strength if both EMA and RSI are positive
            if ema_trend:
                trend_strength += 1

            if rsi_trend:
                trend_strength += 1

            if macd_trend:
                trend_strength += 1

            # Add the current pair's trend strength to the total
            total_trend_strength += trend_strength

            # Store total trend strength for dogwifcoin
            trend_strength_dict[base_coin] = total_trend_strength

            # Print or store the trend strength for this pair
            print(f"Trend strength for {base_coin} vs {comp_coin}: {trend_strength}")

            # # Plotting the data
            # fig, ax1 = plt.subplots()

            # # Plot Ratio and EMAs on the primary y-axis
            # ax1.plot(merged_df['date'], merged_df['ratio'], label='Ratio')
            # ax1.plot(merged_df['date'], merged_df['EMA12'], label='EMA 12', linestyle='--', color='orange')
            # ax1.plot(merged_df['date'], merged_df['EMA21'], label='EMA 21', linestyle='--', color='red')
            # ax1.set_xlabel('Date')
            # ax1.set_ylabel('Ratio')
            # ax1.legend(loc='upper left')

            # # Create secondary y-axis for MACD and Signal
            # ax2 = ax1.twinx()
            # ax2.plot(merged_df['date'], merged_df['MACD'], label='MACD', color='blue')
            # ax2.plot(merged_df['date'], merged_df['Signal'], label='Signal', color='green')
            # ax2.set_ylabel('MACD')
            # ax2.legend(loc='upper right')

            # plt.title(f"{base_coin} vs {comp_coin} Ratio")
            # plt.xticks(rotation=45)
            # plt.tight_layout()
            # plt.show()
#_____________________________________________________________________________________________________________________________________

#_____________________________________________________________________________________________________________________________________
# Function to perform ratio analysis for popcat
def analyze_fwog(coins):
    base_coin = 'fwog'
    df_base = fetch_data(base_coin, 'usd', 40)  # Fetch base coin data

    total_trend_strength = 0

    # Calculate the EMA for the base coin
    df_base['EMA'] = df_base['price'].ewm(span=12, adjust=False).mean()  # 12-period EMA

    for comp_coin in coins:
        if comp_coin != base_coin:  # Skip if the comparison coin is the same
            df_comp = fetch_data(comp_coin, 'usd', 40)  # Fetch comparison coin data
            
            # Merge the data on 'date' to ensure alignment
            merged_df = pd.merge_asof(df_base, df_comp, on='date', suffixes=('_fwog', f'_{comp_coin}'))

            # Calculate the ratio
            merged_df['ratio'] = merged_df['price_fwog'] / merged_df[f'price_{comp_coin}']

            # Calculate 12 and 21 EMA
            merged_df['EMA12'] = calculate_ema(merged_df, 'ratio', span = 12)
            merged_df['EMA21'] = calculate_ema(merged_df, 'ratio', span = 21)

            #Calculate RSI
            merged_df['RSI'] = calculate_rsi(merged_df, 'ratio')

            # **Calculate MACD**
            merged_df = calculate_macd(merged_df, 'ratio')

            # Determine trend strength based on EMA and RSI
            ema_trend = merged_df['EMA12'].iloc[-1] > merged_df['EMA21'].iloc[-1]  # True if EMA12 > EMA21 (bullish)
            rsi_trend = merged_df['RSI'].iloc[-1] > 50  # True if RSI > 50 (bullish)
            macd_trend = merged_df['MACD'].iloc[-1] > merged_df['Signal'].iloc[-1]  # True if MACD > Signal (bullish)

              # Initialize trend strength
            trend_strength = 0
            
            # Only add to trend strength if both EMA and RSI are positive
            if ema_trend:
                trend_strength += 1

            if rsi_trend:
                trend_strength += 1

            if macd_trend:
                trend_strength += 1

            # Add the current pair's trend strength to the total
            total_trend_strength += trend_strength

            # Store total trend strength for dogwifcoin
            trend_strength_dict[base_coin] = total_trend_strength

            # Print or store the trend strength for this pair
            print(f"Trend strength for {base_coin} vs {comp_coin}: {trend_strength}")

            # # Plotting the data
            # fig, ax1 = plt.subplots()

            # # Plot Ratio and EMAs on the primary y-axis
            # ax1.plot(merged_df['date'], merged_df['ratio'], label='Ratio')
            # ax1.plot(merged_df['date'], merged_df['EMA12'], label='EMA 12', linestyle='--', color='orange')
            # ax1.plot(merged_df['date'], merged_df['EMA21'], label='EMA 21', linestyle='--', color='red')
            # ax1.set_xlabel('Date')
            # ax1.set_ylabel('Ratio')
            # ax1.legend(loc='upper left')

            # # Create secondary y-axis for MACD and Signal
            # ax2 = ax1.twinx()
            # ax2.plot(merged_df['date'], merged_df['MACD'], label='MACD', color='blue')
            # ax2.plot(merged_df['date'], merged_df['Signal'], label='Signal', color='green')
            # ax2.set_ylabel('MACD')
            # ax2.legend(loc='upper right')

            # plt.title(f"{base_coin} vs {comp_coin} Ratio")
            # plt.xticks(rotation=45)
            # plt.tight_layout()
            # plt.show()
#_____________________________________________________________________________________________________________________________________

#_____________________________________________________________________________________________________________________________________
# Function to perform ratio analysis for popcat
def analyze_mumu(coins):
    base_coin = 'mumu-the-bull-3'
    df_base = fetch_data(base_coin, 'usd', 40)  # Fetch base coin data

    total_trend_strength = 0

    # Calculate the EMA for the base coin
    df_base['EMA'] = df_base['price'].ewm(span=12, adjust=False).mean()  # 12-period EMA

    for comp_coin in coins:
        if comp_coin != base_coin:  # Skip if the comparison coin is the same
            df_comp = fetch_data(comp_coin, 'usd', 40)  # Fetch comparison coin data
            
            # Merge the data on 'date' to ensure alignment
            merged_df = pd.merge_asof(df_base, df_comp, on='date', suffixes=('_mumu-the-bull-3', f'_{comp_coin}'))

            # Calculate the ratio
            merged_df['ratio'] = merged_df['price_mumu-the-bull-3'] / merged_df[f'price_{comp_coin}']

            # Calculate 12 and 21 EMA
            merged_df['EMA12'] = calculate_ema(merged_df, 'ratio', span = 12)
            merged_df['EMA21'] = calculate_ema(merged_df, 'ratio', span = 21)

            #Calculate RSI
            merged_df['RSI'] = calculate_rsi(merged_df, 'ratio')

            # **Calculate MACD**
            merged_df = calculate_macd(merged_df, 'ratio')

            # Determine trend strength based on EMA and RSI
            ema_trend = merged_df['EMA12'].iloc[-1] > merged_df['EMA21'].iloc[-1]  # True if EMA12 > EMA21 (bullish)
            rsi_trend = merged_df['RSI'].iloc[-1] > 50  # True if RSI > 50 (bullish)
            macd_trend = merged_df['MACD'].iloc[-1] > merged_df['Signal'].iloc[-1]  # True if MACD > Signal (bullish)

              # Initialize trend strength
            trend_strength = 0
            
            # Only add to trend strength if both EMA and RSI are positive
            if ema_trend:
                trend_strength += 1

            if rsi_trend:
                trend_strength += 1

            if macd_trend:
                trend_strength += 1

            # Add the current pair's trend strength to the total
            total_trend_strength += trend_strength

            # Store total trend strength for dogwifcoin
            trend_strength_dict[base_coin] = total_trend_strength

            # Print or store the trend strength for this pair
            print(f"Trend strength for {base_coin} vs {comp_coin}: {trend_strength}")

            # # Plotting the data
            # fig, ax1 = plt.subplots()

            # # Plot Ratio and EMAs on the primary y-axis
            # ax1.plot(merged_df['date'], merged_df['ratio'], label='Ratio')
            # ax1.plot(merged_df['date'], merged_df['EMA12'], label='EMA 12', linestyle='--', color='orange')
            # ax1.plot(merged_df['date'], merged_df['EMA21'], label='EMA 21', linestyle='--', color='red')
            # ax1.set_xlabel('Date')
            # ax1.set_ylabel('Ratio')
            # ax1.legend(loc='upper left')

            # # Create secondary y-axis for MACD and Signal
            # ax2 = ax1.twinx()
            # ax2.plot(merged_df['date'], merged_df['MACD'], label='MACD', color='blue')
            # ax2.plot(merged_df['date'], merged_df['Signal'], label='Signal', color='green')
            # ax2.set_ylabel('MACD')
            # ax2.legend(loc='upper right')

            # plt.title(f"{base_coin} vs {comp_coin} Ratio")
            # plt.xticks(rotation=45)
            # plt.tight_layout()
            # plt.show()
#_____________________________________________________________________________________________________________________________________

#_____________________________________________________________________________________________________________________________________
# Function to perform ratio analysis for popcat
def analyze_ponke(coins):
    base_coin = 'ponke'
    df_base = fetch_data(base_coin, 'usd', 40)  # Fetch base coin data

    total_trend_strength = 0

    # Calculate the EMA for the base coin
    df_base['EMA'] = df_base['price'].ewm(span=12, adjust=False).mean()  # 12-period EMA

    for comp_coin in coins:
        if comp_coin != base_coin:  # Skip if the comparison coin is the same
            df_comp = fetch_data(comp_coin, 'usd', 40)  # Fetch comparison coin data
            
            # Merge the data on 'date' to ensure alignment
            merged_df = pd.merge_asof(df_base, df_comp, on='date', suffixes=('_ponke', f'_{comp_coin}'))

            # Calculate the ratio
            merged_df['ratio'] = merged_df['price_ponke'] / merged_df[f'price_{comp_coin}']

            # Calculate 12 and 21 EMA
            merged_df['EMA12'] = calculate_ema(merged_df, 'ratio', span = 12)
            merged_df['EMA21'] = calculate_ema(merged_df, 'ratio', span = 21)

            #Calculate RSI
            merged_df['RSI'] = calculate_rsi(merged_df, 'ratio')

            # **Calculate MACD**
            merged_df = calculate_macd(merged_df, 'ratio')

            # Determine trend strength based on EMA and RSI
            ema_trend = merged_df['EMA12'].iloc[-1] > merged_df['EMA21'].iloc[-1]  # True if EMA12 > EMA21 (bullish)
            rsi_trend = merged_df['RSI'].iloc[-1] > 50  # True if RSI > 50 (bullish)
            macd_trend = merged_df['MACD'].iloc[-1] > merged_df['Signal'].iloc[-1]  # True if MACD > Signal (bullish)

              # Initialize trend strength
            trend_strength = 0
            
            # Only add to trend strength if both EMA and RSI are positive
            if ema_trend:
                trend_strength += 1

            if rsi_trend:
                trend_strength += 1

            if macd_trend:
                trend_strength += 1

            # Add the current pair's trend strength to the total
            total_trend_strength += trend_strength

            # Store total trend strength for dogwifcoin
            trend_strength_dict[base_coin] = total_trend_strength

            # Print or store the trend strength for this pair
            print(f"Trend strength for {base_coin} vs {comp_coin}: {trend_strength}")

            #  # Plotting the data
            # fig, ax1 = plt.subplots()

            # # Plot Ratio and EMAs on the primary y-axis
            # ax1.plot(merged_df['date'], merged_df['ratio'], label='Ratio')
            # ax1.plot(merged_df['date'], merged_df['EMA12'], label='EMA 12', linestyle='--', color='orange')
            # ax1.plot(merged_df['date'], merged_df['EMA21'], label='EMA 21', linestyle='--', color='red')
            # ax1.set_xlabel('Date')
            # ax1.set_ylabel('Ratio')
            # ax1.legend(loc='upper left')

            # # Create secondary y-axis for MACD and Signal
            # ax2 = ax1.twinx()
            # ax2.plot(merged_df['date'], merged_df['MACD'], label='MACD', color='blue')
            # ax2.plot(merged_df['date'], merged_df['Signal'], label='Signal', color='green')
            # ax2.set_ylabel('MACD')
            # ax2.legend(loc='upper right')

            # plt.title(f"{base_coin} vs {comp_coin} Ratio")
            # plt.xticks(rotation=45)
            # plt.tight_layout()
            # plt.show()
#_____________________________________________________________________________________________________________________________________

#_____________________________________________________________________________________________________________________________________
# Function to perform ratio analysis for popcat
def analyze_nub(coins):
    base_coin = 'sillynubcat'
    df_base = fetch_data(base_coin, 'usd', 40)  # Fetch base coin data

    total_trend_strength = 0

    # Calculate the EMA for the base coin
    df_base['EMA'] = df_base['price'].ewm(span=12, adjust=False).mean()  # 12-period EMA

    for comp_coin in coins:
        if comp_coin != base_coin:  # Skip if the comparison coin is the same
            df_comp = fetch_data(comp_coin, 'usd', 40)  # Fetch comparison coin data
            
            # Merge the data on 'date' to ensure alignment
            merged_df = pd.merge_asof(df_base, df_comp, on='date', suffixes=('_sillynubcat', f'_{comp_coin}'))

            # Calculate the ratio
            merged_df['ratio'] = merged_df['price_sillynubcat'] / merged_df[f'price_{comp_coin}']

            # Calculate 12 and 21 EMA
            merged_df['EMA12'] = calculate_ema(merged_df, 'ratio', span = 12)
            merged_df['EMA21'] = calculate_ema(merged_df, 'ratio', span = 21)

            #Calculate RSI
            merged_df['RSI'] = calculate_rsi(merged_df, 'ratio')

            # **Calculate MACD**
            merged_df = calculate_macd(merged_df, 'ratio')

            # Determine trend strength based on EMA and RSI
            ema_trend = merged_df['EMA12'].iloc[-1] > merged_df['EMA21'].iloc[-1]  # True if EMA12 > EMA21 (bullish)
            rsi_trend = merged_df['RSI'].iloc[-1] > 50  # True if RSI > 50 (bullish)
            macd_trend = merged_df['MACD'].iloc[-1] > merged_df['Signal'].iloc[-1]  # True if MACD > Signal (bullish)

              # Initialize trend strength
            trend_strength = 0
            
            # Only add to trend strength if both EMA and RSI are positive
            if ema_trend:
                trend_strength += 1

            if rsi_trend:
                trend_strength += 1

            if macd_trend:
                trend_strength += 1

            # Add the current pair's trend strength to the total
            total_trend_strength += trend_strength

            # Store total trend strength for dogwifcoin
            trend_strength_dict[base_coin] = total_trend_strength

            # Print or store the trend strength for this pair
            print(f"Trend strength for {base_coin} vs {comp_coin}: {trend_strength}")

            #  # Plotting the data
            # fig, ax1 = plt.subplots()

            # # Plot Ratio and EMAs on the primary y-axis
            # ax1.plot(merged_df['date'], merged_df['ratio'], label='Ratio')
            # ax1.plot(merged_df['date'], merged_df['EMA12'], label='EMA 12', linestyle='--', color='orange')
            # ax1.plot(merged_df['date'], merged_df['EMA21'], label='EMA 21', linestyle='--', color='red')
            # ax1.set_xlabel('Date')
            # ax1.set_ylabel('Ratio')
            # ax1.legend(loc='upper left')

            # # Create secondary y-axis for MACD and Signal
            # ax2 = ax1.twinx()
            # ax2.plot(merged_df['date'], merged_df['MACD'], label='MACD', color='blue')
            # ax2.plot(merged_df['date'], merged_df['Signal'], label='Signal', color='green')
            # ax2.set_ylabel('MACD')
            # ax2.legend(loc='upper right')

            # plt.title(f"{base_coin} vs {comp_coin} Ratio")
            # plt.xticks(rotation=45)
            # plt.tight_layout()
            # plt.show()
#_____________________________________________________________________________________________________________________________________

#_____________________________________________________________________________________________________________________________________
# Function to perform ratio analysis for popcat
def analyze_aura(coins):
    base_coin = 'aura-on-sol'
    df_base = fetch_data(base_coin, 'usd', 40)  # Fetch base coin data

    total_trend_strength = 0

    # Calculate the EMA for the base coin
    df_base['EMA'] = df_base['price'].ewm(span=12, adjust=False).mean()  # 12-period EMA

    for comp_coin in coins:
        if comp_coin != base_coin:  # Skip if the comparison coin is the same
            df_comp = fetch_data(comp_coin, 'usd', 40)  # Fetch comparison coin data
            
            # Merge the data on 'date' to ensure alignment
            merged_df = pd.merge_asof(df_base, df_comp, on='date', suffixes=('_aura-on-sol', f'_{comp_coin}'))

            # Calculate the ratio
            merged_df['ratio'] = merged_df['price_aura-on-sol'] / merged_df[f'price_{comp_coin}']

            # Calculate 12 and 21 EMA
            merged_df['EMA12'] = calculate_ema(merged_df, 'ratio', span = 12)
            merged_df['EMA21'] = calculate_ema(merged_df, 'ratio', span = 21)

            #Calculate RSI
            merged_df['RSI'] = calculate_rsi(merged_df, 'ratio')

            # **Calculate MACD**
            merged_df = calculate_macd(merged_df, 'ratio')

            # Determine trend strength based on EMA and RSI
            ema_trend = merged_df['EMA12'].iloc[-1] > merged_df['EMA21'].iloc[-1]  # True if EMA12 > EMA21 (bullish)
            rsi_trend = merged_df['RSI'].iloc[-1] > 50  # True if RSI > 50 (bullish)
            macd_trend = merged_df['MACD'].iloc[-1] > merged_df['Signal'].iloc[-1]  # True if MACD > Signal (bullish)

              # Initialize trend strength
            trend_strength = 0
            
            # Only add to trend strength if both EMA and RSI are positive
            if ema_trend:
                trend_strength += 1

            if rsi_trend:
                trend_strength += 1

            if macd_trend:
                trend_strength += 1

            # Add the current pair's trend strength to the total
            total_trend_strength += trend_strength

            # Store total trend strength for dogwifcoin
            trend_strength_dict[base_coin] = total_trend_strength

            # Print or store the trend strength for this pair
            print(f"Trend strength for {base_coin} vs {comp_coin}: {trend_strength}")

            # # Plotting the data
            # fig, ax1 = plt.subplots()

            # # Plot Ratio and EMAs on the primary y-axis
            # ax1.plot(merged_df['date'], merged_df['ratio'], label='Ratio')
            # ax1.plot(merged_df['date'], merged_df['EMA12'], label='EMA 12', linestyle='--', color='orange')
            # ax1.plot(merged_df['date'], merged_df['EMA21'], label='EMA 21', linestyle='--', color='red')
            # ax1.set_xlabel('Date')
            # ax1.set_ylabel('Ratio')
            # ax1.legend(loc='upper left')

            # # Create secondary y-axis for MACD and Signal
            # ax2 = ax1.twinx()
            # ax2.plot(merged_df['date'], merged_df['MACD'], label='MACD', color='blue')
            # ax2.plot(merged_df['date'], merged_df['Signal'], label='Signal', color='green')
            # ax2.set_ylabel('MACD')
            # ax2.legend(loc='upper right')

            # plt.title(f"{base_coin} vs {comp_coin} Ratio")
            # plt.xticks(rotation=45)
            # plt.tight_layout()
            # plt.show()
#_____________________________________________________________________________________________________________________________________

#_____________________________________________________________________________________________________________________________________
# Function to perform ratio analysis for popcat
def analyze_lockin(coins):
    base_coin = 'lock-in'
    df_base = fetch_data(base_coin, 'usd', 40)  # Fetch base coin data

    total_trend_strength = 0

    # Calculate the EMA for the base coin
    df_base['EMA'] = df_base['price'].ewm(span=12, adjust=False).mean()  # 12-period EMA

    for comp_coin in coins:
        if comp_coin != base_coin:  # Skip if the comparison coin is the same
            df_comp = fetch_data(comp_coin, 'usd', 40)  # Fetch comparison coin data
            
            # Merge the data on 'date' to ensure alignment
            merged_df = pd.merge_asof(df_base, df_comp, on='date', suffixes=('_lock-in', f'_{comp_coin}'))

            # Calculate the ratio
            merged_df['ratio'] = merged_df['price_lock-in'] / merged_df[f'price_{comp_coin}']

            # Calculate 12 and 21 EMA
            merged_df['EMA12'] = calculate_ema(merged_df, 'ratio', span = 12)
            merged_df['EMA21'] = calculate_ema(merged_df, 'ratio', span = 21)

            #Calculate RSI
            merged_df['RSI'] = calculate_rsi(merged_df, 'ratio')

            # **Calculate MACD**
            merged_df = calculate_macd(merged_df, 'ratio')

            # Determine trend strength based on EMA and RSI
            ema_trend = merged_df['EMA12'].iloc[-1] > merged_df['EMA21'].iloc[-1]  # True if EMA12 > EMA21 (bullish)
            rsi_trend = merged_df['RSI'].iloc[-1] > 50  # True if RSI > 50 (bullish)
            macd_trend = merged_df['MACD'].iloc[-1] > merged_df['Signal'].iloc[-1]  # True if MACD > Signal (bullish)

              # Initialize trend strength
            trend_strength = 0
            
            # Only add to trend strength if both EMA and RSI are positive
            if ema_trend:
                trend_strength += 1

            if rsi_trend:
                trend_strength += 1

            if macd_trend:
                trend_strength += 1

            # Add the current pair's trend strength to the total
            total_trend_strength += trend_strength

            # Store total trend strength for dogwifcoin
            trend_strength_dict[base_coin] = total_trend_strength

            # Print or store the trend strength for this pair
            print(f"Trend strength for {base_coin} vs {comp_coin}: {trend_strength}")

            # # Plotting the data
            # fig, ax1 = plt.subplots()

            # # Plot Ratio and EMAs on the primary y-axis
            # ax1.plot(merged_df['date'], merged_df['ratio'], label='Ratio')
            # ax1.plot(merged_df['date'], merged_df['EMA12'], label='EMA 12', linestyle='--', color='orange')
            # ax1.plot(merged_df['date'], merged_df['EMA21'], label='EMA 21', linestyle='--', color='red')
            # ax1.set_xlabel('Date')
            # ax1.set_ylabel('Ratio')
            # ax1.legend(loc='upper left')

            # # Create secondary y-axis for MACD and Signal
            # ax2 = ax1.twinx()
            # ax2.plot(merged_df['date'], merged_df['MACD'], label='MACD', color='blue')
            # ax2.plot(merged_df['date'], merged_df['Signal'], label='Signal', color='green')
            # ax2.set_ylabel('MACD')
            # ax2.legend(loc='upper right')

            # plt.title(f"{base_coin} vs {comp_coin} Ratio")
            # plt.xticks(rotation=45)
            # plt.tight_layout()
            # plt.show()
#_____________________________________________________________________________________________________________________________________



# Call the function for coins
analyze_dogwifcoin(coins)
#plt.show()
analyze_bonk(coins)
#plt.show
analyze_myro(coins)
#plt.show()
analyze_popcat(coins)
#plt.show()
analyze_wen(coins)
#plt.show()
analyze_tremp(coins)
#plt.show()
analyze_maga(coins)
#plt.show()
analyze_boden(coins)
#plt.show()
analyze_zack(coins)
#plt.show()
analyze_habibi(coins)
#plt.show()
analyze_daddy(coins)
#plt.show()
analyze_usacoin(coins)
#plt.show()
analyze_peng(coins)
#plt.show()
analyze_giga(coins)
#plt.show()
analyze_monk(coins)
#plt.show()
analyze_fwog(coins)
#plt.show()
analyze_mumu(coins)
#plt.show()
analyze_ponke(coins)
#plt.show()
analyze_nub(coins)
#plt.show()
analyze_aura(coins)
#plt.show()
analyze_lockin(coins)
#plt.show()

# Rank coins by total trend strength
ranked_coins = sorted(trend_strength_dict.items(), key=lambda x: x[1], reverse=True)

# Print the ranked coins
print("Ranked coins by total trend strength:")
for coin, strength in ranked_coins:
    print(f"{coin}: {strength}")


print('')
print('')
print('')
print('_______________________________________________________________')
print('')
print('')
print('')
print('THE PGS HAS SPOKEN:')
print('')

for coin, strength in ranked_coins[:4]:  # [:4] limits the output to the top 4
    print(f"{coin}")

print('')
print('')
print('_______________________________________________________________')
print('')
print('')
print('')
