import pandas as pd
from scipy import stats

for year in range(2013, 2025):
    print(f"Processing year: {year}")
    
    try:
        # Use the correct separator (comma)
        df = pd.read_csv(f"data/{year}_temp.csv", sep=',')
    except Exception as e:
        print(f"Error reading file {year}: {e}")
        continue

    # Clean up column names
    df.columns = df.columns.str.strip()
    if 'date' not in df.columns or 'T' not in df.columns:
        print(f"Error: Missing 'date' or 'T' columns in {year}_temp.csv. Found columns: {df.columns}")
        continue
    
    station = df['cod_staz'].iloc[0] 

    # Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])  # Remove invalid dates

    # Set 'date' as the index
    df.set_index('date', inplace=True)

    # Group by month
    grouped = df.groupby(df.index.month)

    # Calculate statistics
    mean = grouped['T'].mean()
    max_ = grouped['T'].max()
    min_ = grouped['T'].min()

    # Calculate mode (compatible with SciPy 1.11+)
    mode_ = grouped['T'].agg(lambda x: stats.mode(x, keepdims=True)[0][0] if not x.empty else None)

    # Create the resulting DataFrame
    result = pd.DataFrame({
        'Station': station,
        'Year': year,
        'Month': mean.index,
        'Mean': mean.values,
        'Max': max_.values,
        'Min': min_.values,
        'Mode': mode_.values
    })

    # Save to CSV
    result.to_csv(f'output/temperature/{year}_temp.csv', index=False)
    print(f"Saved output/temperature/{year}_temp.csv successfully.")
