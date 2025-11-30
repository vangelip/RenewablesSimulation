import requests
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, HourLocator
from datetime import datetime, timedelta
import numpy as np

# Hornsea 2 wind farms only
HORNSEA_2_FARMS = {
    'HOWBO-1': 440,
    'HOWBO-2': 440,
    'HOWBO-3': 440,
}

HORNSEA_2_CAPACITY = sum(HORNSEA_2_FARMS.values())  # 1320 MW

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"


def fetch_market_index_data(start_date, end_date):
    """Fetch market index pricing data from BMRS API"""
    all_data = []
    current = start_date

    while current < end_date:
        chunk_end = min(current + timedelta(days=6), end_date)

        endpoint = f"{BASE_URL}/balancing/pricing/market-index"
        params = {
            'from': current.strftime('%Y-%m-%dT00:00Z'),
            'to': chunk_end.strftime('%Y-%m-%dT23:59Z'),
            'format': 'json'
        }

        try:
            response = requests.get(endpoint, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and data['data']:
                    df = pd.DataFrame(data['data'])
                    all_data.append(df)
                    print(
                        f"    Fetched market data: {current.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
        except Exception as e:
            print(f"    Error fetching market data for {current}: {e}")

        current = chunk_end + timedelta(days=1)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        # Convert time columns
        for col in ['startTime', 'settlementDate']:
            if col in combined.columns:
                combined[col] = pd.to_datetime(combined[col], errors='coerce')
        return combined

    return pd.DataFrame()


def fetch_bmu_data(bmu_unit, start_date, end_date):
    """Fetch data from BMRS API for a specific BMU"""
    all_data = []
    current = start_date

    while current < end_date:
        chunk_end = min(current + timedelta(days=6), end_date)

        endpoint = f"{BASE_URL}/balancing/physical"
        params = {
            'bmUnit': f"T_{bmu_unit}",
            'from': current.strftime('%Y-%m-%dT00:00Z'),
            'to': chunk_end.strftime('%Y-%m-%dT23:59Z'),
            'format': 'json'
        }

        try:
            response = requests.get(endpoint, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and data['data']:
                    df = pd.DataFrame(data['data'])
                    df['bmUnit'] = bmu_unit
                    all_data.append(df)
        except Exception as e:
            print(f"    Error fetching {bmu_unit} for {current}: {e}")

        current = chunk_end + timedelta(days=1)

    if all_data:
        return pd.concat(all_data, ignore_index=True)

    return pd.DataFrame()


def get_week_data(start_date, end_date):
    """Get all data for a specific week"""
    print(f"  Fetching market index data...")
    market_df = fetch_market_index_data(start_date, end_date)

    print(f"  Fetching Hornsea 2 wind farm data...")
    wind_data = []
    for bmu, capacity in HORNSEA_2_FARMS.items():
        print(f"    Fetching {bmu}...")
        df = fetch_bmu_data(bmu, start_date, end_date)
        if not df.empty:
            df['capacity_mw'] = capacity
            wind_data.append(df)

    wind_df = pd.DataFrame()
    if wind_data:
        wind_df = pd.concat(wind_data, ignore_index=True)
        # Convert time columns
        for col in ['timeFrom', 'timeTo', 'settlementDate', 'startTime']:
            if col in wind_df.columns:
                wind_df[col] = pd.to_datetime(wind_df[col], errors='coerce')

    return market_df, wind_df


def process_wind_data(df):
    """Process wind farm data to get capacity factor"""
    if df.empty:
        return pd.DataFrame()

    # Filter for PN data only
    if 'dataset' in df.columns:
        df = df[df['dataset'] == 'PN'].copy()

    # Use levelFrom as the MW output
    if 'levelFrom' in df.columns:
        df['mw_output'] = pd.to_numeric(df['levelFrom'], errors='coerce')
        df['mw_output'] = df['mw_output'].clip(lower=0)
    else:
        df['mw_output'] = 0

    # Aggregate by timestamp - sum all Hornsea 2 farms
    timeseries = df.groupby('timeFrom')['mw_output'].sum().sort_index()

    # Calculate capacity factor as percentage
    cf_series = (timeseries / HORNSEA_2_CAPACITY) * 100

    result_df = pd.DataFrame({
        'timestamp': cf_series.index,
        'output_mw': timeseries.values,
        'capacity_factor_pct': cf_series.values
    })

    # Remove any rows with NaN values
    result_df = result_df.dropna()

    # Handle >100% capacity factor by setting to previous value
    for i in range(len(result_df)):
        if result_df.loc[i, 'capacity_factor_pct'] > 100:
            if i > 0:
                result_df.loc[i, 'capacity_factor_pct'] = result_df.loc[i - 1, 'capacity_factor_pct']
            else:
                result_df.loc[i, 'capacity_factor_pct'] = 100

    return result_df


def merge_datasets(market_df, wind_df):
    """Merge market and wind data on a common timeline, forward-filling gaps"""
    if market_df.empty or wind_df.empty:
        return market_df, wind_df

    # Create a complete time range at 30-minute intervals
    min_time = min(market_df['timestamp'].min(), wind_df['timestamp'].min())
    max_time = max(market_df['timestamp'].max(), wind_df['timestamp'].max())

    # Create 30-minute frequency time index
    time_range = pd.date_range(start=min_time, end=max_time, freq='30min')

    # Set timestamp as index for both dataframes
    market_indexed = market_df.set_index('timestamp')
    wind_indexed = wind_df.set_index('timestamp')

    # Reindex to the complete time range and forward fill
    market_reindexed = market_indexed.reindex(time_range, method='ffill')
    wind_reindexed = wind_indexed.reindex(time_range, method='ffill')

    # Reset index to get timestamp column back
    market_reindexed = market_reindexed.reset_index().rename(columns={'index': 'timestamp'})
    wind_reindexed = wind_reindexed.reset_index().rename(columns={'index': 'timestamp'})

    # Drop any remaining NaN values
    market_reindexed = market_reindexed.dropna()
    wind_reindexed = wind_reindexed.dropna()

    return market_reindexed, wind_reindexed


def process_market_data(df):
    """Process market index data - filter for APXMIDP data provider only"""
    if df.empty:
        return pd.DataFrame()

    # Filter for APXMIDP data provider only
    if 'dataProvider' in df.columns:
        df = df[df['dataProvider'] == 'APXMIDP'].copy()
        print(f"    Filtered to APXMIDP records: {len(df)}")
    else:
        print(f"    Warning: 'dataProvider' column not found in market data")

    if df.empty:
        return pd.DataFrame()

    # Extract relevant columns
    result_df = pd.DataFrame()

    if 'startTime' in df.columns:
        result_df['timestamp'] = df['startTime']

    # Price data
    if 'price' in df.columns:
        result_df['price'] = pd.to_numeric(df['price'], errors='coerce')

    # Volume data
    if 'volume' in df.columns:
        result_df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

    # Remove rows with NaN values
    result_df = result_df.dropna()
    result_df = result_df.sort_values('timestamp').reset_index(drop=True)

    # Remove duplicate timestamps, keeping the first occurrence
    result_df = result_df.drop_duplicates(subset=['timestamp'], keep='first')

    return result_df


def plot_weekly_analysis(week_num, start_date, end_date, market_df, wind_df):
    """Create combined plot for market data and wind output"""

    week_label = f"Week {week_num}: {start_date.strftime('%b %d')} - {end_date.strftime('%b %d, %Y')}"

    # Merge the datasets on timestamp for CSV export
    combined_df = pd.merge(market_df, wind_df, on='timestamp', how='outer')
    combined_df = combined_df.sort_values('timestamp')

    # Save to CSV
    csv_filename = f'week_{week_num:02d}_{start_date.strftime("%Y%m%d")}_data.csv'
    combined_df.to_csv(csv_filename, index=False)
    print(f"  ✓ Saved data: {csv_filename}")

    # Create figure with multiple y-axes
    fig, ax1 = plt.subplots(figsize=(20, 10))

    # Color scheme
    color_price = 'darkblue'
    color_volume = 'orange'
    color_wind = 'green'

    # Plot 1: Market Index Price (left y-axis)
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Market Index Price (£/MWh)', color=color_price, fontsize=12)
    line1 = ax1.plot(market_df['timestamp'], market_df['price'],
                     color=color_price, linewidth=1.2, alpha=0.7, label='Market Price',
                     marker='.', markersize=2)
    ax1.tick_params(axis='y', labelcolor=color_price)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, market_df['price'].max() * 1.1)

    # Plot 2: Trading Volume (right y-axis 1)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Trading Volume (MWh)', color=color_volume, fontsize=12)
    line2 = ax2.plot(market_df['timestamp'], market_df['volume'],
                     color=color_volume, linewidth=1.2, alpha=0.7, label='Trading Volume',
                     marker='.', markersize=2)
    ax2.tick_params(axis='y', labelcolor=color_volume)
    ax2.set_ylim(0, market_df['volume'].max() * 1.1)

    # Plot 3: Hornsea 2 Capacity Factor (right y-axis 2)
    ax3 = ax1.twinx()
    # Offset the right spine of ax3
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel('Hornsea 2 Output (%)', color=color_wind, fontsize=12)
    line3 = ax3.plot(wind_df['timestamp'], wind_df['capacity_factor_pct'],
                     color=color_wind, linewidth=1.2, alpha=0.7, label='Hornsea 2 Output',
                     marker='.', markersize=2)
    ax3.tick_params(axis='y', labelcolor=color_wind)
    ax3.set_ylim(0, 100)

    # Title
    plt.title(f'Market Index Price, Trading Volume & Hornsea 2 Output\n{week_label}',
              fontsize=14, fontweight='bold', pad=20)

    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=11)

    # Format x-axis
    ax1.xaxis.set_major_formatter(DateFormatter('%m/%d %H:%M'))
    ax1.xaxis.set_major_locator(HourLocator(interval=12))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    fig.tight_layout()

    # Save figure
    filename = f'week_{week_num:02d}_{start_date.strftime("%Y%m%d")}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved plot: {filename}")
    plt.close()

    # Calculate statistics
    stats = {
        'week_num': week_num,
        'start_date': start_date,
        'end_date': end_date,
        'week_label': week_label,
        'avg_price': market_df['price'].mean() if 'price' in market_df.columns else None,
        'max_price': market_df['price'].max() if 'price' in market_df.columns else None,
        'min_price': market_df['price'].min() if 'price' in market_df.columns else None,
        'avg_volume': market_df['volume'].mean() if 'volume' in market_df.columns else None,
        'total_volume': market_df['volume'].sum() if 'volume' in market_df.columns else None,
        'avg_cf': wind_df['capacity_factor_pct'].mean(),
        'max_cf': wind_df['capacity_factor_pct'].max(),
        'min_cf': wind_df['capacity_factor_pct'].min(),
        'avg_output_mw': wind_df['output_mw'].mean(),
    }

    return stats


def create_summary_plot(results_df):
    """Create summary trend plots across all weeks"""

    fig, axes = plt.subplots(3, 1, figsize=(18, 12))

    # Create x-axis labels
    x_labels = [f"W{row['week_num']}" for _, row in results_df.iterrows()]
    x_pos = np.arange(len(x_labels))

    # Plot 1: Average Market Price
    ax1 = axes[0]
    ax1.plot(x_pos, results_df['avg_price'], marker='o', linewidth=2,
             markersize=8, color='darkblue', label='Avg Market Price')
    ax1.axhline(y=results_df['avg_price'].mean(), color='red', linestyle='--',
                linewidth=1.5, alpha=0.5, label=f'Overall Avg: £{results_df["avg_price"].mean():.2f}/MWh')
    ax1.set_ylabel('Price (£/MWh)', fontsize=11)
    ax1.set_title('Average Market Index Price by Week', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Total Trading Volume
    ax2 = axes[1]
    ax2.bar(x_pos, results_df['total_volume'], color='orange', alpha=0.8)
    ax2.set_ylabel('Total Volume (MWh)', fontsize=11)
    ax2.set_title('Total Trading Volume by Week', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Hornsea 2 Average Capacity Factor
    ax3 = axes[2]
    ax3.plot(x_pos, results_df['avg_cf'], marker='s', linewidth=2,
             markersize=8, color='green', label='Avg Capacity Factor')
    ax3.axhline(y=results_df['avg_cf'].mean(), color='red', linestyle='--',
                linewidth=1.5, alpha=0.5, label=f'Overall Avg: {results_df["avg_cf"].mean():.2f}%')
    ax3.set_ylabel('Capacity Factor (%)', fontsize=11)
    ax3.set_title('Hornsea 2 Average Output by Week', fontsize=13, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(x_labels, rotation=45, ha='right')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)

    plt.tight_layout()

    filename = 'weekly_trends_aug_oct_2025.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nTrend plot saved: {filename}")
    plt.close()


def generate_weeks(start_date, end_date):
    """Generate list of week start/end dates"""
    weeks = []
    current = start_date
    week_num = 1

    while current < end_date:
        week_end = min(current + timedelta(days=6, hours=23, minutes=59, seconds=59), end_date)
        weeks.append((week_num, current, week_end))
        current = week_end + timedelta(seconds=1)
        week_num += 1

    return weeks


if __name__ == "__main__":
    # Analyze August to October 2025, week by week
    period_start = datetime(2025, 8, 1)
    period_end = datetime(2025, 10, 31, 23, 59, 59)

    weeks = generate_weeks(period_start, period_end)

    print("\n" + "=" * 80)
    print("WEEKLY MARKET INDEX PRICE, TRADING VOLUME & HORNSEA 2 OUTPUT ANALYSIS")
    print("August - October 2025")
    print("=" * 80 + "\n")

    print(f"Total weeks to analyze: {len(weeks)}\n")

    results = []

    for week_num, start_date, end_date in weeks:
        print(f"{'=' * 80}")
        print(f"Processing Week {week_num}: {start_date.strftime('%B %d')} - {end_date.strftime('%B %d, %Y')}")
        print(f"{'=' * 80}")

        # Get data for the week
        market_df, wind_df = get_week_data(start_date, end_date)

        if market_df.empty or wind_df.empty:
            print(f"  ⚠ Insufficient data for this week")
            continue

        # Process data
        market_df = process_market_data(market_df)
        wind_df = process_wind_data(wind_df)

        print(f"  Market data records: {len(market_df)}")
        print(f"  Wind data records: {len(wind_df)}")

        if market_df.empty or wind_df.empty:
            print(f"  ⚠ Insufficient processed data for analysis")
            continue

        # Merge datasets on common timeline
        market_df, wind_df = merge_datasets(market_df, wind_df)

        print(f"  Merged data records: {len(market_df)}")

        # Create plot and get statistics
        stats = plot_weekly_analysis(week_num, start_date, end_date, market_df, wind_df)
        results.append(stats)

        print(f"  Average market price: £{stats['avg_price']:.2f}/MWh")
        print(f"  Average Hornsea 2 CF: {stats['avg_cf']:.2f}%")
        print()

    if not results:
        print("\n⚠ No data was successfully analyzed!")
        exit(1)

    # Create summary dataframe
    results_df = pd.DataFrame(results)

    # Save to CSV
    csv_filename = 'weekly_summary_aug_oct_2025.csv'
    results_df.to_csv(csv_filename, index=False)
    print(f"{'=' * 80}")
    print(f"Summary saved to: {csv_filename}")
    print(f"{'=' * 80}\n")

    # Create summary trend plot
    create_summary_plot(results_df)

    print("\n" + "=" * 80)
    print("OVERALL SUMMARY (August - October 2025)")
    print("=" * 80)
    print(f"\nMarket Index Price:")
    print(f"  Average: £{results_df['avg_price'].mean():.2f}/MWh")
    print(f"  Max weekly average: £{results_df['avg_price'].max():.2f}/MWh")
    print(f"  Min weekly average: £{results_df['avg_price'].min():.2f}/MWh")
    print(f"\nTrading Volume:")
    print(f"  Total period volume: {results_df['total_volume'].sum():.0f} MWh")
    print(f"  Average weekly volume: {results_df['total_volume'].mean():.0f} MWh")
    print(f"\nHornsea 2 Output:")
    print(f"  Overall average CF: {results_df['avg_cf'].mean():.2f}%")
    print(f"  Best week: Week {results_df.loc[results_df['avg_cf'].idxmax(), 'week_num']} "
          f"({results_df['avg_cf'].max():.2f}%)")
    print(f"  Worst week: Week {results_df.loc[results_df['avg_cf'].idxmin(), 'week_num']} "
          f"({results_df['avg_cf'].min():.2f}%)")
    print(f"  Peak output: {results_df['max_cf'].max():.2f}%")
    print(f"  Lowest output: {results_df['min_cf'].min():.2f}%")
    print("\n" + "=" * 80 + "\n")