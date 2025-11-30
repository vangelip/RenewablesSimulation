import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import numpy as np
import calendar
import requests

# ============== CONFIGURATION PARAMETERS ==============
# Wind Farm Configuration
PROJECT_CAPACITY_MW = 1278 # Total project capacity in MW
NEW_TURBINE_CF_TARGET = 60  # Target capacity factor for new turbines (%)
BASELINE_EXPORT_MW = 767  # Baseline export to grid in MW

# Storage Configuration
# BESS
BESS_POWER_MW = 300  # Maximum charge/discharge power
BESS_CAPACITY_MWH = 600  # Total energy storage capacity
BESS_MIN_SOC = 0.2  # Minimum state of charge (20%)

# LAES
LAES_POWER_MW = 767  # Maximum charge/discharge power
LAES_CAPACITY_MWH = 20000  # Total energy storage capacity

# Storage round-trip efficiencies
BESS_RTE = 0.88  # BESS round-trip efficiency (88%)
LAES_RTE = 0.55  # LAES round-trip efficiency (65%)

# Pricing Configuration
CURTAILMENT_PRICE = 20.0  # Price for energy above 767MW (£/MWh)
PRICE_FLOOR = 0.0  # Don't sell if prices go below this (£/MWh)

# Hornsea 2 reference data
HORNSEA_2_FARMS = {
    'HOWBO-1': 440,
    'HOWBO-2': 440,
    'HOWBO-3': 440,
}
HORNSEA_2_CAPACITY = sum(HORNSEA_2_FARMS.values())  # 1320 MW

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"


# ============== ELEXON API DATA FETCHING ==============

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
                    print(f"      ✓ Market data: {current.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
        except Exception as e:
            print(f"      ✗ Error fetching market data for {current}: {e}")

        current = chunk_end + timedelta(days=1)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        # Convert time columns
        for col in ['startTime', 'settlementDate']:
            if col in combined.columns:
                combined[col] = pd.to_datetime(combined[col], errors='coerce')
        return combined

    return pd.DataFrame()


def process_market_data(df):
    """Process market index data - filter for APXMIDP data provider only"""
    if df.empty:
        return pd.DataFrame()

    # Filter for APXMIDP data provider only
    if 'dataProvider' in df.columns:
        df = df[df['dataProvider'] == 'APXMIDP'].copy()
        print(f"      Filtered to APXMIDP records: {len(df)}")
    else:
        print(f"      Warning: 'dataProvider' column not found in market data")

    if df.empty:
        return pd.DataFrame()

    # Extract relevant columns
    result_df = pd.DataFrame()

    if 'startTime' in df.columns:
        result_df['timestamp'] = df['startTime']

    # Price data
    if 'price' in df.columns:
        result_df['price'] = pd.to_numeric(df['price'], errors='coerce')

    # Remove rows with NaN values
    result_df = result_df.dropna()
    result_df = result_df.sort_values('timestamp').reset_index(drop=True)

    # Remove duplicate timestamps, keeping the first occurrence
    result_df = result_df.drop_duplicates(subset=['timestamp'], keep='first')

    return result_df


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
                    print(f"      ✓ {bmu_unit}: {len(df)} records")
        except Exception as e:
            print(f"      ✗ Error fetching {bmu_unit} for {current}: {e}")

        current = chunk_end + timedelta(days=1)

    if all_data:
        return pd.concat(all_data, ignore_index=True)

    return pd.DataFrame()


def fetch_hornsea_data_month(year, month):
    """Fetch Hornsea 2 wind data for a specific month"""
    # Get month start and end dates
    if month == 2 and year % 4 == 0:
        days = 29
    elif month == 2:
        days = 28
    elif month in [4, 6, 9, 11]:
        days = 30
    else:
        days = 31

    start_date = datetime(year, month, 1)
    end_date = datetime(year, month, days, 23, 59, 59)

    print(f"    Fetching Hornsea 2 wind farm data...")
    wind_data = []

    for bmu, capacity in HORNSEA_2_FARMS.items():
        print(f"      Fetching {bmu}...")
        df = fetch_bmu_data(bmu, start_date, end_date)
        if not df.empty:
            df['capacity_mw'] = capacity
            wind_data.append(df)

    if not wind_data:
        print(f"      ✗ No data retrieved for {calendar.month_name[month]} {year}")
        return pd.DataFrame()

    wind_df = pd.concat(wind_data, ignore_index=True)

    # Convert time columns
    for col in ['timeFrom', 'timeTo', 'settlementDate', 'startTime']:
        if col in wind_df.columns:
            wind_df[col] = pd.to_datetime(wind_df[col], errors='coerce')

    return wind_df


def fetch_market_data_month(year, month):
    """Fetch market index data for a specific month"""
    # Get month start and end dates
    if month == 2 and year % 4 == 0:
        days = 29
    elif month == 2:
        days = 28
    elif month in [4, 6, 9, 11]:
        days = 30
    else:
        days = 31

    start_date = datetime(year, month, 1)
    end_date = datetime(year, month, days, 23, 59, 59)

    print(f"    Fetching market index pricing data...")
    market_df = fetch_market_index_data(start_date, end_date)

    if market_df.empty:
        print(f"      ✗ No market data retrieved for {calendar.month_name[month]} {year}")
        return pd.DataFrame()

    return process_market_data(market_df)


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
        print("      ✗ Warning: 'levelFrom' column not found")
        return pd.DataFrame()

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

    # Handle >100% capacity factor by capping at 100%
    result_df['capacity_factor_pct'] = result_df['capacity_factor_pct'].clip(upper=100)

    return result_df


def scale_capacity_factor(wind_data, target_cf=60):
    """
    Scale capacity factor data to achieve target average CF
    Enhances low-wind performance (simulating newer turbine technology)
    """
    if wind_data.empty:
        return wind_data

    current_avg_cf = wind_data['capacity_factor_pct'].mean()

    print(f"      Original average CF: {current_avg_cf:.2f}%")

    # Apply non-linear scaling that boosts low-wind performance more
    # This simulates newer turbines with better low-wind characteristics
    scaled_cf = np.zeros_like(wind_data['capacity_factor_pct'].values)

    for i, cf in enumerate(wind_data['capacity_factor_pct'].values):
        if cf < 15:  # Very low wind - 2x improvement
            scaled_cf[i] = cf * 2.0
        elif cf < 30:  # Low wind - 1.5x improvement
            scaled_cf[i] = cf * 1.5
        elif cf < 50:  # Medium wind - 1.2x improvement
            scaled_cf[i] = cf * 1.2
        else:  # Good wind - 1.05x improvement
            scaled_cf[i] = cf * 1.05

    # Clip to realistic maximum
    scaled_cf = np.clip(scaled_cf, 0, 95)

    # Final adjustment to hit exact target
    current_scaled_avg = scaled_cf.mean()
    final_scale = target_cf / current_scaled_avg
    scaled_cf = scaled_cf * final_scale
    scaled_cf = np.clip(scaled_cf, 0, 95)

    wind_data['capacity_factor_pct'] = scaled_cf

    print(f"      Scaled average CF: {scaled_cf.mean():.2f}%")

    return wind_data


def get_wind_profile_month(year, month, target_cf=60):
    """
    Get real wind generation profile for a month from Elexon API
    and scale to target capacity factor
    """
    # Fetch real data
    wind_df = fetch_hornsea_data_month(year, month)

    if wind_df.empty:
        print(f"      ✗ No data available for {calendar.month_name[month]} {year}")
        return pd.DataFrame()

    # Process the data
    wind_data = process_wind_data(wind_df)

    if wind_data.empty:
        print(f"      ✗ Failed to process data for {calendar.month_name[month]} {year}")
        return pd.DataFrame()

    print(f"      Processed {len(wind_data)} data points")

    # Scale capacity factor to target
    wind_data = scale_capacity_factor(wind_data, target_cf)

    return wind_data


def merge_market_and_wind(market_df, wind_df):
    """Merge market pricing data with wind generation data"""
    if market_df.empty or wind_df.empty:
        return wind_df

    # Create a complete time range at 30-minute intervals
    min_time = min(market_df['timestamp'].min(), wind_df['timestamp'].min())
    max_time = max(market_df['timestamp'].max(), wind_df['timestamp'].max())

    # Create 30-minute frequency time index
    time_range = pd.date_range(start=min_time, end=max_time, freq='30min')

    # Set timestamp as index and reindex
    market_indexed = market_df.set_index('timestamp')
    wind_indexed = wind_df.set_index('timestamp')

    # Reindex to the complete time range and forward fill
    market_reindexed = market_indexed.reindex(time_range, method='ffill')
    wind_reindexed = wind_indexed.reindex(time_range, method='ffill')

    # Merge the two datasets
    merged = wind_reindexed.merge(market_reindexed, left_index=True, right_index=True, how='left')

    # Forward fill any remaining NaN values in price
    merged['price'] = merged['price'].fillna(method='ffill')

    # If still NaN, use a default value
    merged['price'] = merged['price'].fillna(50.0)  # Default to £50/MWh if no data

    merged = merged.reset_index().rename(columns={'index': 'timestamp'})
    merged = merged.dropna(subset=['capacity_factor_pct'])

    return merged


# ============== STORAGE SIMULATION WITH REVENUE ==============

def simulate_storage_month_with_revenue(timestamps, generation_mw, market_prices,
                                        initial_bess_mwh, initial_laes_mwh):
    """
    Simulate storage operation for a month of generation data with revenue calculation.
    Returns detailed results for each timestamp.
    """
    n_periods = len(timestamps)

    # Assume 30-minute settlement periods
    period_hours = 0.5

    # Initialize arrays to store results
    bess_soc_mwh = np.zeros(n_periods)
    laes_soc_mwh = np.zeros(n_periods)
    grid_export_mw = np.zeros(n_periods)
    bess_charge_mw = np.zeros(n_periods)
    laes_charge_mw = np.zeros(n_periods)
    bess_discharge_mw = np.zeros(n_periods)
    laes_discharge_mw = np.zeros(n_periods)
    shortfall_mw = np.zeros(n_periods)

    # Revenue tracking arrays
    baseline_export_mw = np.zeros(n_periods)
    excess_export_mw = np.zeros(n_periods)
    baseline_revenue = np.zeros(n_periods)
    excess_revenue = np.zeros(n_periods)
    total_revenue = np.zeros(n_periods)

    # Set initial states
    bess_soc_mwh[0] = initial_bess_mwh
    laes_soc_mwh[0] = initial_laes_mwh

    for i in range(n_periods):
        if i > 0:
            # Carry forward storage states
            bess_soc_mwh[i] = bess_soc_mwh[i - 1]
            laes_soc_mwh[i] = laes_soc_mwh[i - 1]

        gen_mw = generation_mw[i]
        market_price = market_prices[i]

        if gen_mw > BASELINE_EXPORT_MW:
            # Excess generation - charge storage
            excess_mw = gen_mw - BASELINE_EXPORT_MW
            grid_export_mw[i] = BASELINE_EXPORT_MW

            # Charge BESS first (faster response)
            bess_headroom_mwh = BESS_CAPACITY_MWH - bess_soc_mwh[i]
            bess_charge_limit_mw = min(BESS_POWER_MW,
                                       bess_headroom_mwh / period_hours)
            bess_charge_mw[i] = min(excess_mw, bess_charge_limit_mw)
            bess_soc_mwh[i] += bess_charge_mw[i] * period_hours * BESS_RTE

            # Remaining excess to LAES
            remaining_excess_mw = excess_mw - bess_charge_mw[i]
            if remaining_excess_mw > 0:
                laes_headroom_mwh = LAES_CAPACITY_MWH - laes_soc_mwh[i]
                laes_charge_limit_mw = min(LAES_POWER_MW,
                                           laes_headroom_mwh / period_hours)
                laes_charge_mw[i] = min(remaining_excess_mw, laes_charge_limit_mw)
                laes_soc_mwh[i] += laes_charge_mw[i] * period_hours * LAES_RTE

        else:
            # Deficit - discharge storage
            deficit_mw = BASELINE_EXPORT_MW - gen_mw

            # Use BESS first (down to minimum SOC)
            bess_min_mwh = BESS_CAPACITY_MWH * BESS_MIN_SOC
            bess_available_mwh = max(0, bess_soc_mwh[i] - bess_min_mwh)
            bess_discharge_limit_mw = min(BESS_POWER_MW,
                                          bess_available_mwh / period_hours)
            bess_discharge_mw[i] = min(deficit_mw, bess_discharge_limit_mw)
            bess_soc_mwh[i] -= bess_discharge_mw[i] * period_hours

            # Remaining deficit from LAES
            remaining_deficit_mw = deficit_mw - bess_discharge_mw[i]
            if remaining_deficit_mw > 0:
                laes_available_mwh = laes_soc_mwh[i]
                laes_discharge_limit_mw = min(LAES_POWER_MW,
                                              laes_available_mwh / period_hours)
                laes_discharge_mw[i] = min(remaining_deficit_mw, laes_discharge_limit_mw)
                laes_soc_mwh[i] -= laes_discharge_mw[i] * period_hours

                # Check if LAES is depleted and use remaining BESS
                final_deficit_mw = remaining_deficit_mw - laes_discharge_mw[i]
                if final_deficit_mw > 0 and bess_soc_mwh[i] > 0:
                    # Use remaining BESS capacity (below minimum SOC)
                    emergency_bess_mwh = bess_soc_mwh[i]
                    emergency_bess_mw = min(BESS_POWER_MW - bess_discharge_mw[i],
                                            emergency_bess_mwh / period_hours,
                                            final_deficit_mw)
                    bess_discharge_mw[i] += emergency_bess_mw
                    bess_soc_mwh[i] -= emergency_bess_mw * period_hours
                    final_deficit_mw -= emergency_bess_mw

                # Any remaining deficit is a shortfall
                shortfall_mw[i] = max(0, final_deficit_mw)

            # Calculate actual grid export
            grid_export_mw[i] = gen_mw + bess_discharge_mw[i] + laes_discharge_mw[i]

        # Calculate revenue for this period
        actual_export = grid_export_mw[i]

        # Split into baseline and excess
        if actual_export <= BASELINE_EXPORT_MW:
            baseline_export_mw[i] = actual_export
            excess_export_mw[i] = 0
        else:
            baseline_export_mw[i] = BASELINE_EXPORT_MW
            excess_export_mw[i] = actual_export - BASELINE_EXPORT_MW

        # Calculate revenue based on pricing rules
        # Only sell if price is above floor (0 or positive)
        if market_price >= PRICE_FLOOR:
            # Baseline export sold at market price
            baseline_revenue[i] = baseline_export_mw[i] * market_price * period_hours

            # Excess export sold at curtailment price or market price (whichever is lower)
            if market_price < CURTAILMENT_PRICE:
                excess_price = market_price
            else:
                excess_price = CURTAILMENT_PRICE
            excess_revenue[i] = excess_export_mw[i] * excess_price * period_hours
        else:
            # Negative prices - don't sell
            baseline_revenue[i] = 0
            excess_revenue[i] = 0

        total_revenue[i] = baseline_revenue[i] + excess_revenue[i]

    # Create results dataframe
    results = pd.DataFrame({
        'timestamp': timestamps,
        'generation_mw': generation_mw,
        'grid_export_mw': grid_export_mw,
        'market_price': market_prices,
        'bess_soc_mwh': bess_soc_mwh,
        'laes_soc_mwh': laes_soc_mwh,
        'bess_charge_mw': bess_charge_mw,
        'laes_charge_mw': laes_charge_mw,
        'bess_discharge_mw': bess_discharge_mw,
        'laes_discharge_mw': laes_discharge_mw,
        'shortfall_mw': shortfall_mw,
        'baseline_export_mw': baseline_export_mw,
        'excess_export_mw': excess_export_mw,
        'baseline_revenue': baseline_revenue,
        'excess_revenue': excess_revenue,
        'total_revenue': total_revenue
    })

    # Add status column for visualization
    conditions = [
        (results['bess_charge_mw'] > 0) | (results['laes_charge_mw'] > 0),
        results['bess_discharge_mw'] > 0,
        results['laes_discharge_mw'] > 0,
        results['shortfall_mw'] > 0
    ]
    choices = ['Charging', 'BESS Discharge', 'LAES Discharge', 'Shortfall']
    results['status'] = np.select(conditions, choices, default='Normal')

    return results


# ============== VISUALIZATION ==============

def plot_monthly_results_with_revenue(results, year, month):
    """Create visualization for monthly simulation results including revenue"""

    fig, axes = plt.subplots(5, 1, figsize=(16, 18))
    month_name = calendar.month_name[month]

    # Color scheme
    colors = {
        'Charging': 'green',
        'BESS Discharge': 'gold',
        'LAES Discharge': 'blue',
        'Shortfall': 'red',
        'Normal': 'lightgray'
    }

    # Plot 1: Generation and Grid Export
    ax1 = axes[0]
    ax1.plot(results['timestamp'], results['generation_mw'],
             label='Wind Generation', color='darkgreen', linewidth=1.5, alpha=0.7)
    ax1.plot(results['timestamp'], results['grid_export_mw'],
             label='Grid Export', color='black', linewidth=1.5)
    ax1.axhline(y=BASELINE_EXPORT_MW, color='red', linestyle='--',
                label=f'Target Export ({BASELINE_EXPORT_MW} MW)', alpha=0.5)
    ax1.set_ylabel('Power (MW)', fontsize=11)
    ax1.set_title(f'Wind Generation and Grid Export - {month_name} {year}',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Market Price
    ax2 = axes[1]
    ax2.plot(results['timestamp'], results['market_price'],
             label='Market Price', color='purple', linewidth=1.5)
    ax2.axhline(y=CURTAILMENT_PRICE, color='orange', linestyle='--',
                label=f'Curtailment Price (£{CURTAILMENT_PRICE}/MWh)', alpha=0.5)
    ax2.axhline(y=0, color='red', linestyle=':', label='Zero Price', alpha=0.5)
    ax2.set_ylabel('Price (£/MWh)', fontsize=11)
    ax2.set_title('Market Index Price', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Storage State of Charge (as percentage)
    ax3 = axes[2]
    # Convert to percentages
    bess_soc_pct = (results['bess_soc_mwh'] / BESS_CAPACITY_MWH) * 100
    laes_soc_pct = (results['laes_soc_mwh'] / LAES_CAPACITY_MWH) * 100

    ax3.plot(results['timestamp'], bess_soc_pct,
             label=f'BESS ({BESS_CAPACITY_MWH} MWh)', color='gold', linewidth=2)
    ax3.plot(results['timestamp'], laes_soc_pct,
             label=f'LAES ({LAES_CAPACITY_MWH} MWh)', color='blue', linewidth=2)
    ax3.axhline(y=BESS_MIN_SOC * 100, color='gold',
                linestyle=':', label='BESS Min SOC', alpha=0.5)
    ax3.set_ylabel('State of Charge (%)', fontsize=11)
    ax3.set_ylim(0, 105)
    ax3.set_title('Storage State of Charge', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Revenue Generation
    ax4 = axes[3]
    # Calculate cumulative revenue in £
    cumulative_revenue = results['total_revenue'].cumsum()
    ax4.fill_between(results['timestamp'], 0, cumulative_revenue / 1000,
                     color='green', alpha=0.5, label='Cumulative Revenue')
    ax4.plot(results['timestamp'], cumulative_revenue / 1000,
             color='darkgreen', linewidth=2, label='Total Revenue')
    ax4.set_ylabel('Revenue (£ thousands)', fontsize=11)
    ax4.set_title('Cumulative Revenue', fontsize=13, fontweight='bold')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)

    # Plot 5: System Status (Color-coded)
    ax5 = axes[4]

    # Create color-coded background
    for i in range(len(results) - 1):
        ax5.axvspan(results['timestamp'].iloc[i],
                    results['timestamp'].iloc[i + 1],
                    facecolor=colors[results['status'].iloc[i]],
                    alpha=0.3)

    # Plot shortfall if any
    ax5.fill_between(results['timestamp'], 0, results['shortfall_mw'],
                     color='red', alpha=0.7, label='Supply Shortfall')

    ax5.set_ylabel('Shortfall (MW)', fontsize=11)
    ax5.set_xlabel('Date', fontsize=11)
    ax5.set_title('System Status', fontsize=13, fontweight='bold')

    # Create custom legend
    legend_elements = [
        mpatches.Patch(color='green', alpha=0.3, label='Charging Storage'),
        mpatches.Patch(color='gold', alpha=0.3, label='BESS Discharging'),
        mpatches.Patch(color='blue', alpha=0.3, label='LAES Discharging'),
        mpatches.Patch(color='red', alpha=0.3, label='Supply Shortfall')
    ]
    ax5.legend(handles=legend_elements, loc='upper right')
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    filename = f'storage_simulation_revenue_{year}_{month:02d}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved plot: {filename}")
    plt.close()

    return filename


# ============== MAIN SIMULATION ==============

def main():
    print("\n" + "=" * 70)
    print("RENEWABLE GENERATION WITH STORAGE & REVENUE SIMULATION")
    print("Using Real Hornsea 2 Data and Market Pricing from Elexon BMRS API")
    print(f"Project Capacity: {PROJECT_CAPACITY_MW} MW")
    print(f"Target CF: {NEW_TURBINE_CF_TARGET}% (scaled from Hornsea 2 baseline)")
    print(f"Target Export: {BASELINE_EXPORT_MW} MW")
    print(f"BESS: {BESS_POWER_MW} MW / {BESS_CAPACITY_MWH} MWh (RTE: {BESS_RTE * 100:.0f}%)")
    print(f"LAES: {LAES_POWER_MW} MW / {LAES_CAPACITY_MWH} MWh (RTE: {LAES_RTE * 100:.0f}%)")
    print(f"Baseline Price: Market Rate")
    print(f"Excess Price: £{CURTAILMENT_PRICE}/MWh")
    print("=" * 70 + "\n")

    # Define simulation period
    simulation_months = []
    current = datetime(2024, 10, 1)  # Start October 2024
    end = datetime(2025, 10, 31)  # End October 2025

    while current <= end:
        simulation_months.append((current.year, current.month))
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)

    # Initialize storage states
    bess_soc_mwh = 0
    laes_soc_mwh = 0

    all_results = []
    monthly_summaries = []

    for year, month in simulation_months:
        print(f"\n{'=' * 60}")
        print(f"Processing {calendar.month_name[month]} {year}")
        print(f"{'=' * 60}")

        # Fetch real wind data from Elexon API
        wind_data = get_wind_profile_month(year, month, NEW_TURBINE_CF_TARGET)

        if wind_data.empty:
            print(f"  ✗ Skipping {calendar.month_name[month]} {year} - no wind data available")
            continue

        # Fetch market pricing data
        market_data = fetch_market_data_month(year, month)

        if market_data.empty:
            print(f"  ⚠ Warning: No market data for {calendar.month_name[month]} {year}, using default £50/MWh")
            # Create default market data
            market_data = pd.DataFrame({
                'timestamp': wind_data['timestamp'],
                'price': 50.0
            })

        # Merge wind and market data
        combined_data = merge_market_and_wind(market_data, wind_data)

        if combined_data.empty:
            print(f"  ✗ Skipping {calendar.month_name[month]} {year} - failed to merge data")
            continue

        # Scale to project capacity
        generation_mw = (combined_data['capacity_factor_pct'].values / 100) * PROJECT_CAPACITY_MW

        print(f"  Average CF: {combined_data['capacity_factor_pct'].mean():.2f}%")
        print(f"  Average Market Price: £{combined_data['price'].mean():.2f}/MWh")
        print(f"  Initial storage: BESS={bess_soc_mwh:.1f} MWh, LAES={laes_soc_mwh:.1f} MWh")

        # Run storage simulation with revenue
        results = simulate_storage_month_with_revenue(
            combined_data['timestamp'],
            generation_mw,
            combined_data['price'].values,
            bess_soc_mwh,
            laes_soc_mwh
        )

        # Update storage states for next month
        bess_soc_mwh = results['bess_soc_mwh'].iloc[-1]
        laes_soc_mwh = results['laes_soc_mwh'].iloc[-1]

        # Calculate statistics
        total_shortfall_mwh = results['shortfall_mw'].sum() * 0.5  # 30-min periods
        availability = (1 - results['shortfall_mw'].sum() /
                        (len(results) * BASELINE_EXPORT_MW)) * 100

        # Revenue statistics
        total_revenue = results['total_revenue'].sum()
        baseline_revenue = results['baseline_revenue'].sum()
        excess_revenue = results['excess_revenue'].sum()
        total_energy_sold = (results['baseline_export_mw'].sum() + results['excess_export_mw'].sum()) * 0.5
        avg_realized_price = total_revenue / total_energy_sold if total_energy_sold > 0 else 0

        # Calculate maximum potential revenue (if 767MW always available)
        max_potential_revenue = 0
        for i in range(len(results)):
            market_price = results['market_price'].iloc[i]
            if market_price >= PRICE_FLOOR:
                # Always sell 767MW at market price
                max_baseline_revenue = BASELINE_EXPORT_MW * market_price * 0.5

                # Sell any generation above 767MW at curtailment price (or market if lower)
                gen_mw = results['generation_mw'].iloc[i]
                if gen_mw > BASELINE_EXPORT_MW:
                    excess_mw = gen_mw - BASELINE_EXPORT_MW
                    excess_price = min(market_price, CURTAILMENT_PRICE)
                    max_excess_revenue = excess_mw * excess_price * 0.5
                else:
                    max_excess_revenue = 0

                max_potential_revenue += max_baseline_revenue + max_excess_revenue

        print(f"  Final storage: BESS={bess_soc_mwh:.1f} MWh, LAES={laes_soc_mwh:.1f} MWh")
        print(f"  Total shortfall: {total_shortfall_mwh:.1f} MWh")
        print(f"  Grid supply availability: {availability:.2f}%")
        print(f"  Total Revenue: £{total_revenue:,.2f}")
        print(f"    - Baseline (≤767MW): £{baseline_revenue:,.2f}")
        print(f"    - Excess (>767MW): £{excess_revenue:,.2f}")
        print(f"  Maximum Potential Revenue: £{max_potential_revenue:,.2f}")
        print(f"  Revenue Achievement: {(total_revenue / max_potential_revenue * 100):.2f}%")
        print(f"  Average Realized Price: £{avg_realized_price:.2f}/MWh")

        # Store results
        all_results.append(results)
        monthly_summaries.append({
            'year': year,
            'month': month,
            'month_name': calendar.month_name[month],
            'avg_generation_mw': results['generation_mw'].mean(),
            'avg_export_mw': results['grid_export_mw'].mean(),
            'total_shortfall_mwh': total_shortfall_mwh,
            'availability_pct': availability,
            'final_bess_mwh': bess_soc_mwh,
            'final_laes_mwh': laes_soc_mwh,
            'avg_market_price': results['market_price'].mean(),
            'total_revenue': total_revenue,
            'baseline_revenue': baseline_revenue,
            'excess_revenue': excess_revenue,
            'total_energy_sold_mwh': total_energy_sold,
            'avg_realized_price': avg_realized_price,
            'max_potential_revenue': max_potential_revenue,
            'revenue_achievement_pct': (total_revenue / max_potential_revenue * 100) if max_potential_revenue > 0 else 0
        })

        # Create monthly plot
        plot_monthly_results_with_revenue(results, year, month)

    # Save summary to CSV
    if monthly_summaries:
        summary_df = pd.DataFrame(monthly_summaries)
        summary_df.to_csv('storage_simulation_revenue_summary.csv', index=False)
        print(f"\nMonthly revenue summary saved to: storage_simulation_revenue_summary.csv")

        # Print overall statistics
        print("\n" + "=" * 70)
        print("SIMULATION SUMMARY")
        print("=" * 70)
        print(f"Average generation: {summary_df['avg_generation_mw'].mean():.1f} MW")
        print(f"Average grid export: {summary_df['avg_export_mw'].mean():.1f} MW")
        print(f"Total shortfall: {summary_df['total_shortfall_mwh'].sum():.1f} MWh")
        print(f"Overall availability: {summary_df['availability_pct'].mean():.2f}%")
        print(f"Months with shortfalls: {(summary_df['total_shortfall_mwh'] > 0).sum()}")
        print(f"\n--- REVENUE SUMMARY ---")
        print(f"Total Annual Revenue: £{summary_df['total_revenue'].sum():,.2f}")
        print(f"  - From Baseline Export: £{summary_df['baseline_revenue'].sum():,.2f}")
        print(f"  - From Excess Export: £{summary_df['excess_revenue'].sum():,.2f}")
        print(f"Average Monthly Revenue: £{summary_df['total_revenue'].mean():,.2f}")
        print(f"Total Energy Sold: {summary_df['total_energy_sold_mwh'].sum():,.1f} MWh")
        print(
            f"Average Realized Price: £{(summary_df['total_revenue'].sum() / summary_df['total_energy_sold_mwh'].sum()):.2f}/MWh")
        print(f"Average Market Price: £{summary_df['avg_market_price'].mean():.2f}/MWh")

        # Best and worst months
        best_month = summary_df.loc[summary_df['total_revenue'].idxmax()]
        worst_month = summary_df.loc[summary_df['total_revenue'].idxmin()]

        print(f"\nBest Revenue Month: {best_month['month_name']} {best_month['year']} "
              f"(£{best_month['total_revenue']:,.2f})")
        print(f"Worst Revenue Month: {worst_month['month_name']} {worst_month['year']} "
              f"(£{worst_month['total_revenue']:,.2f})")

        # Print detailed monthly summary table
        print("\n" + "=" * 70)
        print("DETAILED MONTHLY SUMMARY")
        print("=" * 70)
        print(f"{'Month':<15} {'Availability':<15} {'Actual Revenue':<20} {'Max Revenue':<20} {'Achievement':<12}")
        print("-" * 90)

        for _, row in summary_df.iterrows():
            month_label = f"{row['month_name'][:3]} {row['year']}"
            print(f"{month_label:<15} {row['availability_pct']:>13.2f}% "
                  f"£{row['total_revenue']:>18,.2f} "
                  f"£{row['max_potential_revenue']:>18,.2f} "
                  f"{row['revenue_achievement_pct']:>10.2f}%")

        print("-" * 90)
        print(f"{'AVERAGE':<15} {summary_df['availability_pct'].mean():>13.2f}% "
              f"£{summary_df['total_revenue'].mean():>18,.2f} "
              f"£{summary_df['max_potential_revenue'].mean():>18,.2f} "
              f"{summary_df['revenue_achievement_pct'].mean():>10.2f}%")
        print("-" * 90)
        print(f"{'TOTAL':<15} {'':<15} "
              f"£{summary_df['total_revenue'].sum():>18,.2f} "
              f"£{summary_df['max_potential_revenue'].sum():>18,.2f} "
              f"{(summary_df['total_revenue'].sum() / summary_df['max_potential_revenue'].sum() * 100):>10.2f}%")

        print("\n" + "=" * 70)
    else:
        print("\n✗ No data was successfully processed!")


if __name__ == "__main__":
    main()