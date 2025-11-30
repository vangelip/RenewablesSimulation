import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime, timedelta
import calendar
import requests
import warnings

warnings.filterwarnings('ignore')

# Set matplotlib style
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# ============== CONFIGURATION PARAMETERS ==============
# Wind Farm Configuration
NUM_TURBINES = 108
TURBINE_CAPACITY_MW = 14
INSTALLED_CAPACITY_MW = NUM_TURBINES * TURBINE_CAPACITY_MW  # 1,512 MW
BASELINE_EXPORT_MW = 767  # Target baseload delivery
CAPACITY_FACTOR_TARGET = 60  # Target CF after scaling (%)
PROJECT_LIFETIME_YEARS = 30
STORAGE_LIFETIME_YEARS = 25

# Storage Configuration
BESS_POWER_MW = 300
BESS_CAPACITY_MWH = 600
BESS_RTE = 0.86  # 86% round-trip efficiency
BESS_MIN_SOC = 0.10  # 10% minimum state of charge (lower for arbitrage)
BESS_MAX_CHARGE_MW = BESS_POWER_MW  # Grid import limit = discharge rate

LAES_POWER_MW = 767
LAES_CAPACITY_MWH = 20000
LAES_RTE = 0.55  # 55% round-trip efficiency
LAES_MIN_SOC = 0.05  # 5% minimum
LAES_MAX_CHARGE_MW = LAES_POWER_MW  # Grid import limit = discharge rate

# Economic Parameters
DISCOUNT_RATE = 0.065  # 6.5% IRR/WACC

# CAPEX (£ millions)
WIND_CAPEX_M = 2840
BESS_CAPEX_M = 300
LAES_CAPEX_M = 1250
TOTAL_CAPEX_M = WIND_CAPEX_M + BESS_CAPEX_M + LAES_CAPEX_M

# OPEX
WIND_FIXED_OPEX = 65.5e6
WIND_VAR_OPEX_PER_MWH = 1.0
BESS_FIXED_OPEX = 1.8e6
BESS_VAR_OPEX_PER_MWH = 1.0
LAES_FIXED_OPEX = 12.3e6
LAES_VAR_OPEX_PER_MWH = 4.5

# CfD pricing
CFD_AR6_PRICE = 82.0
CFD_AR7_PRICE = 73.0

# Storage ancillary revenues (Modo Energy 2024)
BESS_ANNUAL_REVENUE_PER_MW = 50000  # £50k/MW/year
LAES_ANNUAL_REVENUE_PER_MW = 35000  # £35k/MW/year

# BESS augmentation
BESS_AUGMENTATION_COST_M = 105

# ============== GRID CHARGING STRATEGIES ==============
STRATEGIES = {
    'A_Fixed': {
        'name': 'Fixed Threshold',
        'description': 'Charge < £30/MWh, Discharge > £80/MWh',
        'bess_charge_threshold': 30.0,
        'bess_discharge_threshold': 80.0,
        'laes_charge_threshold': 25.0,  # LAES slightly lower (longer duration, needs bigger spread)
        'laes_discharge_threshold': 75.0,
        'dynamic': False
    },
    'B_Dynamic': {
        'name': 'Dynamic Threshold',
        'description': 'Charge < 50% daily avg, Discharge > 150% daily avg',
        'charge_pct': 0.50,
        'discharge_pct': 1.50,
        'min_charge_price': 40.0,  # Never charge above £40 even if dynamic says so
        'min_discharge_price': 60.0,  # Never discharge below £60
        'dynamic': True
    },
    'C_Negative': {
        'name': 'Negative Prices Only',
        'description': 'Charge < £0/MWh (get paid!), Discharge > £50/MWh',
        'bess_charge_threshold': 0.0,
        'bess_discharge_threshold': 50.0,
        'laes_charge_threshold': 0.0,
        'laes_discharge_threshold': 50.0,
        'dynamic': False
    }
}

# Hornsea 2 reference
HORNSEA_2_FARMS = {
    'HOWBO-1': 440,
    'HOWBO-2': 440,
    'HOWBO-3': 440,
}
HORNSEA_2_CAPACITY = sum(HORNSEA_2_FARMS.values())

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"

print("=" * 70)
print("YORKSHIRE WIND PLUS STORAGE - GRID CHARGING COMPARISON (v3)")
print("=" * 70)
print(f"\nComparing {len(STRATEGIES)} grid charging strategies:")
for key, strat in STRATEGIES.items():
    print(f"  {key}: {strat['name']} - {strat['description']}")
print("=" * 70)


# ============== API FUNCTIONS (same as before) ==============

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
        except Exception as e:
            pass

        current = chunk_end + timedelta(days=1)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        for col in ['startTime', 'settlementDate']:
            if col in combined.columns:
                combined[col] = pd.to_datetime(combined[col], errors='coerce')
        return combined

    return pd.DataFrame()


def process_market_data(df):
    """Process market index data - filter for APXMIDP"""
    if df.empty:
        return pd.DataFrame()

    if 'dataProvider' in df.columns:
        df = df[df['dataProvider'] == 'APXMIDP'].copy()

    if df.empty:
        return pd.DataFrame()

    result_df = pd.DataFrame()

    if 'startTime' in df.columns:
        result_df['timestamp'] = df['startTime']

    if 'price' in df.columns:
        result_df['price'] = pd.to_numeric(df['price'], errors='coerce')

    result_df = result_df.dropna()
    result_df = result_df.sort_values('timestamp').reset_index(drop=True)
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
        except Exception as e:
            pass

        current = chunk_end + timedelta(days=1)

    if all_data:
        return pd.concat(all_data, ignore_index=True)

    return pd.DataFrame()


def fetch_hornsea_data_month(year, month):
    """Fetch Hornsea 2 wind data for a specific month"""
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

    wind_data = []

    for bmu, capacity in HORNSEA_2_FARMS.items():
        df = fetch_bmu_data(bmu, start_date, end_date)
        if not df.empty:
            df['capacity_mw'] = capacity
            wind_data.append(df)

    if not wind_data:
        return pd.DataFrame()

    wind_df = pd.concat(wind_data, ignore_index=True)

    for col in ['timeFrom', 'timeTo', 'settlementDate', 'startTime']:
        if col in wind_df.columns:
            wind_df[col] = pd.to_datetime(wind_df[col], errors='coerce')

    return wind_df


def fetch_market_data_month(year, month):
    """Fetch market index data for a specific month"""
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

    market_df = fetch_market_index_data(start_date, end_date)

    if market_df.empty:
        return pd.DataFrame()

    return process_market_data(market_df)


def process_wind_data(df):
    """Process wind farm data to get capacity factor"""
    if df.empty:
        return pd.DataFrame()

    if 'dataset' in df.columns:
        df = df[df['dataset'] == 'PN'].copy()

    if 'levelFrom' in df.columns:
        df['mw_output'] = pd.to_numeric(df['levelFrom'], errors='coerce')
        df['mw_output'] = df['mw_output'].clip(lower=0)
    else:
        return pd.DataFrame()

    timeseries = df.groupby('timeFrom')['mw_output'].sum().sort_index()
    cf_series = (timeseries / HORNSEA_2_CAPACITY) * 100

    result_df = pd.DataFrame({
        'timestamp': cf_series.index,
        'output_mw': timeseries.values,
        'capacity_factor_pct': cf_series.values
    })

    result_df = result_df.dropna()
    result_df['capacity_factor_pct'] = result_df['capacity_factor_pct'].clip(upper=100)

    return result_df


def scale_capacity_factor(wind_data, target_cf=60):
    """Scale capacity factor data to achieve target average CF"""
    if wind_data.empty:
        return wind_data

    scaled_cf = np.zeros_like(wind_data['capacity_factor_pct'].values)

    for i, cf in enumerate(wind_data['capacity_factor_pct'].values):
        if cf < 15:
            scaled_cf[i] = cf * 2.0
        elif cf < 30:
            scaled_cf[i] = cf * 1.5
        elif cf < 50:
            scaled_cf[i] = cf * 1.2
        else:
            scaled_cf[i] = cf * 1.05

    scaled_cf = np.clip(scaled_cf, 0, 95)

    current_scaled_avg = scaled_cf.mean()
    if current_scaled_avg > 0:
        final_scale = target_cf / current_scaled_avg
        scaled_cf = scaled_cf * final_scale
        scaled_cf = np.clip(scaled_cf, 0, 95)

    wind_data['capacity_factor_pct'] = scaled_cf

    return wind_data


def get_wind_profile_month(year, month, target_cf=60):
    """Get real wind generation profile for a month"""
    wind_df = fetch_hornsea_data_month(year, month)

    if wind_df.empty:
        return pd.DataFrame()

    wind_data = process_wind_data(wind_df)

    if wind_data.empty:
        return pd.DataFrame()

    wind_data = scale_capacity_factor(wind_data, target_cf)

    return wind_data


def merge_market_and_wind(market_df, wind_df):
    """Merge market pricing data with wind generation data"""
    if market_df.empty or wind_df.empty:
        return wind_df

    min_time = min(market_df['timestamp'].min(), wind_df['timestamp'].min())
    max_time = max(market_df['timestamp'].max(), wind_df['timestamp'].max())

    time_range = pd.date_range(start=min_time, end=max_time, freq='30min')

    market_indexed = market_df.set_index('timestamp')
    wind_indexed = wind_df.set_index('timestamp')

    market_reindexed = market_indexed.reindex(time_range, method='ffill')
    wind_reindexed = wind_indexed.reindex(time_range, method='ffill')

    merged = wind_reindexed.merge(market_reindexed, left_index=True, right_index=True, how='left')

    merged['price'] = merged['price'].fillna(method='ffill')
    merged['price'] = merged['price'].fillna(50.0)

    merged = merged.reset_index().rename(columns={'index': 'timestamp'})
    merged = merged.dropna(subset=['capacity_factor_pct'])

    return merged


def fetch_all_monthly_data():
    """Pre-fetch all wind and market data"""
    simulation_months = []
    current = datetime(2024, 10, 1)
    end = datetime(2025, 10, 31)

    while current <= end:
        simulation_months.append((current.year, current.month))
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)

    print("\nPre-fetching wind and market data from Elexon BMRS API...")
    monthly_data = {}

    for year, month in simulation_months:
        print(f"  Fetching {calendar.month_name[month]} {year}...")

        wind_data = get_wind_profile_month(year, month, CAPACITY_FACTOR_TARGET)
        if wind_data.empty:
            print(f"    ✗ No wind data for {calendar.month_name[month]} {year}")
            continue

        market_data = fetch_market_data_month(year, month)
        if market_data.empty:
            print(f"    ⚠ No market data, using default £50/MWh")
            market_data = pd.DataFrame({
                'timestamp': wind_data['timestamp'],
                'price': 50.0
            })

        combined_data = merge_market_and_wind(market_data, wind_data)
        if combined_data.empty:
            print(f"    ✗ Failed to merge data for {calendar.month_name[month]} {year}")
            continue

        monthly_data[(year, month)] = combined_data
        avg_cf = combined_data['capacity_factor_pct'].mean()
        avg_price = combined_data['price'].mean()
        min_price = combined_data['price'].min()
        max_price = combined_data['price'].max()
        neg_hours = (combined_data['price'] < 0).sum() * 0.5
        print(f"    ✓ {len(combined_data)} points | CF: {avg_cf:.1f}% | "
              f"Price: £{min_price:.0f}-£{max_price:.0f} (avg £{avg_price:.1f}) | "
              f"Neg hours: {neg_hours:.1f}h")

    return monthly_data


# ============== STORAGE SIMULATION WITH GRID CHARGING ==============

def calculate_dynamic_thresholds(prices, strategy):
    """Calculate dynamic thresholds based on daily averages"""
    rolling_avg = pd.Series(prices).rolling(window=48, min_periods=1, center=True).mean().values

    charge_threshold = rolling_avg * strategy['charge_pct']
    discharge_threshold = rolling_avg * strategy['discharge_pct']

    charge_threshold = np.minimum(charge_threshold, strategy['min_charge_price'])
    discharge_threshold = np.maximum(discharge_threshold, strategy['min_discharge_price'])

    return charge_threshold, discharge_threshold


def simulate_storage_with_grid_charging(timestamps, generation_mw, market_prices,
                                        initial_bess_mwh, initial_laes_mwh,
                                        strategy):
    """
    Simulate storage operation WITH grid charging capability.

    Key changes:
    - Storage can charge from GRID when prices are low
    - Cheapest energy prioritized (Option B)
    - BESS: Arbitrage focus
    - LAES: Wind firming primary, arbitrage secondary
    """
    n_periods = len(timestamps)
    period_hours = 0.5

    # Get thresholds based on strategy
    if strategy['dynamic']:
        bess_charge_thresh, bess_discharge_thresh = calculate_dynamic_thresholds(market_prices, strategy)
        laes_charge_thresh, laes_discharge_thresh = calculate_dynamic_thresholds(market_prices, strategy)
    else:
        bess_charge_thresh = np.full(n_periods, strategy['bess_charge_threshold'])
        bess_discharge_thresh = np.full(n_periods, strategy['bess_discharge_threshold'])
        laes_charge_thresh = np.full(n_periods, strategy['laes_charge_threshold'])
        laes_discharge_thresh = np.full(n_periods, strategy['laes_discharge_threshold'])

    # Initialize arrays
    bess_soc_mwh = np.zeros(n_periods)
    laes_soc_mwh = np.zeros(n_periods)
    grid_export_mw = np.zeros(n_periods)
    grid_import_mw = np.zeros(n_periods)

    bess_charge_wind_mw = np.zeros(n_periods)
    bess_charge_grid_mw = np.zeros(n_periods)
    laes_charge_wind_mw = np.zeros(n_periods)
    laes_charge_grid_mw = np.zeros(n_periods)

    bess_discharge_mw = np.zeros(n_periods)
    laes_discharge_mw = np.zeros(n_periods)
    shortfall_mw = np.zeros(n_periods)
    curtailed_mw = np.zeros(n_periods)

    wind_direct_mw = np.zeros(n_periods)
    storage_discharge_mw = np.zeros(n_periods)

    wind_cfd_ar6_revenue = np.zeros(n_periods)
    storage_market_revenue = np.zeros(n_periods)
    grid_charge_cost = np.zeros(n_periods)

    bess_cycles = np.zeros(n_periods)
    laes_cycles = np.zeros(n_periods)
    negative_price_charge_mwh = np.zeros(n_periods)

    bess_soc_mwh[0] = initial_bess_mwh
    laes_soc_mwh[0] = initial_laes_mwh

    for i in range(n_periods):
        if i > 0:
            bess_soc_mwh[i] = bess_soc_mwh[i - 1]
            laes_soc_mwh[i] = laes_soc_mwh[i - 1]

        gen_mw = generation_mw[i]
        price = market_prices[i]

        bess_charge_th = bess_charge_thresh[i]
        bess_discharge_th = bess_discharge_thresh[i]
        laes_charge_th = laes_charge_thresh[i]
        laes_discharge_th = laes_discharge_thresh[i]

        bess_headroom_mwh = BESS_CAPACITY_MWH - bess_soc_mwh[i]
        laes_headroom_mwh = LAES_CAPACITY_MWH - laes_soc_mwh[i]

        bess_min_mwh = BESS_CAPACITY_MWH * BESS_MIN_SOC
        laes_min_mwh = LAES_CAPACITY_MWH * LAES_MIN_SOC
        bess_available_mwh = max(0, bess_soc_mwh[i] - bess_min_mwh)
        laes_available_mwh = max(0, laes_soc_mwh[i] - laes_min_mwh)

        # === CASE 1: Price is LOW - CHARGE storage ===
        if price <= bess_charge_th or price <= laes_charge_th:

            if gen_mw >= BASELINE_EXPORT_MW:
                wind_direct_mw[i] = BASELINE_EXPORT_MW
                grid_export_mw[i] = BASELINE_EXPORT_MW
                excess_wind = gen_mw - BASELINE_EXPORT_MW

                if price <= bess_charge_th and bess_headroom_mwh > 0:
                    bess_charge_limit = min(BESS_POWER_MW, bess_headroom_mwh / period_hours)
                    bess_charge_wind_mw[i] = min(excess_wind, bess_charge_limit)
                    bess_soc_mwh[i] += bess_charge_wind_mw[i] * period_hours * BESS_RTE
                    excess_wind -= bess_charge_wind_mw[i]
                    bess_headroom_mwh = BESS_CAPACITY_MWH - bess_soc_mwh[i]

                if price <= laes_charge_th and laes_headroom_mwh > 0 and excess_wind > 0:
                    laes_charge_limit = min(LAES_POWER_MW, laes_headroom_mwh / period_hours)
                    laes_charge_wind_mw[i] = min(excess_wind, laes_charge_limit)
                    laes_soc_mwh[i] += laes_charge_wind_mw[i] * period_hours * LAES_RTE
                    excess_wind -= laes_charge_wind_mw[i]
                    laes_headroom_mwh = LAES_CAPACITY_MWH - laes_soc_mwh[i]

                curtailed_mw[i] = excess_wind

            else:
                wind_direct_mw[i] = gen_mw
                grid_export_mw[i] = gen_mw

            # Grid charging
            if price <= bess_charge_th and bess_headroom_mwh > 0:
                remaining_bess_capacity = min(BESS_MAX_CHARGE_MW - bess_charge_wind_mw[i],
                                              bess_headroom_mwh / period_hours)
                if remaining_bess_capacity > 0:
                    bess_charge_grid_mw[i] = remaining_bess_capacity
                    bess_soc_mwh[i] += bess_charge_grid_mw[i] * period_hours * BESS_RTE
                    grid_import_mw[i] += bess_charge_grid_mw[i]
                    grid_charge_cost[i] += bess_charge_grid_mw[i] * price * period_hours

                    if price < 0:
                        negative_price_charge_mwh[i] += bess_charge_grid_mw[i] * period_hours

            if price <= laes_charge_th and laes_headroom_mwh > 0:
                remaining_laes_capacity = min(LAES_MAX_CHARGE_MW - laes_charge_wind_mw[i],
                                              laes_headroom_mwh / period_hours)
                if remaining_laes_capacity > 0:
                    laes_charge_grid_mw[i] = remaining_laes_capacity
                    laes_soc_mwh[i] += laes_charge_grid_mw[i] * period_hours * LAES_RTE
                    grid_import_mw[i] += laes_charge_grid_mw[i]
                    grid_charge_cost[i] += laes_charge_grid_mw[i] * price * period_hours

                    if price < 0:
                        negative_price_charge_mwh[i] += laes_charge_grid_mw[i] * period_hours

        # === CASE 2: Price is HIGH - DISCHARGE for arbitrage ===
        elif price >= bess_discharge_th or price >= laes_discharge_th:

            if gen_mw >= BASELINE_EXPORT_MW:
                wind_direct_mw[i] = BASELINE_EXPORT_MW
                excess_wind = gen_mw - BASELINE_EXPORT_MW

                if bess_headroom_mwh > 0:
                    bess_charge_limit = min(BESS_POWER_MW, bess_headroom_mwh / period_hours)
                    bess_charge_wind_mw[i] = min(excess_wind, bess_charge_limit)
                    bess_soc_mwh[i] += bess_charge_wind_mw[i] * period_hours * BESS_RTE
                    excess_wind -= bess_charge_wind_mw[i]

                if laes_headroom_mwh > 0 and excess_wind > 0:
                    laes_charge_limit = min(LAES_POWER_MW, laes_headroom_mwh / period_hours)
                    laes_charge_wind_mw[i] = min(excess_wind, laes_charge_limit)
                    laes_soc_mwh[i] += laes_charge_wind_mw[i] * period_hours * LAES_RTE
                    excess_wind -= laes_charge_wind_mw[i]

                curtailed_mw[i] = excess_wind

                # Recalculate available energy after charging
                bess_available_mwh = max(0, bess_soc_mwh[i] - bess_min_mwh)

                if price >= bess_discharge_th and bess_available_mwh > 0:
                    bess_discharge_limit = min(BESS_POWER_MW, bess_available_mwh / period_hours)
                    bess_discharge_mw[i] = bess_discharge_limit
                    bess_soc_mwh[i] -= bess_discharge_mw[i] * period_hours
                    bess_cycles[i] = bess_discharge_mw[i] * period_hours / BESS_CAPACITY_MWH

                storage_discharge_mw[i] = bess_discharge_mw[i]
                grid_export_mw[i] = BASELINE_EXPORT_MW + storage_discharge_mw[i]

            else:
                deficit_mw = BASELINE_EXPORT_MW - gen_mw
                wind_direct_mw[i] = gen_mw

                # LAES primary for wind firming
                if laes_available_mwh > 0:
                    laes_discharge_limit = min(LAES_POWER_MW, laes_available_mwh / period_hours)
                    laes_discharge_mw[i] = min(deficit_mw, laes_discharge_limit)
                    laes_soc_mwh[i] -= laes_discharge_mw[i] * period_hours
                    laes_cycles[i] = laes_discharge_mw[i] * period_hours / LAES_CAPACITY_MWH
                    deficit_mw -= laes_discharge_mw[i]
                    laes_available_mwh = max(0, laes_soc_mwh[i] - laes_min_mwh)

                if deficit_mw > 0 and bess_available_mwh > 0:
                    bess_discharge_limit = min(BESS_POWER_MW, bess_available_mwh / period_hours)
                    bess_for_firming = min(deficit_mw, bess_discharge_limit)
                    bess_discharge_mw[i] += bess_for_firming
                    bess_soc_mwh[i] -= bess_for_firming * period_hours
                    bess_cycles[i] += bess_for_firming * period_hours / BESS_CAPACITY_MWH
                    deficit_mw -= bess_for_firming
                    bess_available_mwh = max(0, bess_soc_mwh[i] - bess_min_mwh)

                shortfall_mw[i] = deficit_mw

                if price >= bess_discharge_th and bess_available_mwh > 0:
                    remaining_bess = min(BESS_POWER_MW - bess_discharge_mw[i],
                                         bess_available_mwh / period_hours)
                    if remaining_bess > 0:
                        bess_discharge_mw[i] += remaining_bess
                        bess_soc_mwh[i] -= remaining_bess * period_hours
                        bess_cycles[i] += remaining_bess * period_hours / BESS_CAPACITY_MWH

                storage_discharge_mw[i] = bess_discharge_mw[i] + laes_discharge_mw[i]
                grid_export_mw[i] = gen_mw + storage_discharge_mw[i]

        # === CASE 3: Price is MEDIUM - Wind firming only ===
        else:
            if gen_mw >= BASELINE_EXPORT_MW:
                wind_direct_mw[i] = BASELINE_EXPORT_MW
                grid_export_mw[i] = BASELINE_EXPORT_MW
                excess_wind = gen_mw - BASELINE_EXPORT_MW

                if bess_headroom_mwh > 0:
                    bess_charge_limit = min(BESS_POWER_MW, bess_headroom_mwh / period_hours)
                    bess_charge_wind_mw[i] = min(excess_wind, bess_charge_limit)
                    bess_soc_mwh[i] += bess_charge_wind_mw[i] * period_hours * BESS_RTE
                    excess_wind -= bess_charge_wind_mw[i]

                if laes_headroom_mwh > 0 and excess_wind > 0:
                    laes_charge_limit = min(LAES_POWER_MW, laes_headroom_mwh / period_hours)
                    laes_charge_wind_mw[i] = min(excess_wind, laes_charge_limit)
                    laes_soc_mwh[i] += laes_charge_wind_mw[i] * period_hours * LAES_RTE
                    excess_wind -= laes_charge_wind_mw[i]

                curtailed_mw[i] = excess_wind

            else:
                deficit_mw = BASELINE_EXPORT_MW - gen_mw
                wind_direct_mw[i] = gen_mw

                if bess_available_mwh > 0:
                    bess_discharge_limit = min(BESS_POWER_MW, bess_available_mwh / period_hours)
                    bess_discharge_mw[i] = min(deficit_mw, bess_discharge_limit)
                    bess_soc_mwh[i] -= bess_discharge_mw[i] * period_hours
                    bess_cycles[i] = bess_discharge_mw[i] * period_hours / BESS_CAPACITY_MWH
                    deficit_mw -= bess_discharge_mw[i]

                if deficit_mw > 0 and laes_available_mwh > 0:
                    laes_discharge_limit = min(LAES_POWER_MW, laes_available_mwh / period_hours)
                    laes_discharge_mw[i] = min(deficit_mw, laes_discharge_limit)
                    laes_soc_mwh[i] -= laes_discharge_mw[i] * period_hours
                    laes_cycles[i] = laes_discharge_mw[i] * period_hours / LAES_CAPACITY_MWH
                    deficit_mw -= laes_discharge_mw[i]

                shortfall_mw[i] = deficit_mw
                storage_discharge_mw[i] = bess_discharge_mw[i] + laes_discharge_mw[i]
                grid_export_mw[i] = gen_mw + storage_discharge_mw[i]

        # Revenue calculations
        wind_cfd_ar6_revenue[i] = wind_direct_mw[i] * CFD_AR6_PRICE * period_hours
        storage_market_revenue[i] = storage_discharge_mw[i] * price * period_hours

    results = pd.DataFrame({
        'timestamp': timestamps,
        'generation_mw': generation_mw,
        'grid_export_mw': grid_export_mw,
        'grid_import_mw': grid_import_mw,
        'market_price': market_prices,
        'bess_soc_mwh': bess_soc_mwh,
        'laes_soc_mwh': laes_soc_mwh,
        'bess_charge_wind_mw': bess_charge_wind_mw,
        'bess_charge_grid_mw': bess_charge_grid_mw,
        'laes_charge_wind_mw': laes_charge_wind_mw,
        'laes_charge_grid_mw': laes_charge_grid_mw,
        'bess_discharge_mw': bess_discharge_mw,
        'laes_discharge_mw': laes_discharge_mw,
        'shortfall_mw': shortfall_mw,
        'curtailed_mw': curtailed_mw,
        'wind_direct_mw': wind_direct_mw,
        'storage_discharge_mw': storage_discharge_mw,
        'wind_cfd_ar6_revenue': wind_cfd_ar6_revenue,
        'storage_market_revenue': storage_market_revenue,
        'grid_charge_cost': grid_charge_cost,
        'bess_cycles': bess_cycles,
        'laes_cycles': laes_cycles,
        'negative_price_charge_mwh': negative_price_charge_mwh,
    })

    return results


def simulate_wind_only_month(timestamps, generation_mw, market_prices):
    """Simulate wind-only operation for comparison"""
    n_periods = len(timestamps)
    period_hours = 0.5

    grid_export_mw = np.minimum(generation_mw, BASELINE_EXPORT_MW)
    shortfall_mw = np.maximum(0, BASELINE_EXPORT_MW - generation_mw)
    curtailed_mw = np.maximum(0, generation_mw - BASELINE_EXPORT_MW)

    cfd_ar6_revenue = grid_export_mw * CFD_AR6_PRICE * period_hours

    results = pd.DataFrame({
        'timestamp': timestamps,
        'generation_mw': generation_mw,
        'grid_export_mw': grid_export_mw,
        'market_price': market_prices,
        'shortfall_mw': shortfall_mw,
        'curtailed_mw': curtailed_mw,
        'cfd_ar6_revenue': cfd_ar6_revenue
    })

    return results


# ============== ANALYSIS FUNCTIONS ==============

def npv_factor(rate, years):
    if rate == 0:
        return years
    return (1 - (1 + rate) ** -years) / rate


def run_strategy_simulation(monthly_data, strategy_key, strategy):
    """Run full simulation for a single strategy"""
    print(f"\n{'=' * 60}")
    print(f"Running Strategy {strategy_key}: {strategy['name']}")
    print(f"  {strategy['description']}")
    print('=' * 60)

    bess_soc_mwh = 0
    laes_soc_mwh = 0

    monthly_summaries = []

    for (year, month), combined_data in sorted(monthly_data.items()):
        generation_mw = (combined_data['capacity_factor_pct'].values / 100) * INSTALLED_CAPACITY_MW

        results = simulate_storage_with_grid_charging(
            combined_data['timestamp'].values,
            generation_mw,
            combined_data['price'].values,
            bess_soc_mwh,
            laes_soc_mwh,
            strategy
        )

        bess_soc_mwh = results['bess_soc_mwh'].iloc[-1]
        laes_soc_mwh = results['laes_soc_mwh'].iloc[-1]

        period_hours = 0.5

        monthly_summaries.append({
            'year': year,
            'month': month,
            'month_name': calendar.month_name[month],
            'total_gen_mwh': generation_mw.sum() * period_hours,
            'avg_cf_pct': combined_data['capacity_factor_pct'].mean(),
            'avg_price': combined_data['price'].mean(),
            'min_price': combined_data['price'].min(),
            'max_price': combined_data['price'].max(),
            'availability_pct': (1 - results['shortfall_mw'].sum() /
                                 (len(results) * BASELINE_EXPORT_MW)) * 100,
            'shortfall_mwh': results['shortfall_mw'].sum() * period_hours,
            'curtailed_mwh': results['curtailed_mw'].sum() * period_hours,
            'wind_cfd_revenue': results['wind_cfd_ar6_revenue'].sum(),
            'storage_market_revenue': results['storage_market_revenue'].sum(),
            'grid_charge_cost': results['grid_charge_cost'].sum(),
            'net_storage_revenue': (results['storage_market_revenue'].sum() -
                                    results['grid_charge_cost'].sum()),
            'bess_discharge_mwh': results['bess_discharge_mw'].sum() * period_hours,
            'laes_discharge_mwh': results['laes_discharge_mw'].sum() * period_hours,
            'bess_charge_wind_mwh': results['bess_charge_wind_mw'].sum() * period_hours,
            'bess_charge_grid_mwh': results['bess_charge_grid_mw'].sum() * period_hours,
            'laes_charge_wind_mwh': results['laes_charge_wind_mw'].sum() * period_hours,
            'laes_charge_grid_mwh': results['laes_charge_grid_mw'].sum() * period_hours,
            'bess_cycles': results['bess_cycles'].sum(),
            'laes_cycles': results['laes_cycles'].sum(),
            'negative_price_charge_mwh': results['negative_price_charge_mwh'].sum(),
            'negative_price_hours': (combined_data['price'] < 0).sum() * 0.5,
        })

    return pd.DataFrame(monthly_summaries)


def calculate_strategy_economics(monthly_df, strategy_name):
    """Calculate economics for a strategy"""
    total_gen = monthly_df['total_gen_mwh'].sum()
    availability = monthly_df['availability_pct'].mean()

    wind_cfd_rev = monthly_df['wind_cfd_revenue'].sum()
    storage_market_rev = monthly_df['storage_market_revenue'].sum()
    grid_charge_cost = monthly_df['grid_charge_cost'].sum()
    net_storage_rev = storage_market_rev - grid_charge_cost

    bess_ancillary = BESS_POWER_MW * BESS_ANNUAL_REVENUE_PER_MW
    laes_ancillary = LAES_POWER_MW * LAES_ANNUAL_REVENUE_PER_MW
    total_ancillary = bess_ancillary + laes_ancillary

    total_revenue = wind_cfd_rev + net_storage_rev + total_ancillary

    bess_discharge = monthly_df['bess_discharge_mwh'].sum()
    laes_discharge = monthly_df['laes_discharge_mwh'].sum()
    bess_charge_grid = monthly_df['bess_charge_grid_mwh'].sum()
    laes_charge_grid = monthly_df['laes_charge_grid_mwh'].sum()
    bess_cycles = monthly_df['bess_cycles'].sum()
    laes_cycles = monthly_df['laes_cycles'].sum()

    neg_price_charge = monthly_df['negative_price_charge_mwh'].sum()
    neg_price_hours = monthly_df['negative_price_hours'].sum()

    wind_opex = WIND_FIXED_OPEX + total_gen * WIND_VAR_OPEX_PER_MWH
    bess_opex = BESS_FIXED_OPEX + bess_discharge * BESS_VAR_OPEX_PER_MWH
    laes_opex = LAES_FIXED_OPEX + laes_discharge * LAES_VAR_OPEX_PER_MWH
    total_opex = wind_opex + bess_opex + laes_opex

    net_income = total_revenue - total_opex
    payback = TOTAL_CAPEX_M * 1e6 / net_income if net_income > 0 else float('inf')

    npv_25 = npv_factor(DISCOUNT_RATE, STORAGE_LIFETIME_YEARS)

    bess_npv_costs = (BESS_CAPEX_M * 1e6 +
                      BESS_AUGMENTATION_COST_M * 1e6 / (1 + DISCOUNT_RATE) ** 10 +
                      BESS_AUGMENTATION_COST_M * 1e6 / (1 + DISCOUNT_RATE) ** 20 +
                      bess_opex * npv_25)
    bess_npv_discharge = bess_discharge * npv_25
    bess_lcos_gross = bess_npv_costs / bess_npv_discharge if bess_npv_discharge > 0 else 0
    bess_lcos_net = (bess_npv_costs - bess_ancillary * npv_25) / bess_npv_discharge if bess_npv_discharge > 0 else 0

    laes_npv_costs = LAES_CAPEX_M * 1e6 + laes_opex * npv_25
    laes_npv_discharge = laes_discharge * npv_25
    laes_lcos_gross = laes_npv_costs / laes_npv_discharge if laes_npv_discharge > 0 else 0
    laes_lcos_net = (laes_npv_costs - laes_ancillary * npv_25) / laes_npv_discharge if laes_npv_discharge > 0 else 0

    return {
        'strategy': strategy_name,
        'availability_pct': availability,
        'total_gen_twh': total_gen / 1e6,
        'wind_cfd_revenue_m': wind_cfd_rev / 1e6,
        'storage_market_revenue_m': storage_market_rev / 1e6,
        'grid_charge_cost_m': grid_charge_cost / 1e6,
        'net_storage_revenue_m': net_storage_rev / 1e6,
        'ancillary_revenue_m': total_ancillary / 1e6,
        'total_revenue_m': total_revenue / 1e6,
        'total_opex_m': total_opex / 1e6,
        'net_income_m': net_income / 1e6,
        'payback_years': payback,
        'bess_discharge_gwh': bess_discharge / 1e3,
        'laes_discharge_gwh': laes_discharge / 1e3,
        'bess_charge_grid_gwh': bess_charge_grid / 1e3,
        'laes_charge_grid_gwh': laes_charge_grid / 1e3,
        'bess_cycles': bess_cycles,
        'laes_cycles': laes_cycles,
        'bess_lcos_gross': bess_lcos_gross,
        'bess_lcos_net': bess_lcos_net,
        'laes_lcos_gross': laes_lcos_gross,
        'laes_lcos_net': laes_lcos_net,
        'neg_price_charge_gwh': neg_price_charge / 1e3,
        'neg_price_hours': neg_price_hours,
    }


# ============== VISUALIZATION ==============

def create_strategy_comparison_charts(strategy_economics, wind_only_economics):
    """Create comprehensive comparison charts"""
    strategies = list(strategy_economics.keys())

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    colors = {'A_Fixed': '#3498db', 'B_Dynamic': '#2ecc71', 'C_Negative': '#9b59b6'}

    # 1. Revenue Comparison
    ax1 = axes[0, 0]
    x = np.arange(len(strategies) + 1)
    width = 0.6

    revenues = [wind_only_economics['total_revenue_m']]
    revenues += [strategy_economics[s]['total_revenue_m'] for s in strategies]

    bar_colors = ['#95a5a6'] + [colors[s] for s in strategies]
    labels = ['Wind Only'] + [strategy_economics[s]['strategy'] for s in strategies]

    bars = ax1.bar(x, revenues, width, color=bar_colors, edgecolor='black')
    ax1.set_ylabel('Annual Revenue (£M)', fontsize=11)
    ax1.set_title('Total Annual Revenue by Strategy', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha='right')
    ax1.set_ylim(0, max(revenues) * 1.15)

    for bar, rev in zip(bars, revenues):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f'£{rev:.0f}M', ha='center', va='bottom', fontweight='bold', fontsize=10)

    # 2. Revenue Breakdown
    ax2 = axes[0, 1]
    x = np.arange(len(strategies))
    width = 0.25

    wind_rev = [strategy_economics[s]['wind_cfd_revenue_m'] for s in strategies]
    storage_rev = [strategy_economics[s]['net_storage_revenue_m'] for s in strategies]
    ancillary_rev = [strategy_economics[s]['ancillary_revenue_m'] for s in strategies]

    ax2.bar(x - width, wind_rev, width, label='Wind CfD', color='#3498db')
    ax2.bar(x, storage_rev, width, label='Net Storage (Market-Cost)', color='#2ecc71')
    ax2.bar(x + width, ancillary_rev, width, label='Ancillary', color='#e74c3c')

    ax2.set_ylabel('Revenue (£M)', fontsize=11)
    ax2.set_title('Revenue Breakdown by Strategy', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([STRATEGIES[s]['name'] for s in strategies])
    ax2.legend(loc='upper right')
    ax2.axhline(y=0, color='black', linewidth=0.5)

    # 3. Payback Period
    ax3 = axes[0, 2]
    paybacks = [strategy_economics[s]['payback_years'] for s in strategies]

    bars = ax3.bar(x, paybacks, width * 2.5, color=[colors[s] for s in strategies], edgecolor='black')
    ax3.axhline(y=wind_only_economics['payback_years'], color='#95a5a6', linestyle='--',
                linewidth=2, label=f'Wind Only: {wind_only_economics["payback_years"]:.1f}y')
    ax3.axhline(y=15, color='red', linestyle=':', linewidth=2, label='15-year threshold')

    ax3.set_ylabel('Payback Period (Years)', fontsize=11)
    ax3.set_title('Payback Period Comparison', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([STRATEGIES[s]['name'] for s in strategies])
    ax3.legend(loc='upper right')
    ax3.set_ylim(0, max(paybacks + [wind_only_economics['payback_years']]) * 1.2)

    for bar, pb in zip(bars, paybacks):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{pb:.1f}y', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # 4. BESS Cycles and Grid Charging
    ax4 = axes[1, 0]
    width = 0.35

    bess_cycles = [strategy_economics[s]['bess_cycles'] for s in strategies]
    bess_grid_gwh = [strategy_economics[s]['bess_charge_grid_gwh'] for s in strategies]

    ax4_twin = ax4.twinx()

    bars1 = ax4.bar(x - width / 2, bess_cycles, width, label='BESS Cycles/Year',
                    color='#e74c3c', edgecolor='black')
    bars2 = ax4_twin.bar(x + width / 2, bess_grid_gwh, width, label='Grid Charge (GWh)',
                         color='#3498db', edgecolor='black')

    ax4.set_ylabel('Annual Cycles', color='#e74c3c', fontsize=11)
    ax4_twin.set_ylabel('Grid Charging (GWh)', color='#3498db', fontsize=11)
    ax4.set_title('BESS Utilization by Strategy', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([STRATEGIES[s]['name'] for s in strategies])

    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # 5. LCOS Comparison
    ax5 = axes[1, 1]
    width = 0.2

    bess_lcos_gross = [strategy_economics[s]['bess_lcos_gross'] for s in strategies]
    bess_lcos_net = [strategy_economics[s]['bess_lcos_net'] for s in strategies]
    laes_lcos_net = [strategy_economics[s]['laes_lcos_net'] for s in strategies]

    ax5.bar(x - width, bess_lcos_gross, width, label='BESS LCOS (Gross)', color='#e74c3c', alpha=0.6)
    ax5.bar(x, bess_lcos_net, width, label='BESS LCOS (Net)', color='#e74c3c')
    ax5.bar(x + width, laes_lcos_net, width, label='LAES LCOS (Net)', color='#9b59b6')

    ax5.axhline(y=CFD_AR6_PRICE, color='orange', linestyle='--', linewidth=2,
                label=f'CfD AR6: £{CFD_AR6_PRICE}/MWh')

    ax5.set_ylabel('LCOS (£/MWh)', fontsize=11)
    ax5.set_title('Levelized Cost of Storage', fontsize=13, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels([STRATEGIES[s]['name'] for s in strategies])
    ax5.legend(loc='upper right', fontsize=9)

    # 6. Negative Price Opportunities
    ax6 = axes[1, 2]
    width = 0.35

    neg_hours = [strategy_economics[s]['neg_price_hours'] for s in strategies]
    neg_charge = [strategy_economics[s]['neg_price_charge_gwh'] for s in strategies]

    ax6_twin = ax6.twinx()

    bars1 = ax6.bar(x - width / 2, neg_hours, width, label='Neg Price Hours Available',
                    color='#f39c12', edgecolor='black')
    bars2 = ax6_twin.bar(x + width / 2, neg_charge, width, label='Energy Charged at Neg Price (GWh)',
                         color='#27ae60', edgecolor='black')

    ax6.set_ylabel('Hours Available', color='#f39c12', fontsize=11)
    ax6_twin.set_ylabel('GWh Charged', color='#27ae60', fontsize=11)
    ax6.set_title('Negative Price Opportunity Capture', fontsize=13, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels([STRATEGIES[s]['name'] for s in strategies])

    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6_twin.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig('strategy_comparison_charts.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: strategy_comparison_charts.png")
    plt.close()


def create_summary_table(strategy_economics, wind_only_economics):
    """Print summary comparison table"""
    print("\n" + "=" * 100)
    print("STRATEGY COMPARISON SUMMARY")
    print("=" * 100)

    print(f"\n{'Metric':<35} {'Wind Only':<15} {'A: Fixed':<15} {'B: Dynamic':<15} {'C: Negative':<15}")
    print("-" * 95)

    print(f"{'Availability (%)':<35} {wind_only_economics['availability_pct']:<15.1f} ", end='')
    for s in ['A_Fixed', 'B_Dynamic', 'C_Negative']:
        print(f"{strategy_economics[s]['availability_pct']:<15.1f} ", end='')
    print()

    print(f"{'Total Revenue (£M)':<35} {wind_only_economics['total_revenue_m']:<15.0f} ", end='')
    for s in ['A_Fixed', 'B_Dynamic', 'C_Negative']:
        print(f"{strategy_economics[s]['total_revenue_m']:<15.0f} ", end='')
    print()

    print(f"{'  - Wind CfD (£M)':<35} {wind_only_economics['wind_cfd_revenue_m']:<15.0f} ", end='')
    for s in ['A_Fixed', 'B_Dynamic', 'C_Negative']:
        print(f"{strategy_economics[s]['wind_cfd_revenue_m']:<15.0f} ", end='')
    print()

    print(f"{'  - Net Storage (£M)':<35} {'N/A':<15} ", end='')
    for s in ['A_Fixed', 'B_Dynamic', 'C_Negative']:
        print(f"{strategy_economics[s]['net_storage_revenue_m']:<15.0f} ", end='')
    print()

    print(f"{'  - Ancillary (£M)':<35} {'N/A':<15} ", end='')
    for s in ['A_Fixed', 'B_Dynamic', 'C_Negative']:
        print(f"{strategy_economics[s]['ancillary_revenue_m']:<15.0f} ", end='')
    print()

    print(f"{'Grid Charge Cost (£M)':<35} {'N/A':<15} ", end='')
    for s in ['A_Fixed', 'B_Dynamic', 'C_Negative']:
        print(f"{strategy_economics[s]['grid_charge_cost_m']:<15.1f} ", end='')
    print()

    print(f"{'Total OPEX (£M)':<35} {wind_only_economics['total_opex_m']:<15.0f} ", end='')
    for s in ['A_Fixed', 'B_Dynamic', 'C_Negative']:
        print(f"{strategy_economics[s]['total_opex_m']:<15.0f} ", end='')
    print()

    print(f"{'Net Income (£M)':<35} {wind_only_economics['net_income_m']:<15.0f} ", end='')
    for s in ['A_Fixed', 'B_Dynamic', 'C_Negative']:
        print(f"{strategy_economics[s]['net_income_m']:<15.0f} ", end='')
    print()

    print(f"{'Payback (years)':<35} {wind_only_economics['payback_years']:<15.1f} ", end='')
    for s in ['A_Fixed', 'B_Dynamic', 'C_Negative']:
        print(f"{strategy_economics[s]['payback_years']:<15.1f} ", end='')
    print()

    print("-" * 95)

    print(f"\n{'BESS Cycles/Year':<35} {'N/A':<15} ", end='')
    for s in ['A_Fixed', 'B_Dynamic', 'C_Negative']:
        print(f"{strategy_economics[s]['bess_cycles']:<15.0f} ", end='')
    print()

    print(f"{'BESS Grid Charge (GWh)':<35} {'N/A':<15} ", end='')
    for s in ['A_Fixed', 'B_Dynamic', 'C_Negative']:
        print(f"{strategy_economics[s]['bess_charge_grid_gwh']:<15.1f} ", end='')
    print()

    print(f"{'BESS LCOS Net (£/MWh)':<35} {'N/A':<15} ", end='')
    for s in ['A_Fixed', 'B_Dynamic', 'C_Negative']:
        print(f"{strategy_economics[s]['bess_lcos_net']:<15.0f} ", end='')
    print()

    print(f"{'LAES Cycles/Year':<35} {'N/A':<15} ", end='')
    for s in ['A_Fixed', 'B_Dynamic', 'C_Negative']:
        print(f"{strategy_economics[s]['laes_cycles']:<15.1f} ", end='')
    print()

    print(f"{'LAES LCOS Net (£/MWh)':<35} {'N/A':<15} ", end='')
    for s in ['A_Fixed', 'B_Dynamic', 'C_Negative']:
        print(f"{strategy_economics[s]['laes_lcos_net']:<15.0f} ", end='')
    print()

    print("-" * 95)

    print(f"\n{'Neg Price Hours (total)':<35} {'N/A':<15} ", end='')
    for s in ['A_Fixed', 'B_Dynamic', 'C_Negative']:
        print(f"{strategy_economics[s]['neg_price_hours']:<15.0f} ", end='')
    print()

    print(f"{'Neg Price Charge (GWh)':<35} {'N/A':<15} ", end='')
    for s in ['A_Fixed', 'B_Dynamic', 'C_Negative']:
        print(f"{strategy_economics[s]['neg_price_charge_gwh']:<15.1f} ", end='')
    print()

    print("\n" + "=" * 100)


# ============== MAIN ==============

def main():
    """Run complete analysis comparing all strategies"""
    print("\n" + "=" * 70)
    print("YORKSHIRE WIND + STORAGE - GRID CHARGING ANALYSIS")
    print("=" * 70)

    monthly_data = fetch_all_monthly_data()

    if not monthly_data:
        print("\n✗ No data fetched! Please check API connection.")
        return None

    print(f"\n✓ Fetched data for {len(monthly_data)} months")

    # Wind-only baseline
    print("\n" + "=" * 60)
    print("Running Wind-Only Baseline...")
    print("=" * 60)

    wind_only_summaries = []
    for (year, month), combined_data in sorted(monthly_data.items()):
        generation_mw = (combined_data['capacity_factor_pct'].values / 100) * INSTALLED_CAPACITY_MW
        results = simulate_wind_only_month(
            combined_data['timestamp'].values,
            generation_mw,
            combined_data['price'].values
        )
        period_hours = 0.5
        wind_only_summaries.append({
            'total_gen_mwh': generation_mw.sum() * period_hours,
            'availability_pct': (1 - results['shortfall_mw'].sum() /
                                 (len(results) * BASELINE_EXPORT_MW)) * 100,
            'cfd_ar6_revenue': results['cfd_ar6_revenue'].sum(),
        })

    wind_only_df = pd.DataFrame(wind_only_summaries)
    wind_only_total_gen = wind_only_df['total_gen_mwh'].sum()
    wind_only_availability = wind_only_df['availability_pct'].mean()
    wind_only_cfd_rev = wind_only_df['cfd_ar6_revenue'].sum()
    wind_only_opex = WIND_FIXED_OPEX + wind_only_total_gen * WIND_VAR_OPEX_PER_MWH
    wind_only_net = wind_only_cfd_rev - wind_only_opex
    wind_only_payback = WIND_CAPEX_M * 1e6 / wind_only_net if wind_only_net > 0 else float('inf')

    wind_only_economics = {
        'availability_pct': wind_only_availability,
        'total_revenue_m': wind_only_cfd_rev / 1e6,
        'wind_cfd_revenue_m': wind_only_cfd_rev / 1e6,
        'total_opex_m': wind_only_opex / 1e6,
        'net_income_m': wind_only_net / 1e6,
        'payback_years': wind_only_payback,
    }

    print(f"  Wind-Only Availability: {wind_only_availability:.1f}%")
    print(f"  Wind-Only Revenue: £{wind_only_cfd_rev / 1e6:.0f}M")
    print(f"  Wind-Only Payback: {wind_only_payback:.1f} years")

    # Run each strategy
    strategy_economics = {}
    strategy_monthly = {}

    for strategy_key, strategy in STRATEGIES.items():
        monthly_df = run_strategy_simulation(monthly_data, strategy_key, strategy)
        economics = calculate_strategy_economics(monthly_df, strategy['name'])

        strategy_economics[strategy_key] = economics
        strategy_monthly[strategy_key] = monthly_df

        print(f"\n  {strategy['name']} Results:")
        print(f"    Availability: {economics['availability_pct']:.1f}%")
        print(f"    Total Revenue: £{economics['total_revenue_m']:.0f}M")
        print(f"    Net Storage Revenue: £{economics['net_storage_revenue_m']:.0f}M")
        print(f"    Grid Charge Cost: £{economics['grid_charge_cost_m']:.1f}M")
        print(f"    Payback: {economics['payback_years']:.1f} years")
        print(f"    BESS Cycles: {economics['bess_cycles']:.0f}/year")
        print(f"    BESS LCOS (net): £{economics['bess_lcos_net']:.0f}/MWh")

    # Create charts and tables
    print("\n" + "=" * 60)
    print("Creating Comparison Charts...")
    print("=" * 60)

    create_strategy_comparison_charts(strategy_economics, wind_only_economics)
    create_summary_table(strategy_economics, wind_only_economics)

    # Save results
    for strategy_key, monthly_df in strategy_monthly.items():
        filename = f'monthly_results_{strategy_key}.csv'
        monthly_df.to_csv(filename, index=False)
        print(f"✓ Saved: {filename}")

    summary_data = [{'Strategy': 'Wind Only', **wind_only_economics}]
    for s in ['A_Fixed', 'B_Dynamic', 'C_Negative']:
        summary_data.append({'Strategy': strategy_economics[s]['strategy'], **strategy_economics[s]})

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('strategy_comparison_summary.csv', index=False)
    print("✓ Saved: strategy_comparison_summary.csv")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  • strategy_comparison_charts.png")
    print("  • strategy_comparison_summary.csv")
    print("  • monthly_results_A_Fixed.csv")
    print("  • monthly_results_B_Dynamic.csv")
    print("  • monthly_results_C_Negative.csv")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    best_payback = min(strategy_economics.items(), key=lambda x: x[1]['payback_years'])
    best_revenue = max(strategy_economics.items(), key=lambda x: x[1]['total_revenue_m'])

    print(f"\n  Shortest Payback: {best_payback[1]['strategy']} ({best_payback[1]['payback_years']:.1f} years)")
    print(f"  Highest Revenue: {best_revenue[1]['strategy']} (£{best_revenue[1]['total_revenue_m']:.0f}M)")

    return strategy_economics, wind_only_economics, strategy_monthly


if __name__ == "__main__":
    results = main()