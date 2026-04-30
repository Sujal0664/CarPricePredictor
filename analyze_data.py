import pandas as pd
import numpy as np

df = pd.read_csv('Car_Dataset_Processed.csv')

# Check for data quality issues
print("=== Data Quality Check ===\n")

# Price distribution
print("Price distribution:")
for threshold in [50, 100, 200, 500, 1000, 5000]:
    count = (df['price(in lakhs)'] > threshold).sum()
    print(f"  > {threshold} lakhs: {count} rows")

print(f"\n  Total rows: {len(df)}")

# Engine(cc) - these should be 67 to ~6000 for cars
print("\nEngine(cc) distribution:")
print(f"  Min: {df['engine(cc)'].min()}")
print(f"  Median: {df['engine(cc)'].median()}")
print(f"  P95: {df['engine(cc)'].quantile(0.95)}")
print(f"  P99: {df['engine(cc)'].quantile(0.99)}")
print(f"  Max: {df['engine(cc)'].max()}")
print(f"  > 10000: {(df['engine(cc)'] > 10000).sum()} rows")

# max_power(bhp) - check if it's same as engine(cc)
print("\nmax_power(bhp) vs engine(cc) comparison:")
print(f"  Identical values: {(df['engine(cc)'] == df['max_power(bhp)']).sum()} out of {len(df)}")

# Mileage
print(f"\nMileage > 100: {(df['mileage(kmpl)'] > 100).sum()} rows")
print(f"Mileage > 50: {(df['mileage(kmpl)'] > 50).sum()} rows")

# Torque 
print(f"\nTorque > 1000: {(df['torque(Nm)'] > 1000).sum()} rows")
print(f"Torque > 500: {(df['torque(Nm)'] > 500).sum()} rows")

# Show some problematic rows
print("\n\nRows with extreme engine values:")
extreme = df[df['engine(cc)'] > 5000][['car_name', 'engine(cc)', 'max_power(bhp)', 'torque(Nm)', 'price(in lakhs)']]
print(extreme.head(10).to_string())

print("\n\nRows with extreme mileage:")
extreme_mile = df[df['mileage(kmpl)'] > 50][['car_name', 'mileage(kmpl)', 'engine(cc)', 'price(in lakhs)']]
print(extreme_mile.head(10).to_string())

print("\n\nRows with extreme torque:")
extreme_torq = df[df['torque(Nm)'] > 500][['car_name', 'torque(Nm)', 'engine(cc)', 'price(in lakhs)']]
print(extreme_torq.head(10).to_string())

# Count clean rows with relaxed thresholds
mask = (
    (df['price(in lakhs)'] <= 200) &
    (df['price(in lakhs)'] > 0) &
    (df['mileage(kmpl)'] <= 50)
)
print(f"\n\nRows after price<=200 and mileage<=50 filter: {mask.sum()}")

# Don't filter engine/power/torque yet - they have duplicated column issues
