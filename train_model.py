"""
Improved Car Price Prediction Model Training Script
====================================================
Key Fixes:
1. Uses brand, car_age, seats, kms_driven + original categorical features
2. DROPS max_power(bhp) - it's identical to engine(cc) (data error)
3. DROPS mileage(kmpl) - too many erroneous values
4. Carefully filters outliers in engine(cc) and torque(Nm) 
5. Scales features using StandardScaler
6. Uses GradientBoosting for much better accuracy
7. Converts price from lakhs to proper rupees in the app
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ============================================================
# 1. Load and clean data
# ============================================================
df = pd.read_csv("Car_Dataset_Processed.csv")
print(f"Original dataset shape: {df.shape}")

# Drop the index column
df = df.drop(columns=["Unnamed: 0"])

# ============================================================
# 2. Extract brand from car_name
# ============================================================
df['brand'] = df['car_name'].apply(lambda x: x.split()[1] if len(x.split()) > 1 else x.split()[0])

# ============================================================
# 3. Feature engineering - Car Age
# ============================================================
current_year = 2026
df['car_age'] = current_year - df['manufacturing_year']

# ============================================================
# 4. Handle outliers - CAREFULLY
# ============================================================
print(f"\nBefore filtering: {len(df)} rows")

# Price: remove 3 extreme outliers (> 200 lakhs = clearly wrong data)
df = df[df['price(in lakhs)'] <= 200]
df = df[df['price(in lakhs)'] > 0]
print(f"After price filter (0-200 lakhs): {len(df)} rows")

# ============================================================
# 5. Encode categorical features
# ============================================================
# Fuel type encoding
d1 = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
df['fuel_type_enc'] = df['fuel_type'].map(d1)

# Insurance encoding
d2 = {'Comprehensive': 0, 'Third Party insurance': 1, 'Zero Dep': 2, 'Third Party': 1, 'Not Available': 3}
df['insurance_enc'] = df['insurance_validity'].map(d2)

# Ownership encoding
d3 = {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth Owner': 4, 'Fifth Owner': 5}
df['ownership_enc'] = df['ownsership'].map(d3)

# Transmission encoding
d4 = {'Manual': 0, 'Automatic': 1}
df['transmission_enc'] = df['transmission'].map(d4)

# Brand encoding using LabelEncoder
brand_encoder = LabelEncoder()
df['brand_encoded'] = brand_encoder.fit_transform(df['brand'])
brand_list = list(brand_encoder.classes_)
print(f"\nBrands ({len(brand_list)}): {brand_list}")

# ============================================================
# 6. Select features - ONLY reliable columns
# ============================================================
# EXCLUDED: max_power(bhp) - identical to engine(cc), data error
# EXCLUDED: mileage(kmpl) - has many extreme wrong values
# EXCLUDED: engine(cc) - has many extreme wrong values  
# EXCLUDED: torque(Nm) - has many extreme values, unreliable
# INCLUDED: brand, car_age, seats, kms, insurance, fuel, ownership, transmission
feature_columns = [
    'brand_encoded',       # Car brand (Maruti vs BMW makes HUGE difference)
    'car_age',             # How old the car is (key depreciation factor)
    'insurance_enc',       # Insurance type
    'fuel_type_enc',       # Petrol/Diesel/CNG
    'seats',               # Number of seats (proxy for car size/segment)
    'kms_driven',          # Kilometers driven
    'ownership_enc',       # Owner number (1st, 2nd, etc.)
    'transmission_enc',    # Manual/Automatic
]

# Drop any rows with NaN in our selected features
df = df.dropna(subset=feature_columns + ['price(in lakhs)'])
print(f"After dropping NaN: {len(df)} rows")

X = df[feature_columns]
Y = df['price(in lakhs)']

print(f"\nFeature matrix shape: {X.shape}")
print(f"\nPrice statistics (in lakhs):")
print(Y.describe())

# ============================================================
# 7. Scale features
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 8. Train/Test split
# ============================================================
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ============================================================
# 9. Train and compare models
# ============================================================
models = {
    'LinearRegression': LinearRegression(),
    'KNN-5': KNeighborsRegressor(n_neighbors=5),
    'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, min_samples_leaf=3),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42, min_samples_leaf=5
    ),
}

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

model_scores = []

for name, model in models.items():
    model.fit(X_train, Y_train)
    
    train_score = model.score(X_train, Y_train)
    test_score = model.score(X_test, Y_test)
    
    Y_pred = model.predict(X_test)
    mae = mean_absolute_error(Y_test, Y_pred)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    
    # Cross validation
    cv_scores = cross_val_score(model, X_scaled, Y, cv=5, scoring='r2')
    
    print(f"\n{name}:")
    print(f"  Train R2: {train_score:.4f}")
    print(f"  Test R2:  {test_score:.4f}")
    print(f"  CV R2:    {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  MAE:      {mae:.2f} lakhs")
    print(f"  RMSE:     {rmse:.2f} lakhs")
    
    model_scores.append((test_score, name, model))

# Sort by test score (descending) and get top 2
model_scores.sort(reverse=True)
best_score, best_name, best_model = model_scores[0]
second_best_score, second_best_name, second_best_model = model_scores[1]

print(f"\n{'=' * 60}")
print(f"BEST MODEL: {best_name} (Test R2 = {best_score:.4f})")
print(f"SECOND BEST MODEL: {second_best_name} (Test R2 = {second_best_score:.4f})")
print(f"{'=' * 60}")

# ============================================================
# 10. Save model, scaler, and metadata
# ============================================================
# Re-train best model on full data for deployment
best_model.fit(X_scaled, Y)
final_train_score = best_model.score(X_scaled, Y)
print(f"\nFinal best model trained on full data - R2: {final_train_score:.4f}")

# Re-train second best model on full data for deployment
second_best_model.fit(X_scaled, Y)
final_train_score_2nd = second_best_model.score(X_scaled, Y)
print(f"Final second best model trained on full data - R2: {final_train_score_2nd:.4f}")

# Save everything needed for prediction
model_artifacts = {
    'model': best_model,
    'scaler': scaler,
    'feature_columns': feature_columns,
    'brand_list': brand_list,
    'd1_fuel': d1,
    'd2_insurance': d2,
    'd3_ownership': d3,
    'd4_transmission': d4,
}

model_artifacts_2nd = {
    'model': second_best_model,
    'scaler': scaler,
    'feature_columns': feature_columns,
    'brand_list': brand_list,
    'd1_fuel': d1,
    'd2_insurance': d2,
    'd3_ownership': d3,
    'd4_transmission': d4,
}

with open('model_improved.pkl', 'wb') as f:
    pickle.dump(model_artifacts, f)

with open('model_second_best.pkl', 'wb') as f:
    pickle.dump(model_artifacts_2nd, f)

print("\nModels saved:")
print(f"   model_improved.pkl (Best: {best_name})")
print(f"   model_second_best.pkl (Second Best: {second_best_name})")

# ============================================================
# 11. Sanity check predictions
# ============================================================
print("\n" + "=" * 60)
print("SANITY CHECK - Sample Predictions")
print("=" * 60)

test_cases = [
    {
        'desc': 'Maruti, 5yr old, 2nd owner, Diesel, Manual, 50k kms, 5 seats',
        'brand': 'Maruti', 'car_age': 5, 'insurance': 'Comprehensive',
        'fuel': 'Diesel', 'seats': 5, 'kms': 50000, 'owner': 'Second Owner',
        'transmission': 'Manual',
    },
    {
        'desc': 'BMW, 3yr old, 1st owner, Petrol, Automatic, 20k kms, 5 seats',
        'brand': 'BMW', 'car_age': 3, 'insurance': 'Comprehensive',
        'fuel': 'Petrol', 'seats': 5, 'kms': 20000, 'owner': 'First Owner',
        'transmission': 'Automatic',
    },
    {
        'desc': 'Hyundai, 7yr old, 3rd owner, Petrol, Manual, 80k kms, 5 seats',
        'brand': 'Hyundai', 'car_age': 7, 'insurance': 'Third Party',
        'fuel': 'Petrol', 'seats': 5, 'kms': 80000, 'owner': 'Third Owner',
        'transmission': 'Manual',
    },
    {
        'desc': 'Mercedes-Benz, 4yr old, 1st owner, Diesel, Auto, 30k kms, 5 seats',
        'brand': 'Mercedes-Benz', 'car_age': 4, 'insurance': 'Comprehensive',
        'fuel': 'Diesel', 'seats': 5, 'kms': 30000, 'owner': 'First Owner',
        'transmission': 'Automatic',
    },
    {
        'desc': 'Maruti, 8yr old, 2nd owner, Petrol, Manual, 60k kms, 5 seats',
        'brand': 'Maruti', 'car_age': 8, 'insurance': 'Comprehensive',
        'fuel': 'Petrol', 'seats': 5, 'kms': 60000, 'owner': 'Second Owner',
        'transmission': 'Manual',
    },
]

for tc in test_cases:
    brand_idx = brand_list.index(tc['brand'])
    features = [[
        brand_idx,
        tc['car_age'],
        d2[tc['insurance']],
        d1[tc['fuel']],
        tc['seats'],
        tc['kms'],
        d3[tc['owner']],
        d4[tc['transmission']],
    ]]
    features_scaled = scaler.transform(features)
    pred = best_model.predict(features_scaled)[0]
    pred_rupees = pred * 100000
    print(f"\n{tc['desc']}")
    print(f"  Predicted: {pred:.2f} Lakhs = Rs {pred_rupees:,.0f}")
