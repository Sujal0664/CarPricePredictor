import streamlit as st
import pickle
import numpy as np

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered",
)

# ============================================================
# Custom CSS - Modern Light Theme
# ============================================================
st.markdown("""
<style>
    /* Main container - Light gradient background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8f0f7 50%, #f0f4f8 100%);
    }
    
    /* Title styling - Bold orange gradient */
    .main-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(90deg, #ff6b35 0%, #f7931e 50%, #ff6b35 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Subtitle styling */
    .sub-title {
        text-align: center;
        color: #4a5568;
        font-size: 1.05rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Result card - Bright white with orange accent */
    .result-card {
        background: linear-gradient(135deg, #ffffff 0%, #fafbfc 100%);
        border: 2px solid #ff6b35;
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: 0 10px 40px rgba(255, 107, 53, 0.15);
    }
    
    /* Result label */
    .result-label {
        color: #718096;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.8rem;
        font-weight: 700;
    }
    
    /* Result price - Vibrant orange */
    .result-price {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(90deg, #ff6b35 0%, #f7931e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }
    
    /* Result lakhs */
    .result-lakhs {
        color: #2d3748;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 600;
    }
    
    /* Info box - Teal accent */
    .info-box {
        background: linear-gradient(135deg, #e0f7fa 0%, #e1f5ff 100%);
        border: 2px solid #0097a7;
        border-radius: 15px;
        padding: 1.3rem 1.5rem;
        margin-top: 1rem;
        color: #00695c;
        font-size: 0.9rem;
        line-height: 1.8;
        font-weight: 500;
    }
    
    /* Input and selectbox styling */
    .stSelectbox > div > div {
        background-color: #ffffff;
        border-color: #cbd5e0;
        border-radius: 10px;
    }
    
    .stNumberInput > div > div > input {
        background-color: #ffffff;
        border-color: #cbd5e0;
        border-radius: 10px;
    }
    
    /* Button - Vibrant orange */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #ff6b35 0%, #f7931e 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.9rem 2rem;
        font-size: 1.15rem;
        font-weight: 700;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 107, 53, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(255, 107, 53, 0.4);
    }

    /* Divider */
    .section-divider {
        border-top: 2px solid #cbd5e0;
        margin: 1.5rem 0;
    }
    
    /* Label styling */
    .stSelectbox > label, .stNumberInput > label {
        color: #2d3748;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Load improved model (using second best model)
# ============================================================
@st.cache_resource
def load_model():
    with open('model_second_best.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    return artifacts

artifacts = load_model()
model = artifacts['model']
scaler = artifacts['scaler']
brand_list = artifacts['brand_list']
d1 = artifacts['d1_fuel']
d2 = artifacts['d2_insurance']
d3 = artifacts['d3_ownership']
d4 = artifacts['d4_transmission']

# ============================================================
# App Header
# ============================================================
st.markdown('<div class="main-title">🚗 Car Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Get an accurate estimate of your used car\'s value</div>', unsafe_allow_html=True)

# ============================================================
# Input Form (Interchanged Layout)
# ============================================================
col1, col2 = st.columns(2)

with col1:
    manufacturing_year = st.number_input(
        'Manufacturing Year', 
        min_value=2005, max_value=2026, value=2020, step=1
    )
    insurance = st.selectbox('Insurance Type', ['Comprehensive', 'Third Party insurance', 'Zero Dep', 'Third Party', 'Not Available'])
    transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
    kms_driven = st.number_input(
        'Kilometers Driven', 
        min_value=0, max_value=500000, value=30000, step=1000
    )

with col2:
    brand = st.selectbox('Car Brand', sorted(brand_list), index=sorted(brand_list).index('Maruti'))
    fuel = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG'])
    owner = st.selectbox('Ownership', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth Owner', 'Fifth Owner'])
    seats = st.selectbox('Number of Seats', [4, 5, 6, 7, 8], index=1)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ============================================================
# Prediction
# ============================================================
if st.button('🔍 Predict Price'):
    # Calculate car age
    current_year = 2026
    car_age = current_year - int(manufacturing_year)
    
    # Encode brand
    brand_idx = brand_list.index(brand)
    
    # Build feature vector (matching training order)
    features = [[
        brand_idx,           # brand_encoded
        car_age,             # car_age
        d2[insurance],       # insurance_enc
        d1[fuel],            # fuel_type_enc
        int(seats),          # seats
        float(kms_driven),   # kms_driven
        d3[owner],           # ownership_enc
        d4[transmission],    # transmission_enc
    ]]
    
    # Scale and predict
    features_scaled = scaler.transform(features)
    prediction_lakhs = model.predict(features_scaled)[0]
    
    # Ensure prediction is not negative
    prediction_lakhs = max(prediction_lakhs, 0.5)
    
    # Convert to rupees
    prediction_rupees = prediction_lakhs * 100000
    
    # Format the output
    if prediction_rupees >= 10000000:  # >= 1 crore
        crores = prediction_rupees / 10000000
        formatted_price = f"₹{crores:.2f} Crore"
    else:
        formatted_price = f"₹{prediction_lakhs:.2f} Lakhs"
    
    # Display result
    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Estimated Market Value</div>
        <div class="result-price">{formatted_price}</div>
        <div class="result-lakhs">₹{prediction_rupees:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Summary info
    st.markdown(f"""
    <div class="info-box">
        <strong>Prediction Summary:</strong><br>
        📌 <strong>{brand}</strong> &bull; {manufacturing_year} model ({car_age} years old)<br>
        ⛽ {fuel} &bull; {transmission} &bull; {owner}<br>
        📏 {kms_driven:,} km driven &bull; {seats} seater &bull; {insurance}
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# Footer
# ============================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #555577; font-size: 0.8rem; padding: 1rem 0;">
    <strong>Model Info:</strong> Advanced ML Model with 8 features including brand, car age, and ownership<br>
    Trained on 1496 cars &bull; Price shown in Indian Rupees
</div>
""", unsafe_allow_html=True)
