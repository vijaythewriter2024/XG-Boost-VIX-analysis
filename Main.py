import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)

# Explainability
import shap

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# STEP 1: FETCH AND PREPARE DATA
# ==========================================

def fetch_market_data(ticker="SPY", start_date="2018-01-01", end_date="2026-09-01"):
    """
    Fetch historical price data and compute returns.
    Handles different column naming conventions from yfinance.
    """
    print(f"📊 Fetching data for {ticker} from {start_date} to {end_date}...")
    
    # Download data
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    # Fix: Check for correct column name
    # yfinance sometimes returns 'Adj Close' or 'Adj Close' with different casing
    price_col = None
    for col in ['Adj Close', 'Adj_Close', 'adj close', 'Adj Close']:
        if col in df.columns:
            price_col = col
            break
    
    # If none found, try to use 'Close' as fallback
    if price_col is None:
        for col in ['Close', 'close']:
            if col in df.columns:
                price_col = col
                break
    
    if price_col is None:
        raise KeyError(f"Could not find a price column in the data. Available columns: {list(df.columns)}")
    
    print(f"✅ Using column: '{price_col}' for price data")
    
    # Use the found column for calculations
    df['price'] = df[price_col]
    df['returns'] = df['price'].pct_change()
    df['log_returns'] = np.log(df['price'] / df['price'].shift(1))
    
    # Also store High and Low for volatility calculations
    # Handle different column naming for High/Low
    high_col = 'High' if 'High' in df.columns else None
    low_col = 'Low' if 'Low' in df.columns else None
    
    if high_col and low_col:
        df['High'] = df[high_col]
        df['Low'] = df[low_col]
    
    print(f"✅ Data fetched: {len(df)} trading days")
    return df

# ==========================================
# STEP 2: FEATURE ENGINEERING
# ==========================================

def create_features(df):
    """
    Create technical, volatility, momentum, and macroeconomic features.
    """
    print("🔧 Engineering features...")
    
    # Make a copy to avoid modifying original
    data = df.copy()
    
    # Use 'price' column for all calculations
    price = data['price']
    
    # ----- Technical Indicators -----
    # Moving Averages
    for window in [5, 10, 20, 50, 100, 200]:
        data[f'sma_{window}'] = price.rolling(window).mean()
        data[f'ema_{window}'] = price.ewm(span=window, adjust=False).mean()
    
    # Price-to-MA ratios
    for window in [20, 50, 200]:
        if window in [20, 50, 200]:
            data[f'price_to_sma_{window}'] = price / data[f'sma_{window}']
    
    # ----- Volatility Features -----
    # Rolling standard deviation
    for window in [5, 10, 21, 63]:
        data[f'volatility_{window}d'] = data['returns'].rolling(window).std() * np.sqrt(252)
    
    # Parkinson's volatility (if High and Low are available)
    if 'High' in data.columns and 'Low' in data.columns:
        data['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * (np.log(data['High'] / data['Low']) ** 2)
        ).rolling(21).mean() * np.sqrt(252)
    else:
        data['parkinson_vol'] = data['volatility_21d']  # fallback
    
    # ----- Momentum Features -----
    # Price momentum
    for period in [5, 10, 21, 63, 126]:
        data[f'momentum_{period}d'] = price.pct_change(period)
    
    # RSI (Relative Strength Index)
    def compute_rsi(data, window=14):
        delta = data['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    data['rsi_14'] = compute_rsi(data, 14)
    
    # ----- Market Breadth / Risk Indicators -----
    # VIX proxy - approximate using rolling volatility
    data['vix_proxy'] = data['volatility_21d'] * 100  # Scale to VIX-like levels
    
    # Drawdown
    data['cumulative_max'] = price.expanding().max()
    data['drawdown'] = (price - data['cumulative_max']) / data['cumulative_max']
    
    # ----- Target Variable: Next-Day 1% VaR Exceedance -----
    # We want to predict if tomorrow's return will be less than -1% (a "risk event")
    data['target'] = (data['returns'].shift(-1) < -0.01).astype(int)
    
    # Future volatility (10-day forward)
    data['future_vol_10d'] = data['returns'].rolling(10).std().shift(-10) * np.sqrt(252)
    
    # Drop NaN values
    data = data.dropna()
    
    print(f"✅ Features created: {len(data.columns)} columns, {len(data)} rows")
    return data

# ==========================================
# STEP 3: PREPARE FEATURES AND TARGET
# ==========================================

def prepare_model_data(data, feature_cols=None, target_col='target'):
    """
    Prepare X and y for modeling, with train/test split.
    """
    if feature_cols is None:
        # Select all numeric columns except the target and price columns
        exclude_cols = ['price', 'High', 'Low', 'Open', 'Volume', 
                       'returns', 'log_returns', 'target', 'future_vol_10d']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        feature_cols = [col for col in feature_cols if data[col].dtype in ['float64', 'int64']]
    
    print(f"📋 Using {len(feature_cols)} features")
    
    X = data[feature_cols]
    y = data[target_col]
    
    # Train/Test split (chronological - NO SHUFFLE for time series!)
    split_idx = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"📊 Training: {len(X_train)} samples | Testing: {len(X_test)} samples")
    print(f"🎯 Target distribution (train): {y_train.sum()}/{len(y_train)} risk events ({y_train.mean()*100:.2f}%)")
    
    return X_train, X_test, y_train, y_test, feature_cols

# ==========================================
# STEP 4: TRAIN XGBOOST MODEL
# ==========================================

def train_xgboost_model(X_train, y_train, scale_pos_weight=None):
    """
    Train an XGBoost model with hyperparameter tuning.
    """
    print("🤖 Training XGBoost model...")
    
    # Handle class imbalance if needed
    if scale_pos_weight is None:
        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    
    # Base model
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    
    # Hyperparameter grid for randomized search
    param_dist = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5, 7],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0.5, 1.0, 1.5, 2.0]
    }
    
    # Randomized search (faster than grid search)
    random_search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=20,  # Reduced for speed
        cv=3,
        scoring='f1',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    print(f"✅ Best parameters: {random_search.best_params_}")
    print(f"✅ Best F1 Score: {random_search.best_score_:.4f}")
    
    return best_model

# ==========================================
# STEP 5: EVALUATE MODEL
# ==========================================

def evaluate_model(model, X_test, y_test, X_train=None, y_train=None):
    """
    Evaluate model performance with multiple metrics.
    """
    print("\n" + "="*60)
    print("📊 MODEL EVALUATION")
    print("="*60)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n📈 Performance Metrics:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📋 Confusion Matrix:")
    print(f"   TP: {cm[1,1]}  FP: {cm[0,1]}")
    print(f"   FN: {cm[1,0]}  TN: {cm[0,0]}")
    
    # Lead-Hit Ratio (Early Warning Capability)
    y_prob_series = pd.Series(y_prob, index=y_test.index)
    threshold = np.percentile(y_prob, 90)
    
    hits = []
    for i in range(1, min(4, len(y_prob_series))):
        high_prob_mask = y_prob_series.shift(-i) > threshold
        risk_event_mask = y_test == 1
        if risk_event_mask.sum() > 0:
            lead_hit = (high_prob_mask & risk_event_mask).sum() / risk_event_mask.sum()
            hits.append(lead_hit)
    
    if hits:
        print(f"\n🎯 Lead-Hit Ratio (Early Warning):")
        for i, hit in enumerate(hits, 1):
            print(f"   {i}-day lead: {hit:.4f}")
    
    return {'model': model, 'predictions': y_pred, 'probabilities': y_prob}

# ==========================================
# STEP 6: SHAP EXPLAINABILITY
# ==========================================

def shap_explain(model, X_train, X_test, feature_names):
    """
    Explain model predictions using SHAP.
    """
    print("\n" + "="*60)
    print("🔍 SHAP EXPLANATIONS")
    print("="*60)
    
    try:
        # Create explainer
        explainer = shap.TreeExplainer(model)
        
        # Get SHAP values for a subset of test data
        sample_size = min(100, len(X_test))
        X_sample = X_test.iloc[:sample_size]
        shap_values = explainer.shap_values(X_sample)
        
        # Summary plot (top features)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.title("SHAP Summary Plot - Top Risk Drivers")
        plt.tight_layout()
        plt.savefig("shap_summary.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Bar plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
        plt.title("SHAP Bar Plot - Feature Importance")
        plt.tight_layout()
        plt.savefig("shap_bar.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Return top features
        mean_shap = np.abs(shap_values).mean(axis=0)
        top_features = sorted(zip(feature_names, mean_shap), key=lambda x: x[1], reverse=True)[:10]
        
        print("\n📌 Top 10 Risk Drivers (by SHAP importance):")
        for feat, importance in top_features:
            print(f"   {feat}: {importance:.4f}")
        
        return shap_values
        
    except Exception as e:
        print(f"⚠️ SHAP explanation failed: {e}")
        return None

# ==========================================
# STEP 7: MAIN EXECUTION
# ==========================================

def main():
    """
    Main execution pipeline.
    """
    print("="*60)
    print("🚀 XGBOOST MARKET RISK FORECASTER")
    print("="*60)
    
    # Step 1: Fetch data
    df = fetch_market_data(ticker="SPY", start_date="2018-01-01", end_date="2026-09-01")
    
    # Step 2: Feature engineering
    data = create_features(df)
    
    # Step 3: Prepare for modeling
    X_train, X_test, y_train, y_test, feature_cols = prepare_model_data(data)
    
    # Step 4: Train model
    model = train_xgboost_model(X_train, y_train)
    
    # Step 5: Evaluate model
    results = evaluate_model(model, X_test, y_test, X_train, y_train)
    
    # Step 6: SHAP explainability
    shap_explain(model, X_train, X_test, feature_cols)
    
    print("\n" + "="*60)
    print("✅ Risk forecasting pipeline complete!")
    print("="*60)

if __name__ == "__main__":
    main()
