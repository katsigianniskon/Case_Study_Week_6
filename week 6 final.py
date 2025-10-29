# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 13:56:15 2025

@author: KonMasterPc
"""

# ============================================================================
# TASK 1: MODEL IMPLEMENTATION FOR PORTFOLIO OPTIMIZATION
# Goal: Predict portfolio returns for Sortino/Sharpe optimization
# ============================================================================

import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import os

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = 'data_week6'
TICKERS = ['INTC', 'NVDA', 'AMD', 'QCOM', 'TXN', 'MU', 'AVGO', 'AMAT', 'ASML', 'TSM']
N_FOLDS = 5
RANDOM_STATE = 42
FORWARD_DAYS = 10  # 10-day forward returns (balance between noise and signal)

print("="*90)
print("🤖 TASK 1: TREE-BASED MODELS FOR PORTFOLIO OPTIMIZATION")
print("="*90)
print(f"📊 Target: Equal-weight portfolio {FORWARD_DAYS}-day returns")
print(f"📊 Models: Random Forest, Gradient Boosting, XGBoost")
print(f"📊 Objective: Returns prediction for Sortino/Sharpe optimization")
print(f"⚠️  Note: In finance, R² > 0.05 is considered good!")
print("="*90)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n📥 Loading data...")

prices_df = pd.read_csv(f'{DATA_DIR}/stock_prices.csv', index_col='Date', parse_dates=True)
returns_df = pd.read_csv(f'{DATA_DIR}/stock_returns.csv', index_col='Date', parse_dates=True)

print(f"✅ Loaded: {returns_df.shape}")

# ============================================================================
# FEATURE ENGINEERING - WEEK 5 + KEY PREDICTORS
# ============================================================================

print("\n🔧 Engineering features (Week 5 + portfolio-focused)...")

features_df = pd.DataFrame(index=returns_df.index)

# --- WEEK 5 FEATURES ---
print("  📍 Week 5 Features: HMact, Herd_t, VRSpike")

# HMact (per-asset)
for ticker in TICKERS:
    features_df[f'{ticker}_HMact'] = returns_df[ticker].abs().rolling(10).sum()

# Herd_t (cross-sectional)
herd_mat = np.sign(returns_df[TICKERS])
features_df['Herd_t'] = herd_mat.mean(axis=1)

# VRSpike (per-asset)
for ticker in TICKERS:
    sigma5 = returns_df[ticker].rolling(5).std()
    sigma20 = returns_df[ticker].rolling(20).std()
    features_df[f'{ticker}_VRSpike'] = sigma5 / (sigma20 + 1e-10)

# --- PORTFOLIO-LEVEL FEATURES (CRITICAL FOR OPTIMIZATION) ---
print("  📍 Portfolio-Level Features")

# Portfolio momentum (multiple horizons)
portfolio_ret = returns_df[TICKERS].mean(axis=1)
features_df['portfolio_momentum_5d'] = portfolio_ret.rolling(5).sum()
features_df['portfolio_momentum_10d'] = portfolio_ret.rolling(10).sum()
features_df['portfolio_momentum_20d'] = portfolio_ret.rolling(20).sum()

# Portfolio volatility (for Sortino calculation later)
features_df['portfolio_volatility_20d'] = portfolio_ret.rolling(20).std()
features_df['portfolio_volatility_60d'] = portfolio_ret.rolling(60).std()

# Downside volatility (CRITICAL for Sortino ratio)
negative_rets = portfolio_ret.apply(lambda x: x if x < 0 else 0)
features_df['portfolio_downside_vol_20d'] = negative_rets.rolling(20).std()

# Cross-sectional dispersion
features_df['cross_sectional_vol'] = returns_df[TICKERS].std(axis=1)
features_df['max_min_spread'] = returns_df[TICKERS].max(axis=1) - returns_df[TICKERS].min(axis=1)

# Correlation (diversification proxy)
features_df['avg_correlation_20d'] = returns_df[TICKERS].rolling(20).corr().groupby(level=0).mean().mean(axis=1)

# --- PER-STOCK AGGREGATES (REDUCED SET) ---
print("  📍 Per-Stock Key Features")

# Average momentum across stocks
for window in [10, 20]:
    momentum_cols = [returns_df[t].rolling(window).sum() for t in TICKERS]
    features_df[f'avg_stock_momentum_{window}d'] = pd.concat(momentum_cols, axis=1).mean(axis=1)

# Average volatility across stocks
for window in [20, 60]:
    vol_cols = [returns_df[t].rolling(window).std() for t in TICKERS]
    features_df[f'avg_stock_volatility_{window}d'] = pd.concat(vol_cols, axis=1).mean(axis=1)

# --- TARGET: Forward portfolio return ---
print(f"  📍 Target: {FORWARD_DAYS}-day forward portfolio return")
features_df['target'] = portfolio_ret.rolling(FORWARD_DAYS).sum().shift(-FORWARD_DAYS)

# --- CLEAN ---
print("\n🧹 Cleaning data...")
features_df = features_df.replace([np.inf, -np.inf], np.nan)
features_df = features_df.dropna()

num_features = len(features_df.columns) - 1
print(f"✅ Features: {num_features}")
print(f"✅ Samples: {len(features_df)}")

# ============================================================================
# TRAIN-TEST SPLIT
# ============================================================================

print("\n📊 Train-test split (80/20)...")

X = features_df.drop('target', axis=1)
y = features_df['target']

split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")

# ============================================================================
# MODELS (BALANCED REGULARIZATION)
# ============================================================================

print("\n🤖 Defining models...")

models = {
    'Random Forest': RandomForestRegressor(
        n_estimators=150,
        max_depth=6,
        min_samples_split=30,
        min_samples_leaf=15,
        max_features='sqrt',
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_split=30,
        random_state=RANDOM_STATE
    ),
    'XGBoost': XGBRegressor(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}

print(f"✅ Defined {len(models)} models")

# ============================================================================
# CROSS-VALIDATION
# ============================================================================

print(f"\n🔄 {N_FOLDS}-fold cross-validation...")

tscv = TimeSeriesSplit(n_splits=N_FOLDS)
cv_results = {name: {'cv_scores': [], 'train_r2': None, 'test_r2': None, 
                      'train_mse': None, 'test_mse': None, 'train_mae': None, 'test_mae': None} 
              for name in models.keys()}

for name, model in models.items():
    print(f"\n  🔹 {name}...")
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model_clone = type(model)(**model.get_params())
        model_clone.fit(X_tr, y_tr)
        
        y_val_pred = model_clone.predict(X_val)
        r2 = r2_score(y_val, y_val_pred)
        fold_scores.append(r2)
        
        print(f"     Fold {fold}: R² = {r2:.4f}")
    
    cv_results[name]['cv_scores'] = fold_scores
    print(f"     CV Mean R²: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    
    # Final model
    print(f"     Training final model...")
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    cv_results[name]['train_r2'] = r2_score(y_train, y_train_pred)
    cv_results[name]['test_r2'] = r2_score(y_test, y_test_pred)
    cv_results[name]['train_mse'] = mean_squared_error(y_train, y_train_pred)
    cv_results[name]['test_mse'] = mean_squared_error(y_test, y_test_pred)
    cv_results[name]['train_mae'] = mean_absolute_error(y_train, y_train_pred)
    cv_results[name]['test_mae'] = mean_absolute_error(y_test, y_test_pred)
    
    print(f"     ✅ Train R²: {cv_results[name]['train_r2']:.4f}, Test R²: {cv_results[name]['test_r2']:.4f}")

# ============================================================================
# PAIRED T-TESTS
# ============================================================================

print("\n📊 Paired t-tests...")

model_names = list(models.keys())
p_values = {}

for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        name1, name2 = model_names[i], model_names[j]
        scores1 = cv_results[name1]['cv_scores']
        scores2 = cv_results[name2]['cv_scores']
        
        t_stat, p_val = stats.ttest_rel(scores1, scores2)
        p_values[f"{name1} vs {name2}"] = p_val
        
        sig = "✓ Significant" if p_val < 0.05 else "✗ Not significant"
        print(f"  • {name1} vs {name2}: p = {p_val:.4f} ({sig})")

# ============================================================================
# RESULTS TABLE
# ============================================================================

print("\n" + "="*90)
print("📊 TASK 1 RESULTS - PORTFOLIO OPTIMIZATION")
print("="*90)

results_data = []
for name in model_names:
    if name == 'Random Forest':
        p_val = '—'
    else:
        p_val = p_values.get(f"Random Forest vs {name}", '—')
        if isinstance(p_val, float):
            p_val = f"{p_val:.4f}"
    
    results_data.append({
        'Model': name,
        'Train R²': f"{cv_results[name]['train_r2']:.4f}",
        'Test R²': f"{cv_results[name]['test_r2']:.4f}",
        'Test MSE': f"{cv_results[name]['test_mse']:.6f}",
        'Test MAE': f"{cv_results[name]['test_mae']:.6f}",
        'p-value vs RF': p_val
    })

results_df = pd.DataFrame(results_data)
print("\n" + results_df.to_string(index=False))

# Interpretation
print("\n📈 INTERPRETATION:")
best_model = max(cv_results.items(), key=lambda x: x[1]['test_r2'])[0]
best_r2 = cv_results[best_model]['test_r2']

if best_r2 > 0.05:
    print(f"✅ Best model ({best_model}) achieves Test R² = {best_r2:.4f}")
    print("✅ This is GOOD for financial prediction! (R² > 0.05)")
    print("✅ Models are ready for portfolio optimization (Sortino/Sharpe)")
elif best_r2 > 0:
    print(f"⚠️  Best model ({best_model}) achieves Test R² = {best_r2:.4f}")
    print("⚠️  Low but positive - models have some predictive power")
    print("✅ Can proceed with portfolio optimization cautiously")
else:
    print("❌ Models show no predictive power on test set")
    print("⚠️  Consider: longer horizons, different features, or ensemble")

print("\n💾 Saving results...")
output_file = f'{DATA_DIR}/task1_results_portfolio.csv'
results_df.to_csv(output_file, index=False)
print(f"✅ Saved: {output_file}")

print("\n" + "="*90)
print("✅ TASK 1 COMPLETE - Ready for Portfolio Optimization!")
print("="*90)

# ============================================================================
# TASK 1: RESULTS VISUALIZATION ONLY
# ============================================================================

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

print("\n📊 Creating visualization...")

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# ============================================================================
# CREATE 4-PANEL FIGURE
# ============================================================================

fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# Define colors and model names
colors = ['#66c2a5', '#fc8d62', '#8da0cb']  # Green, Orange, Blue
model_names = ['Random Forest', 'Gradient Boosting', 'XGBoost']

# ============================================================================
# SUBPLOT 1: Test Set R² (Top Left)
# ============================================================================

ax1 = fig.add_subplot(gs[0, 0])

r2_values = [cv_results[name]['test_r2'] for name in model_names]
bars1 = ax1.bar(model_names, r2_values, color=colors, alpha=0.8, 
                edgecolor='black', linewidth=1.5)

# Add value labels on top
for bar, val in zip(bars1, r2_values):
    height = bar.get_height()
    y_pos = height + 0.001 if height > 0 else height - 0.001
    va = 'bottom' if height > 0 else 'top'
    ax1.text(bar.get_x() + bar.get_width()/2., y_pos,
             f'{val:.4f}', ha='center', va=va, fontsize=10, fontweight='bold')

ax1.set_ylabel('R²', fontsize=12, fontweight='bold')
ax1.set_title('Test Set R²', fontsize=14, fontweight='bold', pad=15)
ax1.set_ylim(min(r2_values) - 0.01, max(r2_values) + 0.005)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_axisbelow(True)

# ============================================================================
# SUBPLOT 2: Test Set MSE (Top Right)
# ============================================================================

ax2 = fig.add_subplot(gs[0, 1])

mse_values = [cv_results[name]['test_mse'] for name in model_names]
bars2 = ax2.bar(model_names, mse_values, color=colors, alpha=0.8,
                edgecolor='black', linewidth=1.5)

# Add value labels
for bar, val in zip(bars2, mse_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.00002,
             f'{val:.6f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_ylabel('MSE', fontsize=12, fontweight='bold')
ax2.set_title('Test Set MSE', fontsize=14, fontweight='bold', pad=15)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_axisbelow(True)

# ============================================================================
# SUBPLOT 3: Predicted vs Actual (Bottom Left) - Random Forest
# ============================================================================

ax3 = fig.add_subplot(gs[1, 0])

# Get Random Forest predictions
rf_model = models['Random Forest']
rf_pred = rf_model.predict(X_test)
rf_r2 = cv_results['Random Forest']['test_r2']

# Scatter plot
ax3.scatter(y_test, rf_pred, alpha=0.5, s=50, color='#66c2a5', 
            edgecolors='black', linewidth=0.5)

# Perfect prediction line
min_val = min(y_test.min(), rf_pred.min())
max_val = max(y_test.max(), rf_pred.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
         label='Perfect Prediction', alpha=0.7)

ax3.set_xlabel('Actual 10-Day Return', fontsize=12, fontweight='bold')
ax3.set_ylabel('Predicted 10-Day Return', fontsize=12, fontweight='bold')
ax3.set_title(f"Random Forest - R²={rf_r2:.4f}", fontsize=14, fontweight='bold', pad=15)
ax3.legend(loc='upper left', fontsize=10)
ax3.grid(alpha=0.3, linestyle='--')
ax3.set_axisbelow(True)

# ============================================================================
# SUBPLOT 4: Top 15 Features - Random Forest (Bottom Right)
# ============================================================================

ax4 = fig.add_subplot(gs[1, 1])

# Get feature importances
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False).head(15)

# Reverse order for horizontal bar plot (highest at top)
feature_importance = feature_importance.iloc[::-1]

# Create horizontal bar plot with proper alignment
y_pos = np.arange(len(feature_importance))
bars4 = ax4.barh(y_pos, 
                 feature_importance['importance'].values,
                 color='#8da0cb', alpha=0.8, edgecolor='black', 
                 linewidth=1, height=0.7, align='center')

# Set y-axis with proper alignment
ax4.set_yticks(y_pos)
ax4.set_yticklabels(feature_importance['feature'].values, fontsize=9)
ax4.set_ylim(-0.5, len(feature_importance) - 0.5)  # Proper bounds

# Labels and title
ax4.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax4.set_title('Top 15 Features - Random Forest', fontsize=14, fontweight='bold', pad=15)

# Grid aligned with bars
ax4.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
ax4.set_axisbelow(True)

# Adjust margins
ax4.margins(y=0.01)

# ============================================================================
# MAIN TITLE & SAVE
# ============================================================================

fig.suptitle('Task 1: Portfolio Return Prediction (10-Day Forward Returns)', 
             fontsize=16, fontweight='bold', y=0.995)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.99])

# Save
output_file = f'{DATA_DIR}/task1_results_visualization.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Saved visualization: {output_file}")

plt.show()

print("="*90)
print("✅ VISUALIZATION COMPLETE!")
print("="*90)


# ============================================================================
# TASK 3: FEATURE IMPORTANCE & SHAP ANALYSIS
# Best Model: Random Forest (from Task 1)
# Requirements: Built-in importance, SHAP values, comparison plots, analysis
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')

print("="*90)
print("📊 TASK 3: FEATURE IMPORTANCE & SHAP ANALYSIS")
print("="*90)

# ============================================================================
# PREREQUISITE: Ensure Task 1 objects exist
# ============================================================================
# Required objects from Task 1:
# - models (dict with trained models)
# - cv_results (dict with results)
# - X_train, X_test, y_test
# - DATA_DIR

# Select best model (Random Forest from Task 1)
best_model_name = 'Random Forest'
best_model = models[best_model_name]

print(f"\n🎯 Selected Model: {best_model_name}")
print(f"   Test R²: {cv_results[best_model_name]['test_r2']:.4f}")
print(f"   Test MSE: {cv_results[best_model_name]['test_mse']:.6f}")

# ============================================================================
# STEP 1: EXTRACT BUILT-IN FEATURE IMPORTANCE
# ============================================================================

print("\n📊 Step 1: Extracting built-in feature importance...")

# Get feature importances from Random Forest
feature_names = X_train.columns.tolist()
importances = best_model.feature_importances_

# Create DataFrame
builtin_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print(f"✅ Extracted {len(builtin_importance)} feature importances")
print("\n📊 Top 10 Features (Built-in Importance):")
print(builtin_importance.head(10).to_string(index=False))

# Save to CSV
builtin_file = f'{DATA_DIR}/task3_builtin_importance.csv'
builtin_importance.to_csv(builtin_file, index=False)
print(f"💾 Saved: {builtin_file}")

# ============================================================================
# STEP 2: COMPUTE SHAP VALUES
# ============================================================================

print("\n📊 Step 2: Computing SHAP values (this may take a moment)...")

# Initialize SHAP explainer for tree-based models
explainer = shap.TreeExplainer(best_model)

# Compute SHAP values for test set
print("   Computing SHAP values for test set...")
shap_values = explainer.shap_values(X_test)

# Get expected value (baseline) - handle both scalar and array
expected_value = explainer.expected_value
if isinstance(expected_value, np.ndarray):
    expected_value = expected_value[0] if len(expected_value) > 0 else expected_value.item()

print(f"✅ SHAP values computed")
print(f"   Shape: {shap_values.shape}")
print(f"   Expected value (baseline): {expected_value:.6f}")

# ============================================================================
# STEP 3: SHAP IMPORTANCE (Mean Absolute SHAP)
# ============================================================================

print("\n📊 Step 3: Calculating SHAP-based importance...")

# Calculate mean absolute SHAP value for each feature
shap_importance_values = np.abs(shap_values).mean(axis=0)

# Create DataFrame
shap_importance = pd.DataFrame({
    'Feature': feature_names,
    'SHAP_Importance': shap_importance_values
}).sort_values('SHAP_Importance', ascending=False)

print(f"✅ SHAP importance calculated")
print("\n📊 Top 10 Features (SHAP Importance):")
print(shap_importance.head(10).to_string(index=False))

# Save to CSV
shap_file = f'{DATA_DIR}/task3_shap_importance.csv'
shap_importance.to_csv(shap_file, index=False)
print(f"💾 Saved: {shap_file}")

# ============================================================================
# STEP 4: CREATE SHAP SUMMARY PLOT
# ============================================================================

print("\n📊 Step 4: Creating SHAP summary plot...")

fig, ax = plt.subplots(figsize=(12, 8))

# Create SHAP summary plot (beeswarm)
shap.summary_plot(
    shap_values, 
    X_test, 
    feature_names=feature_names,
    max_display=20,
    show=False
)

plt.title('SHAP Summary Plot - Feature Impact on Model Predictions', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('SHAP Value (impact on model output)', fontsize=12, fontweight='bold')
plt.tight_layout()

# Save
summary_plot_file = f'{DATA_DIR}/task3_shap_summary_plot.png'
plt.savefig(summary_plot_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Saved: {summary_plot_file}")
plt.close()

# ============================================================================
# STEP 5: COMPARE BUILT-IN VS SHAP IMPORTANCE
# ============================================================================

print("\n📊 Step 5: Creating comparison plots...")

# Merge both importance measures
comparison_df = builtin_importance.merge(
    shap_importance, 
    on='Feature', 
    how='inner'
)

# Get top 20 features by SHAP importance
top_features = comparison_df.nlargest(20, 'SHAP_Importance')

# Create comparison plot
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# PLOT 1: Built-in Feature Importance (Top 20)
ax1 = axes[0]
top_builtin = top_features.sort_values('Importance')

y_pos = np.arange(len(top_builtin))
ax1.barh(y_pos, top_builtin['Importance'].values, 
         color='#66c2a5', alpha=0.8, edgecolor='black', linewidth=1)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(top_builtin['Feature'].values, fontsize=9)
ax1.set_xlabel('Built-in Importance (MDI)', fontsize=12, fontweight='bold')
ax1.set_title('Built-in Feature Importance (Random Forest)', 
              fontsize=13, fontweight='bold', pad=15)
ax1.grid(axis='x', alpha=0.3, linestyle='--')
ax1.set_axisbelow(True)

# PLOT 2: SHAP Importance (Top 20)
ax2 = axes[1]
top_shap = top_features.sort_values('SHAP_Importance')

y_pos = np.arange(len(top_shap))
ax2.barh(y_pos, top_shap['SHAP_Importance'].values,
         color='#fc8d62', alpha=0.8, edgecolor='black', linewidth=1)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(top_shap['Feature'].values, fontsize=9)
ax2.set_xlabel('SHAP Importance (Mean |SHAP|)', fontsize=12, fontweight='bold')
ax2.set_title('SHAP-based Feature Importance', 
              fontsize=13, fontweight='bold', pad=15)
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.set_axisbelow(True)

plt.suptitle('Comparison: Built-in vs SHAP Feature Importance (Top 20)', 
             fontsize=15, fontweight='bold', y=1.00)
plt.tight_layout()

# Save
comparison_plot_file = f'{DATA_DIR}/task3_importance_comparison.png'
plt.savefig(comparison_plot_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Saved: {comparison_plot_file}")
plt.close()

# ============================================================================
# STEP 6: SCATTER PLOT - CORRELATION BETWEEN IMPORTANCES (FIXED FOR REAL)
# ============================================================================

print("\n📊 Step 6: Creating correlation scatter plot...")

fig, ax = plt.subplots(figsize=(10, 8))

# Scatter plot
ax.scatter(comparison_df['Importance'], 
           comparison_df['SHAP_Importance'],
           s=100, alpha=0.6, edgecolors='black', linewidth=1, color='#8da0cb')

# Get actual data ranges
x_max = comparison_df['Importance'].max()
y_max = comparison_df['SHAP_Importance'].max()

# Add diagonal reference line through the data range
# Use the MINIMUM of the two maxes to keep it visible
diag_max = min(x_max, y_max)
ax.plot([0, diag_max], [0, diag_max], 'r--', linewidth=2, alpha=0.5, label='Perfect Agreement')

# Annotate top 5 features
top5_features = comparison_df.nlargest(5, 'SHAP_Importance')
for idx, row in top5_features.iterrows():
    ax.annotate(row['Feature'], 
                (row['Importance'], row['SHAP_Importance']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, alpha=0.9, fontweight='bold')

# Calculate correlation
correlation = comparison_df[['Importance', 'SHAP_Importance']].corr().iloc[0, 1]

# Set axis limits with small margins (NO ASPECT RATIO CONSTRAINT)
ax.set_xlim(-0.002, x_max * 1.1)
ax.set_ylim(-0.0002, y_max * 1.15)

# Labels and title
ax.set_xlabel('Built-in Importance (MDI)', fontsize=12, fontweight='bold')
ax.set_ylabel('SHAP Importance (Mean |SHAP|)', fontsize=12, fontweight='bold')
ax.set_title(f'Built-in vs SHAP Importance Correlation (r = {correlation:.3f})', 
             fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=10, loc='upper left')
ax.grid(alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# NO ax.set_aspect('equal') - this was the problem!

plt.tight_layout()

# Save
scatter_plot_file = f'{DATA_DIR}/task3_importance_correlation.png'
plt.savefig(scatter_plot_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Saved: {scatter_plot_file}")
plt.close()

# ============================================================================
# STEP 7: TOP 5 FEATURES DETAILED ANALYSIS
# ============================================================================

print("\n📊 Step 7: Analyzing Top 5 Features...")

# Get top 5 by SHAP importance
top5 = comparison_df.nlargest(5, 'SHAP_Importance')

print("\n" + "="*90)
print("🏆 TOP 5 MOST IMPORTANT FEATURES")
print("="*90)

for rank, (idx, row) in enumerate(top5.iterrows(), 1):
    print(f"\n{rank}. {row['Feature']}")
    print(f"   Built-in Importance: {row['Importance']:.6f}")
    print(f"   SHAP Importance:     {row['SHAP_Importance']:.6f}")

# Save top 5 to CSV
top5_file = f'{DATA_DIR}/task3_top5_features.csv'
top5[['Feature', 'Importance', 'SHAP_Importance']].to_csv(top5_file, index=False)
print(f"\n💾 Saved: {top5_file}")

# ============================================================================
# STEP 8: SHAP DEPENDENCE PLOTS FOR TOP 5
# ============================================================================

print("\n📊 Step 8: Creating SHAP dependence plots for Top 5 features...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (_, row) in enumerate(top5.iterrows()):
    feature_name = row['Feature']
    feature_idx = feature_names.index(feature_name)
    
    # Create dependence plot
    shap.dependence_plot(
        feature_idx,
        shap_values,
        X_test,
        feature_names=feature_names,
        ax=axes[idx],
        show=False,
        alpha=0.5
    )
    axes[idx].set_title(f'{feature_name}', fontsize=12, fontweight='bold')

# Remove extra subplot
axes[-1].axis('off')

plt.suptitle('SHAP Dependence Plots - Top 5 Features (Non-linear Relationships)', 
             fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout()

# Save
dependence_plot_file = f'{DATA_DIR}/task3_shap_dependence_top5.png'
plt.savefig(dependence_plot_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Saved: {dependence_plot_file}")
plt.close()

# ============================================================================
# STEP 9: SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*90)
print("📈 SUMMARY STATISTICS")
print("="*90)

print(f"\n📊 Feature Importance Correlation:")
print(f"   Pearson r = {correlation:.4f}")
print(f"   Interpretation: {'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.4 else 'Weak'} agreement")

print(f"\n📊 Top 5 Features Account For:")
total_builtin = builtin_importance['Importance'].sum()
total_shap = shap_importance['SHAP_Importance'].sum()
top5_builtin_pct = (top5['Importance'].sum() / total_builtin) * 100
top5_shap_pct = (top5['SHAP_Importance'].sum() / total_shap) * 100

print(f"   Built-in Importance: {top5_builtin_pct:.2f}%")
print(f"   SHAP Importance:     {top5_shap_pct:.2f}%")

# ============================================================================
# FINAL OUTPUT
# ============================================================================

print("\n" + "="*90)
print("✅ TASK 3 COMPLETE!")
print("="*90)
print("\n📁 Generated Files:")
print(f"   1. {builtin_file}")
print(f"   2. {shap_file}")
print(f"   3. {summary_plot_file}")
print(f"   4. {comparison_plot_file}")
print(f"   5. {scatter_plot_file}")
print(f"   6. {top5_file}")
print(f"   7. {dependence_plot_file}")
print("="*90)

