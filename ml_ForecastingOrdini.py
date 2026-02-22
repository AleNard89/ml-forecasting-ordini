"""
Forecasting Ordini - Sistema ML Semplice
=========================================
Replica del notebook Jupyter in formato Python script.

Obiettivo: Prevedere il volume giornaliero di ordini usando XGBoost e LightGBM
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Models
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================
# CONFIGURAZIONE
# ============================================

# 📁 Modifica questo path con il tuo file
import os
FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "online_retail", "Online Retail.xlsx")

# Date per split train/validation/test
TRAIN_END_DATE = '2011-10-31'
VAL_END_DATE = '2011-11-15'


# ============================================
# FASE 1: CARICAMENTO DATI
# ============================================

print("\n" + "="*60)
print("FASE 1: CARICAMENTO DATI")
print("="*60)

# Carica file Excel
df = pd.read_excel(FILE_PATH, engine='openpyxl')
print(f"✓ Dati caricati: {df.shape[0]:,} righe × {df.shape[1]} colonne")

# Converti date
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
print(f"✓ Periodo dati: {df['InvoiceDate'].min()} - {df['InvoiceDate'].max()}")


# ============================================
# FASE 2: PULIZIA DATI
# ============================================

print("\n" + "="*60)
print("FASE 2: PULIZIA DATI")
print("="*60)

print(f"Righe iniziali: {len(df):,}")

# Rimuovi date mancanti
df = df.dropna(subset=['InvoiceDate'])
print(f"✓ Dopo rimozione date NaN: {len(df):,}")

# Rimuovi resi (InvoiceNo che inizia con 'C')
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
print(f"✓ Dopo rimozione resi: {len(df):,}")

# Rimuovi quantità negative o zero
df = df[df['Quantity'] > 0]
print(f"✓ Dopo filtro Quantity > 0: {len(df):,}")

# Rimuovi prezzi negativi o zero
df = df[df['UnitPrice'] > 0]
print(f"✓ Dopo filtro UnitPrice > 0: {len(df):,}")

# Calcola Revenue
df['Revenue'] = df['Quantity'] * df['UnitPrice']

# Estrai solo la data (senza ora)
df['Date'] = df['InvoiceDate'].dt.date


# ============================================
# FASE 3: AGGREGAZIONE GIORNALIERA
# ============================================

print("\n" + "="*60)
print("FASE 3: AGGREGAZIONE GIORNALIERA")
print("="*60)

# Aggrega per giorno
daily = df.groupby('Date').agg({
    'InvoiceNo': 'nunique',      # Numero ordini
    'Quantity': 'sum',            # Quantità totale
    'Revenue': 'sum',             # Revenue totale
    'CustomerID': 'nunique',      # Clienti unici
    'StockCode': 'nunique'        # Prodotti unici
}).reset_index()

# Rinomina colonne
daily.columns = ['Date', 'Orders', 'TotalQuantity', 'Revenue', 
                 'UniqueCustomers', 'UniqueProducts']

# Converti Date in datetime
daily['Date'] = pd.to_datetime(daily['Date'])

# Ordina per data
daily = daily.sort_values('Date').reset_index(drop=True)

print(f"✓ Giorni totali: {len(daily)}")
print(f"✓ Media ordini/giorno: {daily['Orders'].mean():.1f}")
print(f"✓ Min ordini: {daily['Orders'].min()}")
print(f"✓ Max ordini: {daily['Orders'].max()}")


# ============================================
# FASE 4: FEATURE ENGINEERING
# ============================================

print("\n" + "="*60)
print("FASE 4: FEATURE ENGINEERING")
print("="*60)

# Feature temporali base
daily['Year'] = daily['Date'].dt.year
daily['Month'] = daily['Date'].dt.month
daily['Day'] = daily['Date'].dt.day
daily['DayOfWeek'] = daily['Date'].dt.dayofweek  # 0=Lunedì
daily['DayOfYear'] = daily['Date'].dt.dayofyear
daily['WeekOfYear'] = daily['Date'].dt.isocalendar().week
daily['Quarter'] = daily['Date'].dt.quarter

# Feature ciclici (per catturare stagionalità)
daily['DayOfWeek_sin'] = np.sin(2 * np.pi * daily['DayOfWeek'] / 7)
daily['DayOfWeek_cos'] = np.cos(2 * np.pi * daily['DayOfWeek'] / 7)
daily['Month_sin'] = np.sin(2 * np.pi * daily['Month'] / 12)
daily['Month_cos'] = np.cos(2 * np.pi * daily['Month'] / 12)

# Feature weekend
daily['IsWeekend'] = (daily['DayOfWeek'] >= 5).astype(int)

# Feature inizio/fine mese
daily['IsMonthStart'] = (daily['Day'] <= 5).astype(int)
daily['IsMonthEnd'] = (daily['Day'] >= 25).astype(int)

print(f"✓ Feature temporali create: 15")

# Lag features (valori passati)
lags = [1, 2, 3, 7, 14, 28]
for lag in lags:
    daily[f'Orders_lag_{lag}'] = daily['Orders'].shift(lag)

print(f"✓ Lag features create: {len(lags)}")

# Rolling features (medie mobili)
windows = [7, 14, 28]
for window in windows:
    # Media mobile
    daily[f'Orders_rolling_mean_{window}'] = daily['Orders'].shift(1).rolling(window=window).mean()
    # Deviazione standard mobile
    daily[f'Orders_rolling_std_{window}'] = daily['Orders'].shift(1).rolling(window=window).std()

print(f"✓ Rolling features create: {len(windows) * 2}")

# Rimuovi righe con NaN (create dai lag e rolling)
daily_clean = daily.dropna().reset_index(drop=True)
print(f"✓ Righe finali (dopo pulizia NaN): {len(daily_clean)}")
print(f"✓ Feature totali: {len(daily_clean.columns) - 2}")  # -2 per Date e Orders


# ============================================
# FASE 5: SPLIT TRAIN/VAL/TEST
# ============================================

print("\n" + "="*60)
print("FASE 5: SPLIT TRAIN/VALIDATION/TEST")
print("="*60)

# Converti date per confronto
train_end = pd.to_datetime(TRAIN_END_DATE)
val_end = pd.to_datetime(VAL_END_DATE)

# Split
train = daily_clean[daily_clean['Date'] <= train_end].copy()
val = daily_clean[(daily_clean['Date'] > train_end) & (daily_clean['Date'] <= val_end)].copy()
test = daily_clean[daily_clean['Date'] > val_end].copy()

print(f"Train set: {len(train)} giorni ({train['Date'].min().date()} - {train['Date'].max().date()})")
print(f"Val set:   {len(val)} giorni ({val['Date'].min().date()} - {val['Date'].max().date()})")
print(f"Test set:  {len(test)} giorni ({test['Date'].min().date()} - {test['Date'].max().date()})")

# Prepara X (features) e y (target)
exclude_cols = ['Date', 'Orders']
feature_cols = [col for col in daily_clean.columns if col not in exclude_cols]

X_train = train[feature_cols]
y_train = train['Orders']

X_val = val[feature_cols]
y_val = val['Orders']

X_test = test[feature_cols]
y_test = test['Orders']

print(f"\n✓ Features utilizzate: {len(feature_cols)}")


# ============================================
# FASE 6: BASELINE MODELS (per confronto)
# ============================================

print("\n" + "="*60)
print("FASE 6: BASELINE MODELS")
print("="*60)

# Baseline 1: Media del training set
baseline_mean = train['Orders'].mean()
preds_mean = np.full(len(val), baseline_mean)
mae_mean = mean_absolute_error(y_val, preds_mean)
rmse_mean = np.sqrt(mean_squared_error(y_val, preds_mean))

print(f"Baseline Mean:")
print(f"  MAE:  {mae_mean:.2f}")
print(f"  RMSE: {rmse_mean:.2f}")

# Baseline 2: Ultimo valore
baseline_last = train['Orders'].iloc[-1]
preds_last = np.full(len(val), baseline_last)
mae_last = mean_absolute_error(y_val, preds_last)
rmse_last = np.sqrt(mean_squared_error(y_val, preds_last))

print(f"\nBaseline Last Value:")
print(f"  MAE:  {mae_last:.2f}")
print(f"  RMSE: {rmse_last:.2f}")

# Baseline 3: Media mobile 7 giorni
baseline_ma7 = train['Orders'].tail(7).mean()
preds_ma7 = np.full(len(val), baseline_ma7)
mae_ma7 = mean_absolute_error(y_val, preds_ma7)
rmse_ma7 = np.sqrt(mean_squared_error(y_val, preds_ma7))

print(f"\nBaseline Moving Avg (7 days):")
print(f"  MAE:  {mae_ma7:.2f}")
print(f"  RMSE: {rmse_ma7:.2f}")


# ============================================
# FASE 7: XGBOOST MODEL
# ============================================

print("\n" + "="*60)
print("FASE 7: XGBOOST MODEL")
print("="*60)

# Parametri XGBoost
xgb_params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

print("Training XGBoost...")
model_xgb = xgb.XGBRegressor(**xgb_params)
model_xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# Predizioni
preds_xgb = model_xgb.predict(X_val)

# Metriche
mae_xgb = mean_absolute_error(y_val, preds_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_val, preds_xgb))
mape_xgb = np.mean(np.abs((y_val - preds_xgb) / y_val)) * 100
r2_xgb = r2_score(y_val, preds_xgb)

print(f"\n✓ XGBoost Results:")
print(f"  MAE:  {mae_xgb:.2f}")
print(f"  RMSE: {rmse_xgb:.2f}")
print(f"  MAPE: {mape_xgb:.2f}%")
print(f"  R²:   {r2_xgb:.4f}")


# ============================================
# FASE 8: LIGHTGBM MODEL
# ============================================

print("\n" + "="*60)
print("FASE 8: LIGHTGBM MODEL")
print("="*60)

# Parametri LightGBM
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbose': -1
}

print("Training LightGBM...")
model_lgb = lgb.LGBMRegressor(**lgb_params)
model_lgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
)

# Predizioni
preds_lgb = model_lgb.predict(X_val)

# Metriche
mae_lgb = mean_absolute_error(y_val, preds_lgb)
rmse_lgb = np.sqrt(mean_squared_error(y_val, preds_lgb))
mape_lgb = np.mean(np.abs((y_val - preds_lgb) / y_val)) * 100
r2_lgb = r2_score(y_val, preds_lgb)

print(f"\n✓ LightGBM Results:")
print(f"  MAE:  {mae_lgb:.2f}")
print(f"  RMSE: {rmse_lgb:.2f}")
print(f"  MAPE: {mape_lgb:.2f}%")
print(f"  R²:   {r2_lgb:.4f}")


# ============================================
# FASE 9: CONFRONTO MODELLI
# ============================================

print("\n" + "="*60)
print("FASE 9: CONFRONTO FINALE")
print("="*60)

# Crea tabella confronto
results = pd.DataFrame({
    'Model': ['Baseline Mean', 'Baseline Last', 'Baseline MA(7)', 'XGBoost', 'LightGBM'],
    'MAE': [mae_mean, mae_last, mae_ma7, mae_xgb, mae_lgb],
    'RMSE': [rmse_mean, rmse_last, rmse_ma7, rmse_xgb, rmse_lgb],
    'MAPE': ['-', '-', '-', f'{mape_xgb:.2f}%', f'{mape_lgb:.2f}%'],
    'R²': ['-', '-', '-', f'{r2_xgb:.4f}', f'{r2_lgb:.4f}']
})

print("\n" + results.to_string(index=False))

# Best model
best_rmse = min(rmse_xgb, rmse_lgb)
best_model_name = 'XGBoost' if rmse_xgb < rmse_lgb else 'LightGBM'
best_model = model_xgb if rmse_xgb < rmse_lgb else model_lgb
best_preds = preds_xgb if rmse_xgb < rmse_lgb else preds_lgb

print(f"\n🏆 BEST MODEL: {best_model_name} (RMSE: {best_rmse:.2f})")


# ============================================
# FASE 10: VISUALIZZAZIONI
# ============================================

print("\n" + "="*60)
print("FASE 10: VISUALIZZAZIONI")
print("="*60)

# Plot 1: Predizioni vs Reali (Validation)
plt.figure(figsize=(14, 6))
plt.plot(val['Date'], y_val, label='Valori Reali', marker='o', linewidth=2)
plt.plot(val['Date'], best_preds, label=f'Predizioni {best_model_name}', marker='x', linewidth=2, linestyle='--')
plt.title(f'Validation Set - Predizioni vs Reali ({best_model_name})', fontsize=14, fontweight='bold')
plt.xlabel('Data')
plt.ylabel('Numero Ordini')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plot_1_validation.png', dpi=150)
print("✓ Salvato: plot_1_validation.png")

# Plot 2: Feature Importance
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(15), importances[indices])
    plt.yticks(range(15), [feature_cols[i] for i in indices])
    plt.xlabel('Importanza')
    plt.title(f'Top 15 Feature più Importanti ({best_model_name})', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('plot_2_feature_importance.png', dpi=150)
    print("✓ Salvato: plot_2_feature_importance.png")

# Plot 3: Confronto RMSE tra modelli
plt.figure(figsize=(10, 6))
models = ['Mean', 'Last', 'MA(7)', 'XGBoost', 'LightGBM']
rmses = [rmse_mean, rmse_last, rmse_ma7, rmse_xgb, rmse_lgb]
colors = ['lightcoral' if rmse > rmse_lgb else 'lightgreen' for rmse in rmses]
plt.bar(models, rmses, color=colors, edgecolor='black')
plt.ylabel('RMSE')
plt.title('Confronto RMSE tra Modelli', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('plot_3_model_comparison.png', dpi=150)
print("✓ Salvato: plot_3_model_comparison.png")


# ============================================
# FASE 11: VALUTAZIONE SU TEST SET
# ============================================

print("\n" + "="*60)
print("FASE 11: VALUTAZIONE FINALE SU TEST SET")
print("="*60)

# Predizioni su test set
preds_test = best_model.predict(X_test)

# Metriche test
mae_test = mean_absolute_error(y_test, preds_test)
rmse_test = np.sqrt(mean_squared_error(y_test, preds_test))
mape_test = np.mean(np.abs((y_test - preds_test) / y_test)) * 100
r2_test = r2_score(y_test, preds_test)

print(f"\n✓ {best_model_name} - Test Set Results:")
print(f"  MAE:  {mae_test:.2f}")
print(f"  RMSE: {rmse_test:.2f}")
print(f"  MAPE: {mape_test:.2f}%")
print(f"  R²:   {r2_test:.4f}")

# Plot test set
plt.figure(figsize=(14, 6))
plt.plot(test['Date'], y_test, label='Valori Reali', marker='o', linewidth=2)
plt.plot(test['Date'], preds_test, label=f'Predizioni {best_model_name}', marker='x', linewidth=2, linestyle='--')
plt.title(f'Test Set - Predizioni vs Reali ({best_model_name})', fontsize=14, fontweight='bold')
plt.xlabel('Data')
plt.ylabel('Numero Ordini')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plot_4_test_set.png', dpi=150)
print("\n✓ Salvato: plot_4_test_set.png")


# ============================================
# FASE 12: SALVATAGGIO MODELLO
# ============================================

print("\n" + "="*60)
print("FASE 12: SALVATAGGIO MODELLO")
print("="*60)

# Salva modello migliore
import joblib
model_filename = f'model_{best_model_name.lower()}.pkl'
joblib.dump(best_model, model_filename)
print(f"✓ Modello salvato: {model_filename}")

# Salva anche le feature columns
import json
with open('feature_columns.json', 'w') as f:
    json.dump(feature_cols, f)
print(f"✓ Feature columns salvate: feature_columns.json")


# ============================================
# RIEPILOGO FINALE
# ============================================

print("\n" + "="*60)
print("✅ PROCESSO COMPLETATO!")
print("="*60)
print(f"\n📊 Risultati finali:")
print(f"   Best Model: {best_model_name}")
print(f"   Validation RMSE: {best_rmse:.2f}")
print(f"   Test RMSE: {rmse_test:.2f}")
print(f"   Test R²: {r2_test:.4f}")
print(f"\n📁 File generati:")
print(f"   - plot_1_validation.png")
print(f"   - plot_2_feature_importance.png")
print(f"   - plot_3_model_comparison.png")
print(f"   - plot_4_test_set.png")
print(f"   - {model_filename}")
print(f"   - feature_columns.json")
print(f"\n🎯 Il tuo modello è pronto per l'uso!")
