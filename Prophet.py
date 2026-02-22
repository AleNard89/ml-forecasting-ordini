import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import numpy as np

# === CONFIGURAZIONE ===
import os
FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "online_retail", "Online Retail.xlsx")
FORECAST_DAYS = 30

# === 1. CARICAMENTO DATI ===
print("1. Caricamento dataset...")
try:
    df = pd.read_excel(FILE_PATH)
    print(f"   ✓ File caricato: {len(df):,} record")
except FileNotFoundError:
    print(f"   ✗ File non trovato!")
    print(f"   → Verifica il path: {FILE_PATH}")
    exit()

# === 2. PULIZIA DATI ===
print("\n2. Pulizia dati...")
records_iniziali = len(df)
df = df.dropna(subset=['CustomerID', 'Quantity', 'InvoiceDate'])
df = df[df['Quantity'] > 0]
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
records_puliti = len(df)
print(f"   ✓ Rimossi {records_iniziali - records_puliti:,} record invalidi")
print(f"   ✓ Record utilizzabili: {records_puliti:,}")

# === 3. AGGREGAZIONE TEMPORALE ===
print("\n3. Aggregazione giornaliera...")
daily_sales = df.groupby(df['InvoiceDate'].dt.date)['Quantity'].sum().reset_index()
daily_sales.columns = ['ds', 'y']
daily_sales['ds'] = pd.to_datetime(daily_sales['ds'])
print(f"   ✓ Creati {len(daily_sales)} giorni di dati")
print(f"   ✓ Periodo: {daily_sales['ds'].min()} → {daily_sales['ds'].max()}")

# === 4. TRAINING MODELLO PROPHET ===
print("\n4. Training modello Prophet...")
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05
)
model.fit(daily_sales)
print(f"   ✓ Modello addestrato")

# === 4.5 ANALISI PATTERN IMPARATI ===
print("\n" + "="*60)
print("PATTERN IMPARATI DAL MODELLO")
print("="*60)

# Pattern Settimanale
print("\n PATTERN SETTIMANALE (quale giorno vende di più?):")
print("-" * 60)
giorni = ['Lunedì', 'Martedì', 'Mercoledì', 'Giovedì', 'Venerdì', 'Sabato', 'Domenica']
week_dates = pd.date_range('2023-01-02', periods=7)
week_df = pd.DataFrame({'ds': week_dates})
weekly_effect = model.predict_seasonal_components(model.setup_dataframe(week_df))['weekly'].values

for i, (giorno, effetto) in enumerate(zip(giorni, weekly_effect)):
    barra = '█' * int(abs(effetto) / 100)
    segno = '+' if effetto > 0 else ''
    print(f"{giorno:12} {segno}{effetto:>8.0f} unità {barra}")

giorno_migliore = giorni[np.argmax(weekly_effect)]
giorno_peggiore = giorni[np.argmin(weekly_effect)]
print(f"\n   → Giorno MIGLIORE: {giorno_migliore} ({weekly_effect.max():+.0f} unità vs media)")
print(f"   → Giorno PEGGIORE: {giorno_peggiore} ({weekly_effect.min():+.0f} unità vs media)")

# Pattern Annuale
print("\n\n PATTERN ANNUALE (quali mesi vendono di più?):")
print("-" * 60)
mesi = ['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu', 
        'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic']
year_dates = pd.date_range('2023-01-01', periods=365)
year_df = pd.DataFrame({'ds': year_dates})
yearly_effect = model.predict_seasonal_components(model.setup_dataframe(year_df))['yearly'].values

# Media mensile
monthly_effect = []
for i in range(12):
    month_start = i * 30
    month_end = min((i + 1) * 30, len(yearly_effect))
    monthly_effect.append(yearly_effect[month_start:month_end].mean())

for mese, effetto in zip(mesi, monthly_effect):
    barra = '█' * int(abs(effetto) / 200)
    segno = '+' if effetto > 0 else ''
    print(f"{mese:4} {segno}{effetto:>8.0f} unità {barra}")

mese_migliore = mesi[np.argmax(monthly_effect)]
mese_peggiore = mesi[np.argmin(monthly_effect)]
print(f"\n   → Mese MIGLIORE: {mese_migliore} ({max(monthly_effect):+.0f} unità vs media)")
print(f"   → Mese PEGGIORE: {mese_peggiore} ({min(monthly_effect):+.0f} unità vs media)")

# === 5. GENERAZIONE FORECAST ===
print(f"\n\n5. Generazione forecast ({FORECAST_DAYS} giorni)...")
future = model.make_future_dataframe(periods=FORECAST_DAYS)
forecast = model.predict(future)
print(f"   ✓ Forecast completato")

# Trend Generale
print("\n TREND GENERALE:")
print("-" * 60)
trend_start = forecast['trend'].iloc[0]
trend_end = forecast['trend'].iloc[-1]
trend_change = trend_end - trend_start
trend_pct = (trend_change / trend_start) * 100

if trend_change > 0:
    print(f"   ↗ Crescita: +{trend_change:.0f} unità ({trend_pct:+.1f}%)")
    print(f"   → Il business sta CRESCENDO nel periodo analizzato")
else:
    print(f"   ↘ Decrescita: {trend_change:.0f} unità ({trend_pct:.1f}%)")
    print(f"   → Il business sta CALANDO nel periodo analizzato")

# Changepoints (punti di svolta) - CORRETTO
print("\n PUNTI DI SVOLTA (quando cambia il trend?):")
print("-" * 60)
try:
    changepoint_effects = model.params['delta'][0]
    changepoints_dates = model.changepoints
    
    # Trova i 5 cambiamenti più significativi
    top_changes_idx = np.argsort(np.abs(changepoint_effects))[-5:][::-1]
    
    for i in top_changes_idx:
        if i < len(changepoints_dates):
            date = changepoints_dates.iloc[i]
            effect = changepoint_effects[i]
            direction = "↗ Accelerazione" if effect > 0 else "↘ Rallentamento"
            print(f"   {date.strftime('%Y-%m-%d')}: {direction} ({effect:+.0f} unità/giorno)")
except Exception as e:
    print(f"   ⚠ Impossibile estrarre changepoints dettagliati")

print("="*60)

# === 6. VISUALIZZAZIONE ===
print("\n6. Creazione grafici...")
fig1 = model.plot(forecast)
plt.title(f'Forecast Magazzino - Prossimi {FORECAST_DAYS} giorni', fontsize=14, pad=20)
plt.ylabel('Quantità ordinata', fontsize=11)
plt.xlabel('Data', fontsize=11)
plt.tight_layout()

fig2 = model.plot_components(forecast)
plt.tight_layout()
plt.show()

# === 7. REPORT PREVISIONI ===
print("\n" + "="*60)
print("PREVISIONI PROSSIMI 7 GIORNI")
print("="*60)

next_week = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)
next_week['ds'] = next_week['ds'].dt.strftime('%Y-%m-%d')
print(next_week.to_string(index=False))

avg_forecast = next_week['yhat'].mean()
max_forecast = next_week['yhat_upper'].mean()
min_forecast = next_week['yhat_lower'].mean()

print("\n" + "-"*60)
print(f"Media prevista:         {avg_forecast:>8.0f} unità/giorno")
print(f"Scenario pessimistico:  {min_forecast:>8.0f} unità/giorno")
print(f"Scenario ottimistico:   {max_forecast:>8.0f} unità/giorno")
print(f"Safety stock suggerito: {max_forecast * 1.1:>7.0f} unità/giorno (+10%)")
print("="*60)
