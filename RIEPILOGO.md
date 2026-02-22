# Forecasting Ordini - Riepilogo Progetto

## Obiettivo
Prevedere il volume giornaliero di ordini di un e-commerce (Online Retail dataset) utilizzando modelli di Machine Learning.

## Dataset
- **File**: `online_retail/Online Retail.xlsx`
- **Fonte**: UCI Machine Learning Repository - Online Retail Dataset
- **Dimensione**: 541,909 righe x 8 colonne
- **Periodo**: 01/12/2010 - 09/12/2011
- **Record utilizzabili dopo pulizia**: ~530,000

## Struttura Progetto

```
ML_Forecasting/
├── ml_ForecastingOrdini.py    # Pipeline ML (XGBoost + LightGBM)
├── Prophet.py                  # Forecasting con Facebook Prophet
├── Forecasting_Ordini.ipynb    # Notebook Jupyter interattivo (completo)
├── requirements.txt            # Dipendenze Python
├── feature_columns.json        # Lista feature utilizzate dai modelli
├── online_retail/
│   └── Online Retail.xlsx      # Dataset
├── Metodologia/
│   └── DOCUMENTO DI SINTESI METODOLOGICA ML 1 1.docx
├── Tesi/
│   └── TESI_Romel.pdf
└── .venv/                      # Virtual environment Python
```

## Pipeline Dati

### 1. Pulizia Dati
- Rimozione date mancanti
- Rimozione resi (InvoiceNo che inizia con 'C')
- Filtro Quantity > 0 e UnitPrice > 0
- Calcolo Revenue = Quantity x UnitPrice

### 2. Aggregazione Giornaliera
| Metrica           | Descrizione                  |
|-------------------|------------------------------|
| Orders            | Numero ordini unici/giorno   |
| TotalQuantity     | Quantita totale/giorno       |
| Revenue           | Fatturato totale/giorno      |
| UniqueCustomers   | Clienti unici/giorno         |
| UniqueProducts    | Prodotti unici/giorno        |

- **Giorni totali**: 305
- **Media ordini/giorno**: 65.4
- **Range ordini**: 11 - 142

### 3. Feature Engineering (30 feature)

**Feature temporali (15)**:
- Year, Month, Day, DayOfWeek, DayOfYear, WeekOfYear, Quarter
- Encoding ciclico: DayOfWeek_sin/cos, Month_sin/cos
- Flag: IsWeekend, IsMonthStart, IsMonthEnd

**Lag features (6)**:
- Orders_lag_1, _2, _3, _7, _14, _28

**Rolling features (6)**:
- Media mobile e deviazione standard su finestre di 7, 14, 28 giorni

**Feature dal dataset (4)**:
- TotalQuantity, Revenue, UniqueCustomers, UniqueProducts

### 4. Split Temporale
| Set        | Giorni | Periodo                       |
|------------|--------|-------------------------------|
| Train      | 243    | 13/01/2011 - 31/10/2011       |
| Validation | 13     | 01/11/2011 - 15/11/2011       |
| Test       | 21     | 16/11/2011 - 09/12/2011       |

## Modelli e Risultati

### Approccio 1: XGBoost + LightGBM (`ml_ForecastingOrdini.py`)

**Risultati su Validation Set:**

| Modello         | MAE    | RMSE   | MAPE   | R2     |
|-----------------|--------|--------|--------|--------|
| Baseline Mean   | 41.93  | 44.08  | -      | -      |
| Baseline Last   | 29.31  | 32.31  | -      | -      |
| Baseline MA(7)  | 16.85  | 20.47  | -      | -      |
| **XGBoost**     | **7.03** | **8.12** | **6.98%** | **0.6437** |
| LightGBM        | 9.94   | 12.20  | 9.59%  | 0.1952 |

**Miglior modello: XGBoost**

**Risultati su Test Set (XGBoost):**
- MAE: 9.99
- RMSE: 12.98
- MAPE: 9.03%
- R2: 0.7373

### Approccio 2: Prophet (`Prophet.py`)

**Pattern rilevati:**
- **Giorno migliore**: Sabato (+7,455 unita vs media)
- **Giorno peggiore**: Domenica (-8,781 unita vs media)
- **Mese migliore**: Dicembre (+6,410 unita vs media)
- **Mese peggiore**: Luglio (-4,093 unita vs media)
- **Trend generale**: Crescita +193.5% nel periodo analizzato

**Forecast 7 giorni**: ~28,399 unita/giorno (media prevista)

## Come Eseguire

```bash
# Attiva il virtual environment
cd ML_Forecasting

# Script Python
.venv/bin/python ml_ForecastingOrdini.py
.venv/bin/python Prophet.py

# Jupyter Notebook (raccomandato)
.venv/bin/jupyter notebook Forecasting_Ordini.ipynb
```

## File Generati dall'Esecuzione
- `plot_1_validation.png` - Predizioni vs Reali (Validation)
- `plot_2_feature_importance.png` - Top 15 Feature Importance
- `plot_3_model_comparison.png` - Confronto RMSE tra modelli
- `plot_4_test_set.png` - Predizioni vs Reali (Test)
- `model_xgboost.pkl` - Modello XGBoost serializzato
- `feature_columns.json` - Lista feature per inferenza

## Dipendenze
```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
optuna>=3.0.0
plotly>=5.14.0
openpyxl>=3.1.0
prophet>=1.1.0
```

**Requisito macOS**: `brew install libomp` (per XGBoost/LightGBM)
