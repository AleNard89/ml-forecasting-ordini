# Forecasting Ordini E-Commerce

Sistema di previsione del volume giornaliero di ordini per un e-commerce, basato su due approcci complementari di Machine Learning.

## Obiettivo

Prevedere il numero di ordini giornalieri partendo da dati transazionali storici, confrontando modelli tree-based (XGBoost, LightGBM) con un modello di decomposizione temporale (Prophet).

## Dataset

**Online Retail Dataset** - [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail)

- ~541.000 transazioni di un e-commerce UK
- Periodo: Dicembre 2010 - Dicembre 2011
- 8 colonne: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

> Il dataset non e incluso nel repository. Scaricarlo dal link sopra e posizionarlo in `online_retail/Online Retail.xlsx`.

## Struttura del Progetto

```
ML_Forecasting/
├── ml_ForecastingOrdini.py      # Pipeline ML completa (XGBoost + LightGBM)
├── Prophet.py                    # Forecasting con Facebook Prophet
├── Forecasting_Ordini.ipynb      # Notebook interattivo con EDA e visualizzazioni
├── requirements.txt              # Dipendenze Python
├── feature_columns.json          # Lista feature usate dal modello
├── RIEPILOGO.md                  # Riepilogo tecnico dettagliato
└── online_retail/
    └── Online Retail.xlsx        # Dataset (da scaricare separatamente)
```

## Pipeline

```
Dati Grezzi → Pulizia → Aggregazione Giornaliera → Feature Engineering → Split Temporale → Modelli → Valutazione
```

### Pulizia Dati
- Rimozione date mancanti e resi (fatture con prefisso `C`)
- Filtro su quantita e prezzi positivi
- Calcolo del revenue per ogni transazione

### Feature Engineering (30 feature)
- **Temporali** (15): componenti calendario + encoding ciclico sin/cos + flag weekend/inizio-fine mese
- **Lag** (6): valori degli ordini a 1, 2, 3, 7, 14, 28 giorni
- **Rolling** (6): media e deviazione standard su finestre di 7, 14, 28 giorni
- **Dataset** (4): quantita totale, revenue, clienti unici, prodotti unici

### Split Temporale

| Set        | Giorni | Periodo                 |
|------------|--------|-------------------------|
| Train      | 243    | Gen 2011 - Ott 2011     |
| Validation | 13     | 1-15 Nov 2011           |
| Test       | 21     | 16 Nov - 9 Dic 2011     |

## Risultati

### XGBoost + LightGBM (target: ordini giornalieri)

| Modello         | MAE   | RMSE  | MAPE   | R²     |
|-----------------|-------|-------|--------|--------|
| Baseline Mean   | 41.93 | 44.08 | -      | -      |
| Baseline Last   | 29.31 | 32.31 | -      | -      |
| Baseline MA(7)  | 16.85 | 20.47 | -      | -      |
| **XGBoost**     | **7.03** | **8.12** | **6.98%** | **0.6437** |
| LightGBM        | 9.94  | 12.20 | 9.59%  | 0.1952 |

**Miglior modello**: XGBoost

**Test Set (XGBoost)**: MAE 9.99 | RMSE 12.98 | MAPE 9.03% | R² 0.7373

### Prophet (target: quantita giornaliera)

- Trend generale: crescita +193.5% nel periodo
- Giorno migliore: Sabato | Giorno peggiore: Domenica
- Mese migliore: Dicembre | Mese peggiore: Luglio

## Installazione

```bash
# Clona il repository
git clone https://github.com/AleNard89/ml-forecasting-ordini.git
cd ml-forecasting-ordini

# Crea virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Installa dipendenze
pip install -r requirements.txt
```

Su macOS, per XGBoost e LightGBM:
```bash
brew install libomp
```

## Esecuzione

```bash
# Pipeline ML (XGBoost + LightGBM)
python ml_ForecastingOrdini.py

# Forecasting con Prophet
python Prophet.py

# Notebook interattivo (consigliato)
jupyter notebook Forecasting_Ordini.ipynb
```

## Tecnologie

- Python 3.10+
- XGBoost, LightGBM (gradient boosting)
- Prophet (time series decomposition)
- pandas, numpy, scikit-learn
- matplotlib, seaborn, plotly

## Licenza

Il dataset Online Retail e distribuito sotto licenza [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
