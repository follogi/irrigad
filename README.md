# AI - Irrigation Optimization System

Sistema intelligente per l'ottimizzazione dell'irrigazione basato su Machine Learning e regole agronomiche FAO-56.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/flask-3.0.0-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## Caratteristiche Principali

- **Machine Learning**: Modello RandomForest trainato su dati storici
- **Fallback Intelligente**: Algoritmo basato su regole FAO-56 per massima affidabilità
- **Spiegazioni Human-Readable**: Ogni raccomandazione include spiegazione dettagliata del processo decisionale
- **Analisi Temporale**: Grafici interattivi con trend umidità suolo e storico irrigazione
- **Feature Importance**: Identificazione automatica dei fattori più influenti
- **Web UI Professionale**: Interfaccia Bootstrap 5 responsive e user-friendly

---

## Quick Start

### 1. Requisiti

- Python 3.9 o superiore
- pip (package manager)

### 2. Installazione

```bash
# Clona il repository
git clone https://github.com/your-org/irrigad-ai.git
cd irrigad-ai

# Installa dipendenze
pip install -r requirements.txt
```

### 3. Avvio Applicazione

```bash
# Avvia il server Flask
python app.py
```

L'applicazione sarà disponibile su: **http://localhost:5000**

### 4. Utilizzo

1. Apri il browser su http://localhost:5000
2. Carica i 4 file richiesti (sm.csv, meteo.csv, valve.csv, previsioni.json)
3. Clicca su "Traina Modello ML"
4. Clicca su "Calcola Raccomandazione"
5. Visualizza i risultati e le spiegazioni dettagliate

---

## Struttura Progetto

```
irrigad-ai/
├── app.py                          # Flask application
├── requirements.txt                # Python dependencies
├── README.md                       # Documentation
├── modules/
│   ├── __init__.py
│   ├── data_loader.py             # CSV/JSON parser and validation
│   ├── feature_engineering.py     # Feature extraction (17+ features)
│   ├── ml_model.py                # RandomForest training/prediction
│   ├── rule_based.py              # FAO-56 fallback algorithm
│   └── data/
│       ├── __init__.py
│       └── tp_metadata.py         # Sensor metadata
├── templates/
│   └── index.html                 # Main UI
├── static/
│   ├── css/
│   │   └── style.css              # Custom styles
│   └── js/
│       └── main.js                # Frontend logic
└── data/
    └── uploads/                   # Uploaded files (created at runtime)
```

---

## File di Input

### 1. sm.csv (Sensori Umidità Suolo)

File CSV con letture dei sensori di umidità volumetrica del suolo.

**Formato:**
```csv
id_device,name,date,value
123,sm1,2025-09-05T00:09:24.140Z,28.4
123,sm2,2025-09-05T00:09:24.140Z,33.75
```

**Sensori richiesti:**
- `sm1`: Volumetric water content 1 (%)
- `sm2`: Volumetric water content 2 (%)

**Periodo minimo:** 60 giorni

---

### 2. meteo.csv (Dati Meteorologici)

File CSV con dati meteo storici dalla stazione meteorologica.

**Formato:**
```csv
id_device,name,date,value
117,precipitation,2025-09-05T00:20:45.006Z,0
117,airtemperaturemean,2025-09-05T00:20:45.006Z,17.57
117,relativehumiditymean,2025-09-05T00:20:45.006Z,97.09
```

**Sensori richiesti:**
- `precipitation`: Precipitazioni (mm)
- `airtemperaturemean`: Temperatura media aria (°C)
- `relativehumiditymean`: Umidità relativa media (%)
- `solarradiation`: Radiazione solare (W/m²)
- `windspeedmean`: Velocità vento media (m/s)
- `soiltemperaturemean`: Temperatura suolo media (°C)

**Periodo minimo:** 60 giorni (sincronizzato con sm.csv)

---

### 3. valve.csv (Storico Irrigazione)

File CSV con dati di irrigazione (contatori acqua e stato valvole).

**Formato:**
```csv
id_device,name,date,value
104,tfm1,2025-06-01T00:00:50.717Z,61215
104,valve1,2025-06-01T00:00:50.717Z,0
```

**Sensori richiesti:**
- `tfm1`: Total flow meter 1 - contatore acqua cumulativo (m³)
- `valve1-4`: Stato valvole (0=chiusa, 1=aperta)

**IMPORTANTE:** `tfm1` è un contatore incrementale. L'acqua irrigata giornaliera è calcolata come: `tfm1_oggi - tfm1_ieri`

**Periodo minimo:** 60 giorni

---

### 4. previsioni.json (Forecast Meteo)

File JSON con previsioni meteorologiche 3 giorni (formato OpenMeteo API).

**Formato:**
```json
{
  "latitude": 46.38,
  "longitude": 11.3,
  "daily": {
    "time": ["2025-11-12", "2025-11-13", "2025-11-14"],
    "temperature_2m_max": [13.0, 13.5, 12.8],
    "temperature_2m_min": [-0.1, 2.3, 1.1],
    "precipitation_sum": [0.0, 0.0, 0.0],
    "et0_fao_evapotranspiration": [1.01, 0.99, 0.97],
    "vapor_pressure_deficit_max": [0.53, 0.54, 0.54]
  }
}
```

**Campi richiesti:**
- `precipitation_sum`: Pioggia prevista (mm)
- `et0_fao_evapotranspiration`: Evapotraspirazione FAO-56 (mm) - **CHIAVE!**
- `vapor_pressure_deficit_max`: VPD massimo (kPa)
- `temperature_2m_max/min`: Temperature previste (°C)

---

## Algoritmo Machine Learning

### Features (17+)

Il modello utilizza oltre 17 features estratte dai dati storici:

**Da Sensori Umidità Suolo:**
- Media umidità suolo (ultimi 3 giorni)
- Trend umidità (variazione giornaliera)
- Deviazione standard umidità (ultimi 7 giorni)

**Da Dati Meteo:**
- Precipitazioni cumulate (3 giorni, 7 giorni)
- Temperature medie, massime, minime
- Radiazione solare media
- Umidità relativa media
- Velocità vento

**Da Storico Irrigazione:**
- Irrigazione cumulata (7 giorni)
- Media irrigazione (14 giorni)
- Giorni dall'ultima irrigazione significativa

**Da Previsioni:**
- Pioggia prevista (3 giorni)
- ET0 prevista (3 giorni)
- VPD massimo
- Temperatura minima (rischio gelo)

**Features Temporali:**
- Mese, giorno dell'anno, stagione

**Features Calcolate:**
- Deficit idrico (ET0 - precipitazioni)
- Deficit umidità suolo (target - attuale)

### Modello: RandomForest

```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
```

**Target:** Acqua irrigata nelle 24 ore successive (m³)

**Metriche di valutazione:**
- R² Score (coefficiente di determinazione)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

### Algoritmo Fallback (Rule-Based FAO-56)

Se il modello ML ha R² < 0.5, viene utilizzato un algoritmo basato su regole agronomiche:

1. Calcolo deficit idrico base: `ET0 - Precipitazioni`
2. Fattore correzione umidità suolo (1.8x se stress, 0.5x se saturo)
3. Fattore climatico (VPD, piogge recenti, rischio gelo)
4. Recupero deficit accumulato
5. Limiti di sicurezza (max 15mm/giorno)

---

## Output JSON

La risposta dell'endpoint `/predict` include:

```json
{
  "success": true,
  "model_used": "RandomForest",
  "training_metrics": {
    "r2_score": 0.78,
    "mae_m3": 45.3,
    "rmse_m3": 62.1
  },
  "prediction": {
    "water_m3": 856.5,
    "water_mm": 8.57,
    "confidence": "ALTA",
    "priority": "MEDIA",
    "recommendation": "Irrigare oggi con 8.6mm"
  },
  "current_situation": {
    "soil_moisture_avg": 29.1,
    "moisture_deficit": 10.9,
    "days_since_irrigation": 1
  },
  "forecast_3days": {
    "rain_expected_mm": 0.0,
    "et0_expected_mm": 2.97,
    "frost_risk": true
  },
  "decision_explanation": {
    "summary": "Raccomandazione aumentata del 15%...",
    "steps": ["1. Analisi storica...", "..."],
    "key_factors": [...],
    "warnings": ["⚠️ ATTENZIONE: rischio gelo"]
  },
  "feature_importance": [...]
}
```

---

## API Endpoints

### POST /upload

Carica e valida i 4 file di input.

**Body:** multipart/form-data con 4 file
- `sm`: sm.csv
- `meteo`: meteo.csv
- `valve`: valve.csv
- `previsioni`: previsioni.json

**Response:**
```json
{
  "success": true,
  "message": "File caricati con successo",
  "summary": {
    "sm": {"days": 99, "start_date": "2025-06-01", "end_date": "2025-09-07"},
    "meteo": {...},
    "valve": {...},
    "forecast": {...}
  }
}
```

---

### POST /train

Traina il modello ML sui dati caricati.

**Response:**
```json
{
  "success": true,
  "message": "Modello trainato con successo",
  "training_metrics": {
    "r2_score": 0.78,
    "mae_m3": 45.3,
    "training_samples": 97
  },
  "use_ml": true,
  "model_type": "RandomForest",
  "feature_importance": [...]
}
```

---

### POST /predict

Genera raccomandazione irrigazione.

**Response:** Vedi sezione "Output JSON"

---

### GET /analytics

Restituisce dati per grafici.

**Response:**
```json
{
  "success": true,
  "soil_moisture_trend": {
    "dates": [...],
    "sm1": [...],
    "sm2": [...],
    "avg": [...]
  },
  "irrigation_history": {
    "dates": [...],
    "values": [...]
  },
  "feature_importance": [...]
}
```

---

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-12T10:30:00",
  "data_loaded": true,
  "model_trained": true
}
```

---

## Tecnologie Utilizzate

### Backend
- **Flask 3.0**: Web framework
- **pandas 2.1**: Data manipulation
- **numpy 1.26**: Numerical computing
- **scikit-learn 1.3**: Machine learning

### Frontend
- **Bootstrap 5**: UI framework
- **Chart.js 4**: Grafici interattivi
- **Bootstrap Icons**: Icon set
- **Font Awesome 6**: Additional icons
- **Vanilla JavaScript**: No frameworks

---

## Validazioni

Il sistema include validazioni robuste:

- Controllo formato file (CSV/JSON)
- Verifica colonne richieste
- Controllo sensori essenziali
- Validazione periodo minimo (60 giorni)
- Verifica sovrapposizione periodi
- Controllo dati mancanti
- Validazione range valori

---

## Configurazione

### Costanti Agronomiche

Modificabili in `modules/rule_based.py`:

```python
OPTIMAL_MOISTURE = 40.0  # % target umidità suolo
MIN_MOISTURE = 25.0      # % soglia stress
MAX_DAILY_IRRIGATION = 15.0  # mm/giorno limite
```

### Area Campo

Modificabile in `modules/ml_model.py` e `modules/rule_based.py`:

```python
FIELD_AREA_HA = 10.0  # Ettari
```

---

## Troubleshooting

### Errore: "Dati insufficienti per training"

**Causa:** File CSV coprono meno di 60 giorni

**Soluzione:** Carica file con almeno 60 giorni di dati continui

---

### Errore: "Sensori essenziali mancanti"

**Causa:** File CSV non contengono tutti i sensori richiesti

**Soluzione:** Verifica che sm.csv contenga sm1 e sm2, meteo.csv contenga precipitation, airtemperaturemean, etc.

---

### Warning: "Training ML fallito (R² troppo basso)"

**Causa:** Dati storici insufficienti o troppo rumorosi

**Soluzione:** Nessuna azione richiesta. Il sistema passa automaticamente all'algoritmo rule-based FAO-56

---

### Errore: "Periodi non coincidono"

**Causa:** sm.csv, meteo.csv e valve.csv coprono periodi diversi

**Soluzione:** Assicurati che tutti i file CSV coprano lo stesso intervallo temporale (overlap minimo 60 giorni)

---

## Sviluppo Futuro

- [ ] Export raccomandazione in PDF
- [ ] API RESTful con autenticazione
- [ ] Confronto multi-scenario
- [ ] Integrazione diretta con OpenMeteo API
- [ ] Supporto multi-campo
- [ ] Dashboard storico raccomandazioni
- [ ] Alert automatici via email/SMS
- [ ] Mobile app

---

## Licenza

MIT License - Vedi file LICENSE per dettagli

---

## Autori

**Irrigad Team**

- Machine Learning: Claude AI
- Agronomic Consulting: FAO-56 Guidelines
- Weather Data: OpenMeteo API

---

## Contatti

Per supporto o domande:
- Email: support@irrigad.ai
- GitHub: https://github.com/irrigad/irrigation-ai
- Website: https://irrigad.ai

---

## Ringraziamenti

- FAO per le linee guida FAO-56
- OpenMeteo per i dati meteorologici
- scikit-learn team per gli strumenti ML
- Bootstrap e Chart.js per i componenti UI

---

**AI v1.0** - Ottimizzazione Intelligente dell'Irrigazione
