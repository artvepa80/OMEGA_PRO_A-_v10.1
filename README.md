# OMEGA PRO AI v10.1

OMEGA PRO AI v12.0
A hybrid prediction system for generating lottery combinations using multiple models (e.g., LSTM, Transformer, ghost_rng, inverse_mining).
Installation
pip install tqdm watchdog pytest

Usage
Run the system with default settings:
python main.py --data_path data/historial_kabala_github.csv --svi_profile default --top_n 10 --export-formats csv json charts

Command-Line Options

--data_path: Path to historical data file (default: data/historial_kabala_github.csv).
--svi_profile: SVI profile (default, conservative, aggressive).
--top_n: Number of final combinations to generate (default: 10).
--enable-models: Models to enable (e.g., ghost_rng inverse_mining, default: all).
--export-formats: Export formats (csv, json, charts).

SVI Profiles

default: Balanced prediction strategy.
conservative: Prioritizes historically frequent numbers.
aggressive: Favors high-risk, high-reward combinations.

Outputs

CSV: results/svi_export_*.csv
JSON: results/svi_export_*.json
HTML with Charts: results/reporte_prediccion.html (source distribution pie, score histogram, SVI vs. score scatter)
Logs: logs/omega_system.log

Configuration
Edit config/viabilidad.json to customize settings (e.g., svi_batch_size, chart_types, progress_bar_style).
Testing
Run unit tests:
pytest tests/

Requirements

Python 3.8+
Libraries: tqdm, watchdog, pytest


Predicción Inteligente para La Kábala – Integración de modelos avanzados de IA, validación estructural y filtros estratégicos.

---

## 🚀 Estructura del Proyecto

```
OMEGA_PRO_AI_v10.1/
├── main.py                            # Punto de entrada
├── modules/
│   ├── montecarlo_model.py           # Generador Monte Carlo
│   ├── lstm_v2.py                    # Modelo LSTM Secuencial
│   ├── clustering_engine.py          # Clustering por centroides
│   ├── lottery_transformer.py        # Arquitectura Transformer PyTorch
│   ├── score_dynamics.py             # Score adaptativo
│   ├── inverse_mining.py             # Minado inverso
│   ├── filters/
│   │   ├── rules_filter.py           # Filtros estructurales
│   │   ├── ghost_rng_observer.py     # Detección técnica del RNG
│   │   ├── ghost_rng_generative.py   # Validación generativa RNG
├── core/
│   ├── predictor.py                  # Motor central de predicción
│   ├── transformer_predictor.py      # Carga Transformer y predice
│   ├── consensus_engine.py           # Combina modelos por consenso
├── utils/
│   ├── log_manager.py                # Logger centralizado
│   ├── viabilidad.py                 # Score de Viabilidad (SVI)
├── enhanced_lottery_transformer.pth # Modelo Transformer entrenado
├── exportador_resultados.py         # Exportación final de combinaciones
├── exportador_svi.py                # Exportación de combinaciones con SVI
├── enhanced_inference.py            # Inferencia probabilística con logits
├── data/                            # CSV históricos y seeds
│   ├── historial_kabala_github.csv
│   ├── jackpots_omega.csv
│   ├── viabilidad_config.json
├── logs/                            # JWT, errores, métricas y outputs
│   ├── jwt_secret.key
│   ├── filtros.log / omega_errors.log
```

---

## 🧠 Modelos activos

- ✅ Monte Carlo adaptativo (`montecarlo_model.py`)
- ✅ LSTM multicapas con Dropout (`lstm_v2.py`)
- ✅ Clustering por centroides y entropía (`clustering_engine.py`)
- ✅ Transformer entrenado (`lottery_transformer.py` + `.pth`)
- ✅ Consenso estratégico (`consensus_engine.py`)
- ✅ Ghost RNG: análisis técnico y generativo

---

## ⚙️ Instrucciones de ejecución

### 1. Crear entorno virtual y activar:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Instalar dependencias:
```bash
pip install -r requirements.txt
```
(Si no tienes `requirements.txt`, usar lista manual de imports visibles en `rules_filter.py`, incluyendo: `fastapi`, `jwt`, `redis`, `psutil`, `tenacity`, `prometheus_client`, etc.)

### 3. Generar clave JWT (una vez):
```bash
python generar_clave_auth.py
```

### 4. Ejecutar sistema:
```bash
python3 main.py
```

---

## 📤 Output del sistema
- Resultados en consola y exportados a CSV
- Logs detallados en `logs/`
- Visualizaciones (gráficos de score y rechazo)
- JSON de combinaciones con score y SVI

---

## 🧪 Validaciones implementadas
- Score dinámico por filtro estratégico
- Validación estructural contra historial y décadas
- Penalización por RNG sospechoso
- SVI (Score de Viabilidad de Inversión)
- Detección de alertas y activación de modo asalto

---

## 📌 Notas
- Este sistema requiere Python 3.9 o superior
- Redis debe estar activo para caching óptimo (opcional)
- La clave JWT (`logs/jwt_secret.key`) debe existir antes de ejecutar filtros

---

© 2025 Narapa LLC – Todos los derechos reservados
