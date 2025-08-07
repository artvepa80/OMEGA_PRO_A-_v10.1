# ====================================================================
# OMEGA_PRO_AI v10.2 · core/consensus_engine.py
# ====================================================================
"""
Motor de consenso de OMEGA: agrupa resultados de los distintos “engines”
(Apriori, Ghost RNG, Inverse Mining, etc.) y decide cuáles combinaciones
proponer como pronóstico final.
⚠️ Este archivo ahora incluye un *shim* de compatibilidad para asegurar
que la función `validate_combination` siempre esté disponible, sin importar
dónde viva realmente (utils.validation ≥v10 | core.validation ≤v9).
"""
# --------------------------- imports base ----------------------------
from __future__ import annotations
from typing import List, Dict, Any
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import logging
import random
import re

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# ────────────────────────────────────────────────────────────────────────────────
# Imports de modelos
# ────────────────────────────────────────────────────────────────────────────────
from modules.genetic_model import generar_combinaciones_geneticas, GeneticConfig
from modules.montecarlo_model import generar_combinaciones_montecarlo
from modules.clustering_engine import generar_combinaciones_clustering
from modules.rng_emulator import emular_rng_combinaciones
from modules.transformer_model import generar_combinaciones_transformer
from modules.apriori_model import generar_combinaciones_apriori
from modules.lstm_model import generar_combinaciones_lstm
from modules.score_dynamics import score_combinations

from modules.filters.rules_filter import FiltroEstrategico
from modules.utils.exportador_rechazos import exportar_rechazos_filtro
from modules.utils.importador_rechazos import importar_combinaciones_rechazadas
from modules.learning.gboost_jackpot_classifier import GBoostJackpotClassifier
from modules.evaluation.evaluador_inteligente import EvaluadorInteligente
from modules.profiling.jackpot_profiler import perfil_jackpot

# ───── Integraciones extra ─────────────────────────────────────────────────────
from modules.learning.auto_retrain import auto_retrain
from modules.learning.retrotracker import RetroTracker
from modules.learning.evaluate_model import evaluate_model_performance
from modules.utils.combinador_maestro import generar_combinacion_maestra
# --------------------------------------------------------------------
# ─────────────────────────────────────────────────────────────────────
# Compat-shim: garantiza que `validate_combination` esté importable
# • Nueva ubicación (>= v10.x) : utils.validation
# • Legacy (<= v9.x) : core.validation
# • Fallback mínimo si ninguna existe
# ─────────────────────────────────────────────────────────────────────
try:
    # Intento nuevo package-layout
    from utils.validation import validate_combination
except ImportError: # ⇢ soportar árbol legacy
    try:
        from core.validation import validate_combination
    except ImportError:
        # Última línea de defensa: implementación ultra-básica
        def validate_combination(draw: Tuple[int, ...] | List[int]) -> bool: # type: ignore
            """
            Fallback de emergencia:
            • Deben ser exactamente 6 enteros únicos en [1, 40].
            """
            try:
                nums = [int(n) for n in draw]
            except Exception: # no convertible a int / iterable
                return False
            return (
                len(nums) == 6
                and len(set(nums)) == 6
                and all(1 <= n <= 40 for n in nums)
            )
# Re-export explícito → permite `from core.consensus_engine import validate_combination`
__all__ = ["validate_combination"]
# ------------------------ configuración logger -----------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)
# --------------------------------------------------------------------
# ========== AQUÍ EMPIEZA EL RESTO DE TU CÓDIGO ORIGINAL ==========
# (Nada más cambia; simplemente mantuve lo que ya tenías. )
# --------------------------------------------------------------------
# ────────────────────────────────────────────────────────────────────────────────
# Logger global del motor de consenso
# ────────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ConsensusEngine")
logger.propagate = False

# ────────────────────────────────────────────────────────────────────────────────
# Flags de activación de modelos
# ────────────────────────────────────────────────────────────────────────────────
USE_MONTECARLO  = True
USE_LSTM        = True
USE_CLUSTERING  = True
USE_GENETICO    = True
USE_RNG         = True
USE_TRANSFORMER = True
USE_APRIORI     = True
USE_GBOOST      = True
USE_PROFILING   = True
USE_EVALUADOR   = True

# ────────────────────────────────────────────────────────────────────────────────
# Pesos por perfil y ajustes dinámicos
# ────────────────────────────────────────────────────────────────────────────────
PESO_MAP = {
    "default":     {"ghost_rng":1.2,"clustering":1.1,"montecarlo":1.0,"lstm_v2":1.0,"transformer":1.0,"inverse_mining":1.0,"genetico":1.0,"apriori":1.0},
    "conservative":{"lstm_v2":1.4,"transformer":1.3,"montecarlo":1.0,"clustering":1.0,"ghost_rng":1.0,"inverse_mining":1.0,"genetico":1.0,"apriori":1.0},
    "aggressive":  {"ghost_rng":1.4,"montecarlo":1.3,"lstm_v2":1.0,"clustering":1.0,"transformer":1.0,"inverse_mining":1.0,"genetico":1.0,"apriori":1.0},
}

PESOS_MODELOS = {
    "clustering": 1.3,"ghost_rng":1.2,"transformer":1.4,"montecarlo":1.1,
    "genetico":1.15,"lstm_v2":1.0,"inverse_mining":1.05,"apriori":1.05,"consensus":1.0
}

FALLBACK = {"combination":[1,2,3,4,5,6],"source":"fallback","score":0.5,"metrics":{},"normalized":0.0}

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────
def validar_historial(df: pd.DataFrame) -> bool:
    """Comprueba que existan ≥6 columnas numéricas y sin NaN/inf."""
    if df.empty: return False
    num_cols = df.select_dtypes(include='number')
    if num_cols.shape[1] < 6: return False
    return not (num_cols.isna().any().any() or np.isinf(num_cols).any().any())

def generar_reporte_consenso(combinaciones, perf_metrics, retro_tracker):
    """Opcional: genera reporte HTML si el módulo está disponible."""
    try:
        from modules.reporting.html_reporter import generar_reporte_completo
        generar_reporte_completo(
            {
                "combinations": combinaciones,
                "eval_metrics": perf_metrics,
                "retro_results": retro_tracker.get_results() if retro_tracker else {}
            },
            output_path="results/consenso_reporte.html"
        )
        logger.info("✅ Reporte HTML generado → results/consenso_reporte.html")
    except Exception as exc:
        logger.warning(f"⚠️ Reporte HTML omitido: {exc}")

# ────────────────────────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ────────────────────────────────────────────────────────────────────────────────
def generar_combinaciones_consenso(
    historial_df: pd.DataFrame,
    cantidad: int = 60,
    perfil_svi: str = "default",
    logger: logging.Logger | None = None,
    use_score_combinations: bool = False,
    retrain: bool = False,
    evaluate: bool = False,
    backtest: bool = False
) -> List[Dict[str, Any]]:

    logger = logger or logging.getLogger("ConsensusEngine")
    logger.info(f"🚀 Starting consensus generation (perfil_svi: {perfil_svi})")

    # ───── 1. Limpieza del historial ──────────────────────────────────────────
    from utils.validation import clean_historial_df       # import local para evitar ciclo
    historial_df = clean_historial_df(historial_df)
    if not validar_historial(historial_df):
        logger.error("🚨 Historial inválido tras limpieza – usando fallback")
        return [FALLBACK.copy()]

    num_cols = historial_df.select_dtypes(include='number').columns[:6]
    imputer = SimpleImputer(strategy="mean")
    historial_df[num_cols] = (
        imputer.fit_transform(historial_df[num_cols])
        .clip(1, 40).round().astype(int)
    )

    # ───── 2. Retrotracker / Evaluación / Reentrenamiento opcional ────────────
    retro_tracker = RetroTracker() if backtest else None
    perf_metrics  = evaluate_model_performance(historial_df) if evaluate else None
    if perf_metrics:
        for m in perf_metrics:
            PESOS_MODELOS[m["model"]] = max(0.5, m.get("accuracy",1)*1.5)
        logger.info("⚙️ Pesos de modelos ajustados con métricas de evaluación")

    if retrain or (perf_metrics and any(m["accuracy"] < .7 for m in perf_metrics)):
        logger.info("♻️ Reentrenando modelos …")
        auto_retrain(historial_df)

    # ───── 3. Plan de cantidades por modelo ───────────────────────────────────
    activos = {
        "montecarlo":USE_MONTECARLO,"lstm_v2":USE_LSTM,"clustering":USE_CLUSTERING,
        "genetico":USE_GENETICO,"ghost_rng":USE_RNG,"transformer":USE_TRANSFORMER,
        "apriori":USE_APRIORI
    }
    modelos_activos = [k for k,v in activos.items() if v]
    if not modelos_activos:
        logger.warning("⚠️ Ningún modelo activo – devolviendo fallback")
        return [FALLBACK.copy()]

    base = max(1, cantidad // len(modelos_activos))
    peso_perfil = PESO_MAP.get(perfil_svi, PESO_MAP["default"])
    cantidades = {m: int(base * peso_perfil.get(m,1.0)) for m in modelos_activos}

    # ───── 4. Ejecución paralela de cada modelo ───────────────────────────────
    results: list[dict[str,Any]] = []
    max_workers = min(4, multiprocessing.cpu_count())
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = []

        if USE_MONTECARLO:
            futs.append(pool.submit(generar_combinaciones_montecarlo,
                                    historial_df[num_cols].values.tolist(),
                                    cantidades["montecarlo"], logger))
        if USE_LSTM:
            futs.append(pool.submit(generar_combinaciones_lstm,
                                    historial_df[num_cols].values.tolist(),
                                    cantidades["lstm_v2"], logger))
        if USE_CLUSTERING:
            futs.append(pool.submit(generar_combinaciones_clustering,
                                    historial_df[num_cols], cantidades["clustering"], logger))
        if USE_GENETICO:
            cfg = GeneticConfig(poblacion_inicial=50, generaciones=20)
            futs.append(pool.submit(generar_combinaciones_geneticas,
                                    historial_df[num_cols], cfg, cantidades["genetico"], logger))
        if USE_RNG:
            futs.append(pool.submit(emular_rng_combinaciones,
                                    historial_df[num_cols], cantidades["ghost_rng"], logger))
        if USE_TRANSFORMER:
            futs.append(pool.submit(generar_combinaciones_transformer,
                                    historial_df[num_cols], cantidades["transformer"], logger))
        if USE_APRIORI:
            futs.append(pool.submit(generar_combinaciones_apriori,
                                    historial_df[num_cols], cantidades["apriori"], logger))

        for fut in as_completed(futs):
            try:
                model_output = fut.result() or []
                results.extend(model_output)
            except Exception as exc:
                logger.error(f"🚨 Modelo falló: {exc}")
                results.append(FALLBACK.copy())

    if not results:
        return [FALLBACK.copy()]

    # ───── 5. Filtrado estratégico + Scoring opcional ────────────────────────
    filtro = FiltroEstrategico()
    filtradas, rechazadas = [], []
    for item in results:
        combo = item["combination"]
        valido, razones = filtro.aplicar_filtros(combo, return_reasons=True)
        if valido:
            filtradas.append(item)
        else:
            rechazadas.append({"combination":combo,"reasons":razones})
    if rechazadas:
        exportar_rechazos_filtro(rechazadas)

    if not filtradas:
        logger.warning("⚠️ Todas las combinaciones fueron rechazadas")
        return [FALLBACK.copy()]

    # Opcional score_combinations sin logger para evitar pickling
    if use_score_combinations:
        try:
            filtradas = score_combinations(filtradas, historial_df[num_cols], perfil_svi, logger=None)
        except Exception as exc:
            logger.error(f"🚨 score_combinations falló: {exc}")

    # ───── 6. Normalizar score y aplicar pesos de modelo ──────────────────────
    for itm in filtradas:
        peso = PESOS_MODELOS.get(itm.get("source","consensus"),1.0)
        itm["normalized"] = itm.get("score",0)*peso

    filtradas.sort(key=lambda x:x["normalized"], reverse=True)
    top = filtradas[:cantidad]

    # ───── 7. Combinación maestra ─────────────────────────────────────────────
    try:
        combos_for_maestra = [{"combinacion":c["combination"],"score":c["score"]} for c in top]
        metadata = generar_combinacion_maestra(combos_for_maestra)
        logger.info(f"✅ Combinación maestra: {metadata['combinacion_maestra']}")
    except Exception as exc:
        logger.warning(f"⚠️ Error generando combinación maestra: {exc}")

    # ───── 8. Reporte opcional ────────────────────────────────────────────────
    generar_reporte_consenso(top, perf_metrics, retro_tracker)

    logger.info(f"✅ {len(top)} combinaciones seleccionadas por consenso final")
    return top
# ====================================================================
# END OF FILE
# ====================================================================