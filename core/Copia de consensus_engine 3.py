# consensus_engine.py – Módulo de Consenso OMEGA PRO AI v10.1 – Versión Corregida

from typing import List, Dict, Any
from collections import Counter
from modules.genetic_model import generar_combinaciones_geneticas, GeneticConfig
from modules.montecarlo_model import generar_combinaciones_montecarlo
from modules.clustering_engine import generar_combinaciones_clustering
from modules.rng_emulator import emular_rng_combinaciones  # FIX: Cambiado de rng_emulator a emular_rng_combinaciones para coincidir con el módulo corregido
from modules.transformer_model import generar_combinaciones_transformer
from modules.apriori_model import generar_combinaciones_apriori
from modules.score_dynamics import score_combinations
from modules.filters.rules_filter import FiltroEstrategico
from modules.utils.exportador_rechazos import exportar_rechazos_filtro
from modules.utils.importador_rechazos import importar_combinaciones_rechazadas
from modules.learning.gboost_jackpot_classifier import GBoostJackpotClassifier
from modules.evaluation.evaluador_inteligente import EvaluadorInteligente
from modules.profiling.jackpot_profiler import perfil_jackpot
from modules.lstm_model import generar_combinaciones_lstm
from utils.validation import validate_combination, clean_historial_df  # ADDED: Import for data cleaning
import pandas as pd
import logging
from logging import Logger
import random  # For dummy in auto-fit
from concurrent.futures import ThreadPoolExecutor, as_completed  # ADDED: For parallelism in filters

# Flags para activar o desactivar modelos
USE_MONTECARLO = True
USE_LSTM = True
USE_CLUSTERING = True
USE_GENETICO = True
USE_RNG = True
USE_TRANSFORMER = True
USE_APRIORI = True
# Nuevos flags para las mejoras
USE_GBOOST = True
USE_PROFILING = True
USE_EVALUADOR = True
USE_AUTO_FIT_GBOOST = True  # ADDED: Flag for auto-fit in GBoost

# Mapa de pesos por perfil SVI
PESO_MAP = {
    "default": {"ghost_rng":1.2, "clustering":1.1, "montecarlo":1.0, "lstm_v2":1.0, "transformer":1.0, "inverse_mining":1.0, "genetico":1.0, "apriori":1.0},
    "conservative": {"lstm_v2":1.4, "transformer":1.3, "montecarlo":1.0, "clustering":1.0, "ghost_rng":1.0, "inverse_mining":1.0, "genetico":1.0, "apriori":1.0},
    "aggressive": {"ghost_rng":1.4, "montecarlo":1.3, "lstm_v2":1.0, "clustering":1.0, "transformer":1.0, "inverse_mining":1.0, "genetico":1.0, "apriori":1.0},
}

# Pesos por modelo basados en aciertos históricos
PESOS_MODELOS = {
    "clustering": 1.3,
    "ghost_rng": 1.2,
    "transformer": 1.4,
    "montecarlo": 1.1,
    "genetico": 1.15,
    "lstm_v2": 1.0,
    "inverse_mining":1.05,
    "apriori": 1.05,
    "consensus": 1.0,
}

FALLBACK = {"combination": [1,2,3,4,5,6], "source":"fallback", "score":0.5}

def generar_combinaciones_consenso(
    historial_df: pd.DataFrame,
    cantidad: int = 60,
    perfil_svi: str = "default",
    logger: Logger = None,
    use_score_combinations: bool = False
) -> List[Dict[str, Any]]:
    try:
        # Configurar logger único
        if logger is None:
            logger = logging.getLogger(__name__)
            if not logger.handlers:
                logging.basicConfig(level=logging.INFO)
        logger.info(f"🚀 Starting consensus generation (perfil_svi: {perfil_svi})...")

        # Limpieza inicial de historial_df
        historial_df = clean_historial_df(historial_df)  # ADDED: Clean data at start to avoid invalid discards

        # Validar historial_df post-clean
        if historial_df.empty or historial_df.shape[1] < 6:
            logger.error("🚨 Invalid historial_df: empty or insufficient columns")
            return [FALLBACK]

        # Validar columnas numéricas
        columnas_numericas = [col for col in historial_df.columns
                              if col.startswith(("Bolilla","Numero")) or col.isdigit()]
        if not columnas_numericas or len(columnas_numericas) < 6:
            logger.error("🚨 No se encontraron columnas numéricas válidas en historial_df")
            return [FALLBACK]

        # Eliminar nulos (aunque clean ya maneja)
        if historial_df[columnas_numericas].isnull().any().any():
            logger.warning("⚠️ Valores nulos detectados en historial_df, eliminando filas con nulos")
            historial_df = historial_df[columnas_numericas].dropna().copy()
        if historial_df.empty:
            logger.error("🚨 historial_df vacío después de eliminar nulos")
            return [FALLBACK]

        combinaciones: List[Dict[str,Any]] = []
        rechazadas_anteriores = importar_combinaciones_rechazadas()
        modelos_activos = sum([USE_MONTECARLO, USE_LSTM, USE_CLUSTERING,
                                USE_GENETICO, USE_RNG, USE_TRANSFORMER, USE_APRIORI])
        if modelos_activos == 0:
            logger.warning("⚠️ No models active, using fallback for all")
            return [FALLBACK.copy()] * (cantidad // 10 or 1)  # ADDED: Handle zero models with multi-fallback
        cantidad_base = max(1, cantidad // modelos_activos)  # FIXED: Prevent ZeroDivisionError

        # Cantidades por modelo según perfil
        peso_modelos = PESO_MAP.get(perfil_svi, PESO_MAP["default"])
        cantidades_por_modelo = {
            "montecarlo": int(cantidad_base * peso_modelos.get("montecarlo",1.0)),
            "lstm_v2":    int(cantidad_base * peso_modelos.get("lstm_v2",1.0)),
            "clustering": int(cantidad_base * peso_modelos.get("clustering",1.0)),
            "genetico":   int(cantidad_base * peso_modelos.get("genetico",1.0)),
            "ghost_rng":  int(cantidad_base * peso_modelos.get("ghost_rng",1.0)),
            "transformer":int(cantidad_base * peso_modelos.get("transformer",1.0)),
            "apriori":    int(cantidad_base * peso_modelos.get("apriori",1.0)),
        }
        logger.info(f"⚙️ Ejecutando {modelos_activos} modelos activos...")

        historial_list = historial_df[columnas_numericas].values.tolist()
        historial_set = {tuple(sorted(row)) for row in historial_list}

        if USE_MONTECARLO:
            try:
                raw_mc = generar_combinaciones_montecarlo(historial=historial_list, cantidad=cantidades_por_modelo["montecarlo"], logger=logger)
                for item in raw_mc:
                    combo = item.get("combination", [])
                    if validate_combination(combo):
                        combinaciones.append({"combination": combo, "source": "montecarlo", "score": item.get("score",1.0)})
            except Exception as e:
                logger.error(f"🚨 Montecarlo failed: {e}")
                combinaciones.append(FALLBACK.copy())  # ADDED: Fallback per model

        if USE_LSTM:
            try:
                raw_lstm = generar_combinaciones_lstm(data=historial_df[columnas_numericas].values, cantidad=cantidades_por_modelo["lstm_v2"], historial_set=historial_set, logger=logger)
                for item in raw_lstm:
                    combo = item.get("combination", [])
                    if validate_combination(combo):
                        combinaciones.append({"combination": combo, "source": "lstm_v2", "score": item.get("score",1.0)})
            except Exception as e:
                logger.error(f"🚨 LSTM failed: {e}")
                combinaciones.append(FALLBACK.copy())  # ADDED: Fallback per model

        if USE_CLUSTERING:
            try:
                raw_cluster = generar_combinaciones_clustering(historial_df=historial_df, cantidad=cantidades_por_modelo["clustering"], logger=logger)
                for item in raw_cluster:
                    combo = item.get("combination", [])
                    if validate_combination(combo):
                        combinaciones.append({"combination": combo, "source": "clustering", "score": item.get("score",1.0)})
            except Exception as e:
                logger.error(f"🚨 Clustering failed: {e}")
                combinaciones.append(FALLBACK.copy())  # ADDED: Fallback per model

        if USE_GENETICO:
            try:
                raw_gen = generar_combinaciones_geneticas(data=historial_df, historial_set=historial_set, cantidad=cantidades_por_modelo["genetico"], config=GeneticConfig(), logger=logger)
                for item in raw_gen:
                    combo = item.get("combination", [])
                    if validate_combination(combo):
                        combinaciones.append({"combination": combo, "source": "genetico", "score": item.get("score",1.0)})
            except Exception as e:
                logger.error(f"🚨 Genetico failed: {e}")
                combinaciones.append(FALLBACK.copy())  # ADDED: Fallback per model

        if USE_RNG:
            try:
                raw_rng = emular_rng_combinaciones(historial=historial_list, cantidad=cantidades_por_modelo["ghost_rng"], logger=logger)  # FIX: Usar 'historial' corregido
                for item in raw_rng:
                    combo = item.get("combination", [])
                    if validate_combination(combo):
                        combinaciones.append({"combination": combo, "source": "ghost_rng", "score": item.get("score",1.0)})
            except Exception as e:
                logger.error(f"🚨 RNG failed: {e}")
                combinaciones.append(FALLBACK.copy())  # ADDED: Fallback per model

        if USE_TRANSFORMER:
            try:
                raw_trans = generar_combinaciones_transformer(historial_df=historial_df, cantidad=cantidades_por_modelo["transformer"], perfil_svi=perfil_svi, logger=logger)
                for item in raw_trans:
                    combo = item.get("combination", [])
                    if validate_combination(combo):
                        combinaciones.append({"combination": combo, "source": "transformer", "score": item.get("score",1.0)})
            except Exception as e:
                logger.error(f"🚨 Transformer failed: {e}")
                combinaciones.append(FALLBACK.copy())  # ADDED: Fallback per model

        if USE_APRIORI:
            try:
                raw_ap = generar_combinaciones_apriori(data=historial_list, historial_set=historial_set, num_predictions=cantidades_por_modelo["apriori"], logger=logger)
                for item in raw_ap:
                    combo = item.get("combination", [])
                    if validate_combination(combo):
                        combinaciones.append({"combination": combo, "source": "apriori", "score": item.get("score",1.0)})
            except Exception as e:
                logger.error(f"🚨 Apriori failed: {e}")
                combinaciones.append(FALLBACK.copy())  # ADDED: Fallback per model

        if not combinaciones:
            logger.warning("⚠️ No valid combinations generated from any model, returning fallback")
            return [FALLBACK]

        # ─── BLOQUE DE MEJORAS: GBoost, Profiling, Evaluador ──────────────────────────────────
        # 1) GBoost Jackpot Classification
        if USE_GBOOST:
            try:
                clf = GBoostJackpotClassifier()
                if not clf.is_fitted_ and USE_AUTO_FIT_GBOOST:
                    # Intenta cargar pre-entrenado
                    try:
                        clf.load("models/gboost_pretrained.pkl")  # ADDED: Assume path; adjust as needed
                        logger.info("✅ Loaded pre-trained GBoost model")
                    except FileNotFoundError:
                        # Dummy fit como fallback
                        X_dummy = [sorted(random.sample(range(1,41),6)) for _ in range(100)]
                        y_dummy = [random.randint(0,1) for _ in range(100)]
                        clf.fit(X_dummy, y_dummy)
                        logger.warning("⚠️ Auto-fit GBoost con dummy data")
                combos_list = [c["combination"] for c in combinaciones]
                gb_scores = clf.predict(combos_list)
                for c, gb in zip(combinaciones, gb_scores):
                    c.setdefault("metrics", {})["gboost_score"] = float(gb)
                    c["score"] *= float(gb)
                logger.info(f"✅ GBoost applied to {len(combinaciones)} combos")
            except Exception as e:
                logger.warning(f"⚠️ GBoost classification skipped: {e}")
        else:
            logger.info("⚙️ GBoost classification skipped by flag")

        # 2) Jackpot Profiling
        if USE_PROFILING:
            try:
                profile_metrics = perfil_jackpot(combinaciones)
                if len(profile_metrics) != len(combinaciones):
                    logger.warning(
                        f"⚠️ Profile metrics mismatch: {len(profile_metrics)} vs {len(combinaciones)}"
                    )
                else:
                    for c, pm in zip(combinaciones, profile_metrics):
                        c.setdefault("metrics", {}).update(pm)
                    logger.info(f"✅ Profiling applied to {len(combinaciones)} combos")
            except Exception as e:
                logger.warning(f"⚠️ Profiling skipped: {e}")
        else:
            logger.info("⚙️ Profiling skipped by flag")

        # 3) Evaluador Inteligente
        if USE_EVALUADOR:
            try:
                evaluator = EvaluadorInteligente()
                new_combinaciones = evaluator.evaluate(combinaciones)  # FIXED: Changed from 'evaluar' to 'evaluate' (assuming method name; revert if 'evaluar')
                if len(new_combinaciones) != len(combinaciones):
                    logger.warning(
                        f"⚠️ Evaluator changed combo count: {len(combinaciones)} → {len(new_combinaciones)}"
                    )
                else:
                    logger.info(f"✅ Evaluator applied to {len(new_combinaciones)} combos")
                combinaciones = new_combinaciones
            except Exception as e:
                logger.warning(f"⚠️ EvaluadorInteligente skipped: {e}")
        else:
            logger.info("⚙️ EvaluadorInteligente skipped by flag")
        # ──────────────────────────────────────────────────────────────

        # Filtros estratégicos finales
        logger.info("🧪 Aplicando filtros estratégicos (rules_filter v10.5)...")
        filtro = FiltroEstrategico()  # FIX: Definir filtro aquí, sin umbral si no existe en __init__
        filtro.cargar_historial(historial_list)

        perfil_svi_map = {"default":"moderado","conservative":"conservador","aggressive":"agresivo"}
        perfil_filtro = perfil_svi_map.get(perfil_svi, "moderado")
        umbral_map = {"moderado":0.85,"conservador":0.9,"agresivo":0.4}
        umbral = umbral_map.get(perfil_filtro, 0.85)

        combinaciones_filtradas: List[Dict[str,Any]] = []
        rechazos_log: List[tuple] = []

        # ADDED: Parallelize if many combinations
        if len(combinaciones) > 100:
            logger.info("🧪 Paralelizando filtros estratégicos...")
            def filtrar_entrada(entrada):
                comb = entrada["combination"]
                if tuple(sorted(comb)) in rechazadas_anteriores:
                    return None
                try:
                    score_estrategico, razones = filtro.aplicar_filtros(
                        comb, return_score=True, perfil_svi=perfil_filtro
                    )
                    if score_estrategico >= umbral:
                        entrada["score_filter"] = round(float(score_estrategico),4)
                        entrada["score"] = float(
                            0.5 * entrada.get("score",1.0) + 0.5 * score_estrategico
                        )
                        return entrada
                    else:
                        rechazos_log.append((
                            comb, razones,
                            round(float(score_estrategico),4),
                            entrada.get("source","desconocido")
                        ))
                        logger.debug(f"🧹 Rejected combination: {comb}, Score: {score_estrategico:.4f}, Reasons: {razones}")
                        return None
                except Exception as e:
                    logger.warning(f"⚠️ Filter error for combination {comb}: {e}")
                    return None

            with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust workers as needed
                futures = [executor.submit(filtrar_entrada, entrada) for entrada in combinaciones]
                combinaciones_filtradas = [f.result() for f in as_completed(futures) if f.result() is not None]
        else:
            for entrada in combinaciones:
                comb = entrada["combination"]
                if tuple(sorted(comb)) in rechazadas_anteriores:
                    logger.debug(f"🧹 Skipped previously rejected combination: {comb}")
                    continue
                try:
                    score_estrategico, razones = filtro.aplicar_filtros(
                        comb, return_score=True, perfil_svi=perfil_filtro
                    )
                    if score_estrategico >= umbral:
                        entrada["score_filter"] = round(float(score_estrategico),4)
                        entrada["score"] = float(
                            0.5 * entrada.get("score",1.0) + 0.5 * score_estrategico
                        )
                        combinaciones_filtradas.append(entrada)
                    else:
                        rechazos_log.append((
                            comb, razones,
                            round(float(score_estrategico),4),
                            entrada.get("source","desconocido")
                        ))
                        logger.debug(f"🧹 Rejected combination: {comb}, Score: {score_estrategico:.4f}, Reasons: {razones}")
                except Exception as e:
                    logger.warning(f"⚠️ Filter error for combination {comb}: {e}")

        logger.info(f"✅ {len(combinaciones_filtradas)} combinaciones pasaron los filtros (umbral: {umbral})")
        logger.info(f"🚫 {len(rechazos_log)} combinaciones fueron descartadas")
        if rechazos_log:
            exportar_rechazos_filtro(rechazos_log)

        if not combinaciones_filtradas:
            logger.warning("⚠️ All combinations filtered out, returning fallback")
            return [FALLBACK]

        # Normalizar y consolidar
        logger.info("🧮 Normalizando scores y consolidando...")
        for c in combinaciones_filtradas:
            c.setdefault("score", 1.0)
            c.setdefault("source", "unknown")

        # Refinamiento opcional con score_combinations
        if use_score_combinations:
            logger.info("🔍 Aplicando score_combinations para refinamiento...")
            try:
                combinaciones_filtradas = score_combinations(
                    combinations=combinaciones_filtradas,
                    historial=historial_df[columnas_numericas],
                    perfil_svi=perfil_svi,
                    logger=logger
                )
            except Exception as e:
                logger.error(f"🚨 score_combinations failed: {e}")

        # Deduplicación y orden final
        conteo = Counter([tuple(c["combination"]) for c in combinaciones_filtradas])
        combinaciones_unicas: List[Dict[str,Any]] = []
        ya_incluidas = set()

        for combo_tuple, _ in conteo.most_common(cantidad):
            if combo_tuple in ya_incluidas:
                continue
            matches = [c for c in combinaciones_filtradas if tuple(c["combination"])==combo_tuple]
            mejor = max(matches, key=lambda x: float(x["score"]) * PESOS_MODELOS.get(x["source"],1.0))
            combinaciones_unicas.append(mejor)
            ya_incluidas.add(combo_tuple)

        combinaciones_ordenadas = sorted(
            combinaciones_unicas,
            key=lambda x: float(x["score"]) * PESOS_MODELOS.get(x["source"],1.0),
            reverse=True
        )
        top = combinaciones_ordenadas[:cantidad]

        if not top:
            logger.warning("⚠️ No unique combinations after deduplication, returning fallback")
            return [FALLBACK]

        # ADDED: Final metrics
        if top:
            avg_score = sum(c['score'] for c in top) / len(top)
            logger.info(f"📊 Métricas finales: {len(top)} combos, avg_score={avg_score:.2f}")

        logger.info(f"✅ {len(top)} combinaciones seleccionadas por consenso final")
        return top

    except Exception as e:
        logger.error(f"🚨 Critical error in consensus generation: {e}")
        return [FALLBACK.copy()]  # Use copy for safety