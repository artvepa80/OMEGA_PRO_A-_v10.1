# OMEGA_PRO_AI_v10.1/core/consensus_engine.py – Módulo de Consenso OMEGA PRO AI v10.1 – Versión Corregida

from typing import List, Dict, Any
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import logging
from logging import Logger
import random
from sklearn.impute import SimpleImputer  # ADDED: For NaN handling
import multiprocessing  # ADDED: For dynamic max_workers

# Importar módulos de modelos
from modules.genetic_model import generar_combinaciones_geneticas, GeneticConfig
from modules.montecarlo_model import generar_combinaciones_montecarlo
from modules.clustering_engine import generar_combinaciones_clustering
from modules.rng_emulator import emular_rng_combinaciones
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
from utils.validation import validate_combination, clean_historial_df

# Integración de módulos adicionales
from modules.learning.auto_retrain import auto_retrain
from modules.learning.retrotracker import RetroTracker
from modules.learning.evaluate_model import evaluate_model_performance
# Nueva importación para Combinador Maestro
from modules.utils.combinador_maestro import generar_combinacion_maestra

# Configuración inicial de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ConsensusEngine")

# Flags para activar o desactivar modelos
USE_MONTECARLO = True
USE_LSTM = True
USE_CLUSTERING = True
USE_GENETICO = True
USE_RNG = True
USE_TRANSFORMER = True
USE_APRIORI = True
USE_GBOOST = True
USE_PROFILING = True
USE_EVALUADOR = True
USE_AUTO_FIT_GBOOST = True

# Mapa de pesos por perfil SVI
PESO_MAP = {
    "default": {"ghost_rng":1.2, "clustering":1.1, "montecarlo":1.0, "lstm_v2":1.0, "transformer":1.0, "inverse_mining":1.0, "genetico":1.0, "apriori":1.0},
    "conservative": {"lstm_v2":1.4, "transformer":1.3, "montecarlo":1.0, "clustering":1.0, "ghost_rng":1.0, "inverse_mining":1.0, "genetico":1.0, "apriori":1.0},
    "aggressive": {"ghost_rng":1.4, "montecarlo":1.3, "lstm_v2":1.0, "clustering":1.0, "transformer":1.0, "inverse_mining":1.0, "genetico":1.0, "apriori":1.0},
}

# Pesos por modelo basados en aciertos históricos (ajustables dinámicamente)
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

FALLBACK = {"combination": [1,2,3,4,5,6], "source":"fallback", "score":0.5, "metrics":{}, "normalized":0.0}

def validar_historial(df: pd.DataFrame) -> bool:
    """
    Valida la consistencia de historial_df antes de procesar.
    """
    if df.empty or df.shape[1] < 6:
        return False
    if df.select_dtypes(include=['number']).columns.size < 6:
        return False
    return True

def generar_reporte_consenso(combinaciones, perf_metrics, retro_tracker):
    """
    Genera un informe HTML con métricas de evaluación y retroanálisis.
    """
    try:
        from modules.reporting.html_reporter import generar_reporte_completo
        report_data = {
            "combinations": combinaciones,
            "eval_metrics": perf_metrics,
            "retro_results": retro_tracker.get_results() if retro_tracker else {}
        }
        generar_reporte_completo(report_data, output_path="results/consenso_reporte.html")
        logger.info("✅ Reporte HTML generado en results/consenso_reporte.html")
    except Exception as e:
        logger.warning(f"⚠️ Generación de reporte HTML omitida: {e}")

def generar_combinaciones_consenso(
    historial_df: pd.DataFrame,
    cantidad: int = 60,
    perfil_svi: str = "default",
    logger: Logger = None,
    use_score_combinations: bool = False,
    retrain: bool = False,
    evaluate: bool = False,
    backtest: bool = False
) -> List[Dict[str, Any]]:
    try:
        # Configurar logger único
        if logger is None:
            logger = logging.getLogger(__name__)
            if not logger.handlers:
                logging.basicConfig(level=logging.INFO)
        logger.info(f"🚀 Starting consensus generation (perfil_svi: {perfil_svi})...")

        # Limpieza inicial de historial_df
        historial_df = clean_historial_df(historial_df)
        imputer = SimpleImputer(strategy='mean')
        historial_df[historial_df.select_dtypes(include=['number']).columns] = imputer.fit_transform(
            historial_df.select_dtypes(include=['number'])
        )

        # Validar historial_df post-clean
        if not validar_historial(historial_df):
            logger.error("🚨 Invalid historial_df: empty or insufficient columns")
            return [FALLBACK]

        # Validar columnas numéricas
        columnas_numericas = [col for col in historial_df.columns
                              if col.startswith(("Bolilla","Numero")) or col.isdigit()]
        if not columnas_numericas or len(columnas_numericas) < 6:
            logger.error("🚨 No se encontraron columnas numéricas válidas en historial_df")
            return [FALLBACK]

        # Inicializar RetroTracker si backtest está habilitado
        retro_tracker = RetroTracker() if backtest else None

        # Evaluar rendimiento de modelos si evaluate está habilitado
        perf_metrics = None
        if evaluate:
            perf_metrics = evaluate_model_performance(historial_df)
            logger.info(f"✅ Evaluación de modelos completada: {len(perf_metrics)} métricas obtenidas")

        # Ajustar dinámicamente PESOS_MODELOS según perf_metrics
        if perf_metrics:
            for metric in perf_metrics:
                PESOS_MODELOS[metric["model"]] = metric.get("accuracy", 1.0) * 1.5
            logger.info("⚙️ PESOS_MODELOS ajustados dinámicamente según métricas de evaluación")

        # Reentrenar modelos si retrain=True o si las métricas indican bajo rendimiento
        if retrain or (perf_metrics and any(metric["accuracy"] < 0.7 for metric in perf_metrics)):
            logger.info("⚙️ Reentrenando modelos debido a configuración o bajo rendimiento")
            auto_retrain(historial_df)

        # Generar combinaciones
        combinaciones: List[Dict[str,Any]] = []
        rechazadas_anteriores = importar_combinaciones_rechazadas()
        modelos_activos = sum([USE_MONTECARLO, USE_LSTM, USE_CLUSTERING,
                               USE_GENETICO, USE_RNG, USE_TRANSFORMER, USE_APRIORI])
        if modelos_activos == 0:
            logger.warning("⚠️ No models active, using fallback for all")
            return [FALLBACK.copy()] * (cantidad // 10 or 1)

        cantidad_base = max(1, cantidad // modelos_activos)
        peso_modelos = PESO_MAP.get(perfil_svi, PESO_MAP["default"])
        cantidades_por_modelo = {
            "montecarlo": int(cantidad_base * peso_modelos.get("montecarlo",1.0)),
            "lstm_v2": int(cantidad_base * peso_modelos.get("lstm_v2",1.0)),
            "clustering": int(cantidad_base * peso_modelos.get("clustering",1.0)),
            "genetico": int(cantidad_base * peso_modelos.get("genetico",1.0)),
            "ghost_rng": int(cantidad_base * peso_modelos.get("ghost_rng",1.0)),
            "transformer": int(cantidad_base * peso_modelos.get("transformer",1.0)),
            "apriori": int(cantidad_base * peso_modelos.get("apriori",1.0)),
        }
        logger.info(f"⚙️ Ejecutando {modelos_activos} modelos activos...")

        historial_list = historial_df[columnas_numericas].values.tolist()
        historial_set = {tuple(sorted(row)) for row in historial_list}

        # Ejecutar modelos en paralelo
        max_workers = min(4, multiprocessing.cpu_count())  # ADDED: Dynamic max_workers
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            if USE_MONTECARLO:
                futures.append(executor.submit(
                    generar_combinaciones_montecarlo, historial=historial_list, cantidad=cantidades_por_modelo["montecarlo"], logger=logger
                ))
            if USE_LSTM:
                futures.append(executor.submit(
                    generar_combinaciones_lstm, data=historial_df[columnas_numericas].values, cantidad=cantidades_por_modelo["lstm_v2"], historial_set=historial_set, logger=logger
                ))
            if USE_CLUSTERING:
                futures.append(executor.submit(
                    generar_combinaciones_clustering, historial_df=historial_df, cantidad=cantidades_por_modelo["clustering"], logger=logger
                ))
            if USE_GENETICO:
                futures.append(executor.submit(
                    generar_combinaciones_geneticas, data=historial_df, historial_set=historial_set, cantidad=cantidades_por_modelo["genetico"], config=GeneticConfig(), logger=logger
                ))
            if USE_RNG:
                futures.append(executor.submit(
                    emular_rng_combinaciones, historial=historial_list, cantidad=cantidades_por_modelo["ghost_rng"], logger=logger
                ))
            if USE_TRANSFORMER:
                futures.append(executor.submit(
                    generar_combinaciones_transformer, historial_df=historial_df, cantidad=cantidades_por_modelo["transformer"], perfil_svi=perfil_svi, logger=logger
                ))
            if USE_APRIORI:
                futures.append(executor.submit(
                    generar_combinaciones_apriori, data=historial_list, historial_set=historial_set, num_predictions=cantidades_por_modelo["apriori"], logger=logger
                ))

            for future in as_completed(futures):
                try:
                    raw = future.result()
                    source = None
                    # Determinar source basado en el orden de futures (ajustar según el orden de append)
                    if future == futures[0] and USE_MONTECARLO:
                        source = "montecarlo"
                    elif future == futures[1] and USE_LSTM:
                        source = "lstm_v2"
                    elif future == futures[2] and USE_CLUSTERING:
                        source = "clustering"
                    elif future == futures[3] and USE_GENETICO:
                        source = "genetico"
                    elif future == futures[4] and USE_RNG:
                        source = "ghost_rng"
                    elif future == futures[5] and USE_TRANSFORMER:
                        source = "transformer"
                    elif future == futures[6] and USE_APRIORI:
                        source = "apriori"

                    for item in raw:
                        combo = item.get("combination", [])
                        if validate_combination(combo):
                            combinaciones.append({
                                "combination": combo,
                                "source": source,
                                "score": item.get("score", 1.0),
                                "metrics": item.get("metrics", {}),
                                "normalized": 0.0
                            })
                except Exception as e:
                    logger.error(f"🚨 Model failed: {e}")
                    combinaciones.append(FALLBACK.copy())

        if not combinaciones:
            logger.warning("⚠️ No valid combinations generated from any model, returning fallback")
            return [FALLBACK]

        # GBoost Jackpot Classification
        if USE_GBOOST:
            try:
                clf = GBoostJackpotClassifier()
                if not clf.is_fitted_ and USE_AUTO_FIT_GBOOST:
                    try:
                        clf.load("models/gboost_pretrained.pkl")
                        logger.info("✅ Loaded pre-trained GBoost model")
                    except FileNotFoundError:
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

        # Jackpot Profiling
        if USE_PROFILING:
            try:
                profile_metrics = perfil_jackpot(combinaciones)
                if len(profile_metrics) != len(combinaciones):
                    logger.warning(f"⚠️ Profile metrics mismatch: {len(profile_metrics)} vs {len(combinaciones)}")
                else:
                    for c, pm in zip(combinaciones, profile_metrics):
                        c.setdefault("metrics", {}).update(pm)
                        c["score"] *= pm.get("jackpot_prob", 1.0)
                    logger.info(f"✅ Profiling applied to {len(combinaciones)} combos")
            except Exception as e:
                logger.warning(f"⚠️ Profiling skipped: {e}")

        # Evaluador Inteligente
        if USE_EVALUADOR:
            try:
                evaluator = EvaluadorInteligente()
                new_combinaciones = evaluator.evaluate(combinaciones)
                if len(new_combinaciones) != len(combinaciones):
                    logger.warning(f"⚠️ Evaluator changed combo count: {len(combinaciones)} → {len(new_combinaciones)}")
                else:
                    logger.info(f"✅ Evaluator applied to {len(new_combinaciones)} combos")
                combinaciones = new_combinaciones
            except Exception as e:
                logger.warning(f"⚠️ EvaluadorInteligente skipped: {e}")

        # Filtros estratégicos finales
        logger.info("🧪 Aplicando filtros estratégicos (rules_filter v10.5)...")
        filtro = FiltroEstrategico()
        filtro.cargar_historial(historial_list)

        perfil_svi_map = {"default":"moderado","conservative":"conservador","aggressive":"agresivo"}
        perfil_filtro = perfil_svi_map.get(perfil_svi, "moderado")
        umbral_map = {"moderado":0.85,"conservador":0.9,"agresivo":0.4}
        umbral = umbral_map.get(perfil_filtro, 0.85)

        combinaciones_filtradas: List[Dict[str,Any]] = []
        rechazos_log: List[tuple] = []

        # Paralelizar filtrado si hay muchas combinaciones
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
                        entrada["score"] = float(0.5 * entrada.get("score",1.0) + 0.5 * score_estrategico)
                        return entrada
                    else:
                        rechazos_log.append((
                            comb, razones,
                            round(float(score_estrategico),4),
                            entrada.get("source","desconocido")
                        ))
                        return None
                except Exception as e:
                    logger.warning(f"⚠️ Filter error for combination {comb}: {e}")
                    return None

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(filtrar_entrada, entrada) for entrada in combinaciones]
                combinaciones_filtradas = [f.result() for f in as_completed(futures) if f.result() is not None]
        else:
            for entrada in combinaciones:
                comb = entrada["combination"]
                if tuple(sorted(comb)) in rechazadas_anteriores:
                    continue
                try:
                    score_estrategico, razones = filtro.aplicar_filtros(
                        comb, return_score=True, perfil_svi=perfil_filtro
                    )
                    if score_estrategico >= umbral:
                        entrada["score_filter"] = round(float(score_estrategico),4)
                        entrada["score"] = float(0.5 * entrada.get("score",1.0) + 0.5 * score_estrategico)
                        combinaciones_filtradas.append(entrada)
                    else:
                        rechazos_log.append((
                            comb, razones,
                            round(float(score_estrategico),4),
                            entrada.get("source","desconocido")
                        ))
                except Exception as e:
                    logger.warning(f"⚠️ Filter error for combination {comb}: {e}")

        logger.info(f"✅ {len(combinaciones_filtradas)} combinaciones pasaron los filtros (umbral: {umbral})")
        logger.info(f"🚫 {len(rechazos_log)} combinaciones fueron descartadas")
        if rechazos_log:
            exportar_rechazos_filtro(rechazos_log)

        if not combinaciones_filtradas:
            logger.warning("⚠️ All combinations filtered out, returning fallback")
            return [FALLBACK]

        # Refinamiento opcional con score_combinations
        if use_score_combinations:
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

        # Registrar predicciones con RetroTracker
        if retro_tracker:
            retro_tracker.log_predictions(top)
            logger.info("✅ Predicciones registradas con RetroTracker")

        # Ajustar scores según métricas de evaluación
        if perf_metrics:
            for combo in top:
                source = combo["source"]
                for metric in perf_metrics:
                    if metric["model"] == source:
                        combo["score"] *= metric.get("accuracy", 1.0)
                        combo["metrics"]["eval_accuracy"] = metric.get("accuracy", 1.0)
                        break

        # Generar reporte HTML con métricas y retroanálisis
        generar_reporte_consenso(top, perf_metrics, retro_tracker)

        # Métricas finales
        if top:
            avg_score = sum(c['score'] for c in top) / len(top)
            logger.info(f"📊 Métricas finales: {len(top)} combos, avg_score={avg_score:.2f}")

        # Nueva integración: Combinador Maestro
        try:
            # Preparar formato para combinador_maestro
            combos_for_maestra = [{"combinacion": c["combination"], "score": c["score"]} for c in top]
            metadata_maestra = generar_combinacion_maestra(combos_for_maestra)
            logger.info(f"✅ Combinación maestra generada: {metadata_maestra['combinacion_maestra']}")
        except Exception as e:
            logger.warning(f"⚠️ Error generando combinación maestra: {e}")

        logger.info(f"✅ {len(top)} combinaciones seleccionadas por consenso final")
        return top

    except Exception as e:
        logger.error(f"🚨 Critical error in consensus generation: {e}")
        return [FALLBACK.copy()]