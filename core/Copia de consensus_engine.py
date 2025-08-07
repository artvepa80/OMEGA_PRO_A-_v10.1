# consensus_engine.py – Versión v12.4.3 con corrección para Apriori y optimizaciones

from modules.genetic_model import generar_combinaciones_geneticas, GeneticConfig
from typing import List, Dict, Any
from modules.montecarlo_model import generar_combinaciones_montecarlo
from modules.lstm_v2 import generar_combinaciones_lstm_v2 as generar_combinaciones_lstm
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
from utils.validation import validate_combination
from collections import Counter
import pandas as pd
import logging
from logging import Logger

# Flags para activar o desactivar modelos
USE_MONTECARLO   = True
USE_LSTM         = True
USE_CLUSTERING   = True
USE_GENETICO     = True
USE_RNG          = True
USE_TRANSFORMER  = True
USE_APRIORI      = True

# Mapa de pesos por perfil SVI
PESO_MAP = {
    "default":      {"ghost_rng":1.2, "clustering":1.1, "montecarlo":1.0, "lstm_v2":1.0, "transformer":1.0, "inverse_mining":1.0, "genetico":1.0, "apriori":1.0},
    "conservative": {"lstm_v2":1.4, "transformer":1.3, "montecarlo":1.0, "clustering":1.0, "ghost_rng":1.0, "inverse_mining":1.0, "genetico":1.0, "apriori":1.0},
    "aggressive":   {"ghost_rng":1.4, "montecarlo":1.3, "lstm_v2":1.0, "clustering":1.0, "transformer":1.0, "inverse_mining":1.0, "genetico":1.0, "apriori":1.0},
}

# Pesos por modelo basados en aciertos históricos
PESOS_MODELOS = {
    "clustering":    1.3,
    "ghost_rng":     1.2,
    "transformer":   1.4,
    "montecarlo":    1.1,
    "genetico":      1.15,
    "lstm_v2":       1.0,
    "inverse_mining":1.05,
    "apriori":       1.05,
    "consensus":     1.0,
}

def generar_combinaciones_consenso(
    historial_df: pd.DataFrame,
    cantidad: int = 60,
    perfil_svi: str = "default",
    logger: Logger = None,
    use_score_combinations: bool = False
) -> List[Dict[str, Any]]:
    try:
        # Configurar logger
        if logger is None:
            logger = logging.getLogger(__name__)
            if not logger.handlers:
                logging.basicConfig(level=logging.INFO)
        logger.info(f"🚀 Starting consensus generation (perfil_svi: {perfil_svi})...")

        # Validar historial_df
        if historial_df.empty or historial_df.shape[1] < 6:
            logger.error("🚨 Invalid historial_df: empty or insufficient columns")
            return [{"combination":[1,2,3,4,5,6], "source":"fallback", "score":0.5}]

        # Validar columnas numéricas
        columnas_numericas = [col for col in historial_df.columns
                              if col.startswith(("Bolilla","Numero")) or col.isdigit()]
        if not columnas_numericas or len(columnas_numericas) < 6:
            logger.error("🚨 No se encontraron columnas numéricas válidas en historial_df")
            return [{"combination":[1,2,3,4,5,6], "source":"fallback", "score":0.5}]

        # Eliminar nulos
        if historial_df[columnas_numericas].isnull().any().any():
            logger.warning("⚠️ Valores nulos detectados en historial_df, eliminando filas con nulos")
            historial_df = historial_df[columnas_numericas].dropna().copy()
        if historial_df.empty:
            logger.error("🚨 historial_df vacío después de eliminar nulos")
            return [{"combination":[1,2,3,4,5,6], "source":"fallback", "score":0.5}]

        combinaciones: List[Dict[str,Any]] = []
        rechazadas_anteriores = importar_combinaciones_rechazadas()
        modelos_activos = sum([USE_MONTECARLO, USE_LSTM, USE_CLUSTERING,
                                USE_GENETICO, USE_RNG, USE_TRANSFORMER, USE_APRIORI])
        cantidad_base = max(1, cantidad // modelos_activos)

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

        # Montecarlo
        if USE_MONTECARLO:
            logger.info("🎲 Monte Carlo...")
            try:
                historial_list = historial_df[columnas_numericas].values.tolist()
                resultados = generar_combinaciones_montecarlo(
                    historial_list,
                    cantidad=cantidades_por_modelo["montecarlo"],
                    logger=logger
                )
                valid = [r for r in resultados if validate_combination(r.get("combination",[]))]
                if len(valid) < len(resultados):
                    logger.warning(f"⚠️ Skipped {len(resultados)-len(valid)} invalid Monte Carlo combinations")
                for r in valid:
                    logger.debug(f"[MONTECARLO] Combinación válida: {r['combination']}")
                combinaciones += [
                    {"combination":r["combination"],"source":"montecarlo","score":float(r.get("score",1.0))}
                    for r in valid
                ]
                logger.info(f"✅ [MONTECARLO] Generadas {len(valid)}/{cantidades_por_modelo['montecarlo']} combinaciones")
            except Exception as e:
                logger.error(f"⚠️ Monte Carlo falló: {e}")

        # LSTM
        if USE_LSTM:
            logger.info("🔁 LSTM...")
            try:
                resultados = generar_combinaciones_lstm(
                    historial_df=historial_df,
                    cantidad=cantidades_por_modelo["lstm_v2"],
                    fuerza_baja=0.3,
                    version="v10.4",
                    posicional=True,
                    perfil_svi=perfil_svi,
                    logger=logger
                )
                valid = [r for r in resultados if validate_combination(r.get("combination",[]))]
                if len(valid) < len(resultados):
                    logger.warning(f"⚠️ Skipped {len(resultados)-len(valid)} invalid LSTM combinations")
                for r in valid:
                    logger.debug(f"[LSTM] Combinación válida: {r['combination']}")
                combinaciones += [
                    {"combination":r["combination"],"source":"lstm_v2","score":float(r.get("score",1.0))}
                    for r in valid
                ]
                logger.info(f"✅ [LSTM] Generadas {len(valid)}/{cantidades_por_modelo['lstm_v2']} combinaciones")
            except Exception as e:
                logger.error(f"⚠️ LSTM falló: {e}")

        # Clustering
        if USE_CLUSTERING:
            logger.info("🧩 Clustering...")
            try:
                resultados = generar_combinaciones_clustering(
                    historial_df=historial_df,
                    cantidad=cantidades_por_modelo["clustering"],
                    logger=logger
                )
                valid = [r for r in resultados if validate_combination(r.get("combination",[]))]
                if len(valid) < len(resultados):
                    logger.warning(f"⚠️ Skipped {len(resultados)-len(valid)} invalid Clustering combinations")
                for r in valid:
                    logger.debug(f"[CLUSTERING] Combinación válida: {r['combination']}")
                combinaciones += [
                    {"combination":r["combination"],"source":"clustering","score":float(r.get("score",1.0))}
                    for r in valid
                ]
                logger.info(f"✅ [CLUSTERING] Generadas {len(valid)}/{cantidades_por_modelo['clustering']} combinaciones")
            except Exception as e:
                logger.error(f"⚠️ Clustering falló: {e}")

        # ——————————————————————————————————————
        # BLOQUE GENÉTICO CORREGIDO
        # ——————————————————————————————————————
        if USE_GENETICO:
            logger.info("🧬 Genético (nuevo)…")
            try:
                # 1) Reconstruyo el set histórico (CORRECCIÓN: paréntesis extra eliminado)
                historial_set = {
                    tuple(sorted(map(int, row)))  # <-- Paréntesis corregido aquí
                    for row in historial_df[columnas_numericas].to_numpy()
                }

                # 2) Creo la config tipada
                config_gen = GeneticConfig(
                    pop_size        = 50,
                    generations     = 100,
                    tournament_size = 3,
                    elite_fraction  = 0.1,
                    mutation_rate   = 0.25,
                    verbose         = logger.isEnabledFor(logging.DEBUG)
                )

                # 3) Llamo al módulo genético
                resultados = generar_combinaciones_geneticas(
                    data      = historial_df,
                    historial = historial_set,
                    cantidad  = cantidades_por_modelo["genetico"],
                    config    = config_gen,
                    logger    = logger
                )

                # 4) Filtro inválidas
                valid = [
                    r for r in resultados
                    if validate_combination(r["combination"])
                ]
                if len(valid) < len(resultados):
                    logger.warning(
                        "⚠️ Skipped %d invalid Genético combinations",
                        len(resultados) - len(valid)
                    )

                # 5) Inyecto al pool de consenso
                combinaciones += [
                    {
                        "combination": r["combination"],
                        "source":      "genetico",
                        "score":       float(
                            r.get("score", r.get("fitness", 0) / 100)
                        )
                    }
                    for r in valid
                ]

                logger.info(
                    "✅ [GENETICO] Generadas %d/%d combinaciones",
                    len(valid),
                    cantidades_por_modelo["genetico"]
                )
            except Exception as e:
                logger.error("⚠️ Genético falló: %s", str(e))

        # RNG Emulado
        if USE_RNG:
            logger.info("🎰 RNG Emulado...")
            try:
                historial_list = historial_df[columnas_numericas].values.tolist()
                resultados = emular_rng_combinaciones(
                    historical_combinations=historial_list,
                    cantidad=cantidades_por_modelo["ghost_rng"],
                    logger=logger,
                    seed=42,
                    config={
                        "peso_score":      0.75,
                        "peso_frecuencia": 0.25,
                        "min_score":       0.6
                    }
                )
                valid = [r for r in resultados if validate_combination(r.get("combination",[]))]
                if len(valid) < len(resultados):
                    logger.warning(f"⚠️ Skipped {len(resultados)-len(valid)} invalid RNG combinations")
                for r in valid:
                    logger.debug(f"[RNG EMULADO] Combinación válida: {r['combination']}")
                combinaciones += [
                    {"combination":r["combination"],"source":"ghost_rng","score":float(r.get("score",1.0)),"ghost_ok":True}
                    for r in valid
                ]
                logger.info(f"✅ [RNG] Generadas {len(valid)}/{cantidades_por_modelo['ghost_rng']} combinaciones")
            except Exception as e:
                logger.error(f"⚠️ RNG Emulado falló: {e}")

        # Transformer
        if USE_TRANSFORMER:
            logger.info("🧠 Transformer...")
            try:
                resultados = generar_combinaciones_transformer(
                    historial_df=historial_df,
                    cantidad=cantidades_por_modelo["transformer"],
                    perfil_svi=perfil_svi,
                    logger=logger
                )
                valid = [r for r in resultados if validate_combination(r.get("combination",[]))]
                if len(valid) < len(resultados):
                    logger.warning(f"⚠️ Skipped {len(resultados)-len(valid)} invalid Transformer combinations")
                for r in valid:
                    logger.debug(f"[TRANSFORMER] Combinación válida: {r['combination']}")
                combinaciones += [
                    {"combination":r["combination"],"source":"transformer_deep","score":float(r.get("score",1.0))}
                    for r in valid
                ]
                logger.info(f"✅ [TRANSFORMER] Generadas {len(valid)}/{cantidades_por_modelo['transformer']} combinaciones")
            except Exception as e:
                logger.error(f"⚠️ Transformer falló: {e}")

        # Apriori
        if USE_APRIORI:
            logger.info("📊 Apriori...")
            try:
                historial_set = {
                    tuple(sorted(map(int, row)))  # <-- Misma corrección aquí
                    for row in historial_df[columnas_numericas].to_numpy()
                }
                resultados = generar_combinaciones_apriori(
                    data=historial_df[columnas_numericas],
                    historial_set=historial_set,
                    num_predictions=cantidades_por_modelo["apriori"],
                    logger=logger,
                    config={'min_support_frequent': 0.015}
                )
                valid = [r for r in resultados if validate_combination(r.get("combination",[]))]
                if len(valid) < len(resultados):
                    logger.warning(f"⚠️ Skipped {len(resultados)-len(valid)} invalid Apriori combinations")
                for r in valid:
                    logger.debug(f"[APRIORI] Combinación válida: {r['combination']}")
                combinaciones += [
                    {"combination":r["combination"],"source":"apriori","score":float(r.get("score",1.0))}
                    for r in valid
                ]
                logger.info(f"✅ [APRIORI] Generadas {len(valid)}/{cantidades_por_modelo['apriori']} combinaciones")
            except Exception as e:
                logger.error(f"⚠️ Apriori falló: {e}")
                logger.debug(f"⚠️ Detalles del error de Apriori: {repr(e)}")
                combinaciones += [{"combination":[1,2,3,4,5,6],"source":"apriori_fallback","score":0.5}]

        # Si no generó nada
        if not combinaciones:
            logger.warning("⚠️ No valid combinations generated from any model, returning fallback")
            return [{"combination":[1,2,3,4,5,6],"source":"fallback","score":0.5}]

        # Filtros estratégicos finales
        logger.info("🧪 Aplicando filtros estratégicos (rules_filter v10.5)...")
        filtro = FiltroEstrategico()
        filtro.cargar_historial(historial_df[columnas_numericas].values.tolist())

        perfil_svi_map = {"default":"moderado","conservative":"conservador","aggressive":"agresivo"}
        perfil_filtro = perfil_svi_map.get(perfil_svi, "moderado")
        umbral_map = {"moderado":0.85,"conservador":0.9,"agresivo":0.4}
        umbral = umbral_map.get(perfil_filtro, 0.85)

        combinaciones_filtradas: List[Dict[str,Any]] = []
        rechazos_log: List[tuple] = []

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
            return [{"combination":[1,2,3,4,5,6],"source":"fallback","score":0.5}]

        # Normalizar y consolidar
        logger.info("🧮 Normalizando scores y consolidando...")
        for c in combinaciones_filtradas:
            c.setdefault("score", 1.0)
            c.setdefault("source", "unknown")

        logger.info("📊 Balance de fuentes:")
        summary = Counter([c["source"] for c in combinaciones_filtradas])
        for fuente, cuenta in summary.items():
            logger.info(f"  - {fuente}: {cuenta}")

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
            return [{"combination":[1,2,3,4,5,6],"source":"fallback","score":0.5}]

        logger.info(f"✅ {len(top)} combinaciones seleccionadas por consenso final")
        return top

    except Exception as e:
        logger.error(f"🚨 Critical error in consensus generation: {e}")
        return [{"combination":[1,2,3,4,5,6],"source":"fallback","score":0.5}]