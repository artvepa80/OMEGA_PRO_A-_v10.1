# OMEGA_PRO_AI_v10.1/main.py – OMEGA PRO AI v12.4 HÍBRIDO – Lanzador Principal – Versión Mejorada

# ---------------------------------------------------------------------------
# 1. Creación automática de directorios (antes de cualquier operación I/O)
# ---------------------------------------------------------------------------
import os

PATHS_TO_CREATE = [
    'core', 'modules/utils', 'modules/filters', 'modules/learning', 'modules/evaluation',
    'modules/profiling', 'modules/reporting', 'utils', 'backup', 'data', 'config',
    'models', 'outputs', 'results', 'logs', 'temp'
]
for _p in PATHS_TO_CREATE:
    os.makedirs(_p, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Imports estándar y de terceros
# ---------------------------------------------------------------------------
import sys
import time
import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections import Counter
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

# ---------------------------------------------------------------------------
# 3. Logging global con rotación
# ---------------------------------------------------------------------------
import logging
from logging.handlers import RotatingFileHandler

APP_NAME = "OMEGA PRO AI"
APP_VERSION = "v12.4"  # Incrementado tras el parche

def setup_rotating_logger(
    log_file="logs/omega.log",
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    level=logging.INFO
):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger_ = logging.getLogger()
    if logger_.handlers:                       # Evita duplicar handlers
        return logger_
    logger_.setLevel(level)

    fh = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger_.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger_.addHandler(ch)
    return logger_

logger = setup_rotating_logger()

# ---------------------------------------------------------------------------
# 4. Imports internos (después de tener logger disponible para fallback stubs)
# ---------------------------------------------------------------------------
try:
    from modules.learning.auto_retrain import auto_retrain
except ImportError:
    def auto_retrain(*args, **kwargs):
        logger.warning("⚠️ auto_retrain no disponible – usando stub")

try:
    from modules.learning.retrotracker import RetroTracker
except ImportError:
    class RetroTracker:
        def __init__(self, *_, **__):
            logger.warning("⚠️ RetroTracker no disponible – usando stub")
        def get_results(self):
            return {}

try:
    from modules.learning.evaluate_model import evaluate_model_performance
except ImportError:
    def evaluate_model_performance(*args, **kwargs):
        logger.warning("⚠️ evaluate_model_performance no disponible – usando stub")
        return []

from core.predictor import HybridOmegaPredictor as OmegaPredictor
from modules.score_dynamics import clean_combination
from utils.viabilidad import batch_calcular_svi, cargar_viabilidad, parallel_svi
from core.consensus_engine import validate_combination
from modules.exporters.exportador_svi import exportar_combinaciones_svi
from modules.utils.combinador_maestro import generar_combinacion_maestra
from utils.validation import clean_historial_df

# ---------------------------------------------------------------------------
# 5. ANSI para salida coloreada
# ---------------------------------------------------------------------------
ANSI = {
    "reset": "\033[0m", "cyan": "\033[96m", "green": "\033[92m",
    "yellow": "\033[93m", "red": "\033[91m", "bold": "\033[1m",
    "blue": "\033[94m", "magenta": "\033[95m", "underline": "\033[4m"
}

# ---------------------------------------------------------------------------
# Fallback para SimpleImputer (scikit-learn opcional)
# ---------------------------------------------------------------------------
try:
    from sklearn.impute import SimpleImputer  # Manejo de NaN
except ImportError:
    logger.warning("⚠️ scikit-learn no disponible – usando SimpleImputer básico")

    class SimpleImputer:
        """Implementación mínima de imputación media (solo strategy='mean')."""
        def __init__(self, strategy: str = "mean"):
            if strategy != "mean":
                raise ValueError("Solo strategy='mean' soportado en stub")
        def fit_transform(self, X):
            import numpy as np
            col_means = np.nanmean(X, axis=0)
            return np.where(np.isnan(X), col_means, X)

# ---------------------------------------------------------------------------
# 6. Utilidades varias
# ---------------------------------------------------------------------------
def get_fallback_item():
    """Retorna una combinación de fallback consistente."""
    return {
        "combination": [1, 2, 3, 4, 5, 6],
        "score": 0.5,
        "svi_score": 0.5,
        "source": "fallback",
        "original_score": 0.5
    }

def reportar_progreso(i, total, config=None):
    """Barra de progreso segura con cierre automático."""
    config = config or {}
    try:
        if not hasattr(reportar_progreso, "_pbar"):
            bar_style = config.get("progress_bar_style", {"desc": "Procesando", "unit": "combo"})
            reportar_progreso._pbar = tqdm(total=total, **bar_style)
        reportar_progreso._pbar.update(1)
        if i == total:
            reportar_progreso._pbar.close()
            delattr(reportar_progreso, "_pbar")
    except Exception as exc:
        logger.error(f"Error en barra de progreso: {exc}")
        if hasattr(reportar_progreso, "_pbar"):
            reportar_progreso._pbar.close()
            delattr(reportar_progreso, "_pbar")

class ConfigHandler(PatternMatchingEventHandler):
    """Observa cambios en archivos de configuración."""
    def __init__(self, callback, patterns):
        super().__init__(patterns=patterns)
        self.callback = callback
    def on_modified(self, event):
        if not event.is_directory:
            self.callback()

def start_config_watcher(config_path, callback):
    observer = Observer()
    observer.schedule(ConfigHandler(callback, patterns=[config_path.name]),
                      str(config_path.parent), recursive=False)
    observer.start()
    return observer

def print_summary_stats(stats, config, logger):
    """Resumen estructurado de métricas de ejecución."""
    logger.info(f"\n{'=' * 60}")
    logger.info("📊 RESUMEN ESTADÍSTICO:")
    logger.info(f"{'=' * 60}")
    logger.info(f"🔢 Total combinaciones: {stats['total']}")
    logger.info(f"🏆 Mejor score: {stats['max_score']:.3f}")
    logger.info(f"⚖️ Peor score: {stats['min_score']:.3f}")
    logger.info(f"📈 Score promedio: {stats['avg_score']:.3f} (Pred: {stats['avg_original']:.3f}, SVI: {stats['avg_svi']:.3f})")
    logger.info(f"⚖️ Pesos: Pred={stats['pred_weight']*100:.0f}%, SVI={stats['svi_weight']*100:.0f}%")
    logger.info(f"📊 Perfil SVI: {stats['svi_profile']}")

    logger.info("\n🌐 DISTRIBUCIÓN POR FUENTE:")
    emojis = config.get("emojis", {})
    for fuente, cantidad in stats["source_counter"].most_common():
        pct = (cantidad / stats['total']) * 100 if stats['total'] else 0
        logger.info(f" - {emojis.get(fuente, '🔹')} {fuente:<15}: {cantidad} ({pct:.1f}%)")

    if config.get("visualize_summary", False):
        logger.info(f"\n{'=' * 60}")
        logger.info("📊 VISUALIZACIÓN DE DISTRIBUCIÓN:")
        logger.info(f"{'=' * 60}")
        max_val = max(stats['source_counter'].values()) if stats['source_counter'] else 1
        for fuente, cantidad in stats['source_counter'].most_common():
            bar = '█' * int(config.get("max_bar_length", 50) * cantidad / max_val)
            pct = (cantidad / stats['total']) * 100 if stats['total'] else 0
            logger.info(f"{emojis.get(fuente, '🔹')} {fuente:<12} {bar} {pct:.1f}% ({cantidad})")

# ---------------------------------------------------------------------------
# 7. FUNCIÓN PRINCIPAL
# ---------------------------------------------------------------------------
def main(
    data_path="data/historial_kabala_github.csv",
    svi_profile="default",
    top_n=30,
    enable_models=None,
    export_formats=None,
    viabilidad_config="config/viabilidad.json",
    retrain=True,  # Cambiado a False por defecto
    evaluate=True,  # Cambiado a False por defecto
    backtest=True,  # Cambiado a False por defecto
    disable_multiprocessing=True  # Cambiado a False por defecto
):
    """Ejecuta el sistema de predicción OMEGA PRO AI."""
    enable_models = enable_models or ["all"]
    export_formats = export_formats or ["csv", "json"]

    logger.info(f"🚀 {APP_NAME} {APP_VERSION} arrancando")

    # -----------------------------------------------------------------------
    # 7.1 CARGA Y LIMPIEZA DE DATOS (MEJORADA)
    # -----------------------------------------------------------------------
    try:
        logger.info(f"📂 Cargando datos: {data_path}")
        df_raw = pd.read_csv(data_path)
        df_raw = clean_historial_df(df_raw)             # Limpieza de duplicados, formatos, etc.

        cols = [c for c in df_raw.columns if "bolilla" in c.lower()]
        if len(cols) < 6:
            raise ValueError("No se encontraron ≥6 columnas con 'bolilla'")

        # Manejo mejorado de NaN: rellena con 0 antes de conversión
        df_numbers = df_raw[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        imputer = SimpleImputer(strategy="mean")
        historial_df = pd.DataFrame(imputer.fit_transform(df_numbers), columns=cols).round().astype(int)

        if not all(historial_df[col].between(1, 40).all() for col in cols):
            raise ValueError("Datos fuera de rango [1, 40]")

        logger.info(f"✅ Dataset limpio: {len(historial_df)} registros")
    except Exception as exc:
        logger.exception(f"⚠️ Error al procesar datos: {exc}")
        return [get_fallback_item()]

    # -----------------------------------------------------------------------
    # 7.2 REENTRENAR / EVALUAR / BACKTEST (OPCIONALES)
    # -----------------------------------------------------------------------
    if retrain:
        logger.info("♻️ Reentrenando modelos …")
        auto_retrain(historial_df)

    perf_metrics = None
    if evaluate:
        logger.info("🧪 Evaluando rendimiento …")
        perf_metrics = evaluate_model_performance(historial_df)
        logger.info(f"✅ Métricas obtenidas: {len(perf_metrics)}")

    retro_tracker = RetroTracker() if backtest else None

    # -----------------------------------------------------------------------
    # 7.3 CARGAR CONFIGURACIÓN DE VIABILIDAD
    # -----------------------------------------------------------------------
    config_file = Path(viabilidad_config)
    required_keys = ["combo_length", "combo_range_min", "combo_range_max", "svi_batch_size"]
    try:
        global config_viabilidad
        config_viabilidad = cargar_viabilidad(watch_changes=False)
        if not all(k in config_viabilidad for k in required_keys):
            raise KeyError("Faltan claves requeridas en config")
    except Exception as exc:
        logger.exception(f"⚠️ Configuración inválida: {exc}")
        return [get_fallback_item()]

    observer = None
    if config_viabilidad.get("watch_changes", False):
        def reload_config():
            try:
                global config_viabilidad
                logger.info("🔄 Detectado cambio de configuración, recargando …")
                new_cfg = cargar_viabilidad(watch_changes=False)
                if all(k in new_cfg for k in required_keys):
                    config_viabilidad = new_cfg
                    logger.info("✅ Config recargada")
            except Exception as exc_:
                logger.error(f"⚠️ Recarga fallida: {exc_}")

        observer = start_config_watcher(config_file, reload_config)
        logger.info(f"👁️ Observando cambios en {config_file}")

    # -----------------------------------------------------------------------
    # 7.4 INICIALIZAR PREDICTOR
    # -----------------------------------------------------------------------
    try:
        predictor = OmegaPredictor(
            data_path=data_path,
            cantidad_final=top_n,
            historial_df=historial_df,
            perfil_svi=svi_profile,
            logger=logger
        )
        predictor.set_positional_analysis(True)
        predictor.set_auto_export(True)
        predictor.set_logging_level("INFO")

        valid_models = [
            "consensus", "ghost_rng", "inverse_mining", "lstm_v2",
            "montecarlo", "apriori", "transformer_deep", "clustering", "genetico"
        ]
        enable_models = valid_models if "all" in enable_models else [m for m in enable_models if m in valid_models]
        for m in valid_models:
            predictor.usar_modelos[m] = m in enable_models

        predictor.set_ghost_rng_params(max_seeds=8, cantidad_por_seed=4, training_mode=False)

        perfiles_svi_validos = ["default", "conservative", "aggressive"]
        svi_profile = svi_profile if svi_profile in perfiles_svi_validos else "default"
        predictor.set_svi_profile(svi_profile)

        logger.info(f"⚙️ Modelos activos: {', '.join(enable_models)}")
        logger.info(f"⚙️ Perfil SVI: {svi_profile}")
    except Exception as exc:
        logger.exception(f"⚠️ Error al inicializar predictor: {exc}")
        return [get_fallback_item()]

    # -----------------------------------------------------------------------
    # 7.5 EJECUTAR MODELOS
    # -----------------------------------------------------------------------
    try:
        logger.info("🧠 Ejecutando modelos …")
        t0 = time.time()
        combinaciones_finales = predictor.run_all_models()
        logger.info(f"✅ {len(combinaciones_finales)} combinaciones en {time.time() - t0:.2f}s")
    except Exception as exc:
        logger.exception(f"⚠️ Error en predictor: {exc}")
        return [get_fallback_item()]

    if not combinaciones_finales:
        logger.warning("⚠️ Predictor no devolvió combinaciones")
        return [get_fallback_item()]

    # -----------------------------------------------------------------------
    # 7.6 VALIDACIÓN, LIMPIEZA Y DEDUPLICADO
    # -----------------------------------------------------------------------
    seen, validas = set(), []
    combo_len     = config_viabilidad["combo_length"]
    cmin          = config_viabilidad["combo_range_min"]
    cmax          = config_viabilidad["combo_range_max"]

    for itm in combinaciones_finales:
        combo = clean_combination(itm.get("combination", []), logger)
        if len(combo) != combo_len \
           or not validate_combination(combo) \
           or not all(cmin <= n <= cmax for n in combo):
            logger.warning(f"⚠️ Combinación descartada: {combo}")
            continue
        key = tuple(sorted(combo))
        if key not in seen:
            seen.add(key)
            itm["combination"] = combo
            validas.append(itm)

    if not validas:
        logger.warning("⚠️ Todas las combinaciones fueron descartadas")
        return [get_fallback_item()]

    combinaciones_finales = validas

    # -----------------------------------------------------------------------
    # 7.7 COMBINACIÓN MAESTRA
    # -----------------------------------------------------------------------
    try:
        combos_maestra = [{"combinacion": x["combination"], "score": x["score"]} for x in combinaciones_finales]
        maestra = generar_combinacion_maestra(combos_maestra)
        logger.info(f"🏆 Maestra: {maestra['combinacion_maestra']}  Score={maestra['score']:.3f}  SVI={maestra['svi']:.3f}")
        logger.info(f"⚠️ Riesgo: {maestra['riesgo']['alerta']} – {maestra['riesgo']['recomendacion']}")

        combinaciones_finales.insert(0, {
            "combination": maestra["combinacion_maestra"],
            "score": maestra["score"],
            "svi_score": maestra["svi"],
            "source": "maestra",
            "original_score": maestra["score"],
            "perfil": maestra["perfil"],
            "riesgo": maestra["riesgo"]
        })
    except Exception as exc:
        logger.warning(f"⚠️ Error al generar maestra: {exc}")

    # -----------------------------------------------------------------------
    # 7.8 CALCULAR SVI PARA TODAS LAS COMBINACIONES
    # -----------------------------------------------------------------------
    prediction_weight = config_viabilidad.get("prediction_weight", 0.7)
    svi_weight        = config_viabilidad.get("svi_weight", 0.3)

    combos_list = [x["combination"] for x in combinaciones_finales]
    executor_cls = ThreadPoolExecutor if disable_multiprocessing else ProcessPoolExecutor
    try:
        logger.info("🧮 Calculando SVI …")
        t_svi = time.time()
        resultados_svi = parallel_svi(
            combos_list,
            perfil_rng=config_viabilidad.get("preferencia_rangos", "B"),
            validacion_ghost=any(x.get("source") == "ghost_rng" for x in combinaciones_finales),
            score_historico=config_viabilidad.get("score_historico_base", 3.0),
            progress_callback=reportar_progreso,
            config=config_viabilidad,
            logger=logger,
            executor=executor_cls
        )
        logger.info(f"✅ SVI en {time.time() - t_svi:.2f}s")
    except Exception as exc:
        logger.exception(f"⚠️ Error en SVI: {exc}. Asignando SVI=0.5 a todos.")
        resultados_svi = [(json.dumps(c), 0.5) for c in combos_list]

    for itm, (_, svi_val) in zip(combinaciones_finales, resultados_svi):
        original = itm.get("score", 0)
        itm["svi_score"] = svi_val
        itm["original_score"] = original
        itm["score"] = (original * prediction_weight) + (svi_val * svi_weight)

    # -----------------------------------------------------------------------
    # 7.9 ORDENAR, MOSTRAR RESULTADOS Y ESTADÍSTICAS
    # -----------------------------------------------------------------------
    combinaciones_finales.sort(key=lambda x: x["score"], reverse=True)

    emojis = config_viabilidad.get("emojis", {})
    color_fuente = config_viabilidad.get("color_fuente", {})
    thresholds = config_viabilidad.get("score_thresholds", {"high": 0.9, "medium": 0.6})

    logger.info(f"\n{'='*60}")
    logger.info(f"🎯 COMBINACIONES TOP ({len(combinaciones_finales)}):")
    logger.info(f"{'='*60}")

    for idx, itm in enumerate(combinaciones_finales, 1):
        fuente = itm.get("source", "consensus")
        combo_str = " - ".join(f"{n:02d}" for n in itm["combination"])
        score = itm["score"]
        svi   = itm["svi_score"]
        color = color_fuente.get(fuente, "")
        emoji = emojis.get(fuente, "🔹")
        sc_color = ANSI["green"] if score > thresholds["high"] else \
                   ANSI["yellow"] if score > thresholds["medium"] else ANSI["red"]

        if fuente == "maestra":
            logger.info(f"{'⭐'*3} COMBINACIÓN MAESTRA {'⭐'*3}")
        logger.info(f"{idx:2d}. {color}{emoji} {fuente.upper():<18} | "
                    f"{combo_str} | Score: {sc_color}{score:.3f}{ANSI['reset']} "
                    f"(SVI: {svi:.3f}, Pred: {itm['original_score']:.3f}){ANSI['reset']}")
        if fuente == "maestra":
            logger.info(f"   Perfil: {itm['perfil']}  "
                        f"Riesgo: {itm['riesgo']['alerta']} – {itm['riesgo']['recomendacion']}")
            logger.info(f"{'⭐'*9}")

    # Estadísticas
    fuentes_ctr = Counter(itm.get("source", "consensus") for itm in combinaciones_finales)
    print_summary_stats(
        stats={
            "total": len(combinaciones_finales),
            "max_score": max(itm["score"] for itm in combinaciones_finales),
            "min_score": min(itm["score"] for itm in combinaciones_finales),
            "avg_score": sum(itm["score"] for itm in combinaciones_finales) / len(combinaciones_finales),
            "avg_svi": sum(itm["svi_score"] for itm in combinaciones_finales) / len(combinaciones_finales),
            "avg_original": sum(itm["original_score"] for itm in combinaciones_finales) / len(combinaciones_finales),
            "pred_weight": prediction_weight,
            "svi_weight": svi_weight,
            "svi_profile": svi_profile,
            "source_counter": fuentes_ctr
        },
        config={
            "emojis": emojis,
            "visualize_summary": config_viabilidad.get("visualize_summary", False),
            "max_bar_length": config_viabilidad.get("max_bar_length", 50)
        },
        logger=logger
    )

    # -----------------------------------------------------------------------
    # 7.10 EXPORTAR RESULTADOS
    # -----------------------------------------------------------------------
    output_dir = Path(config_viabilidad.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata = {
        "app": APP_NAME,
        "version": APP_VERSION,
        "export_date": datetime.now().isoformat(),
        "parameters": {
            "data_path": data_path,
            "svi_profile": svi_profile,
            "top_n": top_n,
            "enable_models": enable_models
        }
    }

    if "csv" in export_formats and config_viabilidad.get("export_csv", False):
        csv_path = output_dir / f"combinaciones_{timestamp}.csv"
        try:
            with open(csv_path, "w", encoding="utf-8") as fh:
                fh.write(f"# {APP_NAME} {APP_VERSION}\n")
                fh.write(f"# {metadata['export_date']}\n")
                fh.write(f"# params: {json.dumps(metadata['parameters'], ensure_ascii=False)}\n")
            exportar_combinaciones_svi(combinaciones_finales, str(csv_path))
            logger.info(f"📁 CSV exportado: {csv_path}")
        except Exception as exc:
            logger.error(f"⚠️ Error al exportar CSV: {exc}")

    if "json" in export_formats and config_viabilidad.get("export_json", False):
        json_path = output_dir / f"combinaciones_{timestamp}.json"
        try:
            export_data = {"metadata": metadata, "combinations": combinaciones_finales}
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(export_data, fh, indent=4, ensure_ascii=False)
            logger.info(f"📁 JSON exportado: {json_path}")
        except Exception as exc:
            logger.error(f"⚠️ Error al exportar JSON: {exc}")

    logger.info(f"\n{'='*60}")
    logger.info("🚀 Ejecución completada – ¡Buena suerte!")
    logger.info(f"{'='*60}")

    if observer:
        observer.stop()
        observer.join()

    return combinaciones_finales

# ---------------------------------------------------------------------------
# 8. CLI (MEJORADA CON MANEJO DE ERRORES)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"{APP_NAME} {APP_VERSION}")
    parser.add_argument("--data_path", type=str, default="data/historial_kabala_github.csv")
    parser.add_argument("--svi_profile", type=str, default="default", choices=["default", "conservative", "aggressive"])
    parser.add_argument("--top_n", type=int, default=30)
    parser.add_argument("--enable-models", nargs="+", default=["all"])
    parser.add_argument("--export-formats", nargs="+", default=["csv", "json"], choices=["csv", "json"])
    parser.add_argument("--viabilidad-config", type=str, default="config/viabilidad.json")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--disable-multiprocessing", action="store_true")
    args = parser.parse_args()

    try:
        resultado = main(
            data_path=args.data_path,
            svi_profile=args.svi_profile,
            top_n=args.top_n,
            enable_models=args.enable_models,
            export_formats=args.export_formats,
            viabilidad_config=args.viabilidad_config,
            retrain=args.retrain,
            evaluate=args.evaluate,
            backtest=args.backtest,
            disable_multiprocessing=args.disable_multiprocessing
        )
        if len(resultado) == 1 and resultado[0]["source"] == "fallback":
            logger.warning("⚠️ Se devolvió fallback – revisar logs")
            sys.exit(0)
    except Exception as exc:
        logger.exception(f"❌ Error fatal: {exc}")
        sys.exit(1)

    sys.exit(0)