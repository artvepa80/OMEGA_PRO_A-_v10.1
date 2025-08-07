# predictor.py – HybridOmegaPredictor OMEGA PRO AI v10.1 – Versión Corregida con Robustez Anti-Crash

import os
import re
import math
import random
import logging
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Intentar carga de psutil para ajuste dinámico de memoria
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Intentar carga de cryptography para autenticación
try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

# Importar motores de generación de combinaciones
from core.consensus_engine import generar_combinaciones_consenso
from modules.montecarlo_model import generar_combinaciones_montecarlo
from modules.apriori_model import generar_combinaciones_apriori
from modules.transformer_model import generar_combinaciones_transformer
from modules.filters.rules_filter import FiltroEstrategico
from modules.filters.ghost_rng_generative import get_seeds
from modules.inverse_mining_engine import ejecutar_minado_inverso
from modules.score_dynamics import score_combinations, clean_combination
from utils.viabilidad import calcular_svi
from modules.exporters.exportador_svi import exportar_combinaciones_svi
from modules.clustering_engine import generar_combinaciones_clustering
from modules.genetic_model import generar_combinaciones_geneticas, GeneticConfig
from modules.evaluation.evaluador_inteligente import EvaluadorInteligente
from modules.profiling.jackpot_profiler import JackpotProfiler
from modules.lstm_model import generar_combinaciones_lstm
from modules.learning.gboost_jackpot_classifier import GBoostJackpotClassifier

# Configuración inicial de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("OmegaPredictor")

class HybridOmegaPredictor:
    """
    Predictor híbrido para números de lotería que combina múltiples modelos y técnicas.
    
    Esta clase integra varios modelos de predicción incluyendo motor de consenso, LSTM, 
    Monte Carlo, Apriori, Transformer, clustering, algoritmos genéticos y técnicas
    especializadas como Ghost RNG y minado inverso. También incorpora mejoras nuevas
    incluyendo clasificación GBoost, perfilado de jackpot y evaluación inteligente.
    
    Atributos:
        VALID_POSITIONS: Posiciones válidas de bolillas.
        MIN_VALUE: Valor mínimo para una bolilla.
        MAX_VALUE: Valor máximo para una bolilla.
        VALID_SVI_PROFILES: Perfiles SVI válidos.
        DEFAULT_DATA_PATHS: Rutas predeterminadas para datos históricos.
    
    Flags para activar/desactivar modelos (en el diccionario usar_modelos):
        - ghost_rng: Usar modelo Ghost RNG
        - inverse_mining: Usar minado inverso
        - svi: Usar cálculo SVI (Índice de Viabilidad Estadística)
        - lstm_v2: Usar modelo LSTM
        - montecarlo: Usar modelo Monte Carlo
        - apriori: Usar modelo Apriori
        - transformer_deep: Usar modelo Transformer
        - clustering: Usar modelo clustering
        - genetico: Usar modelo de algoritmos genéticos
        - gboost: Usar clasificador GBoost para predicción de jackpot (nuevo)
        - profiling: Usar perfilado de jackpot (nuevo)
        - evaluador: Usar evaluador inteligente (nuevo)
    
    Args:
        data_path: Ruta a datos históricos
        cantidad_final: Número de combinaciones finales a retornar
        historial_df: DataFrame de datos históricos precargado
        perfil_svi: Perfil SVI ('default', 'conservative', 'aggressive')
        logger: Instancia de logger personalizada
        seed: Semilla aleatoria
    """
    
    VALID_POSITIONS = {'B1', 'B2', 'B3', 'B4', 'B5', 'B6'}
    MIN_VALUE = 1
    MAX_VALUE = 40
    VALID_SVI_PROFILES = {'default', 'conservative', 'aggressive'}
    DEFAULT_DATA_PATHS = [
        "data/historial_kabala_github.csv",
        "backup/historial_kabala_github.csv",
        "https://raw.githubusercontent.com/omega-pro-ai/historial/main/historial_kabala_github.csv"
    ]

    def __init__(
        self,
        data_path: Optional[str] = None,
        cantidad_final: int = 30,
        historial_df: Optional[pd.DataFrame] = None,
        perfil_svi: str = 'default',
        logger: Optional[logging.Logger] = None,
        seed: int = 42
    ):
        if cantidad_final <= 0:
            cantidad_final = 30
            (logger or logging.getLogger("OmegaPredictor")).warning(
                f"⚠️ cantidad_final inválida; usando valor predeterminado 30"
            )

        self.data_path = data_path or self.DEFAULT_DATA_PATHS[0]
        self.cantidad_final = cantidad_final
        self.logger = logger or logging.getLogger("OmegaPredictor")

        # Cargar datos
        self.data = historial_df if historial_df is not None else self.cargar_datos()
        self._validate_historial_df()  # Validación adicional de datos

        # Filtro estratégico
        self.filtro = FiltroEstrategico()
        self.filtro.cargar_historial(self.data.values.tolist() if not self.data.empty else [])

        # Parámetros iniciales
        self.use_positional = True
        self.perfil_svi = perfil_svi if perfil_svi in self.VALID_SVI_PROFILES else 'default'
        self.auto_export = True
        self._internal_token = None
        self.log_level = 'INFO'
        self._cached_rng_seeds = None

        # Configuración de ghost RNG e inverse mining
        self.ghost_rng_params = {
            'max_seeds': 8,
            'cantidad_por_seed': 4,
            'training_mode': False
        }
        self.inverse_mining_params = {
            'boost_strategy': 'high_values',
            'penalize': [1, 2, 3],
            'focus_positions': ['B3', 'B5'],
            'count': 12
        }

        # Config centralizada para LSTM
        self.lstm_config = {
            'n_steps': 5,
            'seed': seed,
            'epochs': 100,
            'batch_size': 16,
            'min_history': 100
        }

        # Flags para activar/desactivar modelos
        self.usar_modelos = {
            "ghost_rng": True,
            "inverse_mining": True,
            "svi": True,
            "lstm_v2": True,
            "montecarlo": True,
            "apriori": True,
            "transformer_deep": True,
            "clustering": True,
            "genetico": True,
            # Nuevos flags para las mejoras
            "gboost": True,
            "profiling": True,
            "evaluador": True
        }

        # Inicialización de opciones
        self.set_positional_analysis(True)
        self.set_ghost_rng_usage(True)
        self.set_svi_profile(self.perfil_svi)

        # CORRECCIÓN: Instancia de JackpotProfiler con ambos paths
        self.jackpot_profiler = JackpotProfiler(
            model_path="models/jackpot_profiler.pkl",
            mlb_path="models/jackpot_profiler_mlb.pkl"
        )

        self.logger.info(f"✅ Predictor inicializado con {self.data.shape[0]} sorteos históricos")
        self.logger.debug(f"⚙️ Configuración LSTM: {self.lstm_config}")

    def _validate_historial_df(self):
        """Validación robusta del DataFrame histórico con fallback anti-crash"""
        if self.data.empty or self.data.shape[1] < 6:
            self.logger.error("🚨 historial_df inválido: vacío o columnas insuficientes. Usando datos dummy.")
            self.data = self._create_dummy_data()

        if np.isnan(self.data.values).any():
            self.logger.warning("⚠️ NaN en historial_df, imputando con mean.")
            self.data = self.data.fillna(self.data.mean())
            # Convertir a enteros después de imputación
            self.data = self.data.astype(int)

    def _create_dummy_data(self, rows=100) -> pd.DataFrame:
        """Crea datos dummy para prevenir crashes con datos inválidos"""
        self.logger.warning("⚠️ Generando datos dummy de respaldo")
        dummy_data = np.random.randint(self.MIN_VALUE, self.MAX_VALUE+1, size=(rows, 6))
        return pd.DataFrame(dummy_data, columns=[f'B{i+1}' for i in range(6)])

    def set_lstm_config(
        self,
        n_steps: int = 5,
        seed: int = 42,
        epochs: int = 100,
        batch_size: int = 16,
        min_history: int = 100
    ):
        self.lstm_config.update({
            'n_steps': n_steps,
            'seed': seed,
            'epochs': epochs,
            'batch_size': batch_size,
            'min_history': min_history
        })
        self.logger.info(f"⚙️ Configuración LSTM actualizada: {self.lstm_config}")

    def set_inverse_mining_usage(self, enable: bool):
        self.usar_modelos["inverse_mining"] = enable
        self.logger.info(f"⚙️ Minado inverso {'activado' if enable else 'desactivado'}")

    def set_ghost_rng_params(
        self,
        max_seeds: int = 8,
        cantidad_por_seed: int = 4,
        training_mode: bool = False
    ):
        if max_seeds <= 0 or cantidad_por_seed <= 0:
            self.logger.warning(
                f"⚠️ Parámetros Ghost RNG inválidos; usando valores predeterminados"
            )
            max_seeds, cantidad_por_seed = 8, 4
        self.ghost_rng_params.update({
            'max_seeds': max_seeds,
            'cantidad_por_seed': cantidad_por_seed,
            'training_mode': training_mode
        })
        self.logger.info(f"⚙️ Parámetros Ghost RNG establecidos: {self.ghost_rng_params}")

    def set_ghost_rng_usage(self, enable: bool):
        self.usar_modelos["ghost_rng"] = enable
        self.logger.info(f"⚙️ Ghost RNG {'activado' if enable else 'desactivado'}")

    # Nuevos métodos para control de flags
    def set_gboost_usage(self, enable: bool):
        self.usar_modelos["gboost"] = enable
        self.logger.info(f"⚙️ Clasificador GBoost {'activado' if enable else 'desactivado'}")

    def set_profiling_usage(self, enable: bool):
        self.usar_modelos["profiling"] = enable
        self.logger.info(f"⚙️ Perfilado de jackpot {'activado' if enable else 'desactivado'}")

    def set_evaluador_usage(self, enable: bool):
        self.usar_modelos["evaluador"] = enable
        self.logger.info(f"⚙️ Evaluador inteligente {'activado' if enable else 'desactivado'}")

    def validate_data_path(self, path: str) -> Optional[str]:
        """
        Valida si path es local o URL, descarga si es necesario,
        busca rutas alternativas y retorna la ruta final o None.
        """
        temp_path = None
        try:
            if path.startswith("http://") or path.startswith("https://"):
                self.logger.info(f"🌐 Descargando datos desde: {path}")
                session = requests.Session()
                retry = Retry(total=3, backoff_factor=1,
                              status_forcelist=[429, 500, 502, 503, 504],
                              allowed_methods=["GET"])
                session.mount("https://", HTTPAdapter(max_retries=retry))
                session.mount("http://", HTTPAdapter(max_retries=retry))

                resp = session.get(path, timeout=15)
                resp.raise_for_status()

                os.makedirs("temp", exist_ok=True)
                temp_path = f"temp/{os.path.basename(path)}"
                with open(temp_path, "wb") as f:
                    f.write(resp.content)
                self.logger.info(f"📥 Datos descargados: {temp_path}")
                return temp_path

            if os.path.exists(path):
                return path

            self.logger.warning(f"⚠️ Archivo no encontrado: {path}, probando alternativas...")
            for alt in self.DEFAULT_DATA_PATHS:
                if alt == path: continue
                if alt.startswith("http"):
                    continue
                if os.path.exists(alt):
                    self.logger.info(f"📂 Usando ruta alternativa: {alt}")
                    return alt

            parent = os.path.join("..", path)
            if os.path.exists(parent):
                self.logger.info(f"📂 Usando ruta en directorio padre: {parent}")
                return parent

            raise FileNotFoundError(f"🚨 Archivo de datos no encontrado")

        except Exception as e:
            self.logger.error(f"🚨 Error en descarga/validación: {e}", exc_info=True)
            return None
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    self.logger.debug(f"🧹 Archivo temporal eliminado: {temp_path}")
                except Exception as e:
                    self.logger.warning(f"⚠️ No se pudo eliminar archivo temporal: {e}")

    def set_logging_level(self, level: str = 'INFO'):
        valid = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO,
                 'WARNING': logging.WARNING, 'ERROR': logging.ERROR,
                 'CRITICAL': logging.CRITICAL}
        lvl = valid.get(level.upper(), logging.INFO)
        self.logger.setLevel(lvl)
        logging.getLogger().setLevel(lvl)
        self.log_level = level.upper()
        self.logger.info(f"⚙️ Nivel de logging establecido a: {self.log_level}")

    def set_svi_profile(self, perfil: str):
        if perfil not in self.VALID_SVI_PROFILES:
            self.logger.warning(f"⚠️ Perfil SVI inválido '{perfil}', usando 'default'")
            perfil = 'default'
        self.perfil_svi = perfil
        self.logger.info(f"⚙️ Perfil SVI establecido: {self.perfil_svi}")

    def auto_select_svi_profile(
        self,
        sum_threshold_high: int = 120,
        sum_threshold_low: int = 80,
        std_dev_threshold_high: int = 15,
        std_dev_threshold_low: int = 10
    ) -> str:
        if self.data.empty:
            self.logger.warning("⚠️ Sin datos para selección automática de SVI")
            return self.perfil_svi

        sums = self.data.sum(axis=1)
        avg, std = sums.mean(), sums.std()
        if avg > sum_threshold_high and std > std_dev_threshold_high:
            profile = 'aggressive'
        elif avg < sum_threshold_low and std < std_dev_threshold_low:
            profile = 'conservative'
        else:
            profile = 'default'
        self.set_svi_profile(profile)
        return profile

    def set_auto_export(self, enable: bool):
        self.auto_export = enable
        self.logger.info(f"⚙️ Auto-exportación {'activada' if enable else 'desactivada'}")

    def set_positional_analysis(self, flag: bool):
        self.use_positional = flag
        self.logger.info(f"⚙️ Análisis posicional {'activado' if flag else 'desactivado'}")
        # Reactivar todos los modelos posicionales si fuera necesario
        for key in ["ghost_rng", "inverse_mining", "svi", "lstm_v2",
                    "montecarlo", "apriori", "transformer_deep",
                    "clustering", "genetico"]:
            self.usar_modelos[key] = flag

    def set_inverse_mining_params(
        self,
        boost_strategy: str = 'high_values',
        penalize: Optional[List[int]] = None,
        focus_positions: Optional[List[str]] = None,
        count: int = 12
    ):
        if boost_strategy not in ('high_values', 'last_value'):
            raise ValueError(f"🚨 Estrategia de boost inválida: {boost_strategy}")
        if focus_positions:
            invalid = set(focus_positions) - self.VALID_POSITIONS
            if invalid:
                raise ValueError(f"🚨 Posiciones inválidas: {invalid}")
        if count <= 0:
            raise ValueError(f"🚨 Count debe ser positivo: {count}")
        if penalize:
            bad = [x for x in penalize if not (self.MIN_VALUE <= x <= self.MAX_VALUE)]
            if bad:
                raise ValueError(f"🚨 Números de penalización inválidos: {bad}")

        self.inverse_mining_params.update({
            'boost_strategy': boost_strategy,
            'penalize': penalize or [1,2,3],
            'focus_positions': focus_positions or ['B3','B5'],
            'count': count
        })
        self.logger.info("⚙️ Parámetros de minado inverso actualizados")
        self.logger.debug(f"🔧 {self.inverse_mining_params}")

    def validar_columnas_bolillas(
        self, df: pd.DataFrame
    ) -> Tuple[List[str], pd.DataFrame]:
        pattern = re.compile(r'^(bolilla[_\s]*\d+|B\d+)$', re.IGNORECASE)
        cols = [c for c in df.columns if pattern.match(c)]
        if len(cols) < 6:
            nums = df.select_dtypes(include='number').columns.tolist()
            if len(nums) >= 6:
                cols = nums[:6]
                self.logger.warning(f"⚠️ Usando primeras 6 columnas numéricas: {cols}")
            else:
                raise ValueError(f"🚨 No hay suficientes columnas de bolillas: {cols}")
        return cols, df[cols].dropna()

    def filtrar_validos(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.astype(float)
        mask = data.apply(lambda row: (
            len(row)==6 and all(
                not math.isnan(x) and float(x).is_integer() and
                (self.MIN_VALUE <= x <= self.MAX_VALUE)
                for x in row
            )
        ), axis=1)
        if not mask.all():
            self.logger.warning(f"🚨 {mask.size - mask.sum()} filas inválidas")
            data = data[mask]
        if data.shape[0] < 50:
            self.logger.warning("⚠️ <50 sorteos válidos después de filtrar")
        return data.astype(int)

    def cargar_datos(self) -> pd.DataFrame:
        path = self.validate_data_path(self.data_path)
        if not path:
            raise FileNotFoundError(f"🚨 No se pudo encontrar datos: {self.data_path}")

        try:
            size = os.path.getsize(path) if os.path.exists(path) else 0
            chunk = None
            if size > 50*1024*1024:
                chunk = 500
            elif size > 10*1024*1024:
                chunk = 1000
            elif size > 5*1024*1024:
                chunk = 2000

            if chunk:
                self.logger.info(f"📦 Cargando en chunks: {chunk}")
                frames = []
                for ch in pd.read_csv(path, chunksize=chunk, encoding="utf-8"):
                    cols, dfch = self.validar_columnas_bolillas(ch)
                    if len(cols)>=6:
                        frames.append(dfch)
                if not frames:
                    raise ValueError("🚨 No hay chunks válidos")
                df = pd.concat(frames, ignore_index=True)
            else:
                df = pd.read_csv(path, encoding="utf-8")
                cols, df = self.validar_columnas_bolillas(df)

            df = df.select_dtypes(include='number')
            for c in df.columns:
                if not pd.api.types.is_numeric_dtype(df[c]):
                    df[c] = pd.to_numeric(df[c], errors='coerce')
            df = self.filtrar_validos(df)
            if df.shape[0] < 50:
                raise ValueError("⚠️ <50 sorteos después de carga completa")
            self.logger.info(f"📊 Datos cargados: {df.shape[0]} sorteos")
            self.logger.debug(f"Muestra:\n{df.head()}")
            return df

        except Exception as e:
            self.logger.error(f"🚨 Error cargando datos: {e}", exc_info=True)
            raise

    def aplicar_ghost_rng(
        self,
        resultados_rng: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        combinaciones = []
        if not resultados_rng or not self.usar_modelos["ghost_rng"]:
            self.logger.info("⚠️ Ghost RNG omitido")
            return combinaciones

        valid = []
        for r in resultados_rng:
            if not isinstance(r, dict):
                continue
            if 'seed' not in r or 'draw' not in r:
                continue
            r.setdefault('composite_score', 1.0)
            valid.append(r)

        if not valid:
            self.logger.warning("⚠️ No hay seeds RNG válidas")
            return combinaciones

        top = sorted(valid, key=lambda x: x['composite_score'], reverse=True)[: self.ghost_rng_params['max_seeds']]
        for r in top:
            clean = clean_combination(r['draw'], self.logger)
            if len(clean) != 6:
                continue
            score = r['composite_score'] * 1.10
            combinaciones.append({
                "combination": clean,
                "source": "ghost_rng",
                "score": score,
                "ghost_ok": True,
                "seed": r['seed'],
                "metrics": {"composite_score": r['composite_score']},
                "normalized": 0.0
            })

        self.logger.info(f"✅ Ghost RNG: {len(combinaciones)} combos")
        return combinaciones

    def aplicar_minado_inverso(self, ultima_combinacion: List[int]) -> List[Dict[str, Any]]:
        if not self.usar_modelos["inverse_mining"]:
            return []

        clean_last = clean_combination(ultima_combinacion, self.logger)
        if len(clean_last) != 6:
            return []

        boost = (
            [n for n in clean_last if n > 20]
            if self.inverse_mining_params['boost_strategy']=='high_values'
            else [clean_last[-1]]
        )

        try:
            raw = ejecutar_minado_inverso(
                seed=clean_last,
                boost=boost,
                penalize=self.inverse_mining_params['penalize'],
                focus_positions=self.inverse_mining_params['focus_positions'],
                count=self.inverse_mining_params['count'],
                historial_df=self.data,
                mostrar=False
            )
        except Exception as e:
            self.logger.error(f"🚨 Error en minado inverso: {e}", exc_info=True)
            return []

        result = []
        for item in raw:
            combo = clean_combination(item.get("combination", []), self.logger)
            if len(combo)!=6:
                continue
            score = item.get("score", 1.0) * (
                1.15 if self.inverse_mining_params['boost_strategy']=='high_values' else 1.10
            )
            result.append({
                "combination": combo,
                "source": "inverse_mining",
                "score": score,
                "ghost_ok": False,
                "metrics": {"minado_score": item.get("score", 0)},
                "normalized": 0.0
            })

        self.logger.info(f"✅ Minado inverso: {len(result)} combos")
        return result

    def _obtener_token(self) -> str:
        if not CRYPTO_AVAILABLE:
            return "simulated_token"
        if self._internal_token:
            return self._internal_token
        key_path = "logs/auth_key.key"
        if not os.path.exists(key_path):
            raise FileNotFoundError("🚨 Falta clave de autenticación")
        with open(key_path, "rb") as f:
            key = f.read()
        fernet = Fernet(key)
        token = fernet.encrypt(b"usuario_autorizado").decode()
        self._internal_token = token
        return token

    def _map_svi_profile(self, profile: str) -> str:
        mapping = {
            'default': 'moderado',
            'conservative': 'conservador',
            'aggressive': 'agresivo'
        }
        return mapping.get(profile, 'moderado')

    def _calcular_svi_individual(self, combo: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        perfil_rng = self._map_svi_profile(self.perfil_svi)
        combo_str = str(combo["combination"])
        svi_score = calcular_svi(
            combinacion=combo_str,
            perfil_rng=perfil_rng,
            validacion_ghost=combo.get("ghost_ok", False),
            score_historico=combo.get("score", 1.0)
        )
        return combo, svi_score

    def filtrar_combinaciones(
        self,
        combinaciones: List[Dict[str, Any]],
        token_externo: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        token = token_externo or self._obtener_token()
        final = []
        for idx, item in enumerate(combinaciones):
            combo = clean_combination(item.get("combination", []), self.logger)
            if len(combo)!=6:
                continue
            try:
                score, razones = self.filtro.aplicar_filtros(
                    combo,
                    return_score=True,
                    perfil_svi=self._map_svi_profile(self.perfil_svi)
                )
            except Exception as e:
                continue
            threshold = {'moderado':0.7,'conservador':0.8,'agresivo':0.4}.get(self._map_svi_profile(self.perfil_svi),0.7)
            if score >= threshold:
                item["combination"] = combo
                item["score"] *= score
                item["metrics"] = item.get("metrics", {})
                item["metrics"]["filtro_score"] = score
                item["normalized"] = 0.0
                final.append(item)
        return final

    def calcular_svi_para_combinaciones(
        self, combinaciones: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not self.usar_modelos["svi"]:
            return combinaciones

        valid = []
        for idx, combo in enumerate(combinaciones):
            c = combo.get("combination", [])
            clean = clean_combination(c, self.logger)
            if len(clean)!=6:
                continue
            combo["combination"] = clean
            valid.append(combo)

        results = []
        try:
            for combo in valid:
                combo_str = str(combo["combination"])
                svi = calcular_svi(
                    combinacion=combo_str,
                    perfil_rng=self._map_svi_profile(self.perfil_svi),
                    validacion_ghost=combo.get("ghost_ok", False),
                    score_historico=combo.get("score",1.0)
                )
                combo["svi_score"] = svi
                combo["score"] *= svi
                combo["metrics"] = combo.get("metrics",{})
                combo["metrics"]["svi_score"] = svi
                combo["normalized"] = 0.0
                results.append(combo)
        except Exception:
            # Fallback paralelo
            with ThreadPoolExecutor() as exec:
                futures = {exec.submit(self._calcular_svi_individual, c): c for c in valid}
                for fut in as_completed(futures):
                    try:
                        combo, svi = fut.result()
                        combo["svi_score"] = svi
                        combo["score"] *= svi
                        combo["metrics"] = combo.get("metrics",{})
                        combo["metrics"]["svi_score"] = svi
                        combo["normalized"] = 0.0
                        results.append(combo)
                    except Exception:
                        continue

        return results

    def load_svi_config(self, config_path: str = "config/svi_config.json"):
        if not os.path.exists(config_path):
            self.logger.warning(f"⚠️ Configuración SVI no encontrada: {config_path}")
            return
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            self.auto_select_svi_profile(
                sum_threshold_high=cfg.get("sum_threshold_high",120),
                sum_threshold_low=cfg.get("sum_threshold_low",80),
                std_dev_threshold_high=cfg.get("std_dev_threshold_high",15),
                std_dev_threshold_low=cfg.get("std_dev_threshold_low",10)
            )
            self.logger.info(f"⚙️ Configuración SVI cargada desde {config_path}")
        except Exception as e:
            self.logger.error(f"🚨 Error cargando configuración SVI: {e}")

    def generate_source_score_chart(self, combinaciones: List[Dict[str, Any]]) -> dict:
        source_scores: Dict[str, List[float]] = {}
        for c in combinaciones:
            src = c.get("source", "unknown")
            source_scores.setdefault(src, []).append(c.get("score", 0.0))
        avg_scores = {s: sum(lst)/len(lst) if lst else 0.0 for s, lst in source_scores.items()}
        labels = list(avg_scores.keys())
        data = list(avg_scores.values())
        return {
            "type": "bar",
            "data": {"labels": labels, "datasets": [{"label": "Avg Score", "data": data}]},
            "options": {
                "scales": {"y": {"beginAtZero": True, "title": {"display": True, "text": "Avg Score"}}},
                "plugins": {"title": {"display": True, "text": "Puntaje Promedio por Fuente"}}
            }
        }

    def generate_source_count_chart(self, combinaciones: List[Dict[str, Any]]) -> dict:
        source_counts: Dict[str, int] = {}
        for c in combinaciones:
            src = c.get("source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1
        labels = list(source_counts.keys())
        data = list(source_counts.values())
        return {
            "type": "bar",
            "data": {"labels": labels, "datasets": [{"label": "Count", "data": data}]},
            "options": {
                "scales": {"y": {"beginAtZero": True, "title": {"display": True, "text": "Count"}}},
                "plugins": {"title": {"display": True, "text": "Combinaciones por Fuente"}}
            }
        }

    def export_chart(self, chart_name: str, chart_type: str, chart_data: dict, output_dir: str = "results"):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{chart_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(chart_data, f, indent=2)
        self.logger.info(f"📊 Gráfico exportado: {path}")

    def _run_consensus(self, max_comb: int) -> List[Dict[str, Any]]:
        try:
            # Asegurar al menos 6 columnas numéricas
            df = self.data.select_dtypes(include='number')
            if df.shape[1] < 6:
                raise RuntimeError("Historial sin al menos 6 columnas numéricas para consenso")

            raw = generar_combinaciones_consenso(
                historial_df=df,
                cantidad=max_comb,
                perfil_svi=self._map_svi_profile(self.perfil_svi),
                logger=self.logger,
                use_score_combinations=False
            )
            results = []
            for item in raw:
                combo = clean_combination(item["combination"], self.logger)
                if len(combo) == 6:
                    results.append({
                        "combination": combo,
                        "source": item.get("source", "consensus"),
                        "score": item.get("score", 1.0),
                        "metrics": item.get("metrics", {}),
                        "normalized": 0.0
                    })
            return results

        except Exception as e:
            self.logger.error(f"🚨 Error en consenso: {e}", exc_info=True)
            return []

    def _run_lstm(self, max_comb: int) -> List[Dict[str, Any]]:
        try:
            if self.data.shape[0] < self.lstm_config['min_history']:
                return []
            if self.data.shape[0] < self.lstm_config['n_steps'] + 1:
                return []

            historial = self.data.values.tolist()
            historial_set = {tuple(sorted(map(int, d))) for d in historial}

            raw = generar_combinaciones_lstm(
                data=self.data.values,
                cantidad=max_comb,
                historial_set=historial_set,
                logger=self.logger,
                config=self.lstm_config
            )
            results, invalid = [], 0
            for item in raw:
                clean = clean_combination(item.get("combination", []), self.logger)
                if len(clean)!=6:
                    invalid += 1
                    continue
                results.append({
                    "combination": clean,
                    "source": "lstm_v2",
                    "score": item.get("score", 1.0),
                    "metrics": item.get("metrics", {}),
                    "normalized": 0.0
                })
            return results
        except Exception as e:
            self.logger.error(f"🚨 Error en LSTM: {e}", exc_info=True)
            fallback = sorted(random.sample(range(1,41),6))
            return [{
                "combination": fallback,
                "source": "lstm_fallback",
                "score": 0.5,
                "metrics": {"fallback_reason": str(e)},
                "normalized": 0.0
            }]

    def _run_montecarlo(self, max_comb: int) -> List[Dict[str, Any]]:
        try:
            raw = generar_combinaciones_montecarlo(
                historial=self.data.values.tolist(),
                cantidad=max_comb,
                logger=self.logger
            )
            results, invalid = [], 0
            for item in raw:
                clean = clean_combination(item.get("combination", []), self.logger)
                if len(clean)!=6:
                    invalid += 1
                    continue
                results.append({
                    "combination": clean,
                    "source": "montecarlo",
                    "score": item.get("score", 1.0),
                    "metrics": item.get("metrics", {}),
                    "normalized": 0.0
                })
            return results
        except Exception as e:
            self.logger.error(f"🚨 Error en Montecarlo: {e}", exc_info=True)
            return []

    def _run_apriori(self, max_comb: int) -> List[Dict[str, Any]]:
        try:
            historial = self.data.values.tolist()
            raw = generar_combinaciones_apriori(
                data=historial,
                historial_set={tuple(sorted(c)) for c in historial},
                num_predictions=max_comb,
                logger=self.logger
            )
            results, invalid = [], 0
            for item in raw:
                clean = clean_combination(item.get("combination", []), self.logger)
                if len(clean)!=6:
                    invalid += 1
                    continue
                results.append({
                    "combination": clean,
                    "source": "apriori",
                    "score": item.get("score", 1.0),
                    "metrics": item.get("metrics", {}),
                    "normalized": 0.0
                })
            return results
        except Exception as e:
            self.logger.error(f"🚨 Error en Apriori: {e}", exc_info=True)
            return []

    def _run_transformer(self, max_comb: int) -> List[Dict[str, Any]]:
        try:
            raw = generar_combinaciones_transformer(
                historial_df=self.data,
                cantidad=max_comb,
                perfil_svi=self._map_svi_profile(self.perfil_svi),
                logger=self.logger
            )
            results, invalid = [], 0
            for item in raw:
                clean = clean_combination(item.get("combination", []), self.logger)
                if len(clean)!=6:
                    invalid += 1
                    continue
                results.append({
                    "combination": clean,
                    "source": "transformer_deep",
                    "score": item.get("score", 1.0),
                    "metrics": item.get("metrics", {}),
                    "normalized": 0.0
                })
            return results
        except Exception as e:
            self.logger.error(f"🚨 Error en Transformer: {e}", exc_info=True)
            return []

    def _run_clustering(self, max_comb: int) -> List[Dict[str, Any]]:
        try:
            raw = generar_combinaciones_clustering(
                historial_df=self.data,
                cantidad=max_comb,
                logger=self.logger
            )
            results, invalid = [], 0
            for item in raw:
                clean = clean_combination(item.get("combination", []), self.logger)
                if len(clean)!=6:
                    invalid += 1
                    continue
                results.append({
                    "combination": clean,
                    "source": "clustering",
                    "score": item.get("score", 1.0),
                    "metrics": item.get("metrics", {}),
                    "normalized": 0.0
                })
            return results
        except Exception as e:
            self.logger.error(f"🚨 Error en Clustering: {e}", exc_info=True)
            return []

    def _run_genetico(self, max_comb: int) -> List[Dict[str, Any]]:
        try:
            # Convert historical data to a set of sorted tuples
            historial_set = {tuple(sorted(map(int, x))) for x in self.data.values.tolist()}
            raw = generar_combinaciones_geneticas(
                data=self.data,
                historial_set=historial_set,  # FIX: Cambiado 'historial' a 'historial_set'
                cantidad=max_comb,
                config=GeneticConfig(),
                logger=self.logger
            )
            results, invalid = [], 0
            for item in raw:
                clean = clean_combination(item.get("combination", []), self.logger)
                if len(clean) != 6:
                    invalid += 1
                    continue
                score = item.get("score", item.get("fitness", 0) / 100)
                results.append({
                    "combination": clean,
                    "source": "genetico",
                    "score": score,
                    "metrics": item.get("metrics", {"fitness": item.get("fitness", 0)}),
                    "normalized": 0.0
                })
            return results
        except Exception as e:
            self.logger.error(f"🚨 Error en Genético: {e}", exc_info=True)
            return []

    def run_all_models(self) -> List[Dict[str, Any]]:
        self.logger.info("🚀 Iniciando pipeline de predicción")
        max_comb = 500
        if PSUTIL_AVAILABLE:
            mem = psutil.virtual_memory().available / 1024**2
            max_comb = min(500, int(mem/10))
            self.logger.debug(f"🔧 max_comb ajustado: {max_comb}")

        models = [
            ("consensus", self._run_consensus, max_comb//8),
            ("lstm_v2", self._run_lstm, max_comb//8),
            ("montecarlo", self._run_montecarlo, max_comb//8),
            ("apriori", self._run_apriori, max_comb//8),
            ("transformer_deep", self._run_transformer, max_comb//8),
            ("clustering", self._run_clustering, max_comb//8),
            ("genetico", self._run_genetico, max_comb//8)
        ]

        combinaciones: List[Dict[str, Any]] = []
        for name, fn, cnt in models:
            if self.usar_modelos.get(name, True):
                self.logger.info(f"⚙️ Ejecutando modelo: {name}")
                try:
                    model_results = fn(cnt)
                    combinaciones.extend(model_results)
                    del model_results  # Liberar memoria pronto
                except Exception as e:
                    self.logger.error(f"🚨 {name} falló: {e}", exc_info=True)

        # Ghost RNG
        if self.usar_modelos.get("ghost_rng", True):
            try:
                if self._cached_rng_seeds is None:
                    self._cached_rng_seeds = get_seeds(
                        historial_csv_path=self.data_path,
                        sorteos_reales_path=None,
                        perfil_svi=self._map_svi_profile(self.perfil_svi),
                        max_seeds=self.ghost_rng_params['max_seeds'],
                        training_mode=self.ghost_rng_params['training_mode']
                    )
                
                # Validar que las seeds sean válidas
                if not self._cached_rng_seeds or not isinstance(self._cached_rng_seeds, list):
                    self.logger.warning("⚠️ Seeds RNG inválidas, generando alternativas")
                    self._cached_rng_seeds = [{
                        'seed': random.randint(1000, 9999), 
                        'draw': random.sample(range(1, 41), 6),
                        'composite_score': random.uniform(0.7, 0.95)
                    } for _ in range(self.ghost_rng_params['max_seeds'])]
                
                combinaciones.extend(
                    self.aplicar_ghost_rng(self._cached_rng_seeds)
                )
            except Exception as e:
                self.logger.error(f"🚨 Error generando Ghost RNG: {e}", exc_info=True)

        # Minado inverso
        if self.usar_modelos.get("inverse_mining", True):
            try:
                ultima = self.data.values.tolist()[-1].copy() if not self.data.empty else [1,2,3,4,5,6]
                combinaciones.extend(self.aplicar_minado_inverso(ultima))
            except Exception as e:
                self.logger.error(f"🚨 Error generando minado inverso: {e}", exc_info=True)

        if not combinaciones:
            self.logger.warning("⚠️ No se generaron combinaciones; retornando fallback")
            return [{
                "combination": sorted(random.sample(range(1, 41), 6)),
                "source": "fallback",
                "score": 0.5,
                "svi_score": 0.5,
                "metrics": {},
                "normalized": 0.0
            }]

        # Calcular SVI
        combinaciones = self.calcular_svi_para_combinaciones(combinaciones)

        # Dynamic scoring (opcional)
        try:
            scored = score_combinations(
                combinations=combinaciones,
                historial=self.data,
                cluster_data=None,
                perfil_svi=self._map_svi_profile(self.perfil_svi),
                logger=self.logger
            )
            # Guardar métrica dinámica antes de modificar
            for c in scored:
                # Preservar el score dinámico original
                dyn_score = c.get("score", 0.0)
                svi_score = c.get("svi_score", 0.5)
                
                # Guardar métrica dinámica
                c.setdefault("metrics", {})["dynamic_score"] = dyn_score
                
                # Combinar scores: 0.6 * svi + 0.4 * (dyn_score/5.0)
                c["score"] = 0.6 * svi_score + 0.4 * (dyn_score/5.0)
                c["normalized"] = 0.0
                
            combinaciones = scored
            self.logger.info("✅ Dynamic scoring aplicado")
        except Exception as e:
            self.logger.error(f"🚨 Error en dynamic scoring: {e}", exc_info=True)
        
        # 1) Clasificación GBoost Jackpot
        if self.usar_modelos.get("gboost", True):
            try:
                clf = GBoostJackpotClassifier()
                combos_list = [c["combination"] for c in combinaciones]
                gb_scores = clf.predict(combos_list)
                for idx, c in enumerate(combinaciones):
                    gb_score = float(gb_scores[idx])
                    c.setdefault("metrics", {})["gboost_score"] = gb_score
                    # Aplicar factor multiplicativo
                    c["score"] *= gb_score
                self.logger.info(f"✅ GBoost aplicado a {len(combinaciones)} combos")
            except Exception as e:
                self.logger.warning(f"⚠️ Clasificación GBoost omitida: {e}")
        else:
            self.logger.info("⚙️ Clasificación GBoost omitida por flag")
        
        # 2) Perfilado de Jackpot - CORRECCIÓN APLICADA
        if self.usar_modelos.get("profiling", True):
            try:
                # Obtener métricas de perfilado
                perfil_metrics = self.jackpot_profiler.profile([c["combination"] for c in combinaciones])
                
                # Actualizar cada combinación con las métricas
                for combo_dict, perf_metrics in zip(combinaciones, perfil_metrics):
                    # Asegurarnos de guardar todas las métricas
                    combo_dict.setdefault("metrics", {}).update(perf_metrics)
                    
                    # Aplicar factor de probabilidad de jackpot al score
                    jackpot_prob = perf_metrics.get("jackpot_prob", 1.0)
                    combo_dict["score"] *= jackpot_prob
                    
                self.logger.info(f"✅ Perfilado aplicado a {len(combinaciones)} combos")
            except Exception as e:
                self.logger.warning(f"⚠️ Perfilado omitido: {e}")
        else:
            self.logger.info("⚙️ Perfilado omitido por flag")
        
        # 3) Evaluador Inteligente - FIXED TO MATCH ACTUAL SIGNATURE
        if self.usar_modelos.get("evaluador", True):
            try:
                evaluator = EvaluadorInteligente()
                evaluated = []
                for combo in combinaciones:
                    # Evaluar cada combinación individualmente con manejo de errores
                    try:
                        result = evaluator.evaluar(combo["combination"])
                        combo["metrics"]["evaluador"] = result
                    except Exception as e:
                        self.logger.warning(f"⚠️ Error evaluando combo {combo['combination']}: {e}")
                        combo["metrics"]["evaluador_error"] = str(e)
                    evaluated.append(combo)
                combinaciones = evaluated
                self.logger.info(f"✅ Evaluador aplicado a {len(combinaciones)} combos")
            except Exception as e:
                self.logger.warning(f"⚠️ EvaluadorInteligente omitido: {e}")
        else:
            self.logger.info("⚙️ EvaluadorInteligente omitido por flag")

        # Filtrar
        combinaciones = self.filtrar_combinaciones(combinaciones)
        if not combinaciones:
            self.logger.warning("⚠️ Todos los combos filtrados; usando fallback")
            return [{
                "combination": sorted(random.sample(range(1, 41), 6)),
                "source": "fallback",
                "score": 0.5,
                "svi_score": 0.5,
                "metrics": {},
                "normalized": 0.0
            }]

        # Ordenar y deduplicar
        combinaciones.sort(key=lambda x: x["score"], reverse=True)
        unique: Dict[Tuple[int,...], Dict[str, Any]] = {}
        for c in combinaciones:
            tup = tuple(sorted(c["combination"]))
            if tup not in unique:
                unique[tup] = c
        final = list(unique.values())[: self.cantidad_final]

        # Normalizar
        max_score = max(c["score"] for c in final) or 1.0
        for c in final:
            c["normalized"] = c["score"] / max_score

        # Auto exportación
        if self.auto_export:
            os.makedirs("results", exist_ok=True)
            try:
                exportar_combinaciones_svi(final, f"results/svi_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            except Exception:
                pass

        # Cerrar profiler para liberar recursos
        self.jackpot_profiler.close()

        self.logger.info(f"🏁 Pipeline completado: {len(final)} combos finales")
        return final

    def generar_informe_html(
        self,
        combinaciones: List[Dict[str, Any]],
        duplicate_sources: Dict[str, int],
        source_score_chart: dict,
        source_count_chart: dict
    ):
        try:
            from modules.reporting.html_reporter import generar_reporte_completo
            generar_reporte_completo(
                combinaciones,
                self.data,
                perfil_svi=self.perfil_svi,
                output_path="results/reporte_prediccion.html",
                duplicate_sources=duplicate_sources,
                source_score_chart=source_score_chart,
                source_count_chart=source_count_chart
            )
        except ImportError:
            self._generar_informe_html_basico(combinaciones, duplicate_sources,
                                              source_score_chart, source_count_chart)
        except Exception as e:
            self.logger.error(f"🚨 Error en reporte HTML: {e}", exc_info=True)
            self._generar_informe_html_basico(combinaciones, duplicate_sources,
                                              source_score_chart, source_count_chart)

    def _generar_informe_html_basico(
        self,
        combinaciones: List[Dict[str, Any]],
        duplicate_sources: Dict[str, int],
        source_score_chart: dict,
        source_count_chart: dict
    ):
        try:
            labels = [str(c["combination"]) for c in combinaciones]
            scores = [c["score"] for c in combinaciones]
            svi_scores = [c.get("svi_score", 0.0) for c in combinaciones]
            # Recuperar la métrica dinámica histórica
            dyn_scores = [c.get("metrics", {}).get("dynamic_score", 0.0) for c in combinaciones]
            sources = [c["source"] for c in combinaciones]

            report_path = "results/reporte_prediccion.html"
            os.makedirs(os.path.dirname(report_path), exist_ok=True)

            html = f"""<!DOCTYPE html>
<html lang="es">
<head><meta charset="UTF-8"><title>OMEGA PRO AI Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
 body {{ font-family: Arial,sans-serif; margin:20px; }}
 table {{ width:100%; border-collapse:collapse; }}
 th,td {{ width:1px solid #ddd; padding:8px; text-align:center; }}
 th {{ background:#f2f2f2; }}
</style>
</head><body>
<h1>Predicciones OMEGA PRO AI</h1>
<p>Generado: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
<table><tr><th>Combo</th><th>Score</th><th>SVI</th><th>Dynamic</th><th>Source</th></tr>
{"".join(f"<tr><td>{labels[i]}</td><td>{scores[i]:.4f}</td><td>{svi_scores[i]:.4f}</td><td>{dyn_scores[i]:.4f}</td><td>{sources[i]}</td></tr>" for i in range(len(labels)))}
</table>
<div style="width:45%;display:inline-block"><canvas id="scoreChart"></canvas></div>
<div style="width:45%;display:inline-block"><canvas id="sviChart"></canvas></div>
<script>
// Score Chart
new Chart(document.getElementById('scoreChart'), {{
    type:'bar',
    data:{{labels:{labels}, datasets:[{{label:'Score', data:{scores}}}]}},
    options:{{scales:{{y:{{beginAtZero:true}}}}}}
}});
// SVI Chart
new Chart(document.getElementById('sviChart'), {{
    type:'radar',
    data:{{labels:{labels}, datasets:[{{label:'SVI',data:{svi_scores}}}]}}
}});
</script>
</body></html>"""
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(html)
            self.logger.info(f"📊 HTML básico generado: {report_path}")
        except Exception as e:
            self.logger.error(f"🚨 Error generando HTML básico: {e}", exc_info=True)