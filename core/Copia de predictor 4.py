# OMEGA_PRO_AI_v12.1/core/predictor.py
# HybridOmegaPredictor – v12.1 – Versión Mejorada con Validaciones y Logging Mejorado

import os
import gc
import json
import random
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from logging.handlers import RotatingFileHandler
from copy import deepcopy

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.impute import SimpleImputer
from io import StringIO

# Excepción personalizada para errores de carga de datos
class DataLoadError(Exception):
    """Excepción para errores críticos en carga de datos"""
    pass

# Dependencias opcionales
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

# Módulos de lógica de negocio
from core.consensus_engine import generar_combinaciones_consenso
from modules.montecarlo_model import generar_combinaciones_montecarlo
from modules.apriori_model import generar_combinaciones_apriori
from modules.transformer_model import generar_combinaciones_transformer
from modules.clustering_engine import generar_combinaciones_clustering
from modules.genetic_model import generar_combinaciones_geneticas, GeneticConfig
from modules.lstm_model import generar_combinaciones_lstm
from modules.filters.rules_filter import FiltroEstrategico
from modules.score_dynamics import score_combinations, clean_combination
from modules.evaluation.evaluador_inteligente import EvaluadorInteligente
from modules.profiling.jackpot_profiler import JackpotProfiler
from modules.exporters.exportador_svi import exportar_combinaciones_svi
from utils.validation import clean_historial_df

# Logger global rotativo
global_logger = logging.getLogger("OmegaPredictor")
global_logger.setLevel(logging.INFO)
handler = RotatingFileHandler(
    "omega_predictor.log", maxBytes=10*1024*1024, backupCount=5
)
handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s")
)
global_logger.addHandler(handler)

class HybridOmegaPredictor:
    """
    Predictor híbrido de OMEGA PRO AI con:
    - Config JSON + defaults merge
    - Robustez anti-crash
    - Paralelismo (procesos/hilos)
    - Logging consistente
    - Fallbacks de datos y modelos
    - Validación de configuraciones
    """
    MIN_VALUE = 1
    MAX_VALUE = 40
    VALID_SVI_PROFILES = {'default', 'conservative', 'aggressive'}
    DEFAULT_DATA_PATHS = [
        "data/historial_kabala_github.csv",
        "backup/historial_kabala_github.csv",
        "https://raw.githubusercontent.com/omega-pro-ai/historial/main/historial_kabala_github.csv"
    ]
    
    # Rangos válidos para parámetros de configuración
    CONFIG_VALIDATION_RULES = {
        'lstm_config': {
            'batch_size': (1, 256),
            'epochs': (1, 1000),
            'n_steps': (1, 20)
        },
        'ghost_rng_params': {
            'max_seeds': (1, 100),
            'cantidad_por_seed': (1, 100)
        }
    }
    
    DEFAULT_CONFIG = {
        'lstm_config': {'n_steps': 5, 'seed': 42, 'epochs': 100, 'batch_size': 16, 'min_history': 100},
        'ghost_rng_params': {'max_seeds': 8, 'cantidad_por_seed': 4, 'training_mode': False},
        'inverse_mining_params': {
            'boost_strategy': 'high_values', 
            'penalize': [1, 2, 3],
            'focus_positions': ['B3', 'B5'], 
            'count': 12
        },
        'usar_modelos': {key: True for key in [
            'consensus', 'lstm_v2', 'montecarlo', 'apriori', 'transformer_deep',
            'clustering', 'genetico', 'ghost_rng', 'inverse_mining', 'svi',
            'gboost', 'profiling', 'evaluador'
        ]}
    }

    def __init__(
        self,
        data_path: Optional[str] = None,
        cantidad_final: int = 30,
        historial_df: Optional[pd.DataFrame] = None,
        perfil_svi: str = 'default',
        logger: Optional[logging.Logger] = None,
        seed: int = 42,
        config_path: Optional[str] = None
    ):
        # Reproducibilidad
        seed = abs(seed)
        random.seed(seed)
        np.random.seed(seed)
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass

        self.data_path = data_path or self.DEFAULT_DATA_PATHS[0]
        self.cantidad_final = max(cantidad_final, 1)
        self.logger = logger or global_logger

        # Configuración con validación
        self.config = self.load_config(config_path)
        self._validate_config()
        
        # Extraer sub-configs
        self.lstm_config = self.config['lstm_config']
        self.ghost_rng_params = self.config['ghost_rng_params']
        self.inverse_mining_params = self.config['inverse_mining_params']
        self.usar_modelos = self.config['usar_modelos']

        # Dependencias opcionales
        self._setup_optional_deps(seed)

        # Carga de datos con manejo de errores mejorado
        try:
            self.data = historial_df if historial_df is not None else self._cargar_datos()
            self.data = clean_historial_df(self.data)
        except DataLoadError as e:
            self.logger.critical(str(e))
            self.data = pd.DataFrame()
            # Fallback a datos de ejemplo para evitar crash
            sample_data = [random.sample(range(self.MIN_VALUE, self.MAX_VALUE+1), 6) 
                          for _ in range(50)]
            self.data = pd.DataFrame(sample_data, columns=[f"B{i+1}" for i in range(6)])

        # Filtro estratégico
        self.filtro = FiltroEstrategico()
        self.filtro.cargar_historial(
            self.data.values.tolist() if not self.data.empty else []
        )

        # Param SVI
        self.perfil_svi = perfil_svi
        self.set_svi_profile(self.perfil_svi)
        self.auto_export = True

        # Jackpot profiler
        self.jackpot_profiler = JackpotProfiler(
            model_path="models/jackpot_profiler.pkl",
            mlb_path="models/jackpot_profiler_mlb.pkl"
        )

        self.logger.info(f"✅ Predictor inicializado con {self.data.shape[0]} registros")

    def load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Carga y mergea configuración JSON con defaults.
        
        Args:
            config_path: Ruta al archivo de configuración JSON
            
        Returns:
            Dict: Configuración mergeada
        """
        defaults = deepcopy(self.DEFAULT_CONFIG)
        if not config_path:
            self.logger.info("ℹ️ Sin config explícita; usando defaults")
            return defaults
            
        try:
            with open(config_path, 'r') as f:
                user_cfg = json.load(f)
                
            merged = deepcopy(defaults)
            merged.update(user_cfg)
            
            missing = [k for k in defaults if k not in user_cfg]
            if missing:
                self.logger.warning(f"⚠️ Config keys faltantes: {missing}")
            else:
                self.logger.info(f"✅ Config cargada desde {config_path}")
                
            return merged
        except Exception as e:
            # Log completo con stacktrace
            self.logger.exception(f"🛑 Error crítico cargando config {config_path}")
            self.logger.warning("⚠️ Usando configuración por defecto")
            return defaults

    def _validate_config(self) -> None:
        """
        Valida los valores de configuración contra reglas predefinidas.
        Ajusta valores fuera de rango y registra advertencias.
        """
        for section, rules in self.CONFIG_VALIDATION_RULES.items():
            if section not in self.config:
                continue
                
            for param, (min_val, max_val) in rules.items():
                if param not in self.config[section]:
                    continue
                    
                value = self.config[section][param]
                if not (min_val <= value <= max_val):
                    adjusted = min(max(value, min_val), max_val)
                    self.logger.warning(
                        f"⚠️ Valor inválido en config: {section}.{param}={value} "
                        f"(rango válido: {min_val}-{max_val}). Ajustando a {adjusted}"
                    )
                    self.config[section][param] = adjusted

    def set_svi_profile(self, perfil: str) -> None:
        """
        Valida y establece perfil SVI.
        
        Args:
            perfil: Nombre del perfil a usar
        """
        if perfil in self.VALID_SVI_PROFILES:
            self.perfil_svi = perfil
        else:
            self.logger.warning(f"⚠️ Perfil inválido '{perfil}'; usando default")
            self.perfil_svi = 'default'
        self.logger.info(f"⚙️ Perfil SVI configurado: {self.perfil_svi}")

    def _setup_optional_deps(self, seed: int) -> None:
        """Configura dependencias opcionales con validación de recursos"""
        # Ajuste dinámico de batch_size para LSTM
        if PSUTIL_AVAILABLE:
            mem_gb = psutil.virtual_memory().available / (1024 ** 3)
            bs = min(256, max(1, int(mem_gb * 8)))  # Aseguramos rango 1-256
            self.lstm_config['batch_size'] = bs
            self.logger.info(f"⚙️ Batch size ajustado a {bs} basado en {mem_gb:.1f} GB libres")
        else:
            self.logger.warning("⚠️ psutil no disponible; usando batch_size por defecto")
            
        # Token de seguridad crypto
        if CRYPTO_AVAILABLE:
            try:
                key = Fernet.generate_key()
                self._token = Fernet(key).encrypt(b"omega_secure").decode()
            except Exception as e:
                self._token = "fallback_token"
                self.logger.warning(f"⚠️ Error en crypto: {str(e)}")
        else:
            self._token = "fallback_token"
            self.logger.info("ℹ️ Cryptography no disponible; usando token de respaldo")

    def _cargar_datos(self) -> pd.DataFrame:
        """
        Carga datos históricos con múltiples fuentes y reintentos.
        
        Returns:
            pd.DataFrame: Datos históricos limpios
            
        Raises:
            DataLoadError: Si falla la carga de todas las fuentes
        """
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        
        with requests.Session() as session:
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            
            for path in self.DEFAULT_DATA_PATHS:
                try:
                    self.logger.info(f"🔍 Intentando cargar datos desde: {path}")
                    
                    if path.startswith('http'):
                        response = session.get(path, timeout=15)
                        response.raise_for_status()
                        df = pd.read_csv(StringIO(response.text))
                    else:
                        df = pd.read_csv(path)
                    
                    # Procesamiento de columnas
                    cols = [c for c in df.columns if 'bolilla' in c.lower()]
                    if len(cols) < 6:
                        num_cols = df.select_dtypes(include='number').columns
                        cols = list(num_cols[:6]) if len(num_cols) >= 6 else None
                    
                    if not cols or len(cols) < 6:
                        raise ValueError("No se encontraron suficientes columnas numéricas")
                    
                    df = df[cols].dropna().astype(int)
                    
                    # Validación de rango
                    mask = (df >= self.MIN_VALUE) & (df <= self.MAX_VALUE)
                    valid_rows = mask.all(axis=1)
                    
                    if valid_rows.sum() == 0:
                        raise ValueError("No hay filas válidas después del filtrado")
                    
                    df = df[valid_rows]
                    
                    # Imputación de valores faltantes
                    if df.isnull().any().any():
                        imp = SimpleImputer(strategy='median')
                        df = pd.DataFrame(imp.fit_transform(df), columns=df.columns)
                    
                    self.logger.info(f"✅ Datos cargados: {len(df)} filas de {path}")
                    return df
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Error cargando {path}: {str(e)}")
        
        # Si todas las fuentes fallan
        error_msg = "❌ Fallo en todas las fuentes de datos"
        self.logger.critical(error_msg)
        raise DataLoadError(error_msg)

    def _get_fallback_combo(self) -> Dict[str, Any]:
        """
        Genera una combinación de respaldo cuando falla un modelo.
        
        Returns:
            Dict: Combinación con metadatos básicos
        """
        combo = sorted(random.sample(range(self.MIN_VALUE, self.MAX_VALUE+1), 6))
        return {
            "combination": combo,
            "source": "fallback",
            "score": 0.5,
            "metrics": {},
            "normalized": 0.0
        }

    def _run_model_wrapper(self, model: str) -> List[Dict[str, Any]]:
        """
        Ejecuta un modelo específico con manejo de errores.
        
        Args:
            model: Nombre del modelo a ejecutar
            
        Returns:
            List[Dict]: Combinaciones generadas o lista vacía en caso de error
        """
        try:
            if model == 'consensus':
                return generar_combinaciones_consenso(self.data, self.cantidad_final)
            if model == 'lstm_v2':
                return generar_combinaciones_lstm(self.data.values, self.cantidad_final)
            if model == 'montecarlo':
                return generar_combinaciones_montecarlo(self.data.values.tolist(), self.cantidad_final)
            if model == 'apriori':
                return generar_combinaciones_apriori(self.data.values.tolist(), self.cantidad_final)
            if model == 'transformer_deep':
                return generar_combinaciones_transformer(self.data, self.cantidad_final)
            if model == 'clustering':
                return generar_combinaciones_clustering(self.data, self.cantidad_final)
            if model == 'genetico':
                return generar_combinaciones_geneticas(self.data, self.cantidad_final, GeneticConfig())
            return []
        except Exception as e:
            self.logger.error(f"🔥 Modelo {model} falló: {str(e)}", exc_info=True)
            return []

    def run_all_models(self) -> List[Dict[str, Any]]:
        """
        Ejecuta todos los modelos activos en paralelo y procesa resultados.
        
        Returns:
            List[Dict]: Combinaciones finales procesadas y ordenadas
        """
        # Selección de modelos activos
        models = [k for k, v in self.usar_modelos.items() if v]
        self.logger.info(f"🚀 Iniciando ejecución de modelos: {', '.join(models)}")
        
        # Configuración dinámica de paralelismo
        PoolExecutor = ProcessPoolExecutor if PSUTIL_AVAILABLE else ThreadPoolExecutor
        workers = os.cpu_count() // 2 if PSUTIL_AVAILABLE and os.cpu_count() > 1 else 4
        
        combos = []
        mem_before = psutil.virtual_memory().used if PSUTIL_AVAILABLE else 0
        
        with PoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self._run_model_wrapper, model): model for model in models}
            
            for future in as_completed(futures):
                model = futures[future]
                try:
                    result = future.result()
                    if result:
                        combos.extend(result)
                        self.logger.debug(f"✅ Modelo {model} generó {len(result)} combinaciones")
                    else:
                        fallbacks = [self._get_fallback_combo() for _ in range(3)]
                        combos.extend(fallbacks)
                        self.logger.warning(f"⚠️ Modelo {model} no devolvió datos. Usando 3 fallbacks")
                except Exception as e:
                    self.logger.error(f"🚨 Error procesando modelo {model}: {str(e)}", exc_info=True)
                    combos.extend([self._get_fallback_combo() for _ in range(3)])

        # Procesamiento de combinaciones
        valid_combos = []
        for c in combos:
            if 'combination' not in c:
                self.logger.warning("Combinación sin campo 'combination'. Omisión")
                continue
            try:
                cleaned = clean_combination(c["combination"], self.logger)
                valid_combos.append({**c, "combination": cleaned})
            except Exception as e:
                self.logger.error(f"Error limpiando combinación: {str(e)}")

        # Scoring y evaluación
        scored_combos = score_combinations(valid_combos, self.data)
        
        if self.usar_modelos.get('evaluador', True):
            ev = EvaluadorInteligente()
            for c in scored_combos:
                try:
                    c['metrics']['evaluador'] = ev.evaluar(c['combination'])
                except Exception as e:
                    c['metrics']['evaluador_error'] = str(e)
                    self.logger.warning(f"Evaluador falló para combinación: {str(e)}")

        # Filtrado estratégico
        filtered_combos = [
            c for c in scored_combos 
            if self.filtro.aplicar_filtros(c['combination'])[0] >= 0.7
        ]
        
        # Selección final
        final_combos = sorted(
            filtered_combos, 
            key=lambda x: x.get('score', 0), 
            reverse=True
        )[:self.cantidad_final]
        
        # Normalización de scores
        max_score = max(c['score'] for c in final_combos) or 1
        for c in final_combos:
            c['normalized'] = c['score'] / max_score

        # Exportación automática
        if self.auto_export:
            try:
                os.makedirs('results', exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'results/svi_{timestamp}_{os.getpid()}.csv'
                exportar_combinaciones_svi(final_combos, filename)
                self.logger.info(f"💾 Combinaciones exportadas a {filename}")
            except Exception as e:
                self.logger.error(f"📤 Error en exportación: {str(e)}", exc_info=True)

        # Limpieza de memoria con monitoreo
        if PSUTIL_AVAILABLE:
            mem_mid = psutil.virtual_memory().used
            self.logger.debug(f"🧠 Uso de memoria antes GC: {mem_mid / (1024**2):.2f} MB")
            
        gc.collect()
        
        if PSUTIL_AVAILABLE:
            mem_after = psutil.virtual_memory().used
            freed = (mem_mid - mem_after) / (1024**2)
            self.logger.debug(f"🧠 Memoria liberada: {freed:.2f} MB")

        self.jackpot_profiler.close()
        return final_combos