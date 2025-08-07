#OMEGA_PRO_AI_v10.1/core/predictor.py

import os
import re
import time
import json
import math
import random
import pandas as pd
import logging
import requests
import inspect
import traceback
import numpy as np
import shutil
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from core.consensus_engine import generar_combinaciones_consenso
from modules.lstm_v2 import generar_combinaciones_lstm_v2
from modules.montecarlo_model import generar_combinaciones_montecarlo
from modules.apriori_model import generar_combinaciones_apriori
from modules.transformer_model import generar_combinaciones_transformer
from modules.filters.rules_filter import FiltroEstrategico
from modules.filters.ghost_rng_generative import get_seeds
from modules.inverse_mining_engine import ejecutar_minado_inverso
from modules.score_dynamics import score_combinations, clean_combination
from modules.utils.exportador_resultados import exportar_combinaciones
from utils.viabilidad import calcular_svi
from modules.exporters.exportador_svi import exportar_combinaciones_svi
from modules.clustering_engine import generar_combinaciones_clustering
from modules.genetic_model import generar_combinaciones_geneticas, GeneticConfig  # Updated import
from modules.evaluation.evaluador_inteligente import EvaluadorInteligente
from modules.profiling.jackpot_profiler import JackpotProfiler
from modules.lstm_model import train_lstm_model, predict_next_combination
from modules.learning.gboost_jackpot_classifier import GBoostJackpotClassifier

# Configuración inicial de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("OmegaPredictor")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.debug("🔧 psutil not available, CPU monitoring disabled")

try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("⚠️ cryptography not available, authentication disabled")

class HybridOmegaPredictor:
    VALID_POSITIONS = {'B1', 'B2', 'B3', 'B4', 'B5', 'B6'}
    MIN_VALUE = 1
    MAX_VALUE = 40
    VALID_SVI_PROFILES = {'default', 'conservative', 'aggressive'}
    DEFAULT_DATA_PATHS = [
        "data/historial_kabala_github.csv",
        "backup/historial_kabala_github.csv",
        "https://raw.githubusercontent.com/omega-pro-ai/historial/main/historial_kabala_github.csv"
    ]
    
    def __init__(self, data_path: Optional[str] = None, cantidad_final: int = 30, 
                 historial_df: Optional[pd.DataFrame] = None, perfil_svi: str = 'default', 
                 logger: Optional[logging.Logger] = None):
        if cantidad_final <= 0:
            logger.warning(f"⚠️ Invalid cantidad_final: {cantidad_final}. Using 30")
            cantidad_final = 30
            
        self.data_path = data_path or self.DEFAULT_DATA_PATHS[0]
        self.cantidad_final = cantidad_final
        self.logger = logger if logger and isinstance(logger, logging.Logger) else logging.getLogger("OmegaPredictor")
        self.data = historial_df if historial_df is not None else self.cargar_datos()
        self.filtro = FiltroEstrategico()
        self.filtro.cargar_historial(self.data.values.tolist() if not self.data.empty else [])
        self.use_positional = True
        self.perfil_svi = perfil_svi if perfil_svi in self.VALID_SVI_PROFILES else 'default'
        self.auto_export = True
        self._internal_token = None
        self.log_level = 'INFO'
        self._cached_rng_seeds = None
        self.ghost_rng_params = {
            'max_seeds': 8,
            'cantidad_por_seed': 4,
            'training_mode': False
        }
        self.usar_modelos = {
            "ghost_rng": True,
            "inverse_mining": True,
            "svi": True,
            "lstm_v2": True,
            "montecarlo": True,
            "apriori": True,
            "transformer_deep": True,
            "clustering": True,
            "genetico": True
        }
        self.inverse_mining_params = {
            'boost_strategy': 'high_values',
            'penalize': [1, 2, 3],
            'focus_positions': ['B3', 'B5'],
            'count': 12
        }
        
        self.set_positional_analysis(True)
        self.set_ghost_rng_usage(True)
        self.set_svi_profile(perfil_svi)
        self.logger.info(f"✅ Predictor initialized with {self.data.shape[0]} historical draws")
    
    def set_inverse_mining_usage(self, enable: bool):
        self.usar_modelos["inverse_mining"] = enable
        self.logger.info(f"⚙️ Inverse mining {'enabled' if enable else 'disabled'}")

    def set_ghost_rng_params(self, max_seeds: int = 8, cantidad_por_seed: int = 4, training_mode: bool = False):
        if max_seeds <= 0 or cantidad_por_seed <= 0:
            self.logger.warning(f"⚠️ Invalid Ghost RNG params: max_seeds={max_seeds}, cantidad_por_seed={cantidad_por_seed}. Using defaults")
            max_seeds = 8
            cantidad_por_seed = 4
        self.ghost_rng_params = {
            'max_seeds': max_seeds,
            'cantidad_por_seed': cantidad_por_seed,
            'training_mode': training_mode
        }
        self.logger.info(f"⚙️ Ghost RNG params set: {self.ghost_rng_params}")

    def set_ghost_rng_usage(self, enable: bool):
        self.usar_modelos["ghost_rng"] = enable
        self.logger.info(f"⚙️ Ghost RNG {'enabled' if enable else 'disabled'}")

    def validate_data_path(self, path: str) -> Optional[str]:
        temp_path = None
        try:
            if path.startswith("http://") or path.startswith("https://"):
                self.logger.info(f"🌐 Downloading data from: {path}")
                session = requests.Session()
                retry_strategy = Retry(
                    total=3,
                    backoff_factor=1,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["GET"]
                )
                adapter = HTTPAdapter(max_retries=retry_strategy)
                session.mount("https://", adapter)
                session.mount("http://", adapter)
                
                response = session.get(path, timeout=15)
                response.raise_for_status()
                
                os.makedirs("temp", exist_ok=True)
                temp_path = f"temp/{os.path.basename(path)}"
                
                with open(temp_path, "wb") as f:
                    f.write(response.content)
                
                self.logger.info(f"📥 Data downloaded successfully: {temp_path}")
                return temp_path
            
            if os.path.exists(path):
                return path
                
            self.logger.warning(f"⚠️ File not found: {path}. Trying alternative paths...")
            for alt_path in self.DEFAULT_DATA_PATHS:
                if alt_path == path:
                    continue
                if os.path.exists(alt_path):
                    self.logger.info(f"📂 Using alternative path: {alt_path}")
                    return alt_path
                
            parent_path = os.path.join("..", path)
            if os.path.exists(parent_path):
                self.logger.info(f"📂 Using parent directory path: {parent_path}")
                return parent_path
                
            raise FileNotFoundError(f"🚨 Data file not found in any alternative path")
        
        except Exception as e:
            self.logger.error(f"🚨 Download error: {str(e)}")
            return None
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    self.logger.debug(f"🧹 Temporary file removed: {temp_path}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Error removing temporary file: {str(e)}")
    
    def set_logging_level(self, level: str = 'INFO'):
        valid_levels = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO, 'WARNING': logging.WARNING}
        if level not in valid_levels:
            self.logger.warning(f"⚠️ Invalid log level: {level}. Using INFO")
            level = 'INFO'
            
        self.log_level = level
        self.logger.setLevel(valid_levels[level])
        logging.getLogger().setLevel(valid_levels[level])
        self.logger.info(f"⚙️ Log level set to: {level}")
    
    def set_svi_profile(self, perfil: str):
        if perfil not in self.VALID_SVI_PROFILES:
            self.logger.warning(f"⚠️ Invalid SVI profile: {perfil}. Using 'default'")
            perfil = 'default'
            
        self.perfil_svi = perfil
        self.logger.info(f"⚙️ SVI profile set: {perfil}")
    
    def auto_select_svi_profile(self, sum_threshold_high: int = 120, sum_threshold_low: int = 80, 
                               std_dev_threshold_high: int = 15, std_dev_threshold_low: int = 10):
        if any(t <= 0 for t in [sum_threshold_high, sum_threshold_low, 
                                std_dev_threshold_high, std_dev_threshold_low]):
            self.logger.warning("⚠️ Invalid thresholds, using default values")
            sum_threshold_high, sum_threshold_low = 120, 80
            std_dev_threshold_high, std_dev_threshold_low = 15, 10
            
        if not hasattr(self, 'data') or self.data.empty:
            self.logger.warning("⚠️ No data for automatic SVI profile selection")
            return None
            
        sums = self.data.sum(axis=1)
        avg_sum = sums.mean()
        std_dev = sums.std()
        
        if avg_sum > sum_threshold_high and std_dev > std_dev_threshold_high:
            profile = 'aggressive'
        elif avg_sum < sum_threshold_low and std_dev < std_dev_threshold_low:
            profile = 'conservative'
        else:
            profile = 'default'
            
        self.logger.info(f"📊 Historical trend: Avg sum={avg_sum:.1f}, Std dev={std_dev:.1f}")
        self.set_svi_profile(profile)
        return profile
    
    def set_auto_export(self, enable: bool):
        self.auto_export = enable
        self.logger.info(f"⚙️ Auto-export {'enabled' if enable else 'disabled'}")
    
    def set_positional_analysis(self, flag: bool):
        self.use_positional = flag
        self.logger.info(f"⚙️ Positional analysis {'enabled' if flag else 'disabled'}")
        
        self.usar_modelos.update({
            "ghost_rng": True,
            "inverse_mining": True,
            "svi": True,
            "lstm_v2": True,
            "montecarlo": True,
            "apriori": True,
            "transformer_deep": True,
            "clustering": True,
            "genetico": True
        })
        
        self.inverse_mining_params = {
            'boost_strategy': 'high_values',
            'penalize': [1, 2, 3],
            'focus_positions': ['B3', 'B5'],
            'count': 12
        }

    def set_inverse_mining_params(self, boost_strategy: str = 'high_values', penalize: Optional[List[int]] = None, 
                                 focus_positions: Optional[List[str]] = None, count: int = 12):
        if boost_strategy not in ('high_values', 'last_value'):
            raise ValueError(f"🚨 Invalid boost strategy: {boost_strategy}")
        
        if focus_positions:
            invalid_positions = set(focus_positions) - self.VALID_POSITIONS
            if invalid_positions:
                raise ValueError(f"🚨 Invalid positions: {', '.join(invalid_positions)}")
        
        if count <= 0:
            raise ValueError(f"🚨 Count must be positive: {count}")
            
        if penalize:
            invalid_penalize = [x for x in penalize 
                               if not (isinstance(x, int) and 
                                       self.MIN_VALUE <= x <= self.MAX_VALUE)]
            if invalid_penalize:
                raise ValueError(f"🚨 Invalid penalized numbers: {invalid_penalize}")
        
        self.inverse_mining_params['boost_strategy'] = boost_strategy
        self.inverse_mining_params['penalize'] = penalize or [1, 2, 3]
        self.inverse_mining_params['focus_positions'] = focus_positions or ['B3', 'B5']
        self.inverse_mining_params['count'] = count
        
        self.logger.info("⚙️ Inverse mining parameters updated")
        self.logger.debug(f"🔧 New parameters: {self.inverse_mining_params}")

    def validar_columnas_bolillas(self, df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
        pattern = re.compile(r'^(bolilla[_\s]*\d+|B\d+)$', re.IGNORECASE)
        columnas_bolillas = [col for col in df.columns if pattern.match(col)]
        
        if len(columnas_bolillas) < 6:
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            if len(numeric_cols) >= 6:
                columnas_bolillas = numeric_cols[:6]
                self.logger.warning(f"⚠️ Using first 6 numeric columns as fallback: {columnas_bolillas}")
            else:
                raise ValueError(f"🚨 Valid ball columns not found (need 6), found: {columnas_bolillas}")
        
        return columnas_bolillas, df[columnas_bolillas].dropna()

    def filtrar_validos(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.astype(float)
        valid_mask = data.apply(lambda row: (
            len(row) == 6 and
            all(not math.isnan(x) and x.is_integer() and 
                (self.MIN_VALUE <= x <= self.MAX_VALUE) for x in row)
        ), axis=1)
        
        if not valid_mask.all():
            invalid_values = data[~valid_mask]
            self.logger.warning(f"🚨 Found {len(invalid_values)} rows with out-of-range or non-integer values")
            if self.logger.level <= logging.DEBUG:
                self.logger.debug(f"Invalid values sample:\n{invalid_values.head()}")
            data = data[valid_mask]
        
        if data.shape[0] < 50:
            self.logger.warning("⚠️ Insufficient valid rows after filtering")
        
        data = data.astype(int)
        return data

    def cargar_datos(self) -> pd.DataFrame:
        resolved_path = self.validate_data_path(self.data_path)
        if not resolved_path:
            raise FileNotFoundError(f"🚨 Could not find data file: {self.data_path}")
        
        try:
            file_size = os.path.getsize(resolved_path) if os.path.exists(resolved_path) else 0
            chunk_size = None
            
            if file_size > 50 * 1024 * 1024:
                chunk_size = 500
            elif file_size > 10 * 1024 * 1024:
                chunk_size = 1000
            elif file_size > 5 * 1024 * 1024:
                chunk_size = 2000
                
            if chunk_size:
                self.logger.info(f"📦 Loading data in chunks (size: {chunk_size})...")
                chunks = []
                
                for chunk in pd.read_csv(resolved_path, chunksize=chunk_size, encoding="utf-8", delimiter=","):
                    try:
                        columnas_bolillas, chunk_data = self.validar_columnas_bolillas(chunk)
                        if len(columnas_bolillas) >= 6:
                            chunks.append(chunk_data)
                        else:
                            self.logger.warning(f"⚠️ Chunk has insufficient ball columns: {columnas_bolillas}")
                    except Exception as e:
                        self.logger.error(f"🚨 Error processing chunk: {str(e)}")
                
                if not chunks:
                    raise ValueError("🚨 No valid chunks after processing")
                    
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(resolved_path, encoding="utf-8", delimiter=",")
                columnas_bolillas, df = self.validar_columnas_bolillas(df)
            
            df = df.select_dtypes(include='number')
            for col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except Exception:
                        self.logger.warning(f"⚠️ Column '{col}' contains non-numeric values")
            
            df = self.filtrar_validos(df)
            
            # Added debug logs for data validation
            self.logger.debug(f"📊 Raw data sample:\n{df.head()}")
            self.logger.debug(f"🔍 Filtered data sample (valid numbers only):\n{df.head()}")
            self.logger.debug(f"📋 Data types:\n{df.dtypes}")
            
            if df.shape[0] < 50:
                raise ValueError("⚠️ Insufficient history: less than 50 valid draws")
            
            self.logger.info(f"📊 Data loaded: {df.shape[0]} draws")
            self.logger.debug(f"Data sample:\n{df.head()}")
            return df
            
        except Exception as e:
            self.logger.error(f"🚨 Error loading data: {str(e)}")
            self.logger.debug(f"📂 File: {resolved_path}")
            raise

    def aplicar_ghost_rng(self, resultados_rng: List[Dict], cantidad_por_seed: int = 4) -> List[Dict]:
        combinaciones = []
        
        if not resultados_rng or not self.usar_modelos["ghost_rng"]:
            self.logger.info("⚠️ Ghost RNG skipped: No results or disabled")
            return combinaciones
        
        if not isinstance(resultados_rng, list):
            self.logger.warning("⚠️ resultados_rng must be a list")
            return combinaciones
        
        try:
            self.logger.info("⚙️ Processing Ghost RNG results...")
            
            valid_seeds = []
            for r in resultados_rng:
                if not isinstance(r, dict):
                    self.logger.warning(f"⚠️ Invalid RNG entry (not dict): {type(r)}")
                    continue
                if 'seed' not in r or 'draw' not in r:
                    self.logger.warning(f"⚠️ Incomplete RNG seed: {r}")
                    continue
                if 'composite_score' not in r:
                    self.logger.info(f"Adding default composite_score to seed: {r['seed']}")
                    r['composite_score'] = 1.0
                valid_seeds.append(r)
            
            if not valid_seeds:
                self.logger.warning("⚠️ No valid RNG seeds")
                return combinaciones
            
            max_seeds = self.ghost_rng_params['max_seeds']
            top_seeds = sorted(valid_seeds, key=lambda x: x["composite_score"], reverse=True)[:max_seeds]
            
            seed_details = [f"{s['seed']} ({s['composite_score']:.2f})" for s in top_seeds]
            self.logger.debug(f"🔝 Top {len(top_seeds)} seeds: {', '.join(seed_details)}")
            
            for r in top_seeds:
                clean_draw = clean_combination(r["draw"], self.logger)
                if not clean_draw or len(clean_draw) != 6:
                    self.logger.warning(f"⚠️ Invalid Ghost RNG combination: {r['draw']} (after cleaning: {clean_draw})")
                    continue
                    
                boosted_score = r["composite_score"] * 1.10
                combinaciones.append({
                    "combination": clean_draw,
                    "source": "ghost_rng",
                    "score": boosted_score,
                    "ghost_ok": True,
                    "seed": r["seed"],
                    "metrics": {"composite_score": r["composite_score"]},
                    "normalized": 0.0
                })
            
            self.logger.info(f"📊 Generated {len(combinaciones)} Ghost RNG combinations")
            return combinaciones
        
        except Exception as e:
            self.logger.error(f"🚨 Critical error in Ghost RNG: {str(e)}")
            traceback.print_exc()
            return []

    def aplicar_minado_inverso(self, ultima_combinacion: List[int]) -> List[Dict]:
        if not self.usar_modelos["inverse_mining"]:
            return []
        
        clean_ultima = clean_combination(ultima_combinacion, self.logger)
        if not clean_ultima or len(clean_ultima) != 6:
            self.logger.warning(f"⚠️ Invalid seed combination: {ultima_combinacion} (after cleaning: {clean_ultima})")
            return []
        
        try:
            self.logger.info("🔍 Running inverse mining...")
            self.logger.debug(f"⚙️ Mining params: {self.inverse_mining_params}")
            self.logger.debug(f"🎯 Seed combination: {clean_ultima}")
            
            if self.inverse_mining_params['boost_strategy'] == 'high_values':
                boost = [n for n in clean_ultima if n > 20]
            else:
                boost = [clean_ultima[-1]]
                
            self.logger.debug(f"🚀 Applied boost: {boost}")
            
            minado = ejecutar_minado_inverso(
                seed=clean_ultima,
                boost=boost,
                penalize=self.inverse_mining_params['penalize'],
                focus_positions=self.inverse_mining_params['focus_positions'],
                count=self.inverse_mining_params['count'],
                historial_df=self.data,
                mostrar=False
            )
            
            bonus = 1.15 if self.inverse_mining_params['boost_strategy'] == 'high_values' else 1.10
            result = []
            invalid_count = 0
            
            for item in minado:
                combo = item.get("combination")
                cleaned_combination = clean_combination(combo, self.logger) if combo else []
                
                if not cleaned_combination or len(cleaned_combination) != 6:
                    invalid_count += 1
                    continue
                    
                result.append({
                    "combination": cleaned_combination,
                    "source": "inverse_mining",
                    "score": item["score"] * bonus,
                    "ghost_ok": False,
                    "metrics": {"minado_score": item["score"]},
                    "normalized": 0.0
                })
            
            if invalid_count:
                self.logger.warning(f"⚠️ {invalid_count} invalid combinations skipped")
            
            self.logger.info(f"📊 Inverse mining completed: {len(result)} combinations")
            if result:
                top_score = max(item['score'] for item in result)
                self.logger.debug(f"🏆 Max score: {top_score:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"🚨 Error in inverse mining: {str(e)}")
            self.logger.debug(f"🔧 Seed combination: {clean_ultima}")
            traceback.print_exc()
            return []

    def _obtener_token(self) -> str:
        if not CRYPTO_AVAILABLE:
            self.logger.warning("⚠️ cryptography not available, using simulated authentication")
            return "simulated_token"
            
        if self._internal_token:
            return self._internal_token
            
        auth_key_path = "logs/auth_key.key"
        if not os.path.exists(auth_key_path):
            raise FileNotFoundError("🚨 Authentication file not found")
            
        try:
            with open(auth_key_path, "rb") as f:
                key = f.read()
            fernet = Fernet(key)
            self._internal_token = fernet.encrypt(b"usuario_autorizado").decode()
            self.logger.debug("🔧 Authentication token generated")
            return self._internal_token
        except Exception as e:
            self.logger.error(f"🚨 Token generation error: {str(e)}")
            raise

    def _map_svi_profile(self, profile: str) -> str:
        mapping = {
            'default': 'moderado',
            'conservative': 'conservador',
            'aggressive': 'agresivo'
        }
        mapped_profile = mapping.get(profile, 'moderado')
        self.logger.debug(f"🔧 Mapped SVI profile {profile} to {mapped_profile}")
        return mapped_profile

    def _calcular_svi_individual(self, combo: Dict) -> Tuple[Dict, float]:
        """Helper para cálculo paralelo de SVI"""
        perfil_rng = self._map_svi_profile(self.perfil_svi)
        combo_str = str(combo["combination"])
        svi_score = calcular_svi(
            combinacion=combo_str,
            perfil_rng=perfil_rng,
            validacion_ghost=combo.get("ghost_ok", False),
            score_historico=combo.get("score", 1.0)
        )
        return combo, svi_score

    def filtrar_combinaciones(self, combinaciones: List[Dict], token_externo: Optional[str] = None) -> List[Dict]:
        try:
            token = token_externo or self._obtener_token()
            self.logger.debug("🔧 Authentication token ready")
            
            if not combinaciones:
                self.logger.warning("⚠️ No combinations provided for filtering")
                return []
            
            final = []
            invalid_count = 0
            perfil_svi = self._map_svi_profile(self.perfil_svi)
            
            for idx, item in enumerate(combinaciones):
                combo = item.get("combination")
                clean_combo = clean_combination(combo, self.logger) if combo else []
                
                if not clean_combo or len(clean_combo) != 6:
                    self.logger.warning(f"⚠️ Invalid combination at index {idx}: {combo} (after cleaning: {clean_combo})")
                    invalid_count += 1
                    continue
                
                try:
                    score, razones = self.filtro.aplicar_filtros(
                        clean_combo,
                        return_score=True,
                        perfil_svi=perfil_svi
                    )
                    threshold = {'moderado': 0.7, 'conservador': 0.8, 'agresivo': 0.4}.get(perfil_svi, 0.7)
                    if score >= threshold:
                        item["score"] = item.get("score", 1.0) * score
                        item["filtro_razones"] = razones
                        item["combination"] = clean_combo
                        item["metrics"] = item.get("metrics", {})
                        item["metrics"]["filtro_score"] = score
                        item["normalized"] = 0.0
                        final.append(item)
                    else:
                        self.logger.debug(f"🧹 Combination rejected: {clean_combo}, Score: {score:.3f}, Reasons: {razones}")
                except Exception as e:
                    self.logger.error(f"🚨 Filter application error: {str(e)}")
                    self.logger.debug(f"🔧 Combination: {clean_combo}")
            
            if invalid_count:
                self.logger.warning(f"⚠️ {invalid_count} invalid combinations skipped")
                
            self.logger.info(f"🔍 Filtering completed: {len(final)}/{len(combinaciones)} approved, threshold: {threshold}")
            if final:
                self.logger.debug(f"✅ Approved example: {final[0]['combination']}, Score: {final[0]['score']:.3f}")
            else:
                self.logger.warning("⚠️ No combinations passed filtering")
            
            return final
            
        except Exception as e:
            self.logger.error(f"🚨 Critical filtering error: {str(e)}")
            traceback.print_exc()
            return []

    def calcular_svi_para_combinaciones(self, combinaciones: List[Dict]) -> List[Dict]:
        if not self.usar_modelos["svi"]:
            return combinaciones
            
        try:
            self.logger.info("📈 Calculating strategic viability (SVI)...")
            start_time = time.time()
            processed_count = 0
            error_count = 0
            
            valid_combinations = []
            invalid_indices = []
            
            for idx, combo in enumerate(combinaciones):
                if not isinstance(combo, dict):
                    self.logger.warning(f"⚠️ Invalid combo type at index {idx}: {type(combo)}")
                    invalid_indices.append(idx)
                    continue
                    
                combination = combo.get("combination")
                cleaned_combination = clean_combination(combination, self.logger) if combination else []
                
                if not cleaned_combination or len(cleaned_combination) != 6:
                    self.logger.warning(f"⚠️ Invalid combination at index {idx}: {combination} (after cleaning: {cleaned_combination})")
                    invalid_indices.append(idx)
                    continue
                    
                combo["combination"] = cleaned_combination
                valid_combinations.append(combo)
            
            if invalid_indices:
                self.logger.warning(f"🧹 Removed {len(invalid_indices)} invalid combinations before SVI processing")
                if self.logger.level <= logging.DEBUG:
                    for idx in invalid_indices[:5]:
                        self.logger.debug(f"  - Index {idx}: {combinaciones[idx]}")
                    if len(invalid_indices) > 5:
                        self.logger.debug(f"🧹 ... and {len(invalid_indices)-5} additional invalid")
            
            if not valid_combinations:
                self.logger.error("🚨 No valid combinations for SVI calculation")
                return []
            
            try:
                perfil_rng = self._map_svi_profile(self.perfil_svi)
                
                results = []
                for combo in valid_combinations:
                    combo_str = str(combo["combination"])
                    svi_score = calcular_svi(
                        combinacion=combo_str,
                        perfil_rng=perfil_rng,
                        validacion_ghost=combo.get("ghost_ok", False),
                        score_historico=combo.get("score", 1.0)
                    )
                    combo["svi_score"] = svi_score
                    combo["score"] = combo.get("score", 1.0) * svi_score
                    combo["metrics"] = combo.get("metrics", {})
                    combo["metrics"]["svi_score"] = svi_score
                    combo["normalized"] = 0.0
                    results.append(combo)
                    processed_count += 1
                
                self.logger.debug("📊 SVI calculation completed for individual combinations")
                
            except (TypeError, ValueError) as e:
                self.logger.warning(f"⚠️ SVI calculation failed, using parallel fallback: {str(e)}")
                
                if PSUTIL_AVAILABLE:
                    cpu_before = psutil.cpu_percent(interval=None)
                
                with ThreadPoolExecutor() as executor:
                    futures = {executor.submit(self._calcular_svi_individual, c): c 
                              for c in valid_combinations}
                    
                    for future in as_completed(futures):
                        try:
                            combo, svi_score = future.result()
                            combo["score"] = combo.get("score", 1.0) * svi_score
                            combo["svi_score"] = svi_score
                            combo["metrics"] = combo.get("metrics", {})
                            combo["metrics"]["svi_score"] = svi_score
                            combo["normalized"] = 0.0
                            processed_count += 1
                            results.append(combo)
                        except Exception as e:
                            error_count += 1
                            self.logger.error(f"❌ Parallel SVI error: {str(e)}")
                
                if PSUTIL_AVAILABLE:
                    cpu_after = psutil.cpu_percent(interval=None)
                    self.logger.debug(f"🔧 CPU usage during parallel SVI: {max(cpu_before, cpu_after)}%")
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"📊 SVI calculation completed in {elapsed_time:.2f} seconds")
            self.logger.info(f"  → Processed combinations: {processed_count}")
            self.logger.info(f"  → Errors: {error_count}")
            
            if processed_count == 0:
                self.logger.warning("⚠️ No combinations processed successfully, returning empty list")
                return []
            
            return results
            
        except Exception as e:
            self.logger.error(f"🚨 Critical SVI error: {str(e)}")
            traceback.print_exc()
            return []

    def load_svi_config(self, config_path: str = "config/svi_config.json"):
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                self.auto_select_svi_profile(
                    sum_threshold_high=config.get("sum_threshold_high", 120),
                    sum_threshold_low=config.get("sum_threshold_low", 80),
                    std_dev_threshold_high=config.get("std_dev_threshold_high", 15),
                    std_dev_threshold_low=config.get("std_dev_threshold_low", 10)
                )
                self.logger.info(f"⚙️ SVI config loaded from {config_path}")
            except Exception as e:
                self.logger.error(f"🚨 Error loading SVI config: {str(e)}")
        else:
            self.logger.warning(f"⚠️ SVI config file not found: {config_path}")

    def generate_source_score_chart(self, combinaciones: List[Dict]) -> dict:
        source_scores = {}
        for combo in combinaciones:
            source = combo.get("source", "unknown")
            if source not in source_scores:
                source_scores[source] = []
            score = combo.get("score", 0)
            if isinstance(score, (int, float)):
                source_scores[source].append(score)
        
        avg_scores = {}
        for source, scores in source_scores.items():
            avg_scores[source] = sum(scores) / len(scores) if scores else 0
        
        labels = list(avg_scores.keys())
        data = list(avg_scores.values())
        
        color_map = {
            "consensus": "rgba(54, 162, 235, 0.5)",
            "ghost_rng": "rgba(255, 99, 132, 0.5)",
            "inverse_mining": "rgba(75, 192, 192, 0.5)",
            "lstm_v2": "rgba(255, 206, 86, 0.5)",
            "montecarlo": "rgba(153, 102, 255, 0.5)",
            "apriori": "rgba(255, 159, 64, 0.5)",
            "transformer_deep": "rgba(75, 192, 192, 0.5)",
            "clustering": "rgba(100, 100, 100, 0.5)",
            "genetico": "rgba(200, 100, 100, 0.5)",
            "unknown": "rgba(199, 199, 199, 0.5)"
        }
        
        background_colors = [color_map.get(source, "rgba(199, 199, 199, 0.5)") for source in labels]
        border_colors = [color.replace("0.5", "1.0") for color in background_colors]
        
        return {
            "type": "bar",
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": "Average Score",
                    "data": data,
                    "backgroundColor": background_colors,
                    "borderColor": border_colors,
                    "borderWidth": 1
                }]
            },
            "options": {
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "title": {"display": True, "text": "Average Score"}
                    }
                },
                "plugins": {
                    "title": {"display": True, "text": "Average Scores by Source"}
                }
            }
        }

    def generate_source_count_chart(self, combinaciones: List[Dict]) -> dict:
        source_counts = {}
        for combo in combinaciones:
            source = combo.get("source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
        
        labels = list(source_counts.keys())
        data = list(source_counts.values())
        
        color_map = {
            "consensus": "rgba(54, 162, 235, 0.5)",
            "ghost_rng": "rgba(255, 99, 132, 0.5)",
            "inverse_mining": "rgba(75, 192, 192, 0.5)",
            "lstm_v2": "rgba(255, 206, 86, 0.5)",
            "montecarlo": "rgba(153, 102, 255, 0.5)",
            "apriori": "rgba(255, 159, 64, 0.5)",
            "transformer_deep": "rgba(75, 192, 192, 0.5)",
            "clustering": "rgba(100, 100, 100, 0.5)",
            "genetico": "rgba(200, 100, 100, 0.5)",
            "unknown": "rgba(199, 199, 199, 0.5)"
        }
        
        background_colors = [color_map.get(source, "rgba(199, 199, 199, 0.5)") for source in labels]
        border_colors = [color.replace("0.5", "1.0") for color in background_colors]
        
        return {
            "type": "bar",
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": "Number of Combinations",
                    "data": data,
                    "backgroundColor": background_colors,
                    "borderColor": border_colors,
                    "borderWidth": 1
                }]
            },
            "options": {
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "title": {"display: True, text": "Number of Combinations"}
                    }
                },
                "plugins": {
                    "title": {"display": True, "text": "Combinations by Source"}
                }
            }
        }

    def export_chart(self, chart_name: str, chart_type: str, chart_data: dict, output_dir: str = "results"):
        os.makedirs(output_dir, exist_ok=True)
        chart_path = os.path.join(output_dir, f"{chart_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(chart_path, "w", encoding="utf-8") as f:
            json.dump(chart_data, f, indent=2)
        self.logger.info(f"📊 Chart exported: {chart_path}")

    def _run_consensus(self, max_comb: int) -> List[Dict]:
        try:
            consensus_combinations = generar_combinaciones_consenso(
                historial_df=self.data,
                cantidad=max_comb,
                perfil_svi=self.perfil_svi,
                logger=self.logger
            )
            results = []
            for combo in consensus_combinations:
                clean_combo = clean_combination(combo["combination"], self.logger)
                if not clean_combo or len(clean_combo) != 6:
                    self.logger.warning(f"⚠️ Invalid consensus combination: {combo['combination']}")
                    continue
                results.append({
                    "combination": clean_combo,
                    "source": "consensus",
                    "score": combo.get("score", 1.0),
                    "metrics": combo.get("metrics", {}),
                    "normalized": 0.0
                })
            self.logger.info(f"✅ Generated {len(results)} consensus combinations")
            return results
        except Exception as e:
            self.logger.error(f"🚨 Consensus failed: {str(e)}")
            self.logger.debug(f"Stack trace: {traceback.format_exc()}")
            return []

    def _run_lstm(self, max_comb: int) -> List[Dict]:
        try:
            perfil_svi_mapped = self._map_svi_profile(self.perfil_svi)
            lstm_combinations = generar_combinaciones_lstm_v2(
                data_path=self.data_path,
                cantidad=max_comb,
                perfil_svi=perfil_svi_mapped,
                historial_df=self.data,
                logger=self.logger
            )
            results = []
            invalid_count = 0
            for combo in lstm_combinations:
                clean_combo = clean_combination(combo["combination"], self.logger)
                if not clean_combo or len(clean_combo) != 6:
                    self.logger.warning(f"⚠️ Invalid LSTM combination: {combo['combination']}")
                    invalid_count += 1
                    continue
                results.append({
                    "combination": clean_combo,
                    "source": "lstm_v2",
                    "score": combo.get("score", 1.0),
                    "metrics": combo.get("metrics", {}),
                    "normalized": 0.0
                })
            self.logger.info(f"✅ Generated {len(results)} LSTM combinations, {invalid_count} invalid skipped")
            return results
        except Exception as e:
            self.logger.error(f"🚨 LSTM failed: {str(e)}")
            self.logger.debug(f"Stack trace: {traceback.format_exc()}")
            return []

    def _run_montecarlo(self, max_comb: int) -> List[Dict]:
        try:
            historial_list = self.data.values.tolist() if not self.data.empty else []
            montecarlo_combinations = generar_combinaciones_montecarlo(
                historial=historial_list,
                cantidad=max_comb,
                logger=self.logger
            )
            results = []
            invalid_count = 0
            for combo in montecarlo_combinations:
                clean_combo = clean_combination(combo["combination"], self.logger)
                if not clean_combo or len(clean_combo) != 6:
                    self.logger.warning(f"⚠️ Invalid Monte Carlo combination: {combo['combination']}")
                    invalid_count += 1
                    continue
                results.append({
                    "combination": clean_combo,
                    "source": combo["source"],
                    "score": combo.get("score", 1.0),
                    "metrics": combo.get("metrics", {}),
                    "normalized": 0.0
                })
            self.logger.info(f"✅ Generated {len(results)} Monte Carlo combinations, {invalid_count} invalid skipped")
            return results
        except Exception as e:
            self.logger.error(f"🚨 Monte Carlo failed: {str(e)}")
            self.logger.debug(f"Stack trace: {traceback.format_exc()}")
            return []

    def _run_apriori(self, max_comb: int) -> List[Dict]:
        try:
            historial_list = self.data.values.tolist() if not self.data.empty else []
            historial_set = {tuple(sorted(c)) for c in historial_list}
            
            # Updated Apriori call with num_predictions
            self.logger.debug(f"🔍 Apriori function signature: {inspect.signature(generar_combinaciones_apriori)}")
            apriori_combinations = generar_combinaciones_apriori(
                data=historial_list,
                historial_set=historial_set,
                num_predictions=max_comb,  # Changed from cantidad
                logger=self.logger
            )
            
            results = []
            invalid_count = 0
            for combo in apriori_combinations:
                clean_combo = clean_combination(combo["combination"], self.logger)
                if not clean_combo or len(clean_combo) != 6:
                    self.logger.warning(f"⚠️ Invalid Apriori combination: {combo['combination']}")
                    invalid_count += 1
                    continue
                results.append({
                    "combination": clean_combo,
                    "source": combo["source"],
                    "score": combo.get("score", 1.0),
                    "metrics": combo.get("metrics", {}),
                    "normalized": 0.0
                })
            self.logger.info(f"✅ Generated {len(results)} Apriori combinations, {invalid_count} invalid skipped")
            return results
        except Exception as e:
            self.logger.error(f"🚨 Apriori failed: {str(e)}")
            self.logger.debug(f"Stack trace: {traceback.format_exc()}")
            return []

    def _run_transformer(self, max_comb: int) -> List[Dict]:
        try:
            perfil_svi_mapped = self._map_svi_profile(self.perfil_svi)
            transformer_combinations = generar_combinaciones_transformer(
                historial_df=self.data,
                cantidad=max_comb,
                perfil_svi=perfil_svi_mapped,
                logger=self.logger
            )
            results = []
            invalid_count = 0
            for combo in transformer_combinations:
                clean_combo = clean_combination(combo["combination"], self.logger)
                if not clean_combo or len(clean_combo) != 6:
                    self.logger.warning(f"⚠️ Invalid Transformer combination: {combo['combination']}")
                    invalid_count += 1
                    continue
                results.append({
                    "combination": clean_combo,
                    "source": "transformer_deep",
                    "score": combo.get("score", 1.0),
                    "metrics": combo.get("metrics", {}),
                    "normalized": 0.0
                })
            self.logger.info(f"✅ Generated {len(results)} Transformer combinations, {invalid_count} invalid skipped")
            return results
        except Exception as e:
            self.logger.error(f"🚨 Transformer failed: {str(e)}")
            self.logger.debug(f"Stack trace: {traceback.format_exc()}")
            return []

    def _run_clustering(self, max_comb: int) -> List[Dict]:
        try:
            # CORRECCIÓN APLICADA: Usamos historial_df en lugar de data
            clustering_combinations = generar_combinaciones_clustering(
                historial_df=self.data,  # Parámetro corregido
                cantidad=max_comb,
                logger=self.logger
            )
            results = []
            invalid_count = 0
            for combo in clustering_combinations:
                clean_combo = clean_combination(combo["combination"], self.logger)
                if not clean_combo or len(clean_combo) != 6:
                    self.logger.warning(f"⚠️ Invalid Clustering combination: {combo['combination']}")
                    invalid_count += 1
                    continue
                results.append({
                    "combination": clean_combo,
                    "source": "clustering",
                    "score": combo.get("score", 1.0),
                    "metrics": combo.get("metrics", {}),
                    "normalized": 0.0
                })
            self.logger.info(f"✅ Generated {len(results)} Clustering combinations, {invalid_count} invalid skipped")
            return results
        except Exception as e:
            self.logger.error(f"🚨 Clustering failed: {str(e)}")
            self.logger.debug(f"Stack trace: {traceback.format_exc()}")
            return []

    def _run_genetico(self, max_comb: int) -> List[Dict]:
        try:
            historial_list = self.data.values.tolist() if not self.data.empty else []
            historial_set = {tuple(sorted(c)) for c in historial_list}
            genetico_combinations = generar_combinaciones_geneticas(
                data=self.data,
                historial_set=historial_set,
                cantidad=max_comb,
                logger=self.logger,
                config=None  # Use default config from genetic_model.py
            )
            results = []
            invalid_count = 0
            for combo in genetico_combinations:
                clean_combo = clean_combination(combo["combination"], self.logger)
                if not clean_combo or len(clean_combo) != 6:
                    self.logger.warning(f"⚠️ Invalid Genetico combination: {combo['combination']}")
                    invalid_count += 1
                    continue
                results.append({
                    "combination": clean_combo,
                    "source": "genetico",
                    "score": combo.get("score", combo.get("fitness", 1.0) / 100),  # Convert fitness to score
                    "metrics": combo.get("metrics", {"genetic_fitness": combo.get("fitness", 0)}),
                    "normalized": 0.0
                })
            self.logger.info(f"✅ Generated {len(results)} Genetico combinations, {invalid_count} invalid skipped")
            return results
        except Exception as e:
            self.logger.error(f"🚨 Genetico failed: {str(e)}")
            self.logger.debug(f"Stack trace: {traceback.format_exc()}")
            return []

    def run_all_models(self) -> List[Dict]:
        try:
            self.logger.info("🚀 Starting prediction pipeline...")
            max_comb = 500
            if PSUTIL_AVAILABLE:
                available_memory = psutil.virtual_memory().available / 1024 / 1024
                max_comb = min(500, int(available_memory / 10))
                self.logger.debug(f"🔧 Adjusted max_comb to {max_comb} based on {available_memory:.2f} MB available")
            
            models = [
                ("consensus", self._run_consensus, max_comb // 8),
                ("lstm_v2", self._run_lstm, max_comb // 8),
                ("montecarlo", self._run_montecarlo, max_comb // 8),
                ("apriori", self._run_apriori, max_comb // 8),
                ("transformer_deep", self._run_transformer, max_comb // 8),
                ("clustering", self._run_clustering, max_comb // 8),
                ("genetico", self._run_genetico, max_comb // 8)
            ]
            
            combinaciones = []
            perfil_svi_mapped = self._map_svi_profile(self.perfil_svi)
            
            for name, func, count in models:
                if self.usar_modelos.get(name, True):
                    try:
                        self.logger.info(f"⚙️ Ejecutando modelo: {name}")
                        results = func(count)
                        combinaciones.extend(results)
                    except Exception as e:
                        self.logger.error(f"🚨 {name} failed: {str(e)}")
                        self.logger.debug(f"Stack trace: {traceback.format_exc()}")
            
            ghost_combinations = []
            if self.usar_modelos.get("ghost_rng", True):
                self.logger.info("🌐 Generating Ghost RNG combinations...")
                try:
                    if self._cached_rng_seeds is None:
                        valid_profiles = ['moderado', 'conservador', 'agresivo']
                        if perfil_svi_mapped not in valid_profiles:
                            self.logger.warning(f"⚠️ Unsupported Ghost RNG profile: {perfil_svi_mapped}. Using 'moderado'")
                            perfil_svi_mapped = 'moderado'
                            
                        self._cached_rng_seeds = get_seeds(
                            historial_csv_path=self.data_path,
                            perfil_svi=perfil_svi_mapped,
                            max_seeds=self.ghost_rng_params['max_seeds'],
                            training_mode=self.ghost_rng_params['training_mode']
                        )
                    rng_seeds = self._cached_rng_seeds
                    
                    assert isinstance(rng_seeds, list), (
                        f"Ghost RNG seeds must be list, got {type(rng_seeds)}"
                    )
                    
                    if not rng_seeds:
                        self.logger.warning("⚠️ No seeds generated by get_seeds, skipping Ghost RNG")
                    else:
                        valid_seeds = []
                        for seed in rng_seeds:
                            if not isinstance(seed, dict):
                                self.logger.warning(f"⚠️ Invalid seed (not dict): {seed}")
                                continue
                            if 'seed' not in seed or 'draw' not in seed:
                                self.logger.warning(f"⚠️ Incomplete seed: {seed}")
                                continue
                            if 'composite_score' not in seed:
                                self.logger.info(f"Adding default composite_score to seed: {seed['seed']}")
                                seed['composite_score'] = 1.0
                            valid_seeds.append(seed)
                        rng_seeds = valid_seeds
                        
                        ghost_combinations = self.aplicar_ghost_rng(
                            resultados_rng=rng_seeds,
                            cantidad_por_seed=self.ghost_rng_params['cantidad_por_seed']
                        )
                        combinaciones.extend(ghost_combinations)
                        self.logger.info(f"✅ Generated {len(ghost_combinations)} Ghost RNG combinations")
                except Exception as e:
                    self.logger.error(f"🚨 Error generating Ghost RNG combinations: {str(e)}")
                    self.logger.debug(f"Stack trace: {traceback.format_exc()}")
            
            inverse_combinations = []
            if self.usar_modelos.get("inverse_mining", True):
                self.logger.info("🔍 Generating inverse mining combinations...")
                try:
                    historial_list = self.data.values.tolist() if not self.data.empty else []
                    self.logger.debug(f"Historial list sample: {historial_list[:3]}")
                    last_draw = historial_list[-1] if historial_list else [1, 2, 3, 4, 5, 6]
                    inverse_combinations = self.aplicar_minado_inverso(last_draw)
                    combinaciones.extend(inverse_combinations)
                    self.logger.info(f"✅ Generated {len(inverse_combinations)} inverse mining combinations")
                except Exception as e:
                    self.logger.error(f"🚨 Error generating inverse mining combinations: {str(e)}")
                    self.logger.debug(f"Stack trace: {traceback.format_exc()}")
            
            self.logger.info(f"🔄 Total combinations before processing: {len(combinaciones)}")
            
            if not combinaciones:
                self.logger.warning("⚠️ No valid combinations generated, returning fallback combination")
                return [{
                    "combination": sorted(random.sample(range(1, 41), 6)),
                    "source": "fallback",
                    "score": 0.5,
                    "svi_score": 0.5,
                    "metrics": {},
                    "normalized": 0.0
                }]
            
            combinaciones = self.calcular_svi_para_combinaciones(combinaciones)
            
            if combinaciones:
                try:
                    self.logger.info("📈 Applying dynamic scoring...")
                    source_map = {
                        "consensus": "default",
                        "ghost_rng": "ghost_rng",
                        "inverse_mining": "inverse_mining",
                        "lstm_v2": "lstm_v2",
                        "montecarlo": "montecarlo",
                        "montecarlo_backup": "montecarlo",
                        "apriori": "apriori",
                        "apriori_backup": "apriori",
                        "transformer_deep": "transformer_deep",
                        "clustering": "clustering",
                        "genetico": "genetico"
                    }
                    for combo in combinaciones:
                        combo["source"] = source_map.get(combo["source"], "default")
                    
                    # Enhanced cleaning and deduplication
                    cleaned_combinaciones = []
                    seen_combinations = set()
                    for combo in combinaciones:
                        clean_combo = clean_combination(combo["combination"], self.logger)
                        if not clean_combo or len(clean_combo) != 6:
                            self.logger.warning(f"⚠️ Invalid combination skipped: {combo['combination']}")
                            continue
                        
                        combo_tuple = tuple(sorted(clean_combo))
                        if combo_tuple in seen_combinations:
                            self.logger.debug(f"⚠️ Duplicate combination skipped: {clean_combo}")
                            continue
                        
                        seen_combinations.add(combo_tuple)
                        combo["combination"] = clean_combo
                        cleaned_combinaciones.append(combo)
                    
                    if not cleaned_combinaciones:
                        self.logger.warning("⚠️ No valid combinations after cleaning, returning fallback")
                        return [{
                            "combination": sorted(random.sample(range(1, 41), 6)),
                            "source": "fallback",
                            "score": 0.5,
                            "svi_score": 0.5,
                            "metrics": {},
                            "normalized": 0.0
                        }]
                    
                    # Added debug logs for data validation
                    self.logger.debug(f"🔧 First 5 combinations for scoring: {[c['combination'] for c in cleaned_combinaciones[:5]]}")
                    self.logger.debug(f"📋 Historial data sample:\n{self.data.head()}")
                    
                    # Score combinations with error handling
                    self.logger.info(f"⭐ Scoring {len(cleaned_combinaciones)} combinations")
                    try:
                        scored_combinations = score_combinations(
                            combinations=cleaned_combinaciones,
                            historial=self.data,
                            cluster_data=None,
                            perfil_svi=perfil_svi_mapped,
                            logger=self.logger
                        )
                    except Exception as e:
                        self.logger.error(f"🚨 Scoring failed: {str(e)}")
                        self.logger.debug(f"Stack trace: {traceback.format_exc()}")
                        # Fallback to unscored combinations
                        return cleaned_combinaciones
                    
                    for scored_combo in scored_combinations:
                        svi_score = scored_combo.get("svi_score", 0.5)
                        dynamic_score = scored_combo.get("score", 0)
                        combined_score = (0.6 * svi_score) + (0.4 * (dynamic_score / 5.0))
                        scored_combo["score"] = combined_score
                        scored_combo["dynamic_score"] = dynamic_score
                        scored_combo["metrics"] = scored_combo.get("metrics", {})
                        scored_combo["normalized"] = scored_combo.get("normalized", 0.0)
                    
                    combinaciones = scored_combinations
                    self.logger.info(f"✅ Dynamic scoring applied to {len(combinaciones)} combinations")
                except Exception as e:
                    self.logger.error(f"🚨 Error applying dynamic scoring: {str(e)}")
                    traceback.print_exc()
            
            combinaciones = self.filtrar_combinaciones(combinaciones)
            
            if not combinaciones:
                self.logger.warning("⚠️ All combinations filtered out. Using fallback.")
                return [{
                    "combination": sorted(random.sample(range(1, 41), 6)),
                    "source": "fallback",
                    "score": 0.5,
                    "svi_score": 0.5,
                    "metrics": {},
                    "normalized": 0.0
                }]
            
            final_combinations = sorted(
                combinaciones, 
                key=lambda x: x["score"], 
                reverse=True
            )[:self.cantidad_final]
            
            unique_combinations = {}
            duplicates_removed = 0
            duplicate_details = []
            duplicate_sources = {}
            
            for combo in final_combinations:
                combo_tuple = tuple(combo["combination"])
                if combo_tuple in unique_combinations:
                    duplicates_removed += 1
                    existing_source = unique_combinations[combo_tuple]["source"]
                    new_source = combo["source"]
                    key = f"{existing_source}→{new_source}"
                    duplicate_sources[key] = duplicate_sources.get(key, 0) + 1
                    duplicate_details.append(
                        f"🧹 Duplicate: {combo['combination']} "
                        f"(existing source: {existing_source}, new source: {new_source})"
                    )
                else:
                    unique_combinations[combo_tuple] = combo
            
            final_combinations = list(unique_combinations.values())
            
            if duplicates_removed:
                self.logger.info(f"🧹 Removed {duplicates_removed} duplicate combinations")
                for detail in duplicate_details[:min(5, len(duplicate_details))]:
                    self.logger.debug(detail)
                if duplicates_removed > 5:
                    self.logger.debug(f"🧹 ... and {duplicates_removed-5} additional duplicates")
            
            if duplicate_sources:
                analysis = ", ".join([f"{k}: {v}" for k, v in duplicate_sources.items()])
                self.logger.info(f"🔍 Duplicate source analysis: {analysis}")
            
            if final_combinations:
                max_score = max(combo["score"] for combo in final_combinations)
                if max_score > 0:
                    for combo in final_combinations:
                        combo["normalized"] = combo["score"] / max_score
            
            self.logger.info(f"🏁 Pipeline completed: {len(final_combinations)} final combinations")
            
            if self.auto_export:
                os.makedirs("results", exist_ok=True)
                if not os.access("results", os.W_OK):
                    self.logger.error("🚨 No write permissions for results directory")
                    return final_combinations
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"results/svi_export_{timestamp}.csv"
                
                try:
                    exportar_combinaciones_svi(final_combinations, output_path)
                    
                    if os.path.exists(output_path):
                        file_size = os.path.getsize(output_path)
                        self.logger.info(f"📤 Exported {len(final_combinations)} combinations to {output_path}")
                        self.logger.debug(f"📝 File size: {file_size} bytes")
                        
                        source_score_chart = self.generate_source_score_chart(final_combinations)
                        self.export_chart("source_score_chart", "bar", source_score_chart, "results")
                        
                        source_count_chart = self.generate_source_count_chart(final_combinations)
                        self.export_chart("source_count_chart", "bar", source_count_chart, "results")
                        
                        self.generar_informe_html(final_combinations, duplicate_sources, source_score_chart, source_count_chart)
                    else:
                        self.logger.error(f"🚨 Error: Exported file not found: {output_path}")
                except Exception as e:
                    self.logger.error(f"🚨 Export error: {str(e)}")
                    self.logger.debug(f"📂 Path: {output_path}")
            
            return final_combinations
            
        except Exception as e:
            self.logger.error(f"🚨 Critical pipeline error: {str(e)}")
            traceback.print_exc()
            return [{
                "combination": sorted(random.sample(range(1, 41), 6)),
                "source": "fallback",
                "score": 0.5,
                "svi_score": 0.5,
                "metrics": {},
                "normalized": 0.0
            }]

    def generar_informe_html(self, combinaciones: List[Dict], duplicate_sources: Dict[str, int], 
                           source_score_chart: dict, source_count_chart: dict):
        try:
            from modules.reporting.html_reporter import generar_reporte_completo
            report_path = "results/reporte_prediccion.html"
            generar_reporte_completo(
                combinaciones,
                self.data,
                perfil_svi=self.perfil_svi,
                output_path=report_path,
                duplicate_sources=duplicate_sources,
                source_score_chart=source_score_chart,
                source_count_chart=source_count_chart
            )
            self.logger.info(f"📊 Advanced HTML report generated: {report_path}")
        except ImportError:
            self.logger.warning("⚠️ HTML report module not available, generating basic report")
            self._generar_informe_html_basico(combinaciones, duplicate_sources, source_score_chart, source_count_chart)
        except Exception as e:
            self.logger.error(f"🚨 HTML report generation error: {str(e)}")
            self._generar_informe_html_basico(combinaciones, duplicate_sources, source_score_chart, source_count_chart)

    def _generar_informe_html_basico(self, combinaciones: List[Dict], duplicate_sources: Dict[str, int], 
                                    source_score_chart: dict, source_count_chart: dict):
        try:
            report_path = "results/reporte_prediccion.html"
            
            if not combinaciones:
                self.logger.warning("⚠️ No combinations for HTML report")
                return
            
            chart_labels = [str(c["combination"]) for c in combinaciones]
            chart_scores = [c["score"] for c in combinaciones]
            chart_svi = [c.get("svi_score", 0.5) for c in combinaciones]
            chart_dynamic = [c.get("dynamic_score", 0) for c in combinaciones]
            chart_sources = [c["source"] for c in combinaciones]
            
            if not all(isinstance(s, (int, float)) for s in chart_scores + chart_svi + chart_dynamic):
                self.logger.warning("⚠️ Invalid score data, using defaults")
                chart_scores = [1.0] * len(combinaciones)
                chart_svi = [0.5] * len(combinaciones)
                chart_dynamic = [2.5] * len(combinaciones)
            
            source_counts = {}
            for source in chart_sources:
                source_counts[source] = source_counts.get(source, 0) + 1
            
            duplicate_section = ""
            if duplicate_sources:
                duplicate_section = f"""
                <h2>Duplicate Source Analysis</h2>
                <ul>
                    {"".join(f'<li>{k}: {v} duplicates</li>' for k, v in duplicate_sources.items())}
                </ul>
                """
            
            html_content = f"""
            <!DOCTYPE html>
            <html lang="es">
            <head>
                <meta charset="UTF-8">
                <title>OMEGA PRO AI Prediction Report</title>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #2c3e50; }}
                    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .footer {{ margin-top: 30px; font-size: 0.8em; color: #777; }}
                    .charts-container {{ display: flex; flex-wrap: wrap; }}
                    .chart-container {{ width: 45%; margin: 10px; }}
                </style>
            </head>
            <body>
                <h1>OMEGA PRO AI Predictions</h1>
                <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p>SVI Profile: {self.perfil_svi}</p>
                
                <h2>Recommended Combinations</h2>
                <table>
                    <tr>
                        <th>Combination</th>
                        <th>Final Score</th>
                        <th>SVI Score</th>
                        <th>Dynamic Score</th>
                        <th>Source</th>
                    </tr>
                    {"".join(
                        f'<tr><td>{c["combination"]}</td><td>{c["score"]:.4f}</td>'
                        f'<td>{c.get("svi_score", 0.5):.4f}</td>'
                        f'<td>{c.get("dynamic_score", 0):.2f}</td>'
                        f'<td>{c["source"]}</td></tr>'
                        for c in combinaciones
                    )}
                </table>
                
                {duplicate_section}
                
                <div class="charts-container">
                    <div class="chart-container">
                        <h2>Score Distribution</h2>
                        <canvas id="scoreChart"></canvas>
                    </div>
                    <div class="chart-container">
                        <h2>SVI Comparison</h2>
                        <canvas id="sviChart"></canvas>
                    </div>
                    <div class="chart-container">
                        <h2>Source Distribution</h2>
                        <canvas id="sourceChart"></canvas>
                    </div>
                    <div class="chart-container">
                        <h2>Average Scores by Source</h2>
                        <canvas id="sourceScoreChart"></canvas>
                    </div>
                    <div class="chart-container">
                        <h2>Combinations by Source</h2>
                        <canvas id="sourceCountChart"></canvas>
                    </div>
                </div>
                
                <script>
                    const scoreCtx = document.getElementById('scoreChart').getContext('2d');
                    new Chart(scoreCtx, {{
                        type: 'bar',
                        data: {{
                            labels: {json.dumps(chart_labels)},
                            datasets: [{{
                                label: 'Final Score',
                                data: {json.dumps(chart_scores)},
                                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                                borderColor: 'rgba(54, 162, 235, 1)',
                                borderWidth: 1
                            }}]
                        }},
                        options: {{
                            scales: {{ 
                                y: {{ 
                                    beginAtZero: true,
                                    title: {{ display: True, text: 'Score' }}
                                }}
                            }}
                        }}
                    }});
                    
                    const sviCtx = document.getElementById('sviChart').getContext('2d');
                    new Chart(sviCtx, {{
                        type: 'radar',
                        data: {{
                            labels: {json.dumps(chart_labels)},
                            datasets: [
                                {{
                                    label: 'SVI Score',
                                    data: {json.dumps(chart_svi)},
                                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                                    borderColor: 'rgba(54, 162, 235, 1)',
                                    pointBackgroundColor: 'rgba(54, 162, 235, 1)'
                                }},
                                {{
                                    label: 'Dynamic Score',
                                    data: {json.dumps(chart_dynamic)},
                                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                                    borderColor: 'rgba(255, 99, 132, 1)',
                                    pointBackgroundColor: 'rgba(255, 99, 132, 1)'
                                }}
                            ]
                        }},
                        options: {{
                            scales: {{
                                r: {{
                                    beginAtZero: true
                                }}
                            }}
                        }}
                    }});
                    
                    const sourceCtx = document.getElementById('sourceChart').getContext('2d');
                    new Chart(sourceCtx, {{
                        type: 'pie',
                        data: {{
                            labels: {json.dumps(list(source_counts.keys()))},
                            datasets: [{{
                                label: 'Combinations by Source',
                                data: {json.dumps(list(source_counts.values()))},
                                backgroundColor: [
                                    'rgba(54, 162, 235, 0.5)',
                                    'rgba(255, 99, 132, 0.5)',
                                    'rgba(75, 192, 192, 0.5)',
                                    'rgba(255, 206, 86, 0.5)',
                                    'rgba(153, 102, 255, 0.5)',
                                    'rgba(255, 159, 64, 0.5)',
                                    'rgba(100, 100, 100, 0.5)',
                                    'rgba(200, 100, 100, 0.5)'
                                ],
                                borderColor: [
                                    'rgba(54, 162, 235, 1)',
                                    'rgba(255, 99, 132, 1)',
                                    'rgba(75, 192, 192, 1)',
                                    'rgba(255, 206, 86, 1)',
                                    'rgba(153, 102, 255, 1)',
                                    'rgba(255, 159, 64, 1)',
                                    'rgba(100, 100, 100, 1)',
                                    'rgba(200, 100, 100, 1)'
                                ],
                                borderWidth: 1
                            }}]
                        }},
                        options: {{
                            plugins: {{
                                title: {{ display: true, text: 'Combinations by Source' }}
                            }}
                        }}
                    }});
                    
                    const sourceScoreCtx = document.getElementById('sourceScoreChart').getContext('2d');
                    new Chart(sourceScoreCtx, {json.dumps(source_score_chart)});
                    
                    const sourceCountCtx = document.getElementById('sourceCountChart').getContext('2d');
                    new Chart(sourceCountCtx, {json.dumps(source_count_chart)});
                </script>
                
                <div class="footer">
                    <p>Generated by OMEGA PRO AI v10.1</p>
                    <p>Advice: Combine these suggestions with your personal analysis</p>
                    <p>Good luck!</p>
                </div>
            </body>
            </html>
            """
            
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            self.logger.info(f"📊 Basic HTML report generated: {report_path}")
            
        except Exception as e:
            self.logger.error(f"🚨 Basic HTML report generation error: {str(e)}")