# OMEGA_PRO_AI_v10.1/core/transformer_predictor.py

from modules.lottery_transformer import LotteryTransformer
from utils.transformer_data_utils import prepare_advanced_transformer_data
from modules.filters.rules_filter import FiltroEstrategico
import torch
import numpy as np
import logging
import pandas as pd
import os

# Configurar logging principal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger_main = logging.getLogger('transformer_predictor')

def generar_combinaciones_transformer_v2(historial_df, cantidad=5, logger=None):
    """
    Genera combinaciones válidas usando el modelo Transformer avanzado
    """
    # Adaptar logger si es print o None
    if logger is None or not hasattr(logger, "info"):
        class DummyLogger:
            def info(self, msg): print(f"🔹 {msg}")
            def error(self, msg): print(f"❌ {msg}")
            def warning(self, msg): print(f"⚠️ {msg}")
        logger = DummyLogger()
    
    log = logger.info

    try:
        # 1. Cargar modelo
        model_path = "modules/enhanced_lottery_transformer.pth"
        if not os.path.exists(model_path):
            logger.error(f"Modelo no encontrado en: {model_path}")
            return []
        
        logger.info(f"🧠 Cargando modelo Transformer desde: {model_path}")
        model = LotteryTransformer(
            num_numbers=40,
            d_model=128,
            nhead=4,
            num_layers=3,
            dropout=0.1
        )
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        logger.info("✅ Modelo cargado exitosamente")

        # 2. Preprocesamiento avanzado - FIXED: Asegurar dimensiones correctas
        try:
            logger.info("📊 Preparando datos avanzados...")
            X_seq, X_temp, X_pos, _ = prepare_advanced_transformer_data(
                historial_df,
                seq_length=90,
                for_training=False
            )
            
            # FIX: Asegurar batch dimension (dim=3)
            if X_seq.dim() == 2:
                X_seq = X_seq.unsqueeze(0)  # (1, seq_len, features)
            if X_temp.dim() == 2:
                X_temp = X_temp.unsqueeze(0)
            if X_pos.dim() == 2:
                X_pos = X_pos.unsqueeze(0)
            
            # Usar último elemento si es secuencial
            entrada_seq = X_seq[-1].unsqueeze(0) if X_seq.dim() > 2 else X_seq
            entrada_temp = X_temp[-1].unsqueeze(0) if X_temp.dim() > 2 else X_temp
            entrada_pos = X_pos[-1].unsqueeze(0) if X_pos.dim() > 2 else X_pos
            
            # Validar shapes antes de prediction
            logger.debug(f"Shapes: seq={entrada_seq.shape}, temp={entrada_temp.shape}, pos={entrada_pos.shape}")
            if entrada_seq.dim() != 3 or entrada_temp.dim() != 3 or entrada_pos.dim() != 3:
                raise ValueError("Inputs no tienen batch dimension (3D)")
            
            logger.info("✅ Datos avanzados preparados")
        
        except Exception as prep_error:
            logger.warning(f"⚠️ Fallo en preparación avanzada: {str(prep_error)}")
            logger.warning("🔄 Usando método simplificado de preparación")
            
            last_numbers = historial_df.iloc[-1].values.astype(int).tolist()
            entrada_seq = torch.tensor([last_numbers], dtype=torch.long).unsqueeze(0)  # FIXED: Asegurar (1,1,6)
            entrada_temp = torch.tensor([[2023, 7, 15]], dtype=torch.float).unsqueeze(0)
            entrada_pos = torch.tensor([list(range(1, 7))], dtype=torch.long).unsqueeze(0)

        # 3. Predicción - FIXED: Try-except y manejo de dims
        logger.info("🔮 Ejecutando predicción Transformer...")
        try:
            with torch.no_grad():
                number_logits, _ = model(entrada_seq, entrada_temp, entrada_pos)
                
                # FIX: Manejo flexible de dimensiones en softmax
                number_probs = torch.softmax(number_logits, dim=-1)
                ndim = number_probs.dim()
                if ndim == 3:
                    number_probs = number_probs.mean(dim=1)  # Average over seq if 3D
                if ndim >= 2:
                    number_probs = number_probs[0] if number_probs.shape[0] == 1 else number_probs.mean(dim=0)
                number_probs = number_probs.flatten().cpu().numpy()  # Asegurar 1D y CPU
                
                if number_probs.shape[0] != 40:
                    raise ValueError(f"Expected 40 probabilities, got {number_probs.shape}")
                
            logger.info("✅ Predicción completada")
        
        except Exception as pred_error:
            logger.error(f"🔥 Error en predicción Transformer: {str(pred_error)}")
            number_probs = np.ones(40) / 40  # Uniform fallback

        # 4. Filtros y muestreo
        universo = list(range(1, 41))
        filtro = FiltroEstrategico()
        filtro.cargar_historial(historial_df.values.tolist())

        historial_set = set(
            tuple(sorted(comb)) for comb in historial_df['numeros']
        ) if 'numeros' in historial_df.columns else set(
            tuple(sorted(row)) for row in historial_df.values.astype(int)
        )

        combinaciones = []
        intentos = 0
        max_intentos = cantidad * 50
        temperature = 0.7

        # Verificar y normalizar probabilidades
        if number_probs.sum() <= 0 or np.isnan(number_probs).any():
            logger.warning("📉 Distribución de probabilidad inválida. Usando distribución uniforme")
            number_probs = np.ones(40) / 40
        
        pesos_muestreo = number_probs / number_probs.sum()

        logger.info(f"🎯 Generando hasta {cantidad} combinaciones válidas...")
        while len(combinaciones) < cantidad and intentos < max_intentos:
            numeros_muestreados = np.random.choice(
                universo,
                size=15,
                replace=False,
                p=pesos_muestreo
            )

            candidatos = seleccion_optimizada(numeros_muestreados, number_probs)
            combo_tuple = tuple(sorted(candidatos))

            if es_combinacion_valida(candidatos, combo_tuple, historial_set, filtro):
                score = calcular_score(candidatos, number_probs)
                combinaciones.append({
                    "combination": candidatos,
                    "source": "enhanced_transformer",
                    "score": score
                })
                log(f"✨ {candidatos} | Score: {score:.4f}")

            intentos += 1

        combinaciones.sort(key=lambda x: x['score'], reverse=True)
        logger.info(f"✅ Generadas {len(combinaciones)} combinaciones válidas")
        return combinaciones[:cantidad]

    except Exception as e:
        logger.error(f"🔥 Error crítico en Transformer: {str(e)}")
        return []

# =============================
# FUNCIONES AUXILIARES (sin cambios)
# =============================

def seleccion_optimizada(numeros_muestreados, probs):
    """Selección balanceada de números basada en probabilidades"""
    sorted_nums = sorted(numeros_muestreados, key=lambda x: probs[x-1], reverse=True)
    
    top_nums = sorted_nums[:2]
    mid_nums = sorted_nums[len(sorted_nums)//3:2*len(sorted_nums)//3][:2]
    low_nums = [n for n in sorted_nums[-3:] if probs[n-1] > 0.1][:2]
    
    candidatos = list(set(top_nums + mid_nums + low_nums))
    if len(candidatos) < 6:
        adicionales = [n for n in numeros_muestreados if n not in candidatos]
        candidatos += adicionales[:6-len(candidatos)]
    return sorted(candidatos)

def es_combinacion_valida(candidatos, combo_tuple, historial_set, filtro):
    """Verifica si la combinación cumple con todos los criterios"""
    if len(set(candidatos)) != 6 or not all(1 <= num <= 40 for num in candidatos):
        return False
    
    if combo_tuple in historial_set:
        return False
    
    if not filtro.aplicar_filtros(candidatos):
        return False
    
    suma = sum(candidatos)
    if suma < 80 or suma > 160:
        return False
    
    return True

def calcular_score(candidatos, probs):
    """Calcula un score ponderado para la combinación"""
    scores = []
    for num in candidatos:
        prob = probs[num-1]
        if prob < 0.3:
            scores.append(prob * 1.5)
        elif prob > 0.7:
            scores.append(prob * 0.8)
        else:
            scores.append(prob)
    return sum(scores) / len(scores)

# =============================
# TEST DIRECTO (sin cambios)
# =============================

if __name__ == "__main__":
    try:
        logger_main.info("🚀 Iniciando prueba del Transformer Predictor")
        
        logger_main.info("📂 Cargando datos históricos...")
        df = pd.read_csv("data/historial_kabala_github.csv")
        
        if 'fecha' in df.columns:
            df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
            logger_main.info("✅ Fechas convertidas correctamente")

        logger_main.info("🔮 Generando combinaciones con Transformer...")
        combinaciones = generar_combinaciones_transformer_v2(df, cantidad=5, logger=logger_main)

        if combinaciones:
            logger_main.info("\n🎯 COMBINACIONES GENERADAS:")
            for i, c in enumerate(combinaciones, 1):
                logger_main.info(f"{i}. {c['combination']} → Score: {c['score']:.4f}")
        else:
            logger_main.warning("⚠️ No se generaron combinaciones")

    except Exception as e:
        logger_main.error(f"❌ Error en prueba: {str(e)}")
    finally:
        logger_main.info("🏁 Prueba completada")