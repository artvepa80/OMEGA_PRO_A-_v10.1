# core/predictor.py

import os
import random
import pandas as pd
from rules_filter import aplicar_filtros
from modules.cluster_predictor import generar_combinaciones_clustering
from modules.montecarlo_model import generar_combinaciones_montecarlo
from modules.genetic_model import generate_genetic_predictions as generar_combinaciones_genetico
from modules.rng_emulator import emular_rng_combinaciones
from modules.lstm_v2 import generar_combinaciones_lstm_v2

class OmegaPredictor:
    def __init__(self, data_path="data/historial_kabala_github.csv", cantidad_final=5):
        self.data_path = data_path
        self.historial_set = self.cargar_historial()
        self.data = pd.read_csv(self.data_path)[[
            "Bolilla 1", "Bolilla 2", "Bolilla 3",
            "Bolilla 4", "Bolilla 5", "Bolilla 6"
        ]]
        self.cantidad_final = cantidad_final
        self.positional_analysis = False  # NUEVO: Activador del modo posicional

    def set_positional_analysis(self, enabled=True):
        """Activa o desactiva el modo de análisis posicional."""
        self.positional_analysis = enabled

    def cargar_historial(self):
        df = pd.read_csv(self.data_path)
        columnas_bolillas = [
            "Bolilla 1", "Bolilla 2", "Bolilla 3",
            "Bolilla 4", "Bolilla 5", "Bolilla 6"
        ]
        df_bolillas = df[columnas_bolillas]
        historial = set(tuple(sorted(row)) for row in df_bolillas.to_numpy())
        return historial

    def aplicar_modelos(self):
        combinaciones = []

        clustering = generar_combinaciones_clustering(self.historial_set, cantidad=60)
        combinaciones.extend([{"combination": combo, "score": 1.0, "source": "clustering"} for combo in clustering])

        montecarlo = generar_combinaciones_montecarlo(self.historial_set, cantidad=60)
        combinaciones.extend([{"combination": combo, "score": 1.0, "source": "montecarlo"} for combo in montecarlo])

        genetico = generar_combinaciones_genetico(self.data, cantidad=30)
        combinaciones.extend([{"combination": combo, "score": 1.0, "source": "genetico"} for combo in genetico])

        rng_emuladas = emular_rng_combinaciones(self.historial_set, cantidad=30)
        combinaciones.extend([{"combination": combo, "score": 1.0, "source": "rng_emulado"} for combo in rng_emuladas])

        # Aquí pasamos el flag de análisis posicional al LSTM
        lstm = generar_combinaciones_lstm_v2(
            self.data_path, cantidad=30, posicional=self.positional_analysis
)
        combinaciones.extend([{"combination": combo, "score": 1.0, "source": "lstm_v2"} for combo in lstm])

        return combinaciones

    def filtrar_combinaciones(self, combinaciones):
        final = []
        for item in combinaciones:
            combo = item["combination"]
            if not isinstance(combo, list) or len(combo) != 6:
                continue
            if aplicar_filtros(combo, self.historial_set):
                final.append(item)
        return final

    def seleccionar_finales(self, combinaciones_filtradas):
        random.shuffle(combinaciones_filtradas)
        return combinaciones_filtradas[:self.cantidad_final]

    def run_all_models(self):
        crudas = self.aplicar_modelos()
        filtradas = self.filtrar_combinaciones(crudas)
        finales = self.seleccionar_finales(filtradas)
        return finales