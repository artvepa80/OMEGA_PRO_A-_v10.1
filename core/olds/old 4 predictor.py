# predictor.py – Versión v10.7 con penalización de ghost RNG sospechoso

import os
import random
import pandas as pd
from modules.filters.rules_filter import FiltroEstrategico
from modules.filters.ghost_rng_observer import detect_rng_artifacts, simulate_ghost_rng
from core.consensus_engine import generar_combinaciones_consenso

class OmegaPredictor:
    def __init__(self, data_path="data/historial_kabala_github.csv", cantidad_final=5):
        self.data_path = data_path
        self.historial_set = self.cargar_historial()
        self.data = pd.read_csv(self.data_path)[[
            "Bolilla 1", "Bolilla 2", "Bolilla 3",
            "Bolilla 4", "Bolilla 5", "Bolilla 6"
        ]]
        self.cantidad_final = cantidad_final
        self.positional_analysis = False
        self.filtro = FiltroEstrategico()
        self.filtro.cargar_historial(self.data.values.tolist())

    def set_positional_analysis(self, enabled=True):
        self.positional_analysis = enabled

    def cargar_historial(self):
        df = pd.read_csv(self.data_path)
        posibles_columnas = [
            "Bolilla 1", "Bolilla 2", "Bolilla 3",
            "Bolilla 4", "Bolilla 5", "Bolilla 6",
            "1", "2", "3", "4", "5", "6"
        ]
        columnas_bolillas = [col for col in posibles_columnas if col in df.columns]

        if len(columnas_bolillas) < 6:
            raise ValueError("❌ No se encontraron 6 columnas válidas de bolillas en el historial.")

        df_bolillas = df[columnas_bolillas].dropna()
        historial = set(tuple(sorted(map(int, row))) for row in df_bolillas.to_numpy())
        return historial

    def aplicar_modelos_por_consenso(self):
        combinaciones = generar_combinaciones_consenso(
            historial_df=self.data,
            cantidad=60,
            logger=print
        )
        return combinaciones

    def filtrar_combinaciones(self, combinaciones, semillas_sospechosas=None):
        final = []
        for item in combinaciones:
            combo = item["combination"]
            if not isinstance(combo, list) or len(combo) != 6:
                continue

            # ❌ Rechazar si proviene de RNG sospechoso
            if semillas_sospechosas and self.filtro.penalizar_si_rng_sospechoso(combo, semillas_sospechosas):
                continue

            score_filtro, razones = self.filtro.aplicar_filtros(combo, return_score=True)
            if score_filtro >= 0.95:
                item["score_filter"] = round(score_filtro, 4)
                final.append(item)
        return final

    def seleccionar_finales(self, combinaciones_filtradas):
        random.shuffle(combinaciones_filtradas)
        return combinaciones_filtradas[:self.cantidad_final]

    def run_all_models(self):
        historial_real = [list(row) for row in self.data.tail(20).to_numpy()]
        resultados_rng = detect_rng_artifacts(historial_real, threshold=0.006, num_seeds=500)

        combinaciones = []

        if resultados_rng:
            print("👻 Ghost RNG detectó seeds sospechosas:")
            for r in resultados_rng[:3]:
                print(f"🔍 Seed: {r['seed']} – Similitud: {r['similarity_score']:.4f} – FFT: {r['fft_score']:.2f} – Score: {r['composite_score']:.4f}")
            
            ghost_combos = generar_combinaciones_ghost(resultados_rng, cantidad=10)
            combinaciones.extend([{
                "combination": list(c),
                "score": 1.0,
                "source": "ghost_rng",
                "tag": "reforzada"
            } for c in ghost_combos])
        else:
            print("✅ Ghost RNG no detectó patrones significativos.")

        combinaciones += self.aplicar_modelos_por_consenso()

        # 👁️‍🗨️ Penalización activa por combinaciones contaminadas
        semillas_peligrosas = set(r['seed'] for r in resultados_rng) if resultados_rng else set()
        filtradas = self.filtrar_combinaciones(combinaciones, semillas_sospechosas=semillas_peligrosas)
        finales = self.seleccionar_finales(filtradas)
        print("\n🎯 Combinaciones seleccionadas:")
        for i, item in enumerate(finales):
            fuente = item.get('source', 'desconocida')
            print(f"#{i+1}: {item['combination']} – Score: {item.get('score', 1.0)} – Fuente: {fuente}")

        # 📊 Guardar estadísticas de fallos en log
        os.makedirs("logs", exist_ok=True)
        with open("logs/filtros.log", "a") as f:
            f.write("\n--- Resumen de Fallos de Filtros ---\n")
            stats = self.filtro.reporte_estadisticas()
            for filtro, porcentaje in stats.items():
                f.write(f"{filtro}: {porcentaje:.2f}%\n")
            f.write(f"Tasa Aprobación Total: {self.filtro.tasa_aprobacion():.2%}\n")
            f.write(f"Promedio Scores: {self.filtro.promedio_scores():.4f}\n")
            f.write("------------------------------------\n")

        # 🧾 Mostrar tabla en consola
        stats = self.filtro.reporte_estadisticas()
        print("\n📋 Estadísticas de Filtros Aplicados:")
        print("╔════════════════════════════════╦══════════════╗")
        print("║ Filtro                         ║ % Rechazo    ║")
        print("╠════════════════════════════════╬══════════════╣")
        for filtro, porcentaje in stats.items():
            print(f"║ {filtro:<30} ║ {porcentaje:>6.2f} %     ║")
        print("╚════════════════════════════════╩══════════════╝")
        print(f"✔️ Tasa de aprobación total: {self.filtro.tasa_aprobacion():.2%}")
        print(f"✔️ Promedio general de scores: {self.filtro.promedio_scores():.4f}")

        return finales

# Función auxiliar mejorada
def generar_combinaciones_ghost(resultados_rng, cantidad=5):
    combinaciones = []
    top_seeds = sorted(resultados_rng, key=lambda x: x["composite_score"], reverse=True)[:3]
    for r in top_seeds:
        simuladas = simulate_ghost_rng(r["seed"], total_draws=50)
        combinaciones.extend(simuladas[:cantidad])
    return combinaciones[:cantidad]