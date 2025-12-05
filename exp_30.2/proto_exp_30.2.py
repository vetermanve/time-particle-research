"""
ЭКСПЕРИМЕНТ №30.2 — НОРМАЛИЗОВАННЫЕ ПАРАМЕТРЫ
"""

import numpy as np
import random
import math
import json
import time
import matplotlib.pyplot as plt
from datetime import datetime
import os
import networkx as nx
from collections import defaultdict, Counter

# ================= КОНСТАНТЫ =================
EXPERIMENT_NUMBER = 30
VERSION = "30.2"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = f"experiment_{EXPERIMENT_NUMBER}_{VERSION}_{TIMESTAMP}"
os.makedirs(RESULT_DIR, exist_ok=True)

PLANCK_UNIT = 1.0
SYNC_THRESHOLD = 0.01  # Увеличили порог для более редких резонансов

class TemporalThread:
    """НИТЬ ВРЕМЕНИ С НОРМАЛИЗОВАННЫМИ ПАРАМЕТРАМИ"""
    
    def __init__(self, thread_id: int, dimension: int = 3):
        self.id = thread_id
        self.dimension = dimension
        
        # Генерация кода в диапазоне (0, 1)
        self.code = self._generate_normalized_code()
        
        # Динамические параметры
        self.phase = random.uniform(0, 2 * math.pi)
        self.amplitude = 1.0
        self.frequency = self._calculate_base_frequency()
        
        # Состояния
        self.sync_partners = set()
        self.cluster_id = None
        self.sync_strength = 0.0
        
        # Возникающие свойства
        self.effective_mass = 0.0
        self.charge = 0.0
        self.spin = 0.0
        self.color = None
    
    def _generate_normalized_code(self) -> list:
        """Генерация кода в диапазоне (0, 1)"""
        code = []
        for i in range(self.dimension):
            # Используем разные иррациональные числа, но нормализуем
            irrationals = [
                math.sqrt(2)/10, math.sqrt(3)/10, math.sqrt(5)/10,
                math.pi/10, math.e/10, 0.6180339887  # φ⁻¹
            ]
            base = random.choice(irrationals)
            # Добавляем небольшие вариации
            variation = random.uniform(0.9, 1.1)
            value = (base * variation) % 1.0
            if value == 0:
                value = 0.1
            code.append(value)
        
        return code
    
    def _calculate_base_frequency(self) -> float:
        """Частота в диапазоне (0.1, 1.0)"""
        base_freq = math.sqrt(sum(c**2 for c in self.code))
        # Нормализуем к диапазону (0.1, 1.0)
        normalized = 0.1 + 0.9 * (base_freq / math.sqrt(self.dimension))
        return min(max(normalized, 0.1), 1.0)
    
    def evolve(self, delta_t: float = 0.01):
        """Эволюция с небольшими флуктуациями"""
        self.phase += self.frequency * delta_t
        self.phase %= 2 * math.pi
        # Маленькие флуктуации амплитуды
        self.amplitude = 1.0 + 0.01 * math.sin(self.phase * 3)
    
    def __repr__(self):
        return f"Thread_{self.id}[freq={self.frequency:.3f}]"

class ResonanceChecker:
    """ПРОВЕРКА РЕЗОНАНСА С НОРМАЛИЗАЦИЕЙ"""
    
    @staticmethod
    def check_resonance(code1: list, code2: list, max_n: int = 5) -> tuple:
        """Проверка с нормализованной силой резонанса"""
        if len(code1) != len(code2):
            return False, 0.0, {}
        
        resonances = []
        details = {"pairs": []}
        
        for i, (c1, c2) in enumerate(zip(code1, code2)):
            if c2 == 0 or c1 == 0:
                continue
            
            ratio = c1 / c2
            best_error = float('inf')
            best_n, best_m = 0, 0
            
            # Поиск рационального приближения
            for n in range(1, max_n + 1):
                for m in range(1, max_n + 1):
                    approx = n / m
                    error = abs(ratio - approx)
                    
                    if error < best_error:
                        best_error = error
                        best_n, best_m = n, m
            
            if best_error < SYNC_THRESHOLD:
                # НОРМАЛИЗАЦИЯ: сила резонанса = 1/(1 + error)
                strength = 1.0 / (1.0 + best_error * 100)
                resonances.append(strength)
                
                details["pairs"].append({
                    "ratio": ratio,
                    "error": best_error,
                    "strength": strength
                })
        
        if resonances:
            # Средняя сила, но не более 1.0
            avg_strength = min(sum(resonances) / len(resonances), 1.0)
            details["avg_strength"] = avg_strength
            details["num_resonances"] = len(resonances)
            return True, avg_strength, details
        
        return False, 0.0, {"num_resonances": 0}

class SyncEnsemble:
    """АНСАМБЛЬ С ФИЗИЧЕСКИ ОСМЫСЛЕННЫМИ ПАРАМЕТРАМИ"""
    
    ensemble_counter = 0
    
    def __init__(self, thread_ids: list, threads: dict):
        SyncEnsemble.ensemble_counter += 1
        self.ensemble_id = SyncEnsemble.ensemble_counter
        self.thread_ids = thread_ids.copy()
        self.threads = {tid: threads[tid] for tid in thread_ids}
        self.size = len(thread_ids)
        
        # Вычисляем свойства в правильном порядке
        self.sync_matrix = self._calculate_sync_matrix()
        self.avg_sync_strength = self._calculate_avg_sync()
        self.coherence = self._calculate_coherence()
        self.stability = self._calculate_stability()
        
        # Физические свойства с нормализованными формулами
        self.mass = self._calculate_mass()
        self.charge = self._calculate_charge()
        self.spin = self._calculate_spin()
        self.color = self._assign_color()
        self.lifetime = self._estimate_lifetime()
        
        # Обновляем нити
        for tid in thread_ids:
            threads[tid].cluster_id = self.ensemble_id
            threads[tid].sync_strength = self.avg_sync_strength
    
    def _calculate_sync_matrix(self) -> np.ndarray:
        """Матрица синхронизации"""
        n = self.size
        matrix = np.zeros((n, n))
        
        checker = ResonanceChecker()
        thread_list = list(self.threads.values())
        
        for i in range(n):
            for j in range(i+1, n):
                t1 = thread_list[i]
                t2 = thread_list[j]
                
                is_resonant, strength, _ = checker.check_resonance(t1.code, t2.code)
                if is_resonant:
                    matrix[i, j] = matrix[j, i] = strength
        
        return matrix
    
    def _calculate_avg_sync(self) -> float:
        """Средняя сила синхронизации"""
        if self.size < 2:
            return 0.0
        
        upper_tri = self.sync_matrix[np.triu_indices(self.size, k=1)]
        non_zero = upper_tri[upper_tri > 0]
        
        if len(non_zero) > 0:
            return float(np.mean(non_zero))
        return 0.0
    
    def _calculate_coherence(self) -> float:
        """Когерентность в диапазоне (0, 1)"""
        if self.size < 2:
            return 0.0
        
        # Доля связей
        connections = np.sum(self.sync_matrix > 0) / 2
        total_possible = self.size * (self.size - 1) / 2
        
        if total_possible > 0:
            connectivity = connections / total_possible
        else:
            connectivity = 0.0
        
        # Когерентность = среднее связности и силы
        coherence = (connectivity + self.avg_sync_strength) / 2.0
        return min(max(coherence, 0.0), 1.0)
    
    def _calculate_stability(self) -> float:
        """Реалистичная стабильность (0-1)"""
        if self.size < 2:
            return 0.0
        
        # Базовая стабильность из когерентности
        base_stability = self.coherence
        
        # Штраф за неполную связность
        non_zero = np.sum(self.sync_matrix > 0) / 2
        total_possible = self.size * (self.size - 1) / 2
        connectivity = non_zero / total_possible if total_possible > 0 else 0.0
        
        if connectivity < 1.0:
            # Если не все пары связаны - уменьшаем стабильность
            base_stability *= connectivity
        
        # Добавляем небольшой шум
        stability = base_stability * random.uniform(0.95, 1.05)
        return min(max(stability, 0.0), 1.0)
    
    def _calculate_mass(self) -> float:
        """Масса в разумном диапазоне"""
        # Базовая масса = размер * когерентность
        base_mass = self.size * self.coherence
        
        # Поправка на стабильность
        mass = base_mass * (0.5 + self.stability)
        
        # Масштабируем к физическим масштабам (условные единицы)
        scaled_mass = mass * 10.0  # Масштабирующий коэффициент
        
        return scaled_mass
    
    def _calculate_charge(self) -> float:
        """Заряд, возникающий из асимметрии"""
        if self.size == 2:
            # Для мезонов: заряд = разность частот
            threads = list(self.threads.values())
            freq_diff = abs(threads[0].frequency - threads[1].frequency)
            charge = freq_diff * 10.0
            # Округляем к ближайшему 0, ±1
            if charge < 0.33:
                return 0.0
            elif charge < 0.67:
                return 0.33
            else:
                return 0.67
        elif self.size == 3:
            # Для барионов: заряд из фазовых соотношений
            phases = [t.phase for t in self.threads.values()]
            
            # Вычисляем "центр масс" фаз
            phase_sum = sum(phases)
            phase_center = phase_sum / 3.0
            
            # Заряд = мера асимметрии
            asymmetry = 0.0
            for phase in phases:
                asymmetry += abs(phase - phase_center)
            
            charge = asymmetry / (2 * math.pi)
            
            # Квантуем: 0, ±1/3, ±2/3, ±1
            if charge < 0.16:
                return 0.0
            elif charge < 0.33:
                return 0.33
            elif charge < 0.5:
                return 0.67
            else:
                return 1.0
        
        return 0.0
    
    def _calculate_spin(self) -> float:
        """Спин из круговой поляризации"""
        if self.size == 2:
            # Мезоны: спин 0 или 1
            threads = list(self.threads.values())
            phase_diff = abs(threads[0].phase - threads[1].phase) % (2*math.pi)
            
            if abs(phase_diff - math.pi) < 0.1:  # Противоположные фазы
                return 0.0  # Скалярные мезоны
            else:
                return 1.0  # Векторные мезоны
        elif self.size == 3:
            # Барионы: спин 1/2 или 3/2
            # Проверяем, образуют ли фазы "треугольник"
            phases = [t.phase for t in self.threads.values()]
            phase_diffs = []
            
            for i in range(3):
                for j in range(i+1, 3):
                    diff = abs(phases[i] - phases[j]) % (2*math.pi)
                    phase_diffs.append(min(diff, 2*math.pi - diff))
            
            # Если все разности близки к 2π/3 - спин 1/2
            ideal_diff = 2 * math.pi / 3
            deviations = [abs(d - ideal_diff) for d in phase_diffs]
            avg_deviation = sum(deviations) / len(deviations)
            
            if avg_deviation < 0.2:
                return 0.5  # Нуклоны
            else:
                return 1.5  # Возбуждённые состояния
        
        return 0.0
    
    def _assign_color(self) -> str:
        """Цветовая степень свободы"""
        if self.size == 3:
            # Для барионов: случайный цвет из R,G,B
            colors = ['R', 'G', 'B']
            return random.choice(colors)
        elif self.size == 2:
            # Для мезонов: цвет-антицвет
            return 'R-antiR'
        return 'N'
    
    def _estimate_lifetime(self) -> float:
        """Реалистичное время жизни"""
        if self.stability > 0.9:
            return float('inf')  # Стабильная
        elif self.stability > 0.7:
            return 1e10  # Долгоживущая
        elif self.stability > 0.5:
            return 1e3   # Среднее
        elif self.stability > 0.3:
            return 1e-6  # Короткоживущая
        else:
            return 1e-12 # Резонанс
    
    def get_properties(self) -> dict:
        """Свойства ансамбля"""
        return {
            "ensemble_id": self.ensemble_id,
            "size": self.size,
            "avg_sync_strength": self.avg_sync_strength,
            "coherence": self.coherence,
            "stability": self.stability,
            "mass": self.mass,
            "charge": self.charge,
            "spin": self.spin,
            "color": self.color,
            "lifetime": self.lifetime
        }

class Universe:
    """ВСЕЛЕННАЯ С УЛУЧШЕННЫМ ПОИСКОМ АНСАМБЛЕЙ"""
    
    def __init__(self, num_threads: int = 300, code_dim: int = 3):
        self.num_threads = num_threads
        self.code_dim = code_dim
        
        print(f"Создание {num_threads} нитей времени...")
        self.threads = {}
        for i in range(num_threads):
            self.threads[i] = TemporalThread(i, code_dim)
        
        self.sync_graph = nx.Graph()
        self.sync_graph.add_nodes_from(range(num_threads))
        
        self.ensembles = []
        self.free_threads = set(range(num_threads))
        
        self.stats = {
            "total_threads": num_threads,
            "sync_edges": 0,
            "ensembles_found": 0,
            "mesons": 0,
            "baryons": 0,
            "avg_mass": 0.0,
            "avg_stability": 0.0
        }
    
    def build_sync_network(self, max_pairs_check: int = 5000):
        """Построение сети с поиском сильных резонансов"""
        print("Построение сети синхронизации...")
        
        thread_ids = list(self.threads.keys())
        checker = ResonanceChecker()
        edges_added = 0
        
        # Проверяем случайные пары
        for _ in range(min(max_pairs_check, len(thread_ids) * 10)):
            i, j = random.sample(thread_ids, 2)
            
            if not self.sync_graph.has_edge(i, j):
                t1 = self.threads[i]
                t2 = self.threads[j]
                
                is_resonant, strength, _ = checker.check_resonance(t1.code, t2.code)
                
                # Более строгий порог для связей
                if is_resonant and strength > 0.3:
                    self.sync_graph.add_edge(i, j, weight=strength)
                    t1.sync_partners.add(j)
                    t2.sync_partners.add(i)
                    edges_added += 1
        
        self.stats["sync_edges"] = edges_added
        print(f"Добавлено {edges_added} сильных резонансных связей")
        return edges_added
    
    def find_sync_ensembles(self):
        """Поиск ансамблей с улучшенным алгоритмом"""
        print("Поиск ансамблей...")
        
        # 1. Находим все связные компоненты
        components = list(nx.connected_components(self.sync_graph))
        
        ensembles = []
        used_threads = set()
        
        for comp in components:
            comp_nodes = list(comp)
            
            # Если компонента имеет 2 узла - потенциальный мезон
            if len(comp_nodes) == 2:
                edge_weight = self.sync_graph[comp_nodes[0]][comp_nodes[1]].get('weight', 0)
                if edge_weight > 0.5:  # Сильная связь
                    ensemble = SyncEnsemble(comp_nodes, self.threads)
                    if ensemble.stability > 0.5:
                        ensembles.append(ensemble)
                        used_threads.update(comp_nodes)
            
            # Если компонента имеет 3 узла - потенциальный барион
            elif len(comp_nodes) == 3:
                # Проверяем, что это полный треугольник
                subgraph = self.sync_graph.subgraph(comp_nodes)
                if subgraph.number_of_edges() == 3:  # Все 3 связи существуют
                    # Проверяем среднюю силу связей
                    weights = [subgraph[u][v].get('weight', 0) for u, v in subgraph.edges()]
                    avg_weight = sum(weights) / len(weights)
                    
                    if avg_weight > 0.4:  # Достаточно сильные связи
                        ensemble = SyncEnsemble(comp_nodes, self.threads)
                        if ensemble.stability > 0.4:
                            ensembles.append(ensemble)
                            used_threads.update(comp_nodes)
        
        self.ensembles = ensembles
        self.free_threads = set(self.threads.keys()) - used_threads
        
        # Статистика
        self.stats["ensembles_found"] = len(ensembles)
        self.stats["mesons"] = len([e for e in ensembles if e.size == 2])
        self.stats["baryons"] = len([e for e in ensembles if e.size == 3])
        
        if ensembles:
            self.stats["avg_mass"] = np.mean([e.mass for e in ensembles])
            self.stats["avg_stability"] = np.mean([e.stability for e in ensembles])
        
        return ensembles
    
    def evolve(self, steps: int = 10):
        """Простая эволюция"""
        for step in range(steps):
            for thread in self.threads.values():
                thread.evolve(0.1)
    
    def analyze(self):
        """Анализ результатов"""
        if not self.ensembles:
            return None
        
        analysis = {
            "ensembles": [e.get_properties() for e in self.ensembles],
            "mass_dist": [e.mass for e in self.ensembles],
            "charge_dist": [e.charge for e in self.ensembles],
            "spin_dist": [e.spin for e in self.ensembles],
            "stability_dist": [e.stability for e in self.ensembles]
        }
        
        return analysis
    
    def save_results(self):
        """Сохранение результатов"""
        config = {
            "experiment": EXPERIMENT_NUMBER,
            "version": VERSION,
            "timestamp": TIMESTAMP,
            "num_threads": self.num_threads,
            "code_dim": self.code_dim
        }
        
        with open(f"{RESULT_DIR}/config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        with open(f"{RESULT_DIR}/stats.json", "w") as f:
            json.dump(self.stats, f, indent=2)
        
        if self.ensembles:
            ensembles_data = [e.get_properties() for e in self.ensembles]
            with open(f"{RESULT_DIR}/ensembles.json", "w") as f:
                json.dump(ensembles_data, f, indent=2)
        
        return RESULT_DIR
    
    def visualize(self):
        """Визуализация"""
        if not self.ensembles:
            print("Нет данных для визуализации")
            return
        
        analysis = self.analyze()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Эксперимент {EXPERIMENT_NUMBER} v{VERSION}", fontsize=16)
        
        # 1. Массы
        axes[0, 0].hist(analysis["mass_dist"], bins=15, alpha=0.7, color='blue')
        axes[0, 0].set_title("Распределение масс")
        axes[0, 0].set_xlabel("Масса")
        axes[0, 0].set_ylabel("Частота")
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Заряды
        axes[0, 1].hist(analysis["charge_dist"], bins=15, alpha=0.7, color='green')
        axes[0, 1].set_title("Распределение зарядов")
        axes[0, 1].set_xlabel("Заряд")
        axes[0, 1].set_ylabel("Частота")
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Спины
        axes[0, 2].hist(analysis["spin_dist"], bins=15, alpha=0.7, color='red')
        axes[0, 2].set_title("Распределение спинов")
        axes[0, 2].set_xlabel("Спин")
        axes[0, 2].set_ylabel("Частота")
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Стабильность
        axes[1, 0].hist(analysis["stability_dist"], bins=15, alpha=0.7, color='purple')
        axes[1, 0].set_title("Распределение стабильности")
        axes[1, 0].set_xlabel("Стабильность")
        axes[1, 0].set_ylabel("Частота")
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Масса vs Стабильность
        axes[1, 1].scatter(analysis["mass_dist"], analysis["stability_dist"],
                          alpha=0.6, color='orange')
        axes[1, 1].set_title("Масса vs Стабильность")
        axes[1, 1].set_xlabel("Масса")
        axes[1, 1].set_ylabel("Стабильность")
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Размер ансамбля
        sizes = [e.size for e in self.ensembles]
        size_counts = Counter(sizes)
        axes[1, 2].bar(size_counts.keys(), size_counts.values(),
                      alpha=0.7, color='cyan')
        axes[1, 2].set_title("Распределение по размеру")
        axes[1, 2].set_xlabel("Размер ансамбля")
        axes[1, 2].set_ylabel("Количество")
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{RESULT_DIR}/visualization.png", dpi=150)
        plt.show()

def run_experiment_30_v2():
    """Запуск эксперимента 30.2"""
    print("=" * 80)
    print(f"🚀 ЭКСПЕРИМЕНТ №{EXPERIMENT_NUMBER} v{VERSION}")
    print("Нормализованные параметры")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        # 1. СОЗДАНИЕ ВСЕЛЕННОЙ
        universe = Universe(num_threads=200, code_dim=3)  # Меньше нитей для скорости
        
        # 2. ПОСТРОЕНИЕ СЕТИ
        universe.build_sync_network(max_pairs_check=3000)
        
        # 3. ПОИСК АНСАМБЛЕЙ
        ensembles = universe.find_sync_ensembles()
        
        # 4. ЭВОЛЮЦИЯ
        universe.evolve(steps=5)
        
        # 5. АНАЛИЗ
        analysis = universe.analyze()
        
        # 6. СОХРАНЕНИЕ
        result_dir = universe.save_results()
        
        # 7. ВИЗУАЛИЗАЦИЯ
        if ensembles:
            universe.visualize()
        
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("📊 РЕЗУЛЬТАТЫ v30.2:")
        print("=" * 80)
        
        print(f"\nВсего нитей: {universe.stats['total_threads']}")
        print(f"Сильных связей: {universe.stats['sync_edges']}")
        print(f"Найдено ансамблей: {universe.stats['ensembles_found']}")
        print(f"  • Мезоны: {universe.stats['mesons']}")
        print(f"  • Барионы: {universe.stats['baryons']}")
        
        if analysis:
            print(f"\nСредняя масса: {universe.stats['avg_mass']:.3f}")
            print(f"Средняя стабильность: {universe.stats['avg_stability']:.3f}")
            
            if ensembles:
                print(f"\nПЕРВЫЕ 5 АНСАМБЛЕЙ:")
                for i, e in enumerate(ensembles[:5], 1):
                    props = e.get_properties()
                    print(f"{i}. Ансамбль_{props['ensemble_id']}: "
                          f"размер={props['size']}, "
                          f"масса={props['mass']:.3f}, "
                          f"заряд={props['charge']:.3f}, "
                          f"спин={props['spin']}, "
                          f"стабильность={props['stability']:.3f}")
        
        print(f"\n⏱ Время: {elapsed:.2f} сек")
        print(f"📁 Результаты: {result_dir}")
        print("\n" + "=" * 80)
        print("✅ ЭКСПЕРИМЕНТ 30.2 ЗАВЕРШЁН!")
        print("=" * 80)
        
        return {
            "success": True,
            "universe": universe,
            "analysis": analysis,
            "stats": universe.stats,
            "time": elapsed,
            "dir": result_dir
        }
        
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e),
            "time": time.time() - start_time
        }

# ================= ЗАПУСК =================

if __name__ == "__main__":
    print("""
    🌌 ЭКСПЕРИМЕНТ №30.2: НОРМАЛИЗОВАННЫЕ ПАРАМЕТРЫ
    Исправления:
    1. Нормализованные коды и частоты
    2. Реалистичная стабильность
    3. Физически осмысленные массы
    4. Улучшенный поиск барионов
    """)
    
    results = run_experiment_30_v2()