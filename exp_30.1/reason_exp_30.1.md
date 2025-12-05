Отлично! Мы нашли ошибку в порядке инициализации. Нужно вычислить `stability` перед использованием в `_calculate_mass()`. Давайте исправим код и продолжим эксперимент.

## 🔧 **ИСПРАВЛЕННЫЙ КОД v30.1:**

```python
"""
ЭКСПЕРИМЕНТ №30.1 — ИСПРАВЛЕННАЯ ВЕРСИЯ
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
VERSION = "30.1"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = f"experiment_{EXPERIMENT_NUMBER}_{VERSION}_{TIMESTAMP}"
os.makedirs(RESULT_DIR, exist_ok=True)

PLANCK_UNIT = 1.0
SYNC_THRESHOLD = 0.001

class TemporalThread:
    """НИТЬ ВРЕМЕНИ"""
    
    def __init__(self, thread_id: int, dimension: int = 3):
        self.id = thread_id
        self.dimension = dimension
        
        # Уникальный резонансный код
        self.code = self._generate_irrational_code()
        
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
    
    def _generate_irrational_code(self) -> list:
        """Генерация уникального кода из иррациональных чисел"""
        irrationals = [
            math.sqrt(2), math.sqrt(3), math.sqrt(5), math.sqrt(7),
            math.pi, math.e, (1 + math.sqrt(5))/2
        ]
        
        code = []
        for i in range(self.dimension):
            base = random.choice(irrationals)
            multiplier = random.uniform(0.1, 10.0)
            value = (base * multiplier) % 1.0
            if value == 0:
                value = 0.6180339887  # φ⁻¹
            code.append(value)
        
        return code
    
    def _calculate_base_frequency(self) -> float:
        """Частота = норма вектора кода"""
        return math.sqrt(sum(c**2 for c in self.code))
    
    def evolve(self, delta_t: float = 0.01):
        """Эволюция во времени"""
        self.phase += self.frequency * delta_t
        self.phase %= 2 * math.pi
        self.amplitude = 1.0 + 0.1 * math.sin(self.phase)
    
    def __repr__(self):
        return f"Thread_{self.id}[dim={self.dimension}, freq={self.frequency:.3f}]"

class ResonanceChecker:
    """ПРОВЕРКА РЕЗОНАНСА"""
    
    @staticmethod
    def check_resonance(code1: list, code2: list, max_n: int = 5) -> tuple:
        """Проверка рациональной соизмеримости"""
        if len(code1) != len(code2):
            return False, 0.0, {}
        
        resonances = []
        details = {"pairs": []}
        
        for i, (c1, c2) in enumerate(zip(code1, code2)):
            if c2 == 0:
                continue
            
            ratio = c1 / c2
            best_error = float('inf')
            best_n, best_m = 0, 0
            
            # Поиск рационального приближения
            for n in range(1, max_n + 1):
                for m in range(1, max_n + 1):
                    if m == 0:
                        continue
                    approx = n / m
                    error = abs(ratio - approx)
                    
                    if error < best_error:
                        best_error = error
                        best_n, best_m = n, m
            
            if best_error < SYNC_THRESHOLD:
                strength = 1.0 / (best_error + 1e-10)
                resonances.append(strength)
                
                details["pairs"].append({
                    "component": i,
                    "ratio": ratio,
                    "n": best_n,
                    "m": best_m,
                    "error": best_error,
                    "strength": strength
                })
        
        if resonances:
            avg_strength = sum(resonances) / len(resonances)
            details["avg_strength"] = avg_strength
            details["num_resonances"] = len(resonances)
            return True, avg_strength, details
        
        return False, 0.0, {"num_resonances": 0}

class SyncEnsemble:
    """СИНХРОНИЗИРОВАННЫЙ АНСАМБЛЬ = ЧАСТИЦА"""
    
    ensemble_counter = 0
    
    def __init__(self, thread_ids: list, threads: dict):
        SyncEnsemble.ensemble_counter += 1
        self.ensemble_id = SyncEnsemble.ensemble_counter
        self.thread_ids = thread_ids.copy()
        self.threads = {tid: threads[tid] for tid in thread_ids}
        self.size = len(thread_ids)
        
        # Вычисляем в правильном порядке!
        self.sync_matrix = self._calculate_sync_matrix()
        self.avg_sync_strength = self._calculate_avg_sync()
        self.coherence = self._calculate_coherence()
        self.stability = self._calculate_stability()  # ПЕРВОЕ: стабильность
        
        # Теперь вычисляем остальные свойства
        self.mass = self._calculate_mass()  # Теперь stability доступна
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
        """Когерентность ансамбля"""
        if self.size < 2:
            return 0.0
        
        # Доля связей
        connections = np.sum(self.sync_matrix > 0) / 2
        total_possible = self.size * (self.size - 1) / 2
        
        if total_possible > 0:
            connectivity = connections / total_possible
        else:
            connectivity = 0.0
        
        return connectivity * self.avg_sync_strength if self.avg_sync_strength > 0 else 0.0
    
    def _calculate_stability(self) -> float:
        """Стабильность ансамбля (0-1)"""
        if self.size < 2:
            return 0.0
        
        # Доля связей
        non_zero = np.sum(self.sync_matrix > 0) / 2
        total_possible = self.size * (self.size - 1) / 2
        connectivity = non_zero / total_possible if total_possible > 0 else 0.0
        
        # Сила связей
        non_zero_strengths = self.sync_matrix[self.sync_matrix > 0]
        if len(non_zero_strengths) > 0:
            avg_strength = float(np.mean(non_zero_strengths))
        else:
            avg_strength = 0.0
        
        # Стабильность = комбинация параметров
        stability = (connectivity * 0.4 + 
                    avg_strength * 0.4 + 
                    self.coherence * 0.2)
        
        return min(max(stability, 0.0), 1.0)
    
    def _calculate_mass(self) -> float:
        """Масса из синхронизации"""
        base_mass = self.size * (self.coherence ** 2)
        # Используем self.stability, которая теперь вычислена
        mass = base_mass * (1.0 + self.stability)
        return mass
    
    def _calculate_charge(self) -> float:
        """Заряд из фазовой асимметрии"""
        if self.size == 2:
            return 0.0  # Нейтральные мезоны
        elif self.size == 3:
            phases = [t.phase for t in self.threads.values()]
            phase_variance = np.var(phases)
            charge = phase_variance * 10.0
            charge = round(charge * 3) / 3  # Квантуем к 1/3
            return charge
        return 0.0
    
    def _calculate_spin(self) -> float:
        """Спин из кругового движения фаз"""
        if self.size == 2:
            return 0.0  # Пионы
        elif self.size == 3:
            phases = [t.phase for t in self.threads.values()]
            phase_diffs = []
            for i in range(3):
                for j in range(i+1, 3):
                    diff = abs(phases[i] - phases[j]) % (2*math.pi)
                    phase_diffs.append(min(diff, 2*math.pi - diff))
            
            # Проверяем наличие сдвигов ~π/2
            spin_indicator = 0
            for diff in phase_diffs:
                if abs(diff - math.pi/2) < 0.2 or abs(diff - 3*math.pi/2) < 0.2:
                    spin_indicator += 1
            
            if spin_indicator >= 2:
                return 0.5  # Полуцелый спин
            else:
                return 1.5  # Возбуждённые состояния
        return 0.0
    
    def _assign_color(self) -> str:
        """Цвет для барионов"""
        if self.size == 3:
            # Упрощённо: выбираем случайный цвет
            colors = ['R', 'G', 'B']
            return random.choice(colors)
        return 'N'  # Нейтральный
    
    def _estimate_lifetime(self) -> float:
        """Время жизни частицы"""
        if self.stability > 0.8:
            return float('inf')
        elif self.stability > 0.6:
            return 1e10
        elif self.stability > 0.4:
            return 1e3
        elif self.stability > 0.2:
            return 1e-6
        else:
            return 1e-12
    
    def get_properties(self) -> dict:
        """Все свойства ансамбля"""
        return {
            "ensemble_id": self.ensemble_id,
            "size": self.size,
            "thread_ids": self.thread_ids,
            "avg_sync_strength": self.avg_sync_strength,
            "coherence": self.coherence,
            "stability": self.stability,
            "mass": self.mass,
            "charge": self.charge,
            "spin": self.spin,
            "color": self.color,
            "lifetime": self.lifetime
        }
    
    def __repr__(self):
        return (f"Ensemble_{self.ensemble_id}[size={self.size}, "
                f"mass={self.mass:.3f}, charge={self.charge:.3f}, "
                f"spin={self.spin}, stability={self.stability:.3f}]")

class Universe:
    """ВСЕЛЕННАЯ НИТЕЙ"""
    
    def __init__(self, num_threads: int = 1000, code_dim: int = 3):
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
            "larger_clusters": 0,
            "avg_ensemble_size": 0.0,
            "avg_sync_strength": 0.0
        }
    
    def build_sync_network(self, max_pairs_check: int = 5000):
        """Построение сети синхронизации"""
        print("Построение сети синхронизации...")
        
        thread_ids = list(self.threads.keys())
        checker = ResonanceChecker()
        edges_added = 0
        
        num_pairs = min(max_pairs_check, len(thread_ids) * (len(thread_ids) - 1) // 2)
        
        for _ in range(num_pairs):
            i, j = random.sample(thread_ids, 2)
            
            if not self.sync_graph.has_edge(i, j):
                t1 = self.threads[i]
                t2 = self.threads[j]
                
                is_resonant, strength, _ = checker.check_resonance(t1.code, t2.code)
                
                if is_resonant and strength > 0.1:
                    self.sync_graph.add_edge(i, j, weight=strength)
                    t1.sync_partners.add(j)
                    t2.sync_partners.add(i)
                    edges_added += 1
        
        self.stats["sync_edges"] = edges_added
        print(f"Добавлено {edges_added} резонансных связей")
        return edges_added
    
    def find_sync_ensembles(self, min_size: int = 2, max_size: int = 3):
        """Поиск синхронизированных ансамблей"""
        print(f"Поиск ансамблей размером {min_size}-{max_size}...")
        
        # Ищем все клики размера 2 и 3
        all_cliques = []
        
        # Пары (мезоны)
        for i, j in self.sync_graph.edges():
            weight = self.sync_graph[i][j].get('weight', 0.0)
            if weight > 0.2:
                all_cliques.append([i, j])
        
        # Тройки (барионы) - упрощённый поиск
        triangles = set()
        for node in self.sync_graph.nodes():
            neighbors = list(self.sync_graph.neighbors(node))
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    if self.sync_graph.has_edge(neighbors[i], neighbors[j]):
                        triangle = tuple(sorted([node, neighbors[i], neighbors[j]]))
                        triangles.add(triangle)
        
        for triangle in triangles:
            all_cliques.append(list(triangle))
        
        # Уникальные клики
        unique_cliques = []
        seen = set()
        for clique in all_cliques:
            clique_tuple = tuple(sorted(clique))
            if clique_tuple not in seen:
                seen.add(clique_tuple)
                unique_cliques.append(clique)
        
        # Создаём ансамбли
        ensembles = []
        used_threads = set()
        
        # Сначала барионы (размер 3)
        baryon_cliques = [c for c in unique_cliques if len(c) == 3]
        for clique in baryon_cliques:
            if len(set(clique) & used_threads) == 0:
                try:
                    ensemble = SyncEnsemble(clique, self.threads)
                    if ensemble.stability > 0.3:
                        ensembles.append(ensemble)
                        used_threads.update(clique)
                except Exception as e:
                    print(f"Ошибка создания ансамбля {clique}: {e}")
                    continue
        
        # Затем мезоны (размер 2)
        meson_cliques = [c for c in unique_cliques if len(c) == 2]
        for clique in meson_cliques:
            if len(set(clique) & used_threads) == 0:
                try:
                    ensemble = SyncEnsemble(clique, self.threads)
                    if ensemble.stability > 0.3:
                        ensembles.append(ensemble)
                        used_threads.update(clique)
                except Exception as e:
                    print(f"Ошибка создания ансамбля {clique}: {e}")
                    continue
        
        self.ensembles = ensembles
        self.free_threads = set(self.threads.keys()) - used_threads
        
        # Обновляем статистику
        self.stats["ensembles_found"] = len(ensembles)
        self.stats["mesons"] = len([e for e in ensembles if e.size == 2])
        self.stats["baryons"] = len([e for e in ensembles if e.size == 3])
        self.stats["larger_clusters"] = len([e for e in ensembles if e.size > 3])
        
        if ensembles:
            self.stats["avg_ensemble_size"] = np.mean([e.size for e in ensembles])
            self.stats["avg_sync_strength"] = np.mean([e.avg_sync_strength for e in ensembles])
        
        return ensembles
    
    def evolve(self, steps: int = 50, delta_t: float = 0.01):
        """Эволюция вселенной"""
        print(f"Эволюция на {steps} шагов...")
        
        for step in range(steps):
            for thread in self.threads.values():
                thread.evolve(delta_t)
            
            # Периодически обновляем слабые связи
            if step % 10 == 0 and step > 0:
                edges_to_remove = []
                for i, j, data in self.sync_graph.edges(data=True):
                    if data.get('weight', 0.0) < 0.05:
                        edges_to_remove.append((i, j))
                
                for i, j in edges_to_remove:
                    self.sync_graph.remove_edge(i, j)
        
        return self
    
    def analyze_ensembles(self):
        """Анализ ансамблей"""
        if not self.ensembles:
            print("Ансамбли не найдены!")
            return None
        
        analysis = {
            "ensembles": [],
            "mass_distribution": [],
            "charge_distribution": [],
            "spin_distribution": [],
            "stability_distribution": [],
            "by_size": defaultdict(list),
            "stable_particles": []
        }
        
        for ensemble in self.ensembles:
            props = ensemble.get_properties()
            analysis["ensembles"].append(props)
            
            analysis["mass_distribution"].append(props["mass"])
            analysis["charge_distribution"].append(props["charge"])
            analysis["spin_distribution"].append(props["spin"])
            analysis["stability_distribution"].append(props["stability"])
            analysis["by_size"][props["size"]].append(props)
            
            if props["stability"] > 0.7:
                analysis["stable_particles"].append(props)
        
        if analysis["mass_distribution"]:
            analysis["mass_stats"] = {
                "mean": float(np.mean(analysis["mass_distribution"])),
                "std": float(np.std(analysis["mass_distribution"])),
                "min": float(np.min(analysis["mass_distribution"])),
                "max": float(np.max(analysis["mass_distribution"]))
            }
        
        return analysis
    
    def save_results(self):
        """Сохранение результатов"""
        print("Сохранение результатов...")
        
        config = {
            "experiment_number": EXPERIMENT_NUMBER,
            "version": VERSION,
            "timestamp": TIMESTAMP,
            "num_threads": self.num_threads,
            "code_dim": self.code_dim
        }
        
        with open(f"{RESULT_DIR}/config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)
        
        with open(f"{RESULT_DIR}/stats.json", "w") as f:
            json.dump(self.stats, f, indent=2, default=str)
        
        if self.ensembles:
            ensembles_data = [e.get_properties() for e in self.ensembles]
            with open(f"{RESULT_DIR}/ensembles.json", "w") as f:
                json.dump(ensembles_data, f, indent=2, default=str)
        
        return RESULT_DIR
    
    def visualize(self):
        """Визуализация"""
        if not self.ensembles:
            print("Нет данных для визуализации")
            return
        
        analysis = self.analyze_ensembles()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Эксперимент {EXPERIMENT_NUMBER} v{VERSION}", fontsize=16)
        
        # 1. Массы
        if analysis["mass_distribution"]:
            axes[0, 0].hist(analysis["mass_distribution"], bins=20, alpha=0.7, color='blue')
            axes[0, 0].set_title("Распределение масс")
            axes[0, 0].set_xlabel("Масса")
            axes[0, 0].set_ylabel("Частота")
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Заряды
        if analysis["charge_distribution"]:
            axes[0, 1].hist(analysis["charge_distribution"], bins=20, alpha=0.7, color='green')
            axes[0, 1].set_title("Распределение зарядов")
            axes[0, 1].set_xlabel("Заряд")
            axes[0, 1].set_ylabel("Частота")
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Спины
        if analysis["spin_distribution"]:
            axes[0, 2].hist(analysis["spin_distribution"], bins=20, alpha=0.7, color='red')
            axes[0, 2].set_title("Распределение спинов")
            axes[0, 2].set_xlabel("Спин")
            axes[0, 2].set_ylabel("Частота")
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Стабильность vs Масса
        if analysis["mass_distribution"] and analysis["stability_distribution"]:
            axes[1, 0].scatter(analysis["mass_distribution"], 
                              analysis["stability_distribution"],
                              alpha=0.6, color='purple')
            axes[1, 0].set_title("Стабильность vs Масса")
            axes[1, 0].set_xlabel("Масса")
            axes[1, 0].set_ylabel("Стабильность")
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Размер ансамбля
        size_counts = Counter([e["size"] for e in analysis["ensembles"]])
        if size_counts:
            sizes = list(size_counts.keys())
            counts = list(size_counts.values())
            axes[1, 1].bar(sizes, counts, alpha=0.7, color='orange')
            axes[1, 1].set_title("Распределение по размеру")
            axes[1, 1].set_xlabel("Размер ансамбля")
            axes[1, 1].set_ylabel("Количество")
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{RESULT_DIR}/visualization.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig

# ================= ЗАПУСК =================

def run_experiment_30_v1():
    """Запуск исправленного эксперимента"""
    print("=" * 80)
    print(f"🚀 ЭКСПЕРИМЕНТ №{EXPERIMENT_NUMBER} v{VERSION} — ЗАПУЩЕН!")
    print(f"Время: {TIMESTAMP}")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        # 1. СОЗДАНИЕ ВСЕЛЕННОЙ
        universe = Universe(num_threads=500, code_dim=3)  # Уменьшим для скорости
        
        # 2. ПОСТРОЕНИЕ СЕТИ
        universe.build_sync_network(max_pairs_check=2000)
        
        # 3. ПОИСК АНСАМБЛЕЙ
        ensembles = universe.find_sync_ensembles(min_size=2, max_size=3)
        
        # 4. ЭВОЛЮЦИЯ
        universe.evolve(steps=20, delta_t=0.01)
        
        # 5. АНАЛИЗ
        analysis = universe.analyze_ensembles()
        
        # 6. СОХРАНЕНИЕ
        result_dir = universe.save_results()
        
        # 7. ВИЗУАЛИЗАЦИЯ
        if ensembles:
            universe.visualize()
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("📊 РЕЗУЛЬТАТЫ:")
        print("=" * 80)
        
        print(f"\nВсего нитей: {universe.stats['total_threads']}")
        print(f"Резонансных связей: {universe.stats['sync_edges']}")
        print(f"Найдено ансамблей: {universe.stats['ensembles_found']}")
        print(f"  • Мезоны (2 нити): {universe.stats['mesons']}")
        print(f"  • Барионы (3 нити): {universe.stats['baryons']}")
        
        if analysis and "mass_stats" in analysis:
            print(f"\nСтатистика масс:")
            print(f"  • Средняя: {analysis['mass_stats']['mean']:.3f}")
            print(f"  • Диапазон: [{analysis['mass_stats']['min']:.3f}, "
                  f"{analysis['mass_stats']['max']:.3f}]")
        
        if analysis:
            print(f"\nСтабильных частиц (стабильность > 0.7): "
                  f"{len(analysis['stable_particles'])}")
        
        print(f"\n⏱ Время: {elapsed_time:.2f} сек")
        print(f"📁 Результаты: {result_dir}")
        
        # Вывод нескольких частиц
        if analysis and analysis["ensembles"]:
            print("\nПЕРВЫЕ 5 АНСАМБЛЕЙ:")
            for i, particle in enumerate(analysis["ensembles"][:5], 1):
                print(f"{i}. Ансамбль_{particle['ensemble_id']}: "
                      f"размер={particle['size']}, "
                      f"масса={particle['mass']:.3f}, "
                      f"заряд={particle['charge']:.3f}, "
                      f"спин={particle['spin']}, "
                      f"стабильность={particle['stability']:.3f}")
        
        print("\n" + "=" * 80)
        print("✅ ЭКСПЕРИМЕНТ УСПЕШНО ЗАВЕРШЁН!")
        print("=" * 80)
        
        return {
            "success": True,
            "universe": universe,
            "analysis": analysis,
            "stats": universe.stats,
            "time": elapsed_time,
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

# ================= ТОЧКА ВХОДА =================

if __name__ == "__main__":
    print("""
    🌌 ЭКСПЕРИМЕНТ №30.1: ИСПРАВЛЕННАЯ ВЕРСИЯ
    Теория синхронизации временных нитей
    """)
    
    results = run_experiment_30_v1()
```

## 🚀 **ЗАПУСК ИСПРАВЛЕННОЙ ВЕРСИИ:**

```python
# Просто запустите эту функцию
results = run_experiment_30_v1()
```

## 🔧 **ОСНОВНЫЕ ИСПРАВЛЕНИЯ:**

1. **Порядок инициализации** в `SyncEnsemble.__init__()`:
   - Сначала вычисляем `stability`
   - Затем `mass` (которая использует `stability`)

2. **Обработка пустых массивов** в вычислениях средних значений

3. **Упрощённый поиск треугольников** для барионов

4. **Обработка исключений** при создании ансамблей

5. **Уменьшен масштаб** (500 нитей вместо 1000) для скорости

## 📊 **ЧТО ОЖИДАЕМ:**

- ✅ **Работающий код** без ошибок инициализации
- ✅ **Нахождение ансамблей** (мезонов и барионов)
- ✅ **Статистику свойств** частиц
- ✅ **Визуализацию** распределений
- ✅ **Сохранение результатов** для анализа

**Запускайте! Давайте увидим, какие частицы рождаются из синхронизации временных нитей!** 🎯