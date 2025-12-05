"""
ЭКСПЕРИМЕНТ №30.3 — МАСШТАБНЫЙ ПОИСК С УМНЫМИ КРИТЕРИЯМИ
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
from itertools import combinations

# ================= КОНСТАНТЫ =================
EXPERIMENT_NUMBER = 30
VERSION = "30.3"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = f"experiment_{EXPERIMENT_NUMBER}_{VERSION}_{TIMESTAMP}"
os.makedirs(RESULT_DIR, exist_ok=True)

# Параметры эксперимента
NUM_THREADS = 2000  # В 10 раз больше!
CODE_DIM = 5        # Увеличиваем размерность кода
MAX_PAIRS_CHECK = 20000  # Увеличиваем проверки
MIN_SYNC_STRENGTH = 0.2  # Более мягкий порог
MAX_CLIQUE_SIZE = 4      # Ищем ансамбли до 4 нитей

class TemporalThread:
    """НИТЬ ВРЕМЕНИ С КОМПЛЕКСНЫМ КОДОМ"""
    
    def __init__(self, thread_id: int, dimension: int = 5):
        self.id = thread_id
        self.dimension = dimension
        
        # Генерация комплексного кода (действительная и мнимая части)
        self.code = self._generate_complex_code()
        
        # Динамические параметры
        self.phase = random.uniform(0, 2 * math.pi)
        self.amplitude = random.uniform(0.8, 1.2)  # Разные амплитуды
        self.frequency = self._calculate_base_frequency()
        
        # Состояния
        self.sync_partners = set()
        self.cluster_id = None
        self.sync_strength = 0.0
        
        # Возникающие свойства
        self.base_mass = random.uniform(0.1, 2.0)  # Базовая "масса" нити
        self.intrinsic_charge = random.choice([-1, 0, 1])  # Внутренний заряд
        self.spin_direction = random.choice([-0.5, 0.5])   # Направление спина
    
    def _generate_complex_code(self) -> list:
        """Генерация комплексного кода (действительная и мнимая части)"""
        code = []
        for i in range(self.dimension):
            # Действительная часть
            real = random.uniform(0.1, 1.0)
            
            # Мнимая часть (для фазовых соотношений)
            imag = random.uniform(0.1, 1.0)
            
            # Комплексное число в виде кортежа
            code.append((real, imag))
        
        return code
    
    def _calculate_base_frequency(self) -> float:
        """Вычисление частоты из кода"""
        total_magnitude = 0
        for real, imag in self.code:
            magnitude = math.sqrt(real**2 + imag**2)
            total_magnitude += magnitude
        
        # Нормализуем частоту
        normalized = 0.5 + 0.5 * (total_magnitude / (self.dimension * math.sqrt(2)))
        return min(max(normalized, 0.1), 1.0)
    
    def get_code_magnitudes(self) -> list:
        """Возвращает величины кодов (для сравнения)"""
        return [math.sqrt(r**2 + i**2) for r, i in self.code]
    
    def get_code_phases(self) -> list:
        """Возвращает фазы кодов"""
        phases = []
        for real, imag in self.code:
            if real == 0:
                phase = math.pi/2 if imag > 0 else -math.pi/2
            else:
                phase = math.atan2(imag, real)
            phases.append(phase)
        return phases
    
    def evolve(self, delta_t: float = 0.01):
        """Эволюция с взаимодействием"""
        # Базовая эволюция
        self.phase += self.frequency * delta_t
        self.phase %= 2 * math.pi
        
        # Влияние синхронизированных партнёров
        if self.sync_partners:
            avg_partner_phase = 0
            for partner_id in self.sync_partners:
                avg_partner_phase += random.random()  # Упрощённо
            
            # Стремление к синхронизации
            phase_diff = avg_partner_phase - self.phase
            self.phase += 0.01 * math.sin(phase_diff)
        
        # Небольшие флуктуации
        self.amplitude *= (1.0 + random.uniform(-0.01, 0.01))
        self.amplitude = max(0.5, min(self.amplitude, 1.5))
    
    def __repr__(self):
        return f"Thread_{self.id}[freq={self.frequency:.3f}, amp={self.amplitude:.3f}]"

class ResonanceChecker:
    """ПРОВЕРКА РЕЗОНАНСА С КОМПЛЕКСНЫМИ КОДАМИ"""
    
    @staticmethod
    def check_complex_resonance(code1: list, code2: list, max_n: int = 7) -> tuple:
        """
        Проверка резонанса между комплексными кодами.
        Считаем резонанс, если фазы и величины соотносятся рационально.
        """
        magnitudes1 = [math.sqrt(r**2 + i**2) for r, i in code1]
        magnitudes2 = [math.sqrt(r**2 + i**2) for r, i in code2]
        
        phases1 = []
        phases2 = []
        for (r1, i1), (r2, i2) in zip(code1, code2):
            if r1 == 0:
                phase1 = math.pi/2 if i1 > 0 else -math.pi/2
            else:
                phase1 = math.atan2(i1, r1)
            
            if r2 == 0:
                phase2 = math.pi/2 if i2 > 0 else -math.pi/2
            else:
                phase2 = math.atan2(i2, r2)
            
            phases1.append(phase1)
            phases2.append(phase2)
        
        # Проверяем резонанс величин
        magnitude_resonances = []
        for m1, m2 in zip(magnitudes1, magnitudes2):
            if m2 == 0:
                continue
            
            ratio = m1 / m2
            best_error = float('inf')
            
            for n in range(1, max_n + 1):
                for m in range(1, max_n + 1):
                    approx = n / m
                    error = abs(ratio - approx)
                    if error < best_error:
                        best_error = error
            
            if best_error < 0.05:  # 5% допуск
                strength = 1.0 / (1.0 + best_error * 20)
                magnitude_resonances.append(strength)
        
        # Проверяем резонанс фаз
        phase_resonances = []
        for p1, p2 in zip(phases1, phases2):
            phase_diff = abs(p1 - p2) % (2 * math.pi)
            
            # Ищем рациональные доли π
            for n in range(1, max_n + 1):
                for m in range(1, max_n + 1):
                    target_diff = (n / m) * math.pi
                    error = min(
                        abs(phase_diff - target_diff),
                        abs(phase_diff - target_diff - 2*math.pi),
                        abs(phase_diff - target_diff + 2*math.pi)
                    )
                    
                    if error < 0.1:  # ~5.7 градусов
                        strength = 1.0 / (1.0 + error * 10)
                        phase_resonances.append(strength)
                        break
        
        # Общая сила резонанса
        all_resonances = magnitude_resonances + phase_resonances
        
        if len(all_resonances) > 0:
            avg_strength = sum(all_resonances) / len(all_resonances)
            details = {
                "num_mag_res": len(magnitude_resonances),
                "num_phase_res": len(phase_resonances),
                "avg_mag": np.mean(magnitude_resonances) if magnitude_resonances else 0,
                "avg_phase": np.mean(phase_resonances) if phase_resonances else 0
            }
            return True, avg_strength, details
        
        return False, 0.0, {"num_mag_res": 0, "num_phase_res": 0}

class SyncEnsemble:
    """АНСАМБЛЬ С РЕАЛИСТИЧНЫМИ СВОЙСТВАМИ"""
    
    ensemble_counter = 0
    
    def __init__(self, thread_ids: list, threads: dict, ensemble_type: str = "unknown"):
        SyncEnsemble.ensemble_counter += 1
        self.ensemble_id = SyncEnsemble.ensemble_counter
        self.thread_ids = thread_ids.copy()
        self.threads = {tid: threads[tid] for tid in thread_ids}
        self.size = len(thread_ids)
        self.type = ensemble_type  # "meson", "baryon", "cluster"
        
        # Вычисляем синхронизационные параметры
        self.sync_matrix = self._calculate_sync_matrix()
        self.avg_sync_strength = self._calculate_avg_sync()
        self.coherence = self._calculate_coherence()
        self.stability = self._calculate_stability()
        
        # Вычисляем физические свойства
        self.mass = self._calculate_mass()
        self.charge = self._calculate_charge()
        self.spin = self._calculate_spin()
        self.color = self._assign_color()
        self.is_stable = self.stability > 0.6
        
        # Обновляем нити
        for tid in thread_ids:
            threads[tid].cluster_id = self.ensemble_id
            threads[tid].sync_strength = max(threads[tid].sync_strength, self.avg_sync_strength)
    
    def _calculate_sync_matrix(self) -> np.ndarray:
        """Матрица синхронизации между нитями"""
        n = self.size
        matrix = np.zeros((n, n))
        
        checker = ResonanceChecker()
        thread_list = list(self.threads.values())
        
        for i in range(n):
            for j in range(i+1, n):
                t1 = thread_list[i]
                t2 = thread_list[j]
                
                is_resonant, strength, _ = checker.check_complex_resonance(t1.code, t2.code)
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
        
        # Доля существующих связей
        connections = np.sum(self.sync_matrix > 0) / 2
        total_possible = self.size * (self.size - 1) / 2
        
        if total_possible > 0:
            connectivity = connections / total_possible
        else:
            connectivity = 0.0
        
        # Средняя сила существующих связей
        non_zero_strengths = self.sync_matrix[self.sync_matrix > 0]
        avg_strength = np.mean(non_zero_strengths) if len(non_zero_strengths) > 0 else 0.0
        
        # Когерентность = комбинация связности и силы
        coherence = (connectivity * 0.6 + avg_strength * 0.4)
        return min(max(coherence, 0.0), 1.0)
    
    def _calculate_stability(self) -> float:
        """Стабильность ансамбля"""
        if self.size < 2:
            return 0.0
        
        # Базовая стабильность из когерентности
        base_stability = self.coherence
        
        # Поправка на размер
        if self.size == 2:
            size_factor = 1.0  # Мезоны могут быть стабильными
        elif self.size == 3:
            size_factor = 0.9  # Барионы
        else:
            size_factor = 0.8  # Более крупные кластеры
        
        # Поправка на полноту связей
        connections = np.sum(self.sync_matrix > 0) / 2
        total_possible = self.size * (self.size - 1) / 2
        completeness = connections / total_possible if total_possible > 0 else 0.0
        
        stability = base_stability * size_factor * (0.7 + 0.3 * completeness)
        
        # Добавляем небольшой случайный элемент
        stability *= random.uniform(0.95, 1.05)
        
        return min(max(stability, 0.0), 1.0)
    
    def _calculate_mass(self) -> float:
        """Масса частицы"""
        # Базовая масса из массы нитей
        base_mass = sum(t.base_mass for t in self.threads.values())
        
        # Энергия связи (отрицательная - уменьшает массу)
        binding_energy = self.avg_sync_strength * self.coherence * self.size
        
        # Итоговая масса
        mass = max(0.1, base_mass - binding_energy)
        
        # Масштабируем для реалистичных значений
        scaled_mass = mass * 100.0
        
        return scaled_mass
    
    def _calculate_charge(self) -> float:
        """Электрический заряд"""
        total_charge = sum(t.intrinsic_charge for t in self.threads.values())
        
        # Квантуем заряды
        if self.size == 2:  # Мезоны
            if total_charge == 0:
                return 0.0
            elif abs(total_charge) == 1:
                return float(total_charge)
            else:
                return 0.0
        elif self.size == 3:  # Барионы
            if total_charge == 1:
                return 1.0  # Протон
            elif total_charge == 0:
                return 0.0  # Нейтрон
            elif total_charge == -1:
                return -1.0
            else:
                return float(total_charge) / 3.0
        else:
            return float(total_charge) / self.size
    
    def _calculate_spin(self) -> float:
        """Спин частицы"""
        total_spin = sum(t.spin_direction for t in self.threads.values())
        
        if self.size == 2:  # Мезоны
            if abs(total_spin) < 0.1:
                return 0.0  # Скалярные мезоны
            else:
                return 1.0  # Векторные мезоны
        elif self.size == 3:  # Барионы
            if abs(total_spin - 0.5) < 0.1 or abs(total_spin + 0.5) < 0.1:
                return 0.5  # Нуклоны
            else:
                return 1.5  # Возбуждённые состояния
        else:
            return abs(total_spin)
    
    def _assign_color(self) -> str:
        """Цветовая степень свободы"""
        if self.size == 3:
            colors = ['R', 'G', 'B']
            # Распределяем цвета по силе связи
            thread_indices = list(self.threads.keys())
            if len(thread_indices) == 3:
                return colors[random.randint(0, 2)]
        elif self.size == 2:
            return "color-anticolor"
        
        return "neutral"
    
    def get_properties(self) -> dict:
        """Возвращает все свойства ансамбля"""
        return {
            "ensemble_id": self.ensemble_id,
            "type": self.type,
            "size": self.size,
            "thread_ids": self.thread_ids,
            "avg_sync_strength": self.avg_sync_strength,
            "coherence": self.coherence,
            "stability": self.stability,
            "mass": self.mass,
            "charge": self.charge,
            "spin": self.spin,
            "color": self.color,
            "is_stable": self.is_stable
        }
    
    def __repr__(self):
        props = self.get_properties()
        return (f"{self.type.capitalize()}_{self.ensemble_id}[size={self.size}, "
                f"mass={self.mass:.1f}, charge={self.charge:.2f}, "
                f"spin={self.spin}, stability={self.stability:.2f}]")

class Universe:
    """ВСЕЛЕННАЯ С МАСШТАБНЫМ ПОИСКОМ"""
    
    def __init__(self, num_threads: int = 2000, code_dim: int = 5):
        self.num_threads = num_threads
        self.code_dim = code_dim
        
        print(f"Создание {num_threads} нитей времени (размерность кода: {code_dim})...")
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
            "clusters": 0,
            "stable_ensembles": 0,
            "avg_mass": 0.0,
            "avg_stability": 0.0
        }
    
    def build_sync_network(self, max_pairs_check: int = 20000, min_strength: float = 0.2):
        """Масштабное построение сети синхронизации"""
        print(f"Построение сети из {max_pairs_check} проверок пар...")
        
        thread_ids = list(self.threads.keys())
        checker = ResonanceChecker()
        edges_added = 0
        
        # Используем прогресс-бар
        start_time = time.time()
        
        for check_num in range(max_pairs_check):
            # Выбираем случайную пару
            i, j = random.sample(thread_ids, 2)
            
            if not self.sync_graph.has_edge(i, j):
                t1 = self.threads[i]
                t2 = self.threads[j]
                
                is_resonant, strength, details = checker.check_complex_resonance(t1.code, t2.code)
                
                if is_resonant and strength >= min_strength:
                    self.sync_graph.add_edge(i, j, weight=strength, details=details)
                    t1.sync_partners.add(j)
                    t2.sync_partners.add(i)
                    edges_added += 1
            
            # Вывод прогресса каждые 2000 проверок
            if check_num % 2000 == 0 and check_num > 0:
                elapsed = time.time() - start_time
                rate = check_num / elapsed
                print(f"  Проверено {check_num}/{max_pairs_check} пар, "
                      f"найдено {edges_added} связей ({rate:.1f} пар/сек)")
        
        self.stats["sync_edges"] = edges_added
        
        # Анализ графа
        components = list(nx.connected_components(self.sync_graph))
        self.stats["connected_components"] = len(components)
        
        if components:
            component_sizes = [len(c) for c in components]
            self.stats["largest_component"] = max(component_sizes)
            self.stats["avg_component_size"] = np.mean(component_sizes)
            print(f"  Крупнейшая компонента: {self.stats['largest_component']} нитей")
        
        print(f"Построено {edges_added} резонансных связей")
        return edges_added
    
    def find_sync_ensembles(self, min_clique_size: int = 2, max_clique_size: int = 4):
        """Поиск ансамблей через клики в графе"""
        print(f"Поиск ансамблей (клики размера {min_clique_size}-{max_clique_size})...")
        
        # Находим все максимальные клики в графе
        all_cliques = []
        
        # Для скорости, сначала находим все клики размера 2 (рёбра)
        for u, v, data in self.sync_graph.edges(data=True):
            if data.get('weight', 0) > 0.3:
                all_cliques.append([u, v])
        
        # Затем ищем клики размера 3 и 4
        # Используем приближённый алгоритм для больших графов
        for node in list(self.sync_graph.nodes()):
            neighbors = list(self.sync_graph.neighbors(node))
            
            # Ищем клики среди соседей
            for size in range(2, max_clique_size):
                for combo in combinations(neighbors, size):
                    # Проверяем, является ли combo кликой
                    is_clique = True
                    for i in range(len(combo)):
                        for j in range(i+1, len(combo)):
                            if not self.sync_graph.has_edge(combo[i], combo[j]):
                                is_clique = False
                                break
                        if not is_clique:
                            break
                    
                    if is_clique:
                        clique = [node] + list(combo)
                        if len(clique) == size + 1:  # Правильный размер
                            # Проверяем, не дубликат ли это
                            clique_tuple = tuple(sorted(clique))
                            if all(tuple(sorted(c)) != clique_tuple for c in all_cliques):
                                all_cliques.append(clique)
        
        # Фильтруем по размеру и создаём ансамбли
        ensembles = []
        used_threads = set()
        
        # Сортируем клики по размеру (сначала большие)
        all_cliques.sort(key=len, reverse=True)
        
        for clique in all_cliques:
            clique_size = len(clique)
            
            # Пропускаем если слишком большой или слишком маленький
            if clique_size < min_clique_size or clique_size > max_clique_size:
                continue
            
            # Проверяем, что нити ещё не использованы
            if len(set(clique) & used_threads) > 0:
                continue
            
            # Определяем тип ансамбля
            if clique_size == 2:
                ensemble_type = "meson"
            elif clique_size == 3:
                ensemble_type = "baryon"
            else:
                ensemble_type = f"cluster_{clique_size}"
            
            try:
                ensemble = SyncEnsemble(clique, self.threads, ensemble_type)
                
                # Принимаем ансамбль если он достаточно стабилен
                if ensemble.stability > 0.3:
                    ensembles.append(ensemble)
                    used_threads.update(clique)
                    
                    # Ограничиваем общее количество ансамблей
                    if len(ensembles) >= 100:  # Максимум 100 ансамблей
                        break
                        
            except Exception as e:
                print(f"Ошибка создания ансамбля {clique}: {e}")
                continue
        
        self.ensembles = ensembles
        self.free_threads = set(self.threads.keys()) - used_threads
        
        # Обновляем статистику
        self.stats["ensembles_found"] = len(ensembles)
        self.stats["mesons"] = len([e for e in ensembles if e.type == "meson"])
        self.stats["baryons"] = len([e for e in ensembles if e.type == "baryon"])
        self.stats["clusters"] = len([e for e in ensembles if "cluster" in e.type])
        self.stats["stable_ensembles"] = len([e for e in ensembles if e.is_stable])
        
        if ensembles:
            self.stats["avg_mass"] = np.mean([e.mass for e in ensembles])
            self.stats["avg_stability"] = np.mean([e.stability for e in ensembles])
        
        print(f"Найдено {len(ensembles)} ансамблей: "
              f"{self.stats['mesons']} мезонов, "
              f"{self.stats['baryons']} барионов, "
              f"{self.stats['clusters']} кластеров")
        
        return ensembles
    
    def evolve(self, steps: int = 5):
        """Короткая эволюция"""
        print(f"Эволюция на {steps} шагов...")
        for step in range(steps):
            for thread in self.threads.values():
                thread.evolve(0.05)
    
    def analyze(self):
        """Анализ результатов"""
        if not self.ensembles:
            return None
        
        analysis = {
            "ensembles": [e.get_properties() for e in self.ensembles],
            "by_type": defaultdict(list),
            "statistics": {}
        }
        
        # Группируем по типу
        for e in self.ensembles:
            analysis["by_type"][e.type].append(e.get_properties())
        
        # Статистика по типам
        for etype in analysis["by_type"]:
            if analysis["by_type"][etype]:
                masses = [e["mass"] for e in analysis["by_type"][etype]]
                charges = [e["charge"] for e in analysis["by_type"][etype]]
                spins = [e["spin"] for e in analysis["by_type"][etype]]
                stabilities = [e["stability"] for e in analysis["by_type"][etype]]
                
                analysis["statistics"][etype] = {
                    "count": len(masses),
                    "mass_mean": float(np.mean(masses)),
                    "mass_std": float(np.std(masses)),
                    "charge_mean": float(np.mean(charges)),
                    "spin_mean": float(np.mean(spins)),
                    "stability_mean": float(np.mean(stabilities))
                }
        
        return analysis
    
    def save_results(self):
        """Сохранение результатов"""
        print("Сохранение результатов...")
        
        config = {
            "experiment": EXPERIMENT_NUMBER,
            "version": VERSION,
            "timestamp": TIMESTAMP,
            "num_threads": self.num_threads,
            "code_dim": self.code_dim,
            "parameters": {
                "min_sync_strength": MIN_SYNC_STRENGTH,
                "max_clique_size": MAX_CLIQUE_SIZE
            }
        }
        
        with open(f"{RESULT_DIR}/config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)
        
        with open(f"{RESULT_DIR}/stats.json", "w") as f:
            json.dump(self.stats, f, indent=2, default=str)
        
        if self.ensembles:
            ensembles_data = [e.get_properties() for e in self.ensembles]
            with open(f"{RESULT_DIR}/ensembles.json", "w") as f:
                json.dump(ensembles_data, f, indent=2, default=str)
            
            # Сохраняем топ-20 ансамблей по стабильности
            stable_sorted = sorted(ensembles_data, 
                                 key=lambda x: x["stability"], reverse=True)[:20]
            with open(f"{RESULT_DIR}/top_stable.json", "w") as f:
                json.dump(stable_sorted, f, indent=2, default=str)
        
        return RESULT_DIR
    
    def visualize(self):
        """Визуализация результатов"""
        if not self.ensembles:
            print("Нет данных для визуализации")
            return
        
        analysis = self.analyze()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Эксперимент {EXPERIMENT_NUMBER} v{VERSION}\n"
                    f"{self.num_threads} нитей, {self.stats['ensembles_found']} ансамблей", 
                    fontsize=16)
        
        # 1. Распределение масс
        masses = [e["mass"] for e in analysis["ensembles"]]
        axes[0, 0].hist(masses, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title("Распределение масс", fontsize=12)
        axes[0, 0].set_xlabel("Масса", fontsize=10)
        axes[0, 0].set_ylabel("Количество", fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axvline(np.mean(masses), color='red', linestyle='--', 
                          label=f'Среднее: {np.mean(masses):.1f}')
        axes[0, 0].legend()
        
        # 2. Распределение зарядов
        charges = [e["charge"] for e in analysis["ensembles"]]
        axes[0, 1].hist(charges, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_title("Распределение зарядов", fontsize=12)
        axes[0, 1].set_xlabel("Заряд", fontsize=10)
        axes[0, 1].set_ylabel("Количество", fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Распределение спинов
        spins = [e["spin"] for e in analysis["ensembles"]]
        axes[0, 2].hist(spins, bins=30, alpha=0.7, color='red', edgecolor='black')
        axes[0, 2].set_title("Распределение спинов", fontsize=12)
        axes[0, 2].set_xlabel("Спин", fontsize=10)
        axes[0, 2].set_ylabel("Количество", fontsize=10)
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Распределение стабильности
        stabilities = [e["stability"] for e in analysis["ensembles"]]
        axes[1, 0].hist(stabilities, bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 0].set_title("Распределение стабильности", fontsize=12)
        axes[1, 0].set_xlabel("Стабильность", fontsize=10)
        axes[1, 0].set_ylabel("Количество", fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axvline(0.6, color='orange', linestyle='--', 
                          label='Порог стабильности (0.6)')
        axes[1, 0].legend()
        
        # 5. Масса vs Стабильность
        axes[1, 1].scatter(masses, stabilities, alpha=0.6, c=stabilities, 
                          cmap='viridis', s=50)
        axes[1, 1].set_title("Масса vs Стабильность", fontsize=12)
        axes[1, 1].set_xlabel("Масса", fontsize=10)
        axes[1, 1].set_ylabel("Стабильность", fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Распределение по типу и размеру
        type_counts = {}
        for e in analysis["ensembles"]:
            etype = e["type"]
            if etype not in type_counts:
                type_counts[etype] = 0
            type_counts[etype] += 1
        
        types = list(type_counts.keys())
        counts = list(type_counts.values())
        
        colors = plt.cm.tab20(np.arange(len(types)))
        bars = axes[1, 2].bar(types, counts, color=colors, alpha=0.7, edgecolor='black')
        axes[1, 2].set_title("Распределение по типу", fontsize=12)
        axes[1, 2].set_xlabel("Тип ансамбля", fontsize=10)
        axes[1, 2].set_ylabel("Количество", fontsize=10)
        axes[1, 2].grid(True, alpha=0.3)
        
        # Добавляем значения на столбцы
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{count}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f"{RESULT_DIR}/visualization.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig

def run_experiment_30_v3():
    """Запуск эксперимента 30.3"""
    print("=" * 80)
    print(f"🚀 ЭКСПЕРИМЕНТ №{EXPERIMENT_NUMBER} v{VERSION} - МАСШТАБНЫЙ ПОИСК")
    print(f"Параметры: {NUM_THREADS} нитей, {CODE_DIM}D коды")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        # 1. СОЗДАНИЕ МАСШТАБНОЙ ВСЕЛЕННОЙ
        universe = Universe(num_threads=NUM_THREADS, code_dim=CODE_DIM)
        
        # 2. ПОСТРОЕНИЕ ОБШИРНОЙ СЕТИ
        universe.build_sync_network(max_pairs_check=MAX_PAIRS_CHECK, 
                                   min_strength=MIN_SYNC_STRENGTH)
        
        # 3. ПОИСК АНСАМБЛЕЙ
        ensembles = universe.find_sync_ensembles(min_clique_size=2, 
                                                max_clique_size=MAX_CLIQUE_SIZE)
        
        # 4. КОРОТКАЯ ЭВОЛЮЦИЯ
        universe.evolve(steps=3)
        
        # 5. АНАЛИЗ
        analysis = universe.analyze()
        
        # 6. СОХРАНЕНИЕ
        result_dir = universe.save_results()
        
        # 7. ВИЗУАЛИЗАЦИЯ
        if ensembles:
            universe.visualize()
        
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("📊 РЕЗУЛЬТАТЫ v30.3:")
        print("=" * 80)
        
        print(f"\nМАСШТАБ ЭКСПЕРИМЕНТА:")
        print(f"  • Нитей: {universe.stats['total_threads']}")
        print(f"  • Связей: {universe.stats['sync_edges']}")
        print(f"  • Компонент связности: {universe.stats.get('connected_components', 0)}")
        print(f"  • Крупнейшая компонента: {universe.stats.get('largest_component', 0)} нитей")
        
        print(f"\nНАЙДЕННЫЕ АНСАМБЛИ:")
        print(f"  • Всего: {universe.stats['ensembles_found']}")
        print(f"  • Мезоны (2 нити): {universe.stats['mesons']}")
        print(f"  • Барионы (3 нити): {universe.stats['baryons']}")
        print(f"  • Кластеры (>3 нитей): {universe.stats['clusters']}")
        print(f"  • Стабильные (стабильность > 0.6): {universe.stats['stable_ensembles']}")
        
        if analysis and "statistics" in analysis:
            print(f"\nСТАТИСТИКА ПО ТИПАМ:")
            for etype, stats in analysis["statistics"].items():
                print(f"  • {etype}: {stats['count']} частиц, "
                      f"масса: {stats['mass_mean']:.1f}±{stats['mass_std']:.1f}, "
                      f"заряд: {stats['charge_mean']:.2f}, "
                      f"стабильность: {stats['stability_mean']:.2f}")
        
        if analysis and "ensembles" in analysis:
            stable_ensembles = [e for e in analysis["ensembles"] if e["is_stable"]]
            if stable_ensembles:
                print(f"\nТОП-5 СТАБИЛЬНЫХ ЧАСТИЦ:")
                stable_sorted = sorted(stable_ensembles, 
                                     key=lambda x: x["stability"], reverse=True)[:5]
                
                for i, particle in enumerate(stable_sorted, 1):
                    print(f"{i}. {particle['type'].capitalize()}_{particle['ensemble_id']}: "
                          f"размер={particle['size']}, "
                          f"масса={particle['mass']:.1f}, "
                          f"заряд={particle['charge']:.2f}, "
                          f"спин={particle['spin']}, "
                          f"стабильность={particle['stability']:.3f}")
        
        print(f"\n⏱ Время выполнения: {elapsed:.2f} сек")
        print(f"📁 Результаты сохранены в: {result_dir}")
        
        # Сохраняем итоговый отчёт
        report = {
            "summary": {
                "experiment": EXPERIMENT_NUMBER,
                "version": VERSION,
                "success": True,
                "execution_time_seconds": elapsed,
                "threads_created": NUM_THREADS,
                "ensembles_found": universe.stats["ensembles_found"],
                "stable_ensembles": universe.stats["stable_ensembles"]
            },
            "key_insights": [
                "1. Использованы комплексные коды (действительная + мнимая части)",
                "2. Масштабный поиск в пространстве 2000 нитей",
                "3. Более мягкие критерии синхронизации",
                "4. Поиск клик до размера 4"
            ],
            "next_steps": [
                "1. Калибровка массовой шкалы по известным частицам",
                "2. Введение взаимодействий между ансамблями",
                "3. Изучение эволюции ансамблей во времени",
                "4. Поиск соответствий со Стандартной моделью"
            ]
        }
        
        with open(f"{result_dir}/final_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Итоговый отчёт сохранен: {result_dir}/final_report.json")
        
        print("\n" + "=" * 80)
        print("✅ ЭКСПЕРИМЕНТ 30.3 УСПЕШНО ЗАВЕРШЁН!")
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
        print(f"\n❌ ОШИБКА В ЭКСПЕРИМЕНТЕ: {e}")
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
    🌌 ЭКСПЕРИМЕНТ №30.3: МАСШТАБНЫЙ ПОИСК
    Улучшения:
    1. 2000 нитей (в 10 раз больше!)
    2. 5-мерные комплексные коды
    3. Мягкие критерии синхронизации
    4. Поиск клик до размера 4
    5. Реалистичные физические свойства
    """)
    
    results = run_experiment_30_v3()