"""
МОДЕЛЬ v10.0 - КОМПЛЕКСНАЯ МОДЕЛЬ С ГЛУБОКИМ ПОИСКОМ
"""

import numpy as np
import time
import json
import os
import sys
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from itertools import combinations
import math

# ============== КОНСТАНТЫ И УТИЛИТЫ ==============

def save_checkpoint(data, filename):
    """Сохранение чекпоинта"""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, default=json_serializer)

def load_checkpoint(filename):
    """Загрузка чекпоинта"""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def json_serializer(obj):
    """Сериализатор для JSON"""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

# ============== ФИЗИЧЕСКАЯ МОДЕЛЬ ==============

class TimeThread:
    """Нить времени (кварк)"""
    
    def __init__(self, quark_type, params):
        self.type = quark_type
        self.anti = quark_type.startswith('anti_')
        self.base_type = quark_type.replace('anti_', '')
        
        # Базовые параметры
        if self.base_type == 'u':
            self.base_mass = params['m_u']
            self.frequency = params['ω_u']
            self.amplitude = params['A_u']
        elif self.base_type == 'd':
            self.base_mass = params['m_d']
            self.frequency = params['ω_d']
            self.amplitude = params['A_d']
        else:
            self.base_mass = params.get(f'm_{self.base_type}', 2.0)
            self.frequency = params.get(f'ω_{self.base_type}', 1.0)
            self.amplitude = params.get(f'A_{self.base_type}', 1.0)
        
        # Эффективная масса
        self.effective_mass = self.base_mass * self.frequency * self.amplitude
        
        # Цвет и фаза (будут установлены позже)
        self.color = None
        self.phase = None
        
        # Заряд
        if self.base_type == 'u':
            self.charge = 2/3
        elif self.base_type == 'd':
            self.charge = -1/3
        else:
            self.charge = 0
            
        if self.anti:
            self.charge *= -1

class HadronResonator:
    """Резонансная система (адрон)"""
    
    # Цветовые матрицы (упрощенные SU(3))
    COLOR_VECTORS = {
        'R': np.array([1, 0, 0]),
        'G': np.array([0, 1, 0]),
        'B': np.array([0, 0, 1]),
        'anti_R': np.array([-1, 0, 0]),
        'anti_G': np.array([0, -1, 0]),
        'anti_B': np.array([0, 0, -1])
    }
    
    def __init__(self, name, composition, params):
        self.name = name
        self.composition = composition
        self.params = params
        self.is_meson = len(composition) == 2
        
        # Создаем нити (кварки)
        self.threads = self._create_threads()
        
        # Назначаем цвета и фазы
        self._assign_colors()
        self._assign_phases()
        
        # Вычисляем когерентности
        self.color_coherence = self._calculate_color_coherence()
        self.phase_coherence = self._calculate_phase_coherence()
        
        # Базовая масса
        self.base_mass = sum(t.effective_mass for t in self.threads)
        
        # Специфическое усиление
        self.specific_gain = self._get_specific_gain()
        
        # Квантовые параметры
        self.quantum_scale = params.get('quantum_scale', 1.0)
        self.noise_level = params.get('noise_level', 0.001)
    
    def _create_threads(self):
        """Создание нитей времени (кварков)"""
        threads = []
        for q_type in self.composition:
            thread = TimeThread(q_type, self.params)
            threads.append(thread)
        return threads
    
    def _assign_colors(self):
        """Назначение цветов"""
        if self.is_meson:
            # Мезон: кварк и антикварк
            self.threads[0].color = 'R'
            self.threads[1].color = 'anti_R'
        else:
            # Барион: три разных цвета
            colors = ['R', 'G', 'B']
            for i, thread in enumerate(self.threads):
                thread.color = colors[i % 3]
    
    def _assign_phases(self):
        """Назначение фаз"""
        if self.is_meson:
            # Фаза мезона из параметров
            base_phase = self.params.get('φ_meson', 0.0)
            self.threads[0].phase = base_phase
            self.threads[1].phase = base_phase + np.pi  # Противоположная фаза
        else:
            # Фазы барионов
            if self.name == 'proton':
                φ = self.params.get('φ_proton', 0.0)
                # uud: первые два u в фазе, d сдвинут
                self.threads[0].phase = φ  # u
                self.threads[1].phase = φ  # u
                self.threads[2].phase = φ + np.pi/2  # d
            elif self.name == 'neutron':
                φ = self.params.get('φ_neutron', 0.0)
                # udd: u в фазе 0, оба d сдвинуты
                self.threads[0].phase = φ  # u
                self.threads[1].phase = φ + np.pi/2  # d
                self.threads[2].phase = φ + np.pi/2  # d
            else:
                # Для других барионов равномерное распределение
                for i, thread in enumerate(self.threads):
                    thread.phase = i * 2*np.pi / len(self.threads)
    
    def _calculate_color_coherence(self):
        """Расчет цветовой когерентности"""
        if self.is_meson:
            # Для мезона: кварк-антикварк = сильная связь
            vec1 = self.COLOR_VECTORS[self.threads[0].color]
            vec2 = self.COLOR_VECTORS[self.threads[1].color]
            dot = np.dot(vec1, vec2)
            # Нормируем от 0 до 1
            return 1.0 - abs(dot) / 3.0
        else:
            # Для бариона: средняя попарная когерентность
            coherences = []
            for i, j in combinations(range(len(self.threads)), 2):
                vec1 = self.COLOR_VECTORS[self.threads[i].color]
                vec2 = self.COLOR_VECTORS[self.threads[j].color]
                dot = np.dot(vec1, vec2)
                coherence = 1.0 - abs(dot) / 3.0
                coherences.append(coherence)
            return np.mean(coherences)
    
    def _calculate_phase_coherence(self):
        """Расчет фазовой когерентности"""
        phases = [t.phase for t in self.threads]
        
        if len(phases) == 2:  # Мезоны
            phase_diff = abs(phases[0] - phases[1]) % (2*np.pi)
            phase_diff = min(phase_diff, 2*np.pi - phase_diff)
            # Для мезонов: противоположные фазы дают максимальную связь
            coherence = np.cos(phase_diff + np.pi)  # +π для инвертирования
            return (coherence + 1) / 2  # Нормируем от 0 до 1
        else:  # Барионы
            coherences = []
            for i, j in combinations(range(len(phases)), 2):
                phase_diff = abs(phases[i] - phases[j]) % (2*np.pi)
                phase_diff = min(phase_diff, 2*np.pi - phase_diff)
                coherence = np.cos(phase_diff)  # Для барионов без инверсии
                coherences.append((coherence + 1) / 2)  # Нормируем
            return np.mean(coherences)
    
    def _get_specific_gain(self):
        """Специфическое усиление для частицы"""
        if self.name == 'proton':
            return self.params.get('γ_proton', 1.0)
        elif self.name == 'neutron':
            return self.params.get('γ_neutron', 0.3)
        elif self.name in ['pi+', 'pi-']:
            return self.params.get('γ_pi_charged', 4.0)
        elif self.name == 'pi0':
            return self.params.get('γ_pi_neutral', 3.5)
        else:
            return 1.0
    
    def calculate_binding_energy(self):
        """Расчет энергии связи"""
        α = self.params.get('α_color', 1.0)  # Цветовая сила
        β = self.params.get('β_phase', 1.0)  # Фазовая сила
        
        # Комбинированная когерентность
        combined = (α * self.color_coherence + β * self.phase_coherence) / (α + β)
        
        # Энергия связи
        binding = combined * self.specific_gain * self.quantum_scale
        
        # Добавляем квантовые флуктуации
        noise = np.random.normal(0, self.noise_level * binding)
        
        return binding + noise
    
    def calculate_mass(self):
        """Расчет массы частицы"""
        binding = self.calculate_binding_energy()
        
        if self.is_meson:
            # Для мезонов энергия связи уменьшает массу
            raw_mass = (self.base_mass - binding) * 100.0
        else:
            # Для барионов энергия связи увеличивает массу
            raw_mass = (self.base_mass + binding) * 100.0
        
        # Гарантируем положительность
        return max(raw_mass, 1.0)

# ============== ПАРАЛЛЕЛЬНЫЙ ОПТИМИЗАТОР ==============

class DeepSearchOptimizer:
    """Оптимизатор с глубоким поиском"""
    
    def __init__(self, config):
        self.config = config
        
        # Создаем директорию для результатов
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = f"v10_results_{timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Настройка логгирования
        self.log_file = os.path.join(self.result_dir, "search.log")
        self.setup_logging()
        
        # Целевые частицы
        self.targets = {
            'proton': {'mass': 938.272, 'composition': ['u', 'u', 'd']},
            'neutron': {'mass': 939.565, 'composition': ['u', 'd', 'd']},
            'pi+': {'mass': 139.570, 'composition': ['u', 'anti_d']},
            'pi0': {'mass': 134.9768, 'composition': ['u', 'anti_u']},
            'pi-': {'mass': 139.570, 'composition': ['d', 'anti_u']},
        }
        
        # Параметры и их диапазоны
        self.param_ranges = self._init_param_ranges()
        self.param_names = list(self.param_ranges.keys())
        
        # Статистика
        self.iterations_done = 0
        self.start_time = None
        
        # Лучшие результаты
        self.best_error = float('inf')
        self.best_params = None
        self.best_masses = None
        
        # История
        self.history = []
        
        print(f"\n{'='*80}")
        print("МОДЕЛЬ v10.0 - ГЛУБОКИЙ ПОИСК")
        print(f"{'='*80}")
        print(f"Директория результатов: {self.result_dir}")
        print(f"Параметров: {len(self.param_names)}")
        print(f"Целевых частиц: {len(self.targets)}")
        print(f"{'='*80}")
    
    def _init_param_ranges(self):
        """Инициализация диапазонов параметров"""
        ranges = {
            # Базовые массы
            'm_u': (1.5, 3.0),
            'm_d': (3.0, 6.0),
            
            # Колебательные параметры
            'ω_u': (0.5, 1.5),
            'ω_d': (0.5, 1.5),
            'A_u': (0.5, 1.5),
            'A_d': (0.5, 1.5),
            
            # Коэффициенты связи
            'α_color': (0.1, 5.0),
            'β_phase': (0.1, 5.0),
            
            # Специфические усиления
            'γ_proton': (0.5, 3.0),
            'γ_neutron': (0.1, 1.0),
            'γ_pi_charged': (2.0, 8.0),
            'γ_pi_neutral': (1.5, 6.0),
            
            # Квантовые поправки
            'quantum_scale': (0.8, 1.2),
            'noise_level': (0.001, 0.01),
            
            # Фазовые сдвиги
            'φ_proton': (0, 2*np.pi),
            'φ_neutron': (0, 2*np.pi),
            'φ_meson': (0, 2*np.pi),
        }
        
        return ranges
    
    def setup_logging(self):
        """Настройка логгирования"""
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def evaluate_params(self, params):
        """Оценка параметров"""
        masses = {}
        total_error = 0.0
        
        # Вычисляем массы всех частиц
        for name, target in self.targets.items():
            hadron = HadronResonator(name, target['composition'], params)
            mass = hadron.calculate_mass()
            masses[name] = mass
            
            # Относительная ошибка
            target_mass = target['mass']
            rel_error = abs(mass - target_mass) / target_mass
            total_error += rel_error ** 2
        
        # Физические штрафы
        penalties = self._calculate_penalties(params, masses)
        total_error += penalties
        
        return total_error, masses
    
    def _calculate_penalties(self, params, masses):
        """Вычисление штрафов за нарушение физических ограничений"""
        penalties = 0.0
        
        # 1. Нейтрон должен быть тяжелее протона
        if masses['neutron'] <= masses['proton']:
            penalties += 1000.0
        
        # 2. γ_neutron < γ_proton
        if params['γ_neutron'] >= params['γ_proton']:
            penalties += 500.0
        
        # 3. γ_pi_neutral < γ_pi_charged
        if params['γ_pi_neutral'] >= params['γ_pi_charged']:
            penalties += 300.0
        
        # 4. Отношение масс кварков должно быть разумным
        u_eff = params['m_u'] * params['ω_u'] * params['A_u'] * 100
        d_eff = params['m_d'] * params['ω_d'] * params['A_d'] * 100
        ratio = d_eff / u_eff if u_eff > 0 else 1.0
        
        if ratio < 1.3 or ratio > 2.2:
            penalties += 100.0 * abs(ratio - 1.6)
        
        # 5. Разность масс n-p
        mass_diff = abs((masses['neutron'] - masses['proton']) - 1.293)
        penalties += 200.0 * mass_diff
        
        return penalties
    
    def run_hybrid_search(self, total_iterations=10000000, num_workers=8):
        """Запуск гибридного поиска"""
        self.start_time = time.time()
        self.logger.info(f"Начало гибридного поиска")
        self.logger.info(f"Всего итераций: {total_iterations:,}")
        self.logger.info(f"Количество потоков: {num_workers}")
        
        # Создаем пул процессов
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            
            # Запускаем задачи
            for worker_id in range(num_workers):
                future = executor.submit(
                    self._worker_search,
                    worker_id,
                    total_iterations // num_workers
                )
                futures.append(future)
            
            # Собираем результаты
            best_global_error = float('inf')
            best_global_params = None
            
            for future in as_completed(futures):
                try:
                    worker_error, worker_params = future.result()
                    
                    if worker_error < best_global_error:
                        best_global_error = worker_error
                        best_global_params = worker_params
                        
                        # Сохраняем лучший результат
                        self._save_best_result(worker_error, worker_params)
                        
                        self.logger.info(f"Новый лучший результат: {worker_error:.6f}")
                
                except Exception as e:
                    self.logger.error(f"Ошибка в воркере: {e}")
        
        # Финальный отчет
        self._final_report(best_global_error, best_global_params)
        
        return best_global_params, best_global_error
    
    def _worker_search(self, worker_id, iterations):
        """Поиск в отдельном процессе"""
        np.random.seed(worker_id + int(time.time()))
        
        best_error = float('inf')
        best_params = None
        
        # Начальные параметры
        current_params = self._random_params()
        current_error, _ = self.evaluate_params(current_params)
        
        # Параметры отжига
        temperature = 10.0
        cooling_rate = 0.999999
        
        for i in range(iterations):
            # Генерация новых параметров
            new_params = self._mutate_params(current_params, temperature)
            new_error, _ = self.evaluate_params(new_params)
            
            # Критерий Метрополиса
            if new_error < current_error:
                current_params = new_params
                current_error = new_error
            else:
                delta = new_error - current_error
                if delta < 0 or np.random.random() < math.exp(-delta / temperature):
                    current_params = new_params
                    current_error = new_error
            
            # Обновление лучшего
            if new_error < best_error:
                best_error = new_error
                best_params = new_params.copy()
            
            # Охлаждение
            temperature *= cooling_rate
            
            # Прогресс
            if i % 100000 == 0:
                elapsed = time.time() - self.start_time
                self.logger.info(
                    f"Воркер {worker_id}: {i:,}/{iterations:,} | "
                    f"Темп: {temperature:.4f} | "
                    f"Ошибка: {current_error:.3f} | "
                    f"Лучшая: {best_error:.3f}"
                )
        
        return best_error, best_params
    
    def _random_params(self):
        """Случайные параметры в заданных диапазонах"""
        params = {}
        for name, (min_val, max_val) in self.param_ranges.items():
            params[name] = np.random.uniform(min_val, max_val)
        return params
    
    def _mutate_params(self, params, temperature):
        """Мутация параметров с учетом температуры"""
        new_params = params.copy()
        
        for name, (min_val, max_val) in self.param_ranges.items():
            current_val = params[name]
            step = (max_val - min_val) * 0.1 * temperature
            mutation = np.random.normal(0, step)
            new_val = current_val + mutation
            
            # Отражающие границы
            while new_val < min_val or new_val > max_val:
                if new_val < min_val:
                    new_val = 2 * min_val - new_val
                if new_val > max_val:
                    new_val = 2 * max_val - new_val
            
            new_params[name] = new_val
        
        return new_params
    
    def _save_best_result(self, error, params):
        """Сохранение лучшего результата"""
        # Вычисляем массы для лучших параметров
        _, masses = self.evaluate_params(params)
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'error': error,
            'parameters': params,
            'masses': masses,
            'iterations': self.iterations_done
        }
        
        filename = os.path.join(self.result_dir, f"best_result_{error:.6f}.json")
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2, default=json_serializer)
    
    def _final_report(self, error, params):
        """Финальный отчет"""
        if params is None:
            self.logger.error("Не найдено допустимых параметров")
            return
        
        # Вычисляем массы
        _, masses = self.evaluate_params(params)
        
        elapsed = time.time() - self.start_time
        hours = elapsed / 3600
        
        print(f"\n{'='*80}")
        print("ФИНАЛЬНЫЙ ОТЧЕТ v10.0")
        print(f"{'='*80}")
        print(f"Общее время поиска: {hours:.1f} часов")
        print(f"Лучшая ошибка: {error:.6f}")
        
        # Расчет эффективных масс
        u_eff = params['m_u'] * params['ω_u'] * params['A_u'] * 100
        d_eff = params['m_d'] * params['ω_d'] * params['A_d'] * 100
        
        print(f"\nЭффективные массы кварков:")
        print(f"  u: {u_eff:.2f} МэВ")
        print(f"  d: {d_eff:.2f} МэВ")
        print(f"  Отношение d/u: {d_eff/u_eff:.3f}")
        
        print(f"\nМассы частиц:")
        total_error = 0
        for name in self.targets:
            mass = masses[name]
            target = self.targets[name]['mass']
            error_pct = abs(mass - target) / target * 100
            total_error += error_pct
            status = "✓" if error_pct < 0.1 else "⚠" if error_pct < 1.0 else "✗"
            print(f"  {status} {name:6}: {mass:8.3f} МэВ (цель {target:7.3f}) - {error_pct:5.2f}%")
        
        avg_error = total_error / len(self.targets)
        print(f"\nСредняя ошибка: {avg_error:.2f}%")
        
        # Разность масс n-p
        diff = masses['neutron'] - masses['proton']
        print(f"\nРазность масс n-p: {diff:.3f} МэВ (цель 1.293 МэВ)")
        
        print(f"\nРезультаты сохранены в: {self.result_dir}")
        print(f"{'='*80}")

# ============== ЗАПУСК ==============

def main():
    """Основная функция"""
    print("МОДЕЛЬ v10.0 - ГЛУБОКИЙ ФИЗИЧЕСКИЙ ПОИСК")
    print("="*80)
    
    # Конфигурация
    config = {
        'total_iterations': 5000000,  # 5 миллионов итераций
        'num_workers': 8,              # 8 потоков
        'checkpoint_interval': 100000  # Сохранять каждые 100к итераций
    }
    
    # Создаем оптимизатор
    optimizer = DeepSearchOptimizer(config)
    
    try:
        # Запускаем поиск
        best_params, best_error = optimizer.run_hybrid_search(
            total_iterations=config['total_iterations'],
            num_workers=config['num_workers']
        )
        
        return best_params, best_error
        
    except KeyboardInterrupt:
        print("\n\nПоиск прерван пользователем")
        return None, None
    except Exception as e:
        print(f"\nКРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    # Увеличиваем лимит рекурсии для глубоких вычислений
    sys.setrecursionlimit(10000)
    
    # Запускаем
    best_params, best_error = main()