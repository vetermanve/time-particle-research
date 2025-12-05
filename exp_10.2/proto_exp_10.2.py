"""
МОДЕЛЬ v10.0 - КОМПЛЕКСНАЯ МОДЕЛЬ С МОНИТОРИНГОМ ПРОГРЕССА В РЕАЛЬНОМ ВРЕМЕНИ
"""

import numpy as np
import time
import json
import os
import sys
from datetime import datetime, timedelta
import threading
import queue
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
import math

# ============== ВИЗУАЛИЗАЦИЯ ПРОГРЕССА ==============

class ProgressMonitor:
    """Мониторинг прогресса в реальном времени"""
    
    def __init__(self, total_iterations, num_workers, update_interval=2):
        self.total_iterations = total_iterations
        self.num_workers = num_workers
        self.update_interval = update_interval
        
        # Статистика по воркерам
        self.workers = {}
        for i in range(num_workers):
            self.workers[i] = {
                'completed': 0,
                'current_error': float('inf'),
                'best_error': float('inf'),
                'temperature': 10.0,
                'status': 'waiting'
            }
        
        # Общая статистика
        self.global_best_error = float('inf')
        self.start_time = None
        self.last_update = time.time()
        
        # Очередь для обновлений от воркеров
        self.update_queue = queue.Queue()
        
    def start(self):
        """Запуск мониторинга"""
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def update_worker(self, worker_id, data):
        """Обновление данных воркера"""
        self.update_queue.put((worker_id, data))
        
    def _monitor_loop(self):
        """Цикл мониторинга"""
        while True:
            try:
                # Обрабатываем обновления из очереди
                while not self.update_queue.empty():
                    worker_id, data = self.update_queue.get_nowait()
                    if worker_id in self.workers:
                        self.workers[worker_id].update(data)
                        
                        # Обновляем глобальный лучший результат
                        if 'best_error' in data:
                            if data['best_error'] < self.global_best_error:
                                self.global_best_error = data['best_error']
            
                # Выводим прогресс каждые update_interval секунд
                current_time = time.time()
                if current_time - self.last_update >= self.update_interval:
                    self._display_progress()
                    self.last_update = current_time
                    
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Ошибка в мониторе: {e}")
                break
    
    def _display_progress(self):
        """Отображение прогресса"""
        # Очищаем экран (работает в большинстве терминалов)
        print("\033[2J\033[H", end="")
        
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            total_completed = sum(w['completed'] for w in self.workers.values())
            iterations_per_sec = total_completed / elapsed
            
            # Расчет оставшегося времени
            if total_completed > 0:
                remaining = (self.total_iterations - total_completed) / iterations_per_sec
                eta = datetime.now() + timedelta(seconds=remaining)
                eta_str = eta.strftime("%H:%M:%S")
            else:
                remaining = float('inf')
                eta_str = "--:--:--"
            
            # Прогресс-бар
            progress = total_completed / self.total_iterations
            bar_length = 50
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            print("═" * 80)
            print("🚀 МОДЕЛЬ v10.0 - ПРОГРЕСС В РЕАЛЬНОМ ВРЕМЕНИ")
            print("═" * 80)
            print(f"\n📊 ОБЩАЯ СТАТИСТИКА:")
            print(f"   Итераций: {total_completed:,} / {self.total_iterations:,} "
                  f"({progress*100:.1f}%)")
            print(f"   Скорость: {iterations_per_sec:.0f} итер/сек")
            print(f"   Прошло: {elapsed:.0f} сек")
            print(f"   Осталось: {remaining:.0f} сек (ETA: {eta_str})")
            print(f"   Лучшая ошибка: {self.global_best_error:.6f}")
            print(f"\n   [{bar}] {progress*100:.1f}%")
            
            print(f"\n👷 СТАТУС ВОРКЕРОВ ({self.num_workers} потоков):")
            for worker_id, data in self.workers.items():
                status = data['status']
                if status == 'running':
                    status_icon = "🟢"
                elif status == 'finished':
                    status_icon = "✅"
                else:
                    status_icon = "⚪"
                
                worker_progress = data['completed'] / (self.total_iterations / self.num_workers)
                print(f"   {status_icon} Воркер {worker_id}: "
                      f"{data['completed']:,} итер | "
                      f"Темп: {data['temperature']:.2f} | "
                      f"Ошибка: {data['current_error']:.3f} | "
                      f"Лучшая: {data['best_error']:.3f}")
            
            print(f"\n🎯 ЦЕЛЕВЫЕ ПАРАМЕТРЫ:")
            print("   • Точность масс: < 0.1%")
            print("   • Разность масс n-p: 1.293 МэВ")
            print("   • Отношение m_d/m_u: 1.5-2.0")
            print("═" * 80)
            print("ℹ️  Для прерывания нажмите Ctrl+C")
            
            # Автосохранение каждые 30 секунд
            if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                print(f"\n💾 Автосохранение прогресса...")
                sys.stdout.flush()
        
    def stop(self):
        """Остановка мониторинга"""
        if hasattr(self, 'monitor_thread'):
            self._display_progress()  # Финальное отображение
            print("\n\n✅ Мониторинг завершен")

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
        
        # Цвет и фаза
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
        
        # Создаем нити
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
    
    def _create_threads(self):
        threads = []
        for q_type in self.composition:
            thread = TimeThread(q_type, self.params)
            threads.append(thread)
        return threads
    
    def _assign_colors(self):
        if self.is_meson:
            self.threads[0].color = 'R'
            self.threads[1].color = 'anti_R'
        else:
            colors = ['R', 'G', 'B']
            for i, thread in enumerate(self.threads):
                thread.color = colors[i % 3]
    
    def _assign_phases(self):
        if self.is_meson:
            base_phase = self.params.get('φ_meson', 0.0)
            self.threads[0].phase = base_phase
            self.threads[1].phase = base_phase + np.pi
        else:
            if self.name == 'proton':
                φ = self.params.get('φ_proton', 0.0)
                self.threads[0].phase = φ
                self.threads[1].phase = φ
                self.threads[2].phase = φ + np.pi/2
            elif self.name == 'neutron':
                φ = self.params.get('φ_neutron', 0.0)
                self.threads[0].phase = φ
                self.threads[1].phase = φ + np.pi/2
                self.threads[2].phase = φ + np.pi/2
    
    def _calculate_color_coherence(self):
        if self.is_meson:
            vec1 = self.COLOR_VECTORS[self.threads[0].color]
            vec2 = self.COLOR_VECTORS[self.threads[1].color]
            dot = np.dot(vec1, vec2)
            return 1.0 - abs(dot) / 3.0
        else:
            coherences = []
            for i, j in combinations(range(len(self.threads)), 2):
                vec1 = self.COLOR_VECTORS[self.threads[i].color]
                vec2 = self.COLOR_VECTORS[self.threads[j].color]
                dot = np.dot(vec1, vec2)
                coherence = 1.0 - abs(dot) / 3.0
                coherences.append(coherence)
            return np.mean(coherences)
    
    def _calculate_phase_coherence(self):
        phases = [t.phase for t in self.threads]
        
        if len(phases) == 2:
            phase_diff = abs(phases[0] - phases[1]) % (2*np.pi)
            phase_diff = min(phase_diff, 2*np.pi - phase_diff)
            coherence = np.cos(phase_diff + np.pi)
            return (coherence + 1) / 2
        else:
            coherences = []
            for i, j in combinations(range(len(phases)), 2):
                phase_diff = abs(phases[i] - phases[j]) % (2*np.pi)
                phase_diff = min(phase_diff, 2*np.pi - phase_diff)
                coherence = np.cos(phase_diff)
                coherences.append((coherence + 1) / 2)
            return np.mean(coherences)
    
    def _get_specific_gain(self):
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
        α = self.params.get('α_color', 1.0)
        β = self.params.get('β_phase', 1.0)
        
        combined = (α * self.color_coherence + β * self.phase_coherence) / (α + β)
        binding = combined * self.specific_gain * self.params.get('quantum_scale', 1.0)
        
        noise = np.random.normal(0, self.params.get('noise_level', 0.001) * binding)
        return binding + noise
    
    def calculate_mass(self):
        binding = self.calculate_binding_energy()
        
        if self.is_meson:
            raw_mass = (self.base_mass - binding) * 100.0
        else:
            raw_mass = (self.base_mass + binding) * 100.0
        
        return max(raw_mass, 1.0)

# ============== ОПТИМИЗАТОР ==============

class DeepSearchOptimizer:
    def __init__(self, config):
        self.config = config
        
        # Создаем директорию для результатов
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = f"v10_results_{timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Инициализируем монитор прогресса
        self.progress_monitor = ProgressMonitor(
            total_iterations=config['total_iterations'],
            num_workers=config['num_workers'],
            update_interval=config.get('update_interval', 2)
        )
        
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
        
        # Лучшие результаты
        self.best_error = float('inf')
        self.best_params = None
        self.best_masses = None
        
        print(f"\n{'='*80}")
        print("🚀 МОДЕЛЬ v10.0 - ГЛУБОКИЙ ПОИСК С МОНИТОРИНГОМ")
        print(f"{'='*80}")
        print(f"📁 Директория: {self.result_dir}")
        print(f"⚙️  Параметров: {len(self.param_names)}")
        print(f"🎯 Частиц: {len(self.targets)}")
        print(f"🔄 Итераций: {config['total_iterations']:,}")
        print(f"👷 Потоков: {config['num_workers']}")
        print(f"{'='*80}")
        print("\nМониторинг прогресса запускается...")
        print("Для прерывания нажмите Ctrl+C\n")
    
    def _init_param_ranges(self):
        ranges = {
            'm_u': (1.5, 3.0),
            'm_d': (3.0, 6.0),
            'ω_u': (0.5, 1.5),
            'ω_d': (0.5, 1.5),
            'A_u': (0.5, 1.5),
            'A_d': (0.5, 1.5),
            'α_color': (0.1, 5.0),
            'β_phase': (0.1, 5.0),
            'γ_proton': (0.5, 3.0),
            'γ_neutron': (0.1, 1.0),
            'γ_pi_charged': (2.0, 8.0),
            'γ_pi_neutral': (1.5, 6.0),
            'quantum_scale': (0.8, 1.2),
            'noise_level': (0.001, 0.01),
            'φ_proton': (0, 2*np.pi),
            'φ_neutron': (0, 2*np.pi),
            'φ_meson': (0, 2*np.pi),
        }
        return ranges
    
    def evaluate_params(self, params):
        masses = {}
        total_error = 0.0
        
        for name, target in self.targets.items():
            hadron = HadronResonator(name, target['composition'], params)
            mass = hadron.calculate_mass()
            masses[name] = mass
            
            target_mass = target['mass']
            rel_error = abs(mass - target_mass) / target_mass
            total_error += rel_error ** 2
        
        # Физические штрафы
        penalties = self._calculate_penalties(params, masses)
        total_error += penalties
        
        return total_error, masses
    
    def _calculate_penalties(self, params, masses):
        penalties = 0.0
        
        if masses['neutron'] <= masses['proton']:
            penalties += 1000.0
        
        if params['γ_neutron'] >= params['γ_proton']:
            penalties += 500.0
        
        if params['γ_pi_neutral'] >= params['γ_pi_charged']:
            penalties += 300.0
        
        u_eff = params['m_u'] * params['ω_u'] * params['A_u'] * 100
        d_eff = params['m_d'] * params['ω_d'] * params['A_d'] * 100
        ratio = d_eff / u_eff if u_eff > 0 else 1.0
        
        if ratio < 1.3 or ratio > 2.2:
            penalties += 100.0 * abs(ratio - 1.6)
        
        mass_diff = abs((masses['neutron'] - masses['proton']) - 1.293)
        penalties += 200.0 * mass_diff
        
        return penalties
    
    def _worker_search(self, worker_id, iterations, start_params=None):
        """Поиск в отдельном процессе"""
        np.random.seed(worker_id + int(time.time()))
        
        best_error = float('inf')
        best_params = None
        
        # Начальные параметры
        if start_params:
            current_params = start_params.copy()
        else:
            current_params = self._random_params()
        
        current_error, _ = self.evaluate_params(current_params)
        
        # Параметры отжига
        temperature = 10.0
        cooling_rate = 0.999999
        
        for i in range(iterations):
            # Обновляем прогресс каждые 1000 итераций
            if i % 1000 == 0:
                self.progress_monitor.update_worker(worker_id, {
                    'completed': i,
                    'current_error': current_error,
                    'best_error': best_error,
                    'temperature': temperature,
                    'status': 'running'
                })
            
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
        
        # Финальное обновление прогресса
        self.progress_monitor.update_worker(worker_id, {
            'completed': iterations,
            'current_error': current_error,
            'best_error': best_error,
            'temperature': temperature,
            'status': 'finished'
        })
        
        return best_error, best_params
    
    def _random_params(self):
        params = {}
        for name, (min_val, max_val) in self.param_ranges.items():
            params[name] = np.random.uniform(min_val, max_val)
        return params
    
    def _mutate_params(self, params, temperature):
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
    
    def run_hybrid_search(self):
        """Запуск гибридного поиска"""
        # Запускаем монитор прогресса
        self.progress_monitor.start()
        time.sleep(1)  # Даем время монитору запуститься
        
        total_iterations = self.config['total_iterations']
        num_workers = self.config['num_workers']
        iterations_per_worker = total_iterations // num_workers
        
        print(f"\n🎬 Запуск поиска...")
        print(f"   Каждый воркер выполнит {iterations_per_worker:,} итераций")
        print(f"   Общее время ожидания: ~{total_iterations/5000/num_workers:.1f} минут\n")
        
        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                
                # Запускаем воркеров
                for worker_id in range(num_workers):
                    future = executor.submit(
                        self._worker_search,
                        worker_id,
                        iterations_per_worker
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
                    
                    except Exception as e:
                        print(f"❌ Ошибка в воркере: {e}")
            
            # Останавливаем монитор
            self.progress_monitor.stop()
            
            # Финальный отчет
            self._final_report(best_global_error, best_global_params)
            
            return best_global_params, best_global_error
            
        except KeyboardInterrupt:
            print("\n\n⏹️  Поиск прерван пользователем")
            self.progress_monitor.stop()
            return None, None
    
    def _save_best_result(self, error, params):
        """Сохранение лучшего результата"""
        _, masses = self.evaluate_params(params)
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'error': error,
            'parameters': params,
            'masses': masses
        }
        
        filename = os.path.join(self.result_dir, f"best_result_{error:.6f}.json")
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2, default=self._json_serializer)
        
        # Также сохраняем как текущий лучший
        current_best = os.path.join(self.result_dir, "current_best.json")
        with open(current_best, 'w') as f:
            json.dump(result, f, indent=2, default=self._json_serializer)
    
    def _final_report(self, error, params):
        """Финальный отчет"""
        if params is None:
            print("❌ Не найдено допустимых параметров")
            return
        
        # Вычисляем массы
        _, masses = self.evaluate_params(params)
        
        print(f"\n{'='*80}")
        print("🎉 ФИНАЛЬНЫЙ ОТЧЕТ v10.0")
        print(f"{'='*80}")
        print(f"📊 Лучшая ошибка: {error:.6f}")
        
        # Расчет эффективных масс
        u_eff = params['m_u'] * params['ω_u'] * params['A_u'] * 100
        d_eff = params['m_d'] * params['ω_d'] * params['A_d'] * 100
        
        print(f"\n⚛️  ЭФФЕКТИВНЫЕ МАССЫ КВАРКОВ:")
        print(f"   u-кварк: {u_eff:.2f} МэВ")
        print(f"   d-кварк: {d_eff:.2f} МэВ")
        print(f"   Отношение m_d/m_u: {d_eff/u_eff:.3f}")
        
        print(f"\n🔧 ФИЗИЧЕСКИЕ ПАРАМЕТРЫ:")
        print(f"   α_color: {params['α_color']:.4f}")
        print(f"   β_phase: {params['β_phase']:.4f}")
        print(f"   γ_proton: {params['γ_proton']:.4f}")
        print(f"   γ_neutron: {params['γ_neutron']:.4f} (в {params['γ_proton']/params['γ_neutron']:.2f} раз меньше)")
        
        print(f"\n🎯 МАССЫ ЧАСТИЦ:")
        total_error = 0
        for name in self.targets:
            mass = masses[name]
            target = self.targets[name]['mass']
            error_pct = abs(mass - target) / target * 100
            total_error += error_pct
            status = "✅" if error_pct < 0.1 else "⚠️ " if error_pct < 1.0 else "❌"
            print(f"   {status} {name:6}: {mass:8.3f} МэВ (цель {target:7.3f}) - {error_pct:5.2f}%")
        
        avg_error = total_error / len(self.targets)
        print(f"\n📈 Средняя ошибка: {avg_error:.2f}%")
        
        # Разность масс n-p
        diff = masses['neutron'] - masses['proton']
        diff_status = "✅" if abs(diff - 1.293) < 0.01 else "⚠️ " if abs(diff - 1.293) < 0.1 else "❌"
        print(f"\n⚖️  Разность масс n-p: {diff_status} {diff:.3f} МэВ (цель 1.293 МэВ)")
        
        print(f"\n💾 Результаты сохранены в: {self.result_dir}")
        print(f"{'='*80}")
    
    def _json_serializer(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

# ============== ЗАПУСК ==============

def main():
    """Основная функция"""
    print("🚀 МОДЕЛЬ v10.0 - ГЛУБОКИЙ ФИЗИЧЕСКИЙ ПОИСК С МОНИТОРИНГОМ")
    print("="*80)
    
    # Конфигурация
    config = {
        'total_iterations': 1000000,  # 1 миллион итераций для теста
        'num_workers': 4,              # 4 потока (можно увеличить до числа ядер)
        'update_interval': 2,          # Обновление каждые 2 секунды
    }
    
    print(f"\n⚙️  Конфигурация:")
    print(f"   • Итераций: {config['total_iterations']:,}")
    print(f"   • Потоков: {config['num_workers']}")
    print(f"   • Ожидаемое время: ~{config['total_iterations']/5000/config['num_workers']:.1f} минут")
    print(f"\n📺 Мониторинг будет обновляться каждые {config['update_interval']} секунды")
    print("   Для прерывания нажмите Ctrl+C\n")
    
    input("Нажмите Enter для запуска...")
    
    # Создаем оптимизатор
    optimizer = DeepSearchOptimizer(config)
    
    try:
        # Запускаем поиск
        best_params, best_error = optimizer.run_hybrid_search()
        
        if best_params is not None:
            print(f"\n🎉 Поиск завершен успешно!")
            print(f"   Лучшая ошибка: {best_error:.6f}")
        else:
            print("\n❌ Поиск не дал результатов")
        
        return best_params, best_error
        
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    # Увеличиваем лимит рекурсии
    sys.setrecursionlimit(10000)
    
    # Запускаем
    best_params, best_error = main()