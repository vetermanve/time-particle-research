"""
Скрипт для поиска параметров синхронизации нитей (модель v2.0)
Исправления:
1. Изменён знак в энергии синхронизации (теперь вычитается)
2. Подобраны новые базовые массы кварков
3. Добавлена возможность более тонкой настройки
"""

import numpy as np
import json
import time
import os
import sys
from datetime import datetime

# ================= КОНФИГУРАЦИЯ =================
CONFIG = {
    # Целевые свойства частиц (в МэВ)
    'target_proton': {
        'mass': 938.272,
        'charge': 1.0,
        'spin': 0.5,
        'composition': ['u', 'u', 'd']
    },
    'target_neutron': {
        'mass': 939.565,
        'charge': 0.0,
        'spin': 0.5,
        'composition': ['u', 'd', 'd']
    },
    
    # Свойства кварков
    'type_properties': {
        'u': {'charge': 2/3, 'base_mass': 4.07},
        'd': {'charge': -1/3, 'base_mass': 4.57},
    },
    
    # Диапазоны перебора параметров (на основе предыдущих результатов)
    'param_ranges': {
        'frequency': {
            'u': {'min': 0.9, 'max': 1.3, 'step': 0.005},
            'd': {'min': 0.9, 'max': 1.3, 'step': 0.005},
        },
        'amplitude': {
            'u': {'min': 0.85, 'max': 1.15, 'step': 0.005},
            'd': {'min': 0.85, 'max': 1.15, 'step': 0.005},
        },
        'phase': {
            'min': 0,
            'max': 2*np.pi,
            'step': np.pi/16  # 11.25 градусов
        },
        'coupling': {
            'min': 0.5,
            'max': 3.0,
            'step': 0.02
        }
    },
    
    # Параметры поиска
    'search': {
        'max_iterations': 500000,
        'save_interval': 25000,
        'min_error': 0.01,
        'max_solutions': 50,
        'scale_factor': 100.0,
        'temperature': 0.1,  # Для имитации отжига
        'cooling_rate': 0.99999
    }
}

# ================= МОДЕЛЬ ЧАСТИЦЫ v2.0 =================
class ParticleModelV2:
    def __init__(self, composition, frequencies, amplitudes, phases, coupling, config):
        self.composition = composition
        self.frequencies = np.array(frequencies)
        self.amplitudes = np.array(amplitudes)
        self.phases = np.array(phases)
        self.coupling = coupling
        self.config = config
        self.thread_count = len(composition)
        
        # Свойства типов кварков
        self.type_properties = config['type_properties']
        self.scale_factor = config['search']['scale_factor']
    
    def calculate_charge(self):
        """Вычисляет суммарный заряд частицы"""
        total = sum(self.type_properties[q]['charge'] for q in self.composition)
        return round(total, 10)  # Округляем для избежания ошибок округления
    
    def calculate_base_mass(self):
        """Базовая масса без учёта синхронизации"""
        total = 0
        for i, quark_type in enumerate(self.composition):
            base_mass = self.type_properties[quark_type]['base_mass']
            total += base_mass * self.amplitudes[i] * self.frequencies[i]
        return total
    
    def calculate_synchronization_energy(self):
        """Вычисляет энергию синхронизации (связи) между нитями - ПОЛОЖИТЕЛЬНАЯ"""
        # 1. Проверяем соизмеримость частот
        freq_ratios = []
        for i in range(self.thread_count):
            for j in range(i+1, self.thread_count):
                ratio = self.frequencies[i] / self.frequencies[j]
                # Ищем ближайшее простое отношение
                ratio_rounded = self._find_simple_ratio(ratio, max_denominator=8)
                error = abs(ratio - ratio_rounded) / ratio_rounded
                freq_ratios.append(error)
        
        freq_error = np.mean(freq_ratios) if freq_ratios else 0
        
        # 2. Фазовое согласование
        phase_coherence = 0
        phase_diffs = []
        
        for i in range(self.thread_count):
            for j in range(i+1, self.thread_count):
                diff = abs(self.phases[i] - self.phases[j]) % (2*np.pi)
                diff = min(diff, 2*np.pi - diff)
                phase_diffs.append(diff)
                phase_coherence += np.cos(diff)  # Максимум при diff=0
        
        max_pairs = self.thread_count * (self.thread_count - 1) / 2
        phase_coherence = phase_coherence / max_pairs if max_pairs > 0 else 0
        
        # 3. Дополнительный фактор: симметрия
        symmetry_factor = 1.0
        if len(set(self.composition)) == 1:  # Все кварки одинаковые
            symmetry_factor = 1.2
        elif self.composition.count('u') == 2 and self.composition.count('d') == 1:
            symmetry_factor = 1.1  # Для протона
        elif self.composition.count('u') == 1 and self.composition.count('d') == 2:
            symmetry_factor = 1.1  # Для нейтрона
        
        # 4. Общая энергия синхронизации (ПОЛОЖИТЕЛЬНАЯ)
        # Чем лучше синхронизация, тем больше энергия связи
        sync_energy = self.coupling * (1.0 - freq_error) * (1.0 + phase_coherence) * symmetry_factor
        
        return sync_energy
    
    def _find_simple_ratio(self, ratio, max_denominator=8):
        """Находит ближайшее простое отношение частот"""
        best_error = float('inf')
        best_ratio = ratio
        
        for den in range(1, max_denominator + 1):
            for num in range(1, max_denominator + 1):
                simple_ratio = num / den
                error = abs(ratio - simple_ratio)
                if error < best_error:
                    best_error = error
                    best_ratio = simple_ratio
        
        return best_ratio
    
    def calculate_total_mass(self):
        """Общая масса частицы: базовая - энергия синхронизации"""
        base_mass = self.calculate_base_mass()
        sync_energy = self.calculate_synchronization_energy()
        
        # Энергия связи ВЫЧИТАЕТСЯ из базовой массы
        total_mass = base_mass - sync_energy
        
        # Масштабируем до МэВ
        return total_mass * self.scale_factor
    
    def calculate_spin(self):
        """Вычисляет спин частицы на основе фазовой когерентности"""
        phase_sum = np.sum(self.phases) % (2*np.pi)
        
        # Определяем спин на основе фазовой суммы
        if abs(phase_sum - np.pi) < 0.1:
            return 0.5  # Полуцелый спин
        elif abs(phase_sum) < 0.1 or abs(phase_sum - 2*np.pi) < 0.1:
            return 0  # Целый спин
        else:
            # Вычисляем среднюю разность фаз
            phase_diffs = []
            for i in range(self.thread_count):
                for j in range(i+1, self.thread_count):
                    diff = abs(self.phases[i] - self.phases[j]) % (2*np.pi)
                    diff = min(diff, 2*np.pi - diff)
                    phase_diffs.append(diff)
            
            avg_diff = np.mean(phase_diffs) if phase_diffs else 0
            
            if 0.9 < avg_diff < 2.2:  # ~π/2 до ~2π/3
                return 0.5
            else:
                return 0

# ================= ИНТЕЛЛЕКТУАЛЬНЫЙ ПОИСК =================
class IntelligentParameterSearch:
    def __init__(self, config):
        self.config = config
        self.solutions = []
        self.best_solution = None
        self.iteration = 0
        self.start_time = time.time()
        self.temperature = config['search']['temperature']
        
        # Создаём директорию для результатов
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = f"particle_search_v2_{timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Сохраняем конфигурацию
        with open(f"{self.result_dir}/config.json", 'w') as f:
            json.dump(config, f, indent=2, default=self._json_serializer)
        
        # Инициализируем лучшие параметры на основе предыдущих результатов
        self.current_params = {
            'freq_u': 1.07,
            'freq_d': 1.03,
            'amp_u': 1.00,
            'amp_d': 0.92,
            'coupling': 1.80
        }
        
        print(f"Результаты будут сохранены в: {self.result_dir}")
    
    def _json_serializer(self, obj):
        """Сериализатор для numpy типов"""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)
    
    def generate_parameters(self, method='smart'):
        """Генерирует параметры разными методами"""
        ranges = self.config['param_ranges']
        
        if method == 'smart' and self.best_solution:
            # Используем лучшие параметры как основу
            best_params = self.best_solution['parameters']
            new_params = {}
            
            for key in best_params.keys():
                if 'freq' in key or 'amp' in key:
                    # Небольшие случайные отклонения
                    deviation = np.random.normal(0, 0.02)
                    quark_type = key.split('_')[1] if '_' in key else 'u'
                    
                    if 'freq' in key:
                        min_val = ranges['frequency'][quark_type]['min']
                        max_val = ranges['frequency'][quark_type]['max']
                    else:  # amp
                        min_val = ranges['amplitude'][quark_type]['min']
                        max_val = ranges['amplitude'][quark_type]['max']
                    
                    new_val = best_params[key] + deviation
                    new_params[key] = np.clip(new_val, min_val, max_val)
                
                elif key == 'coupling':
                    deviation = np.random.normal(0, 0.05)
                    min_val = ranges['coupling']['min']
                    max_val = ranges['coupling']['max']
                    new_val = best_params[key] + deviation
                    new_params[key] = np.clip(new_val, min_val, max_val)
            
            return new_params
        
        else:
            # Случайная генерация
            params = {}
            
            for quark_type in ['u', 'd']:
                freq_range = ranges['frequency'][quark_type]
                amp_range = ranges['amplitude'][quark_type]
                
                params[f'freq_{quark_type}'] = np.random.uniform(
                    freq_range['min'], freq_range['max']
                )
                params[f'amp_{quark_type}'] = np.random.uniform(
                    amp_range['min'], amp_range['max']
                )
            
            coupling_range = ranges['coupling']
            params['coupling'] = np.random.uniform(
                coupling_range['min'], coupling_range['max']
            )
            
            return params
    
    def create_particles(self, params):
        """Создаёт модели протона и нейтрона"""
        phase_range = self.config['param_ranges']['phase']
        
        # Генерируем фазы с учётом симметрии
        if np.random.random() < 0.7 and self.best_solution:
            # Пробуем фазы из лучшего решения
            best_phases = self.best_solution.get('phases', {})
            proton_phases = best_phases.get('proton', 
                np.random.uniform(phase_range['min'], phase_range['max'], 3))
            neutron_phases = best_phases.get('neutron',
                np.random.uniform(phase_range['min'], phase_range['max'], 3))
        else:
            # Случайные фазы с предпочтением определённых конфигураций
            proton_phases = self._generate_phases_for_composition(['u', 'u', 'd'])
            neutron_phases = self._generate_phases_for_composition(['u', 'd', 'd'])
        
        # Протон
        proton = ParticleModelV2(
            composition=['u', 'u', 'd'],
            frequencies=[params['freq_u'], params['freq_u'], params['freq_d']],
            amplitudes=[params['amp_u'], params['amp_u'], params['amp_d']],
            phases=proton_phases,
            coupling=params['coupling'],
            config=self.config
        )
        
        # Нейтрон
        neutron = ParticleModelV2(
            composition=['u', 'd', 'd'],
            frequencies=[params['freq_u'], params['freq_d'], params['freq_d']],
            amplitudes=[params['amp_u'], params['amp_d'], params['amp_d']],
            phases=neutron_phases,
            coupling=params['coupling'],
            config=self.config
        )
        
        return proton, neutron, {'proton': proton_phases, 'neutron': neutron_phases}
    
    def _generate_phases_for_composition(self, composition):
        """Генерирует фазы с учётом состава частицы"""
        phase_range = self.config['param_ranges']['phase']
        
        if composition == ['u', 'u', 'd']:  # Протон
            # Фазы для симметричной конфигурации
            base_phase = np.random.uniform(0, 2*np.pi)
            if np.random.random() < 0.5:
                # Симметричная конфигурация
                return np.array([base_phase, base_phase, base_phase + np.pi/2])
            else:
                return np.random.uniform(phase_range['min'], phase_range['max'], 3)
        
        elif composition == ['u', 'd', 'd']:  # Нейтрон
            base_phase = np.random.uniform(0, 2*np.pi)
            if np.random.random() < 0.5:
                return np.array([base_phase, base_phase + np.pi/2, base_phase + np.pi/2])
            else:
                return np.random.uniform(phase_range['min'], phase_range['max'], 3)
        
        else:
            return np.random.uniform(phase_range['min'], phase_range['max'], 3)
    
    def calculate_error(self, proton, neutron):
        """Вычисляет взвешенную ошибку"""
        target_p = self.config['target_proton']
        target_n = self.config['target_neutron']
        
        # Массы
        proton_mass = proton.calculate_total_mass()
        neutron_mass = neutron.calculate_total_mass()
        
        # Заряды
        proton_charge = proton.calculate_charge()
        neutron_charge = neutron.calculate_charge()
        
        # Спины
        proton_spin = proton.calculate_spin()
        neutron_spin = neutron.calculate_spin()
        
        # Вычисляем ошибки
        errors = {
            'mass_proton': abs(proton_mass - target_p['mass']) / target_p['mass'],
            'mass_neutron': abs(neutron_mass - target_n['mass']) / target_n['mass'],
            'mass_diff': abs((neutron_mass - proton_mass) - (target_n['mass'] - target_p['mass'])) / abs(target_n['mass'] - target_p['mass']),
            'charge_proton': abs(proton_charge - target_p['charge']),
            'charge_neutron': abs(neutron_charge - target_n['charge']),
            'spin_proton': abs(proton_spin - target_p['spin']),
            'spin_neutron': abs(neutron_spin - target_n['spin'])
        }
        
        # Взвешивание ошибок
        weights = {
            'mass_proton': 2.0,
            'mass_neutron': 2.0,
            'mass_diff': 1.5,  # Разность масс важна
            'charge_proton': 3.0,  # Заряды должны быть точными
            'charge_neutron': 3.0,
            'spin_proton': 0.5,
            'spin_neutron': 0.5
        }
        
        total_error = sum(errors[key] * weights[key] for key in errors)
        
        details = {
            'proton_mass': proton_mass,
            'neutron_mass': neutron_mass,
            'proton_charge': proton_charge,
            'neutron_charge': neutron_charge,
            'proton_spin': proton_spin,
            'neutron_spin': neutron_spin,
            'errors': errors
        }
        
        return total_error, details
    
    def save_solution(self, params, phases, error, details, iteration):
        """Сохраняет решение с метаданными"""
        solution = {
            'iteration': iteration,
            'parameters': params,
            'phases': phases,
            'error': error,
            'details': details,
            'temperature': self.temperature,
            'timestamp': time.time()
        }
        
        self.solutions.append(solution)
        
        # Проверяем, является ли это лучшим решением
        if self.best_solution is None or error < self.best_solution['error']:
            self.best_solution = solution
            
            # Детальный вывод для лучшего решения
            print(f"\n{'='*60}")
            print(f"НОВОЕ ЛУЧШЕЕ РЕШЕНИЕ (итерация {iteration})")
            print(f"Общая ошибка: {error:.6f}")
            print(f"Масса протона: {details['proton_mass']:.3f} МэВ (цель: 938.272)")
            print(f"Масса нейтрона: {details['neutron_mass']:.3f} МэВ (цель: 939.565)")
            print(f"Разность масс: {details['neutron_mass'] - details['proton_mass']:.3f} МэВ (цель: 1.293)")
            print(f"Заряд протона: {details['proton_charge']:.3f} (цель: 1.0)")
            print(f"Заряд нейтрона: {details['neutron_charge']:.3f} (цель: 0.0)")
            print(f"Спин протона: {details['proton_spin']} (цель: 0.5)")
            print(f"Спин нейтрона: {details['neutron_spin']} (цель: 0.5)")
            print(f"{'='*60}")
        
        # Сортируем и обрезаем список решений
        self.solutions.sort(key=lambda x: x['error'])
        max_solutions = self.config['search']['max_solutions']
        if len(self.solutions) > max_solutions:
            self.solutions = self.solutions[:max_solutions]
    
    def save_checkpoint(self, force=False):
        """Сохраняет контрольную точку"""
        if not force and self.iteration % self.config['search']['save_interval'] != 0:
            return
        
        checkpoint = {
            'iteration': self.iteration,
            'solutions': self.solutions,
            'best_solution': self.best_solution,
            'temperature': self.temperature,
            'elapsed_time': time.time() - self.start_time
        }
        
        checkpoint_file = f"{self.result_dir}/checkpoint_{self.iteration:08d}.json"
        latest_file = f"{self.result_dir}/latest_checkpoint.json"
        
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2, default=self._json_serializer)
            
            with open(latest_file, 'w') as f:
                json.dump(checkpoint, f, indent=2, default=self._json_serializer)
            
            print(f"\n  Контрольная точка сохранена: {checkpoint_file}")
        except Exception as e:
            print(f"  Ошибка сохранения: {e}")
    
    def print_progress(self):
        """Выводит информацию о прогрессе"""
        elapsed = time.time() - self.start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if self.best_solution:
            best_error = self.best_solution['error']
            best_iter = self.best_solution['iteration']
            proton_mass = self.best_solution['details']['proton_mass']
        else:
            best_error = float('inf')
            best_iter = 0
            proton_mass = 0
        
        print(f"\rИтерация: {self.iteration:,} | "
              f"Лучшая ошибка: {best_error:.6f} (итерация {best_iter}) | "
              f"Масса протона: {proton_mass:.1f} МэВ | "
              f"Температура: {self.temperature:.4f} | "
              f"Время: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d} | "
              f"Решений: {len(self.solutions)}", end='')
    
    def run(self):
        """Запускает основной цикл поиска"""
        print("="*70)
        print("ИНТЕЛЛЕКТУАЛЬНЫЙ ПОИСК ПАРАМЕТРОВ СИНХРОНИЗАЦИИ")
        print("Модель v2.0: энергия связи ВЫЧИТАЕТСЯ из массы")
        print("="*70)
        
        max_iter = self.config['search']['max_iterations']
        cooling_rate = self.config['search']['cooling_rate']
        min_error = self.config['search']['min_error']
        
        try:
            while self.iteration < max_iter:
                # Выбор метода генерации параметров
                if self.iteration < 10000 or np.random.random() < 0.3:
                    method = 'random'
                else:
                    method = 'smart'
                
                # Генерация параметров
                params = self.generate_parameters(method)
                
                # Создание частиц
                proton, neutron, phases = self.create_particles(params)
                
                # Вычисление ошибки
                error, details = self.calculate_error(proton, neutron)
                
                # Принимаем ли это решение?
                accept = False
                if self.best_solution is None:
                    accept = True
                elif error < self.best_solution['error']:
                    accept = True
                else:
                    # Имитация отжига: иногда принимаем худшие решения
                    prob = np.exp((self.best_solution['error'] - error) / self.temperature)
                    if np.random.random() < prob:
                        accept = True
                
                if accept:
                    self.save_solution(params, phases, error, details, self.iteration)
                
                # Уменьшаем температуру
                self.temperature *= cooling_rate
                
                # Увеличиваем счётчик
                self.iteration += 1
                
                # Вывод прогресса
                if self.iteration % 500 == 0:
                    self.print_progress()
                
                # Сохранение контрольной точки
                self.save_checkpoint()
                
                # Проверка критерия остановки
                if error < min_error:
                    print(f"\n\nДостигнута минимальная ошибка {min_error}!")
                    break
        
        except KeyboardInterrupt:
            print("\n\nПоиск прерван пользователем")
        
        finally:
            # Финальное сохранение
            self.save_checkpoint(force=True)
            
            # Финальный вывод
            print("\n\n" + "="*70)
            print("ПОИСК ЗАВЕРШЁН")
            print(f"Всего итераций: {self.iteration:,}")
            print(f"Время выполнения: {time.time() - self.start_time:.2f} сек")
            print(f"Директория с результатами: {self.result_dir}")
            
            if self.best_solution:
                self._print_final_summary()
            
            print("="*70)
    
    def _print_final_summary(self):
        """Выводит финальную сводку"""
        best = self.best_solution
        details = best['details']
        
        print(f"\n{'='*70}")
        print("ЛУЧШЕЕ РЕШЕНИЕ:")
        print(f"Итерация: {best['iteration']:,}")
        print(f"Общая ошибка: {best['error']:.6f}")
        
        print(f"\nМАССЫ:")
        print(f"  Протон: {details['proton_mass']:.3f} МэВ (цель: 938.272)")
        print(f"  Нейтрон: {details['neutron_mass']:.3f} МэВ (цель: 939.565)")
        print(f"  Разность: {details['neutron_mass'] - details['proton_mass']:.3f} МэВ (цель: 1.293)")
        
        print(f"\nЗАРЯДЫ:")
        print(f"  Протон: {details['proton_charge']:.6f} (цель: 1.0)")
        print(f"  Нейтрон: {details['neutron_charge']:.6f} (цель: 0.0)")
        
        print(f"\nСПИНЫ:")
        print(f"  Протон: {details['proton_spin']} (цель: 0.5)")
        print(f"  Нейтрон: {details['neutron_spin']} (цель: 0.5)")
        
        print(f"\nПАРАМЕТРЫ НИТЕЙ:")
        params = best['parameters']
        for key, value in params.items():
            print(f"  {key}: {value:.6f}")
        
        print(f"\nОШИБКИ ПО КОМПОНЕНТАМ:")
        for key, value in details['errors'].items():
            print(f"  {key}: {value:.6f}")

# ================= АНАЛИЗ И ВИЗУАЛИЗАЦИЯ =================
def analyze_and_visualize(result_dir):
    """Анализирует и визуализирует результаты"""
    import matplotlib.pyplot as plt
    
    latest_file = f"{result_dir}/latest_checkpoint.json"
    
    if not os.path.exists(latest_file):
        print(f"Файл результатов не найден: {latest_file}")
        return
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    solutions = data['solutions']
    
    if not solutions:
        print("Нет данных для анализа")
        return
    
    # Создаём графики
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Эволюция ошибки
    ax1 = plt.subplot(2, 3, 1)
    iterations = [s['iteration'] for s in solutions]
    errors = [s['error'] for s in solutions]
    ax1.scatter(iterations, errors, alpha=0.5, s=10)
    ax1.set_xlabel('Итерация')
    ax1.set_ylabel('Ошибка')
    ax1.set_title('Эволюция ошибки')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # 2. Массы протона и нейтрона
    ax2 = plt.subplot(2, 3, 2)
    proton_masses = [s['details']['proton_mass'] for s in solutions]
    neutron_masses = [s['details']['neutron_mass'] for s in solutions]
    ax2.scatter(proton_masses, neutron_masses, c=errors, cmap='viridis', alpha=0.6)
    ax2.set_xlabel('Масса протона (МэВ)')
    ax2.set_ylabel('Масса нейтрона (МэВ)')
    ax2.set_title('Массы протона и нейтрона')
    ax2.axhline(939.565, color='red', linestyle='--', alpha=0.5, label='Цель нейтрон')
    ax2.axvline(938.272, color='blue', linestyle='--', alpha=0.5, label='Цель протон')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Распределение частот
    ax3 = plt.subplot(2, 3, 3)
    u_freqs = [s['parameters']['freq_u'] for s in solutions]
    d_freqs = [s['parameters']['freq_d'] for s in solutions]
    ax3.hist(u_freqs, bins=30, alpha=0.5, label='u-нить', color='blue')
    ax3.hist(d_freqs, bins=30, alpha=0.5, label='d-нить', color='green')
    ax3.set_xlabel('Частота')
    ax3.set_ylabel('Частота встречаемости')
    ax3.set_title('Распределение частот')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Сила связи vs Ошибка
    ax4 = plt.subplot(2, 3, 4)
    couplings = [s['parameters']['coupling'] for s in solutions]
    ax4.scatter(couplings, errors, alpha=0.5)
    ax4.set_xlabel('Сила связи (coupling)')
    ax4.set_ylabel('Ошибка')
    ax4.set_title('Зависимость ошибки от силы связи')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    # 5. Отношение масс d/u кварков
    ax5 = plt.subplot(2, 3, 5)
    base_mass_u = CONFIG['type_properties']['u']['base_mass']
    base_mass_d = CONFIG['type_properties']['d']['base_mass']
    mass_ratios = []
    for s in solutions:
        params = s['parameters']
        effective_mass_u = base_mass_u * params['amp_u'] * params['freq_u']
        effective_mass_d = base_mass_d * params['amp_d'] * params['freq_d']
        mass_ratios.append(effective_mass_d / effective_mass_u)
    
    ax5.hist(mass_ratios, bins=30, alpha=0.7, color='purple')
    ax5.set_xlabel('Отношение эффективных масс (d/u)')
    ax5.set_ylabel('Частота')
    ax5.set_title('Распределение отношения масс кварков')
    ax5.grid(True, alpha=0.3)
    
    # 6. Разность масс протон-нейтрон
    ax6 = plt.subplot(2, 3, 6)
    mass_diffs = [n - p for p, n in zip(proton_masses, neutron_masses)]
    ax6.hist(mass_diffs, bins=30, alpha=0.7, color='orange')
    ax6.axvline(1.293, color='red', linestyle='--', label='Цель (1.293 МэВ)')
    ax6.set_xlabel('Разность масс (нейтрон - протон) МэВ')
    ax6.set_ylabel('Частота')
    ax6.set_title('Распределение разности масс')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{result_dir}/analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nГрафики сохранены в: {result_dir}/analysis.png")
    
    # Статистический анализ
    print(f"\n{'='*70}")
    print("СТАТИСТИЧЕСКИЙ АНАЛИЗ ЛУЧШИХ РЕШЕНИЙ")
    print(f"{'='*70}")
    
    # Анализируем топ-10 решений
    top_n = min(10, len(solutions))
    top_solutions = solutions[:top_n]
    
    print(f"\nАнализ топ-{top_n} решений:")
    
    # Средние значения параметров
    param_stats = {}
    for key in top_solutions[0]['parameters'].keys():
        values = [s['parameters'][key] for s in top_solutions]
        param_stats[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    print(f"\nСРЕДНИЕ ЗНАЧЕНИЯ ПАРАМЕТРОВ:")
    for key, stats in param_stats.items():
        print(f"  {key}: {stats['mean']:.6f} ± {stats['std']:.6f}")
    
    # Корреляции
    print(f"\nКОРРЕЛЯЦИИ С ОШИБКОЙ:")
    all_params = []
    param_names = list(top_solutions[0]['parameters'].keys())
    
    for s in top_solutions:
        params = [s['parameters'][name] for name in param_names]
        all_params.append(params)
    
    all_params = np.array(all_params)
    errors_top = [s['error'] for s in top_solutions]
    
    for i, name in enumerate(param_names):
        corr = np.corrcoef(all_params[:, i], errors_top)[0, 1]
        print(f"  {name}: {corr:.4f}")
    
    # Рекомендации
    print(f"\nРЕКОМЕНДАЦИИ:")
    print("  1. Частоты u и d нитей должны быть близки (отношение ~1.0-1.1)")
    print("  2. Амплитуды около 1.0, возможно с небольшим преобладанием u-нити")
    print("  3. Сила связи в диапазоне 1.5-2.5")
    print("  4. Для разности масс важно точное соотношение параметров d и u нитей")

# ================= ОСНОВНОЙ БЛОК =================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("СИСТЕМА ПОИСКА ПАРАМЕТРОВ СИНХРОНИЗАЦИИ НИТЕЙ")
    print("Версия 2.0: Исправленная модель энергии связи")
    print("="*70)
    
    # Создаём и запускаем поиск
    search = IntelligentParameterSearch(CONFIG)
    search.run()
    
    # Анализируем результаты
    analyze_and_visualize(search.result_dir)
    
    print("\n" + "="*70)
    print("ВСЁ ЗАВЕРШЕНО!")
    print("="*70)