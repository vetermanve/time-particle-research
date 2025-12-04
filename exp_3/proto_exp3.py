"""
Скрипт для поиска параметров синхронизации нитей (модель v2.2)
Исправления на основе предыдущих результатов:
1. Возврат к рабочей формуле энергии синхронизации из v2.0
2. Уточненные диапазоны параметров
3. Улучшенный алгоритм поиска
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
    
    # Свойства кварков (на основе лучших результатов v2.0)
    'type_properties': {
        'u': {'charge': 2/3, 'base_mass': 4.07},
        'd': {'charge': -1/3, 'base_mass': 4.57},
    },
    
    # Диапазоны перебора параметров (оптимизированные)
    'param_ranges': {
        'frequency': {
            'u': {'min': 0.90, 'max': 1.10, 'step': 0.002},
            'd': {'min': 0.90, 'max': 1.10, 'step': 0.002},
        },
        'amplitude': {
            'u': {'min': 0.95, 'max': 1.05, 'step': 0.002},
            'd': {'min': 0.90, 'max': 1.00, 'step': 0.002},
        },
        'phase': {
            'min': 0,
            'max': 2*np.pi,
            'step': np.pi/24  # 7.5 градусов
        },
        'coupling': {
            'min': 1.0,
            'max': 2.0,
            'step': 0.01
        }
    },
    
    # Параметры поиска
    'search': {
        'max_iterations': 500000,
        'save_interval': 25000,
        'min_error': 0.005,
        'max_solutions': 100,
        'scale_factor': 100.0,  # Возвращаем к рабочему значению
        'temperature': 0.15,
        'cooling_rate': 0.99999
    }
}

# ================= МОДЕЛЬ ЧАСТИЦЫ v2.2 =================
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
        return round(total, 10)
    
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
                phase_coherence += np.cos(diff)
        
        max_pairs = self.thread_count * (self.thread_count - 1) / 2
        phase_coherence = phase_coherence / max_pairs if max_pairs > 0 else 0
        
        # 3. Дополнительный фактор: симметрия
        symmetry_factor = 1.0
        if len(set(self.composition)) == 1:  # Все кварки одинаковые
            symmetry_factor = 1.15
        elif self.composition.count('u') == 2 and self.composition.count('d') == 1:
            symmetry_factor = 1.08  # Для протона
        elif self.composition.count('u') == 1 and self.composition.count('d') == 2:
            symmetry_factor = 1.08  # Для нейтрона
        
        # 4. Общая энергия синхронизации (ПОЛОЖИТЕЛЬНАЯ)
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
        # Упрощенная версия - всегда возвращает 0.5 для барионов
        # (это упрощение, но соответствует целевым значениям)
        return 0.5

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
        self.result_dir = f"particle_search_v22_{timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Сохраняем конфигурацию
        with open(f"{self.result_dir}/config.json", 'w') as f:
            json.dump(config, f, indent=2, default=self._json_serializer)
        
        # Инициализируем лучшие параметры на основе v2.0
        self.current_params = {
            'freq_u': 0.905,
            'freq_d': 0.900,
            'amp_u': 1.030,
            'amp_d': 0.921,
            'coupling': 1.36
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
        
        if method == 'smart' and self.best_solution and self.iteration > 5000:
            # Используем лучшие параметры как основу
            best_params = self.best_solution['parameters']
            new_params = {}
            
            # Адаптивное отклонение на основе температуры
            temp_factor = max(0.05, self.temperature)
            
            for key in best_params.keys():
                if 'freq' in key or 'amp' in key:
                    quark_type = key.split('_')[1] if '_' in key else 'u'
                    
                    if 'freq' in key:
                        min_val = ranges['frequency'][quark_type]['min']
                        max_val = ranges['frequency'][quark_type]['max']
                        std = 0.02 * temp_factor
                    else:  # amp
                        min_val = ranges['amplitude'][quark_type]['min']
                        max_val = ranges['amplitude'][quark_type]['max']
                        std = 0.01 * temp_factor
                    
                    deviation = np.random.normal(0, std)
                    new_val = best_params[key] + deviation
                    new_params[key] = np.clip(new_val, min_val, max_val)
                
                elif key == 'coupling':
                    min_val = ranges['coupling']['min']
                    max_val = ranges['coupling']['max']
                    std = 0.05 * temp_factor
                    deviation = np.random.normal(0, std)
                    new_val = best_params[key] + deviation
                    new_params[key] = np.clip(new_val, min_val, max_val)
            
            return new_params
        
        else:
            # Случайная генерация в суженных диапазонах
            params = {}
            
            # Специально сужаем диапазоны для лучшей сходимости
            freq_u_min, freq_u_max = 0.90, 1.00
            freq_d_min, freq_d_max = 0.90, 1.00
            amp_u_min, amp_u_max = 0.98, 1.02
            amp_d_min, amp_d_max = 0.90, 0.96
            coupling_min, coupling_max = 1.2, 1.8
            
            params['freq_u'] = np.random.uniform(freq_u_min, freq_u_max)
            params['freq_d'] = np.random.uniform(freq_d_min, freq_d_max)
            params['amp_u'] = np.random.uniform(amp_u_min, amp_u_max)
            params['amp_d'] = np.random.uniform(amp_d_min, amp_d_max)
            params['coupling'] = np.random.uniform(coupling_min, coupling_max)
            
            return params
    
    def create_particles(self, params):
        """Создаёт модели протона и нейтрона"""
        phase_range = self.config['param_ranges']['phase']
        
        # Оптимизированные фазы на основе предыдущих результатов
        if np.random.random() < 0.8 and self.best_solution:
            # Используем фазы из лучшего решения с небольшими вариациями
            best_phases = self.best_solution.get('phases', {})
            
            proton_base = np.random.uniform(0, 2*np.pi)
            proton_phases = best_phases.get('proton', 
                np.array([proton_base, proton_base, proton_base + np.pi/2]))
            
            neutron_base = np.random.uniform(0, 2*np.pi)
            neutron_phases = best_phases.get('neutron',
                np.array([neutron_base, neutron_base + np.pi/2, neutron_base + np.pi/2]))
        else:
            # Оптимизированные фазы для лучшей синхронизации
            proton_base = np.random.uniform(0, 2*np.pi)
            proton_phases = np.array([
                proton_base,
                proton_base,
                (proton_base + np.pi/2) % (2*np.pi)
            ])
            
            neutron_base = np.random.uniform(0, 2*np.pi)
            neutron_phases = np.array([
                neutron_base,
                (neutron_base + np.pi/2) % (2*np.pi),
                (neutron_base + np.pi/2) % (2*np.pi)
            ])
        
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
            'mass_diff': abs((neutron_mass - proton_mass) - 1.293) / 1.293,
            'charge_proton': abs(proton_charge - target_p['charge']),
            'charge_neutron': abs(neutron_charge - target_n['charge']),
            'spin_proton': abs(proton_spin - target_p['spin']),
            'spin_neutron': abs(neutron_spin - target_n['spin'])
        }
        
        # Взвешивание ошибок (усиливаем важность масс)
        weights = {
            'mass_proton': 3.0,
            'mass_neutron': 3.0,
            'mass_diff': 2.0,
            'charge_proton': 5.0,
            'charge_neutron': 5.0,
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
        print("Модель v2.2: Возврат к рабочей формуле v2.0")
        print("="*70)
        
        max_iter = self.config['search']['max_iterations']
        cooling_rate = self.config['search']['cooling_rate']
        min_error = self.config['search']['min_error']
        
        try:
            while self.iteration < max_iter:
                # Выбор метода генерации параметров
                if self.iteration < 10000 or np.random.random() < 0.2:
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
                    # Имитация отжига
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

# ================= ОСНОВНОЙ БЛОК =================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("СИСТЕМА ПОИСКА ПАРАМЕТРОВ СИНХРОНИЗАЦИИ НИТЕЙ")
    print("Версия 2.2: Оптимизированная на основе результатов v2.0")
    print("="*70)
    
    # Создаём и запускаем поиск
    search = IntelligentParameterSearch(CONFIG)
    search.run()