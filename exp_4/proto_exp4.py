"""
Скрипт для поиска параметров синхронизации нитей (модель v3.0)
Расширение на легкие мезоны и барионы
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
    'target_particles': {
        # БАРИОНЫ
        'proton': {
            'mass': 938.272,
            'charge': 1.0,
            'spin': 0.5,
            'composition': ['u', 'u', 'd']
        },
        'neutron': {
            'mass': 939.565,
            'charge': 0.0,
            'spin': 0.5,
            'composition': ['u', 'd', 'd']
        },
        'lambda0': {
            'mass': 1115.68,
            'charge': 0.0,
            'spin': 0.5,
            'composition': ['u', 'd', 's']
        },
        'sigma+': {
            'mass': 1189.37,
            'charge': 1.0,
            'spin': 0.5,
            'composition': ['u', 'u', 's']
        },
        'sigma0': {
            'mass': 1192.64,
            'charge': 0.0,
            'spin': 0.5,
            'composition': ['u', 'd', 's']
        },
        'sigma-': {
            'mass': 1197.45,
            'charge': -1.0,
            'spin': 0.5,
            'composition': ['d', 'd', 's']
        },
        'xi0': {
            'mass': 1314.86,
            'charge': 0.0,
            'spin': 0.5,
            'composition': ['u', 's', 's']
        },
        'xi-': {
            'mass': 1321.71,
            'charge': -1.0,
            'spin': 0.5,
            'composition': ['d', 's', 's']
        },
        
        # МЕЗОНЫ
        'pi+': {
            'mass': 139.57,
            'charge': 1.0,
            'spin': 0,
            'composition': ['u', 'anti_d']
        },
        'pi-': {
            'mass': 139.57,
            'charge': -1.0,
            'spin': 0,
            'composition': ['d', 'anti_u']
        },
        'pi0': {
            'mass': 134.98,
            'charge': 0.0,
            'spin': 0,
            'composition': ['u', 'anti_u']  # упрощенно
        },
        'k+': {
            'mass': 493.68,
            'charge': 1.0,
            'spin': 0,
            'composition': ['u', 'anti_s']
        },
        'k-': {
            'mass': 493.68,
            'charge': -1.0,
            'spin': 0,
            'composition': ['s', 'anti_u']
        },
        'k0': {
            'mass': 497.61,
            'charge': 0.0,
            'spin': 0,
            'composition': ['d', 'anti_s']
        },
        'anti_k0': {
            'mass': 497.61,
            'charge': 0.0,
            'spin': 0,
            'composition': ['s', 'anti_d']
        },
        'eta': {
            'mass': 547.86,
            'charge': 0.0,
            'spin': 0,
            'composition': ['u', 'anti_u']  # упрощенно
        },
        'eta_prime': {
            'mass': 957.78,
            'charge': 0.0,
            'spin': 0,
            'composition': ['s', 'anti_s']
        },
    },
    
    # Свойства кварков
    'type_properties': {
        'u': {'charge': 2/3, 'base_mass': 4.07},
        'd': {'charge': -1/3, 'base_mass': 4.57},
        's': {'charge': -1/3, 'base_mass': 94.0},  # Подобрано
        'anti_u': {'charge': -2/3, 'base_mass': 4.07},
        'anti_d': {'charge': 1/3, 'base_mass': 4.57},
        'anti_s': {'charge': 1/3, 'base_mass': 94.0},
    },
    
    # Диапазоны перебора параметров (оптимизированные на основе v2.2)
    'param_ranges': {
        'frequency': {
            'u': {'min': 0.95, 'max': 1.05, 'step': 0.001},
            'd': {'min': 0.92, 'max': 1.02, 'step': 0.001},
            's': {'min': 0.85, 'max': 1.05, 'step': 0.001},
        },
        'amplitude': {
            'u': {'min': 0.98, 'max': 1.02, 'step': 0.001},
            'd': {'min': 0.90, 'max': 0.95, 'step': 0.001},
            's': {'min': 0.90, 'max': 1.10, 'step': 0.001},
        },
        'phase': {
            'min': 0,
            'max': 2*np.pi,
            'step': np.pi/32  # 5.625 градусов
        },
        'coupling_baryon': {
            'min': 1.5,
            'max': 2.5,
            'step': 0.01
        },
        'coupling_meson': {
            'min': 0.5,
            'max': 1.5,
            'step': 0.01
        }
    },
    
    # Параметры поиска
    'search': {
        'max_iterations': 200000,
        'save_interval': 25000,
        'min_error': 0.05,
        'max_solutions': 100,
        'scale_factor': 100.0,
        'temperature': 0.1,
        'cooling_rate': 0.999995,
        'use_focus_search': True,  # Сначала ищем хорошие решения для нуклонов
        'focus_particles': ['proton', 'neutron', 'lambda0', 'pi+', 'k+']
    }
}

# ================= МОДЕЛЬ ЧАСТИЦЫ v3.0 =================
class ParticleModelV3:
    def __init__(self, composition, frequencies, amplitudes, phases, coupling, config, is_meson=False):
        self.composition = composition
        self.frequencies = np.array(frequencies)
        self.amplitudes = np.array(amplitudes)
        self.phases = np.array(phases)
        self.coupling = coupling
        self.config = config
        self.is_meson = is_meson
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
        """Вычисляет энергию синхронизации (связи) между нитями"""
        # 1. Проверяем соизмеримость частот
        freq_ratios = []
        for i in range(self.thread_count):
            for j in range(i+1, self.thread_count):
                ratio = self.frequencies[i] / self.frequencies[j]
                ratio_rounded = self._find_simple_ratio(ratio, max_denominator=8)
                error = abs(ratio - ratio_rounded) / ratio_rounded
                freq_ratios.append(1.0 - error)  # Чем ближе ratio к простому, тем больше
        
        freq_coherence = np.mean(freq_ratios) if freq_ratios else 0.5
        
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
        phase_coherence = (phase_coherence / max_pairs + 1) / 2 if max_pairs > 0 else 0.5
        
        # 3. Дополнительные факторы
        symmetry_factor = 1.0
        
        # Для барионов
        if self.thread_count == 3:
            if len(set(self.composition)) == 1:  # Все кварки одинаковые
                symmetry_factor = 1.15
            elif self.composition.count('u') == 2 and self.composition.count('d') == 1:
                symmetry_factor = 1.08  # Протон
            elif self.composition.count('u') == 1 and self.composition.count('d') == 2:
                symmetry_factor = 1.08  # Нейтрон
            elif 's' in self.composition:
                if self.composition.count('s') == 1:
                    symmetry_factor = 1.05  # Лямбда, Сигма
                elif self.composition.count('s') == 2:
                    symmetry_factor = 1.03  # Кси
        # Для мезонов
        elif self.thread_count == 2:
            if 's' in self.composition or 'anti_s' in self.composition:
                symmetry_factor = 1.1  # Каоны
            else:
                symmetry_factor = 1.05  # Пионы
        
        # 4. Общая энергия синхронизации
        # Для мезонов используем модифицированную формулу
        if self.is_meson:
            # Для мезонов кварк и антикварк должны быть в противофазе
            phase_diff = abs(self.phases[0] - self.phases[1]) % (2*np.pi)
            phase_diff = min(phase_diff, 2*np.pi - phase_diff)
            meson_coherence = np.cos(phase_diff + np.pi)  # Максимум при разности π
            sync_energy = self.coupling * (0.7 * freq_coherence + 0.3 * (1 + meson_coherence)/2) * symmetry_factor
        else:
            sync_energy = self.coupling * (0.6 * freq_coherence + 0.4 * phase_coherence) * symmetry_factor
        
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
        """Вычисляет спин частицы"""
        if self.is_meson:
            return 0.0  # Все мезоны в нашем списке имеют спин 0
        else:
            return 0.5  # Все барионы в нашем списке имеют спин 1/2

# ================= ИНТЕЛЛЕКТУАЛЬНЫЙ ПОИСК v3.0 =================
class IntelligentParameterSearchV3:
    def __init__(self, config):
        self.config = config
        self.solutions = []
        self.best_solution = None
        self.iteration = 0
        self.start_time = time.time()
        self.temperature = config['search']['temperature']
        
        # Создаём директорию для результатов
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = f"particle_search_v3_{timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Сохраняем конфигурацию
        with open(f"{self.result_dir}/config.json", 'w') as f:
            json.dump(config, f, indent=2, default=self._json_serializer)
        
        # Инициализируем лучшие параметры на основе v2.2
        self.current_params = {
            'freq_u': 0.982852,
            'freq_d': 0.951839,
            'freq_s': 0.95,
            'amp_u': 0.988859,
            'amp_d': 0.912337,
            'amp_s': 0.95,
            'coupling_baryon': 1.771949,
            'coupling_meson': 1.0
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
                        std = 0.01 * temp_factor
                    else:  # amp
                        min_val = ranges['amplitude'][quark_type]['min']
                        max_val = ranges['amplitude'][quark_type]['max']
                        std = 0.005 * temp_factor
                    
                    deviation = np.random.normal(0, std)
                    new_val = best_params[key] + deviation
                    new_params[key] = np.clip(new_val, min_val, max_val)
                
                elif 'coupling' in key:
                    if 'baryon' in key:
                        min_val = ranges['coupling_baryon']['min']
                        max_val = ranges['coupling_baryon']['max']
                    else:
                        min_val = ranges['coupling_meson']['min']
                        max_val = ranges['coupling_meson']['max']
                    
                    std = 0.02 * temp_factor
                    deviation = np.random.normal(0, std)
                    new_val = best_params[key] + deviation
                    new_params[key] = np.clip(new_val, min_val, max_val)
            
            return new_params
        
        else:
            # Случайная генерация в оптимизированных диапазонах
            params = {}
            
            for quark_type in ['u', 'd', 's']:
                freq_range = ranges['frequency'][quark_type]
                amp_range = ranges['amplitude'][quark_type]
                
                params[f'freq_{quark_type}'] = np.random.uniform(
                    freq_range['min'], freq_range['max']
                )
                params[f'amp_{quark_type}'] = np.random.uniform(
                    amp_range['min'], amp_range['max']
                )
            
            coupling_baryon_range = ranges['coupling_baryon']
            coupling_meson_range = ranges['coupling_meson']
            
            params['coupling_baryon'] = np.random.uniform(
                coupling_baryon_range['min'], coupling_baryon_range['max']
            )
            params['coupling_meson'] = np.random.uniform(
                coupling_meson_range['min'], coupling_meson_range['max']
            )
            
            return params
    
    def create_particle(self, particle_name, params):
        """Создаёт модель для одной частицы"""
        target = self.config['target_particles'][particle_name]
        composition = target['composition']
        is_meson = (len(composition) == 2)
        
        # Определяем фазы в зависимости от типа частицы
        phase_range = self.config['param_ranges']['phase']
        
        if is_meson:
            # Для мезонов: кварк и антикварк в противофазе
            base_phase = np.random.uniform(0, 2*np.pi)
            phases = np.array([base_phase, (base_phase + np.pi) % (2*np.pi)])
        else:
            # Для барионов: оптимизированные фазы
            base_phase = np.random.uniform(0, 2*np.pi)
            
            if composition == ['u', 'u', 'd']:  # Протон
                phases = np.array([base_phase, base_phase, (base_phase + np.pi/2) % (2*np.pi)])
            elif composition == ['u', 'd', 'd']:  # Нейтрон
                phases = np.array([base_phase, (base_phase + np.pi/2) % (2*np.pi), 
                                 (base_phase + np.pi/2) % (2*np.pi)])
            elif composition == ['u', 'd', 's']:  # Лямбда0
                phases = np.array([base_phase, (base_phase + np.pi/3) % (2*np.pi),
                                 (base_phase + 2*np.pi/3) % (2*np.pi)])
            else:
                phases = np.random.uniform(phase_range['min'], phase_range['max'], 3)
        
        # Собираем частоты и амплитуды для каждого кварка в составе
        frequencies = []
        amplitudes = []
        
        for quark_type in composition:
            base_type = quark_type.replace('anti_', '')
            frequencies.append(params[f'freq_{base_type}'])
            amplitudes.append(params[f'amp_{base_type}'])
        
        # Выбираем силу связи
        coupling = params['coupling_meson'] if is_meson else params['coupling_baryon']
        
        particle = ParticleModelV3(
            composition=composition,
            frequencies=frequencies,
            amplitudes=amplitudes,
            phases=phases,
            coupling=coupling,
            config=self.config,
            is_meson=is_meson
        )
        
        return particle, phases
    
    def create_all_particles(self, params, focus_only=False):
        """Создаёт все частицы из конфигурации"""
        particles = {}
        phases = {}
        
        particle_list = self.config['search']['focus_particles'] if focus_only else self.config['target_particles'].keys()
        
        for particle_name in particle_list:
            if particle_name in self.config['target_particles']:
                particle, particle_phases = self.create_particle(particle_name, params)
                particles[particle_name] = particle
                phases[particle_name] = particle_phases
        
        return particles, phases
    
    def calculate_error(self, particles):
        """Вычисляет взвешенную ошибку"""
        total_error = 0
        details = {}
        
        for name, particle in particles.items():
            target = self.config['target_particles'][name]
            
            # Рассчитываем свойства
            mass = particle.calculate_total_mass()
            charge = particle.calculate_charge()
            spin = particle.calculate_spin()
            
            # Вычисляем ошибки
            mass_error = abs(mass - target['mass']) / target['mass']
            charge_error = abs(charge - target['charge'])
            spin_error = abs(spin - target['spin'])
            
            # Взвешивание ошибок (разное для разных типов частиц)
            if 'pi' in name or 'k' in name or 'eta' in name:
                # Мезоны: меньше штраф за спин
                weights = {'mass': 2.0, 'charge': 3.0, 'spin': 0.5}
            elif name in ['proton', 'neutron']:
                # Нуклоны: самый высокий приоритет
                weights = {'mass': 3.0, 'charge': 5.0, 'spin': 1.0}
            else:
                # Остальные барионы
                weights = {'mass': 2.0, 'charge': 3.0, 'spin': 1.0}
            
            error = (weights['mass'] * mass_error + 
                    weights['charge'] * charge_error + 
                    weights['spin'] * spin_error)
            
            total_error += error
            
            details[name] = {
                'mass': mass,
                'charge': charge,
                'spin': spin,
                'mass_error': mass_error,
                'charge_error': charge_error,
                'spin_error': spin_error,
                'particle_error': error
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
            print(f"\n{'='*70}")
            print(f"НОВОЕ ЛУЧШЕЕ РЕШЕНИЕ (итерация {iteration})")
            print(f"Общая ошибка: {error:.6f}")
            
            # Выводим только основные частицы для краткости
            main_particles = ['proton', 'neutron', 'lambda0', 'pi+', 'k+']
            for name in main_particles:
                if name in details:
                    d = details[name]
                    target = self.config['target_particles'][name]
                    print(f"{name}: масса {d['mass']:.1f} МэВ (цель {target['mass']}), "
                          f"ошибка {d['mass_error']*100:.2f}%")
            
            print(f"{'='*70}")
        
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
            proton_mass = self.best_solution['details'].get('proton', {}).get('mass', 0)
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
        print("Модель v3.0: Расширение на мезоны и странные барионы")
        print("="*70)
        
        max_iter = self.config['search']['max_iterations']
        cooling_rate = self.config['search']['cooling_rate']
        min_error = self.config['search']['min_error']
        
        try:
            focus_phase = True  # Начинаем с фокусировки на основных частицах
            
            while self.iteration < max_iter:
                # Переключаем фазу поиска
                if focus_phase and self.iteration > 50000:
                    focus_phase = False
                    print("\nПереход к полному поиску по всем частицам")
                
                # Выбор метода генерации параметров
                if self.iteration < 10000 or np.random.random() < 0.3:
                    method = 'random'
                else:
                    method = 'smart'
                
                # Генерация параметров
                params = self.generate_parameters(method)
                
                # Создание частиц
                if focus_phase and self.config['search']['use_focus_search']:
                    particles, phases = self.create_all_particles(params, focus_only=True)
                else:
                    particles, phases = self.create_all_particles(params, focus_only=False)
                
                # Вычисление ошибки
                error, details = self.calculate_error(particles)
                
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
        
        print(f"\nОСНОВНЫЕ ЧАСТИЦЫ:")
        main_particles = ['proton', 'neutron', 'lambda0', 'pi+', 'k+']
        for name in main_particles:
            if name in details:
                d = details[name]
                target = self.config['target_particles'][name]
                print(f"\n{name.upper()}:")
                print(f"  Масса: {d['mass']:.3f} МэВ (цель: {target['mass']}, ошибка: {d['mass_error']*100:.2f}%)")
                print(f"  Заряд: {d['charge']:.3f} (цель: {target['charge']})")
                print(f"  Спин: {d['spin']} (цель: {target['spin']})")
        
        print(f"\nПАРАМЕТРЫ НИТЕЙ:")
        params = best['parameters']
        for key, value in params.items():
            print(f"  {key}: {value:.6f}")
        
        # Статистика по ошибкам
        print(f"\nСТАТИСТИКА ОШИБОК:")
        print(f"{'Частица':<15} {'Масса':<10} {'Ошибка массы':<15} {'Заряд':<10} {'Спин':<10}")
        print("-" * 60)
        
        for name, d in sorted(details.items(), key=lambda x: x[1]['particle_error']):
            target = self.config['target_particles'][name]
            print(f"{name:<15} {d['mass']:<10.1f} {d['mass_error']*100:<15.2f}% "
                  f"{d['charge']:<10.1f} {d['spin']:<10.1f}")

# ================= АНАЛИЗ РЕЗУЛЬТАТОВ =================
def analyze_results(result_dir):
    """Анализирует результаты поиска"""
    import matplotlib.pyplot as plt
    
    latest_file = f"{result_dir}/latest_checkpoint.json"
    
    if not os.path.exists(latest_file):
        print(f"Файл результатов не найден: {latest_file}")
        return
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    if 'best_solution' not in data:
        print("Нет данных для анализа")
        return
    
    best = data['best_solution']
    details = best['details']
    params = best['parameters']
    
    # Создаём отчёт
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Сравнение масс частиц
    ax1 = plt.subplot(2, 2, 1)
    particles = list(details.keys())
    masses = [details[p]['mass'] for p in particles]
    target_masses = [CONFIG['target_particles'][p]['mass'] for p in particles]
    
    x = np.arange(len(particles))
    width = 0.35
    
    ax1.bar(x - width/2, masses, width, label='Расчёт', alpha=0.7)
    ax1.bar(x + width/2, target_masses, width, label='Эксперимент', alpha=0.7)
    
    ax1.set_xlabel('Частицы')
    ax1.set_ylabel('Масса (МэВ)')
    ax1.set_title('Сравнение расчётных и экспериментальных масс')
    ax1.set_xticks(x)
    ax1.set_xticklabels(particles, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Относительные ошибки масс
    ax2 = plt.subplot(2, 2, 2)
    errors = [details[p]['mass_error'] * 100 for p in particles]
    colors = ['green' if e < 5 else 'orange' if e < 10 else 'red' for e in errors]
    
    ax2.bar(x, errors, color=colors)
    ax2.axhline(5, color='red', linestyle='--', alpha=0.5, label='5% ошибка')
    ax2.set_xlabel('Частицы')
    ax2.set_ylabel('Ошибка массы (%)')
    ax2.set_title('Относительные ошибки масс')
    ax2.set_xticks(x)
    ax2.set_xticklabels(particles, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Параметры кварков
    ax3 = plt.subplot(2, 2, 3)
    quark_params = {
        'Частота u': params['freq_u'],
        'Частота d': params['freq_d'],
        'Частота s': params['freq_s'],
        'Амплитуда u': params['amp_u'],
        'Амплитуда d': params['amp_d'],
        'Амплитуда s': params['amp_s'],
        'Связь (барионы)': params['coupling_baryon'],
        'Связь (мезоны)': params['coupling_meson']
    }
    
    ax3.bar(range(len(quark_params)), list(quark_params.values()))
    ax3.set_xlabel('Параметр')
    ax3.set_ylabel('Значение')
    ax3.set_title('Параметры модели')
    ax3.set_xticks(range(len(quark_params)))
    ax3.set_xticklabels(list(quark_params.keys()), rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Эффективные массы кварков
    ax4 = plt.subplot(2, 2, 4)
    effective_masses = {
        'u': CONFIG['type_properties']['u']['base_mass'] * params['amp_u'] * params['freq_u'] * 100,
        'd': CONFIG['type_properties']['d']['base_mass'] * params['amp_d'] * params['freq_d'] * 100,
        's': CONFIG['type_properties']['s']['base_mass'] * params['amp_s'] * params['freq_s'] * 100
    }
    
    ax4.bar(effective_masses.keys(), effective_masses.values())
    ax4.set_xlabel('Кварк')
    ax4.set_ylabel('Эффективная масса (МэВ)')
    ax4.set_title('Эффективные массы кварков в модели')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{result_dir}/analysis_v3.png', dpi=150, bbox_inches='tight')
    
    print(f"\nАнализ сохранён в: {result_dir}/analysis_v3.png")
    
    # Выводим сводку
    print(f"\n{'='*70}")
    print("СВОДКА РЕЗУЛЬТАТОВ:")
    print(f"{'='*70}")
    
    print(f"\nПараметры модели:")
    for key, value in params.items():
        print(f"  {key}: {value:.6f}")
    
    print(f"\nЭффективные массы кварков:")
    for quark, mass in effective_masses.items():
        print(f"  {quark}-кварк: {mass:.1f} МэВ")
    
    print(f"\nОтношения масс:")
    print(f"  m_d/m_u: {effective_masses['d']/effective_masses['u']:.3f}")
    print(f"  m_s/m_u: {effective_masses['s']/effective_masses['u']:.3f}")
    print(f"  m_s/m_d: {effective_masses['s']/effective_masses['d']:.3f}")

# ================= ОСНОВНОЙ БЛОК =================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("СИСТЕМА ПОИСКА ПАРАМЕТРОВ СИНХРОНИЗАЦИИ НИТЕЙ")
    print("Версия 3.0: Полная модель лёгких адронов")
    print(f"Целевые частицы: {len(CONFIG['target_particles'])}")
    print("="*70)
    
    # Создаём и запускаем поиск
    search = IntelligentParameterSearchV3(CONFIG)
    search.run()
    
    # Анализируем результаты
    analyze_results(search.result_dir)
    
    print("\n" + "="*70)
    print("МОДЕЛЬ РАСШИРЕНА НА МЕЗОНЫ И СТРАННЫЕ БАРИОНЫ!")
    print("="*70)