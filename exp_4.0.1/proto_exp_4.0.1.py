"""
Скрипт для поиска параметров синхронизации нитей (модель v4.0)
Раздельные модели для барионов и мезонов
"""

import numpy as np
import json
import time
import os
import sys
from datetime import datetime

# ================= КОНФИГУРАЦИЯ =================
CONFIG = {
    # Целевые свойства частиц (в МэВ) - ТОЛЬКО ОСНОВНЫЕ
    'target_particles': {
        # БАРИОНЫ (рабочая модель)
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
        
        # МЕЗОНЫ (новая модель)
        'pi+': {
            'mass': 139.57,
            'charge': 1.0,
            'spin': 0,
            'composition': ['u', 'anti_d'],
            'is_meson': True
        },
        'k+': {
            'mass': 493.68,
            'charge': 1.0,
            'spin': 0,
            'composition': ['u', 'anti_s'],
            'is_meson': True
        }
    },
    
    # Свойства кварков (КОРРЕКТИРОВАННЫЕ)
    'type_properties': {
        'u': {'charge': 2/3, 'base_mass': 4.07},
        'd': {'charge': -1/3, 'base_mass': 4.57},
        's': {'charge': -1/3, 'base_mass': 4.07 * 25},  # m_s/m_u ≈ 25
        'anti_u': {'charge': -2/3, 'base_mass': 4.07},
        'anti_d': {'charge': 1/3, 'base_mass': 4.57},
        'anti_s': {'charge': 1/3, 'base_mass': 4.07 * 25},
    },
    
    # Диапазоны перебора (оптимизированные)
    'param_ranges': {
        # БАРИОНЫ (используем найденные значения как основу)
        'baryon': {
            'frequency': {
                'u': {'min': 0.95, 'max': 1.05, 'step': 0.001},
                'd': {'min': 0.90, 'max': 1.00, 'step': 0.001},
                's': {'min': 0.85, 'max': 1.00, 'step': 0.001},
            },
            'amplitude': {
                'u': {'min': 0.98, 'max': 1.02, 'step': 0.001},
                'd': {'min': 0.90, 'max': 0.95, 'step': 0.001},
                's': {'min': 0.95, 'max': 1.05, 'step': 0.001},
            },
            'coupling': {
                'min': 1.5,
                'max': 2.5,
                'step': 0.01
            }
        },
        # МЕЗОНЫ (новые параметры)
        'meson': {
            'frequency': {
                'u': {'min': 0.95, 'max': 1.05, 'step': 0.001},
                'd': {'min': 0.90, 'max': 1.00, 'step': 0.001},
                's': {'min': 0.80, 'max': 0.95, 'step': 0.001},
            },
            'amplitude': {
                'u': {'min': 0.98, 'max': 1.02, 'step': 0.001},
                'd': {'min': 0.90, 'max': 0.95, 'step': 0.001},
                's': {'min': 0.90, 'max': 1.00, 'step': 0.001},
            },
            'phase_shift': {
                'min': np.pi * 0.8,
                'max': np.pi * 1.2,
                'step': 0.01
            },
            'coupling': {
                'min': 3.0,  # Для мезонов нужна большая энергия связи
                'max': 10.0,
                'step': 0.1
            }
        }
    },
    
    # Параметры поиска
    'search': {
        'max_iterations': 100000,
        'save_interval': 20000,
        'min_error': 0.01,
        'max_solutions': 50,
        'scale_factor': 100.0,
        'temperature': 0.1,
        'cooling_rate': 0.99999
    }
}

# ================= МОДЕЛЬ БАРИОНА (рабочая) =================
class BaryonModel:
    def __init__(self, composition, params, config):
        self.composition = composition
        self.config = config
        self.thread_count = 3
        
        # Параметры из найденных значений
        self.frequencies = [params['freq_' + q] for q in composition]
        self.amplitudes = [params['amp_' + q] for q in composition]
        
        # Оптимальные фазы для барионов
        base_phase = np.random.uniform(0, 2*np.pi)
        if composition == ['u', 'u', 'd']:  # Протон
            self.phases = [base_phase, base_phase, base_phase + np.pi/2]
        elif composition == ['u', 'd', 'd']:  # Нейтрон
            self.phases = [base_phase, base_phase + np.pi/2, base_phase + np.pi/2]
        elif composition == ['u', 'd', 's']:  # Лямбда0
            self.phases = [base_phase, base_phase + np.pi/3, base_phase + 2*np.pi/3]
        else:
            self.phases = [base_phase, base_phase + np.pi/2, base_phase + np.pi]
        
        self.coupling = params['coupling_baryon']
        self.type_properties = config['type_properties']
        self.scale_factor = config['search']['scale_factor']
    
    def calculate_charge(self):
        return sum(self.type_properties[q]['charge'] for q in self.composition)
    
    def calculate_base_mass(self):
        total = 0
        for i, quark_type in enumerate(self.composition):
            base_mass = self.type_properties[quark_type]['base_mass']
            total += base_mass * self.amplitudes[i] * self.frequencies[i]
        return total
    
    def calculate_synchronization_energy(self):
        """Формула из v2.2 (работает для барионов)"""
        # Частотная когерентность
        freq_ratios = []
        for i in range(self.thread_count):
            for j in range(i+1, self.thread_count):
                ratio = self.frequencies[i] / self.frequencies[j]
                ratio_rounded = self._find_simple_ratio(ratio, max_denominator=8)
                error = abs(ratio - ratio_rounded) / ratio_rounded
                freq_ratios.append(error)
        
        freq_error = np.mean(freq_ratios) if freq_ratios else 0
        
        # Фазовая когерентность
        phase_coherence = 0
        for i in range(self.thread_count):
            for j in range(i+1, self.thread_count):
                diff = abs(self.phases[i] - self.phases[j]) % (2*np.pi)
                diff = min(diff, 2*np.pi - diff)
                phase_coherence += np.cos(diff)
        
        max_pairs = 3  # Для 3 кварков всегда 3 пары
        phase_coherence = phase_coherence / max_pairs
        
        # Симметрия
        symmetry_factor = 1.0
        if self.composition.count('u') == 2 and self.composition.count('d') == 1:
            symmetry_factor = 1.08  # Протон
        elif self.composition.count('u') == 1 and self.composition.count('d') == 2:
            symmetry_factor = 1.08  # Нейтрон
        elif 's' in self.composition:
            symmetry_factor = 1.05  # Странные барионы
        
        # Энергия синхронизации
        sync_energy = self.coupling * (1.0 - freq_error) * (1.0 + phase_coherence) * symmetry_factor
        return sync_energy
    
    def _find_simple_ratio(self, ratio, max_denominator=8):
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
        base_mass = self.calculate_base_mass()
        sync_energy = self.calculate_synchronization_energy()
        return (base_mass - sync_energy) * self.scale_factor
    
    def calculate_spin(self):
        return 0.5  # Все барионы в списке имеют спин 1/2

# ================= МОДЕЛЬ МЕЗОНА (новая) =================
class MesonModel:
    def __init__(self, composition, params, config):
        self.composition = composition
        self.config = config
        
        # Кварк и антикварк
        self.quark = composition[0]
        self.antiquark = composition[1]
        
        # Параметры
        self.freq_q = params['freq_' + self.quark.replace('anti_', '')]
        self.freq_aq = params['freq_' + self.antiquark.replace('anti_', '')]
        self.amp_q = params['amp_' + self.quark.replace('anti_', '')]
        self.amp_aq = params['amp_' + self.antiquark.replace('anti_', '')]
        
        # Фазовый сдвиг (специальный параметр для мезонов)
        self.phase_shift = params.get('phase_shift', np.pi)
        
        # Сила связи для мезонов (больше, чем для барионов)
        self.coupling = params['coupling_meson']
        
        self.type_properties = config['type_properties']
        self.scale_factor = config['search']['scale_factor']
    
    def calculate_charge(self):
        return self.type_properties[self.quark]['charge'] + self.type_properties[self.antiquark]['charge']
    
    def calculate_base_mass(self):
        base_mass_q = self.type_properties[self.quark]['base_mass']
        base_mass_aq = self.type_properties[self.antiquark]['base_mass']
        return (base_mass_q * self.amp_q * self.freq_q + 
                base_mass_aq * self.amp_aq * self.freq_aq)
    
    def calculate_synchronization_energy(self):
        """НОВАЯ ФОРМУЛА для мезонов"""
        # 1. Частотное отношение (должно быть близко к 1)
        freq_ratio = self.freq_q / self.freq_aq
        freq_coherence = 1.0 - min(abs(freq_ratio - 1.0), 0.5) / 0.5
        
        # 2. Фазовая синхронизация (кварк и антикварк в ПРОТИВОФАЗЕ)
        # Чем ближе phase_shift к π, тем лучше
        phase_coherence = np.cos(self.phase_shift)  # = -1 при π
        
        # 3. Массовая асимметрия (для K-мезонов важнее)
        mass_asymmetry = 1.0
        if 's' in self.quark or 's' in self.antiquark:
            # Для странных мезонов немного уменьшаем энергию связи
            mass_asymmetry = 0.9
        
        # 4. Общая энергия синхронизации (БОЛЬШАЯ для мезонов)
        # Отрицательный phase_coherence УВЕЛИЧИВАЕТ sync_energy
        sync_energy = self.coupling * (0.4 * freq_coherence + 0.6 * (1 - phase_coherence)/2) * mass_asymmetry
        
        return sync_energy
    
    def calculate_total_mass(self):
        """Для мезонов: базовая масса МИНУС БОЛЬШАЯ энергия синхронизации"""
        base_mass = self.calculate_base_mass()
        sync_energy = self.calculate_synchronization_energy()
        total = (base_mass - sync_energy) * self.scale_factor
        
        # Гарантируем положительную массу
        return max(total, 1.0)
    
    def calculate_spin(self):
        return 0.0  # Все мезоны в списке имеют спин 0

# ================= ИНТЕЛЛЕКТУАЛЬНЫЙ ПОИСК v4.0 =================
class IntelligentParameterSearchV4:
    def __init__(self, config):
        self.config = config
        self.solutions = []
        self.best_solution = None
        self.iteration = 0
        self.start_time = time.time()
        self.temperature = config['search']['temperature']
        
        # Создаём директорию для результатов
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = f"particle_search_v4_{timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Сохраняем конфигурацию
        with open(f"{self.result_dir}/config.json", 'w') as f:
            json.dump(config, f, indent=2, default=self._json_serializer)
        
        # Начальные параметры (из лучшего решения v2.2 + новые)
        self.current_params = {
            'freq_u': 0.982852,
            'freq_d': 0.951839,
            'freq_s': 0.90,  # s-кварк немного медленнее
            'amp_u': 0.988859,
            'amp_d': 0.912337,
            'amp_s': 0.98,   # s-кварк с амплитудой близкой к 1
            'coupling_baryon': 1.771949,
            'coupling_meson': 5.0,  # Больше для мезонов
            'phase_shift': np.pi  # Противофаза для мезонов
        }
        
        print(f"Результаты будут сохранены в: {self.result_dir}")
    
    def _json_serializer(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)
    
    def generate_parameters(self, method='smart'):
        ranges = self.config['param_ranges']
        
        if method == 'smart' and self.best_solution and self.iteration > 5000:
            best_params = self.best_solution['parameters']
            new_params = {}
            temp_factor = max(0.05, self.temperature)
            
            for key in best_params.keys():
                if key in ['freq_u', 'freq_d', 'freq_s']:
                    quark = key.split('_')[1]
                    min_val = ranges['baryon']['frequency'][quark]['min']
                    max_val = ranges['baryon']['frequency'][quark]['max']
                    deviation = np.random.normal(0, 0.01 * temp_factor)
                    new_params[key] = np.clip(best_params[key] + deviation, min_val, max_val)
                
                elif key in ['amp_u', 'amp_d', 'amp_s']:
                    quark = key.split('_')[1]
                    min_val = ranges['baryon']['amplitude'][quark]['min']
                    max_val = ranges['baryon']['amplitude'][quark]['max']
                    deviation = np.random.normal(0, 0.005 * temp_factor)
                    new_params[key] = np.clip(best_params[key] + deviation, min_val, max_val)
                
                elif key == 'coupling_baryon':
                    min_val = ranges['baryon']['coupling']['min']
                    max_val = ranges['baryon']['coupling']['max']
                    deviation = np.random.normal(0, 0.02 * temp_factor)
                    new_params[key] = np.clip(best_params[key] + deviation, min_val, max_val)
                
                elif key == 'coupling_meson':
                    min_val = ranges['meson']['coupling']['min']
                    max_val = ranges['meson']['coupling']['max']
                    deviation = np.random.normal(0, 0.1 * temp_factor)
                    new_params[key] = np.clip(best_params[key] + deviation, min_val, max_val)
                
                elif key == 'phase_shift':
                    min_val = ranges['meson']['phase_shift']['min']
                    max_val = ranges['meson']['phase_shift']['max']
                    deviation = np.random.normal(0, 0.05 * temp_factor)
                    new_params[key] = np.clip(best_params[key] + deviation, min_val, max_val)
                else:
                    new_params[key] = best_params[key]
            
            return new_params
        
        else:
            # Случайная генерация
            params = {}
            
            # Частоты и амплитуды (используем диапазоны для барионов)
            for quark in ['u', 'd', 's']:
                freq_range = ranges['baryon']['frequency'][quark]
                amp_range = ranges['baryon']['amplitude'][quark]
                
                params[f'freq_{quark}'] = np.random.uniform(freq_range['min'], freq_range['max'])
                params[f'amp_{quark}'] = np.random.uniform(amp_range['min'], amp_range['max'])
            
            # Сила связи
            baryon_coupling_range = ranges['baryon']['coupling']
            meson_coupling_range = ranges['meson']['coupling']
            phase_shift_range = ranges['meson']['phase_shift']
            
            params['coupling_baryon'] = np.random.uniform(
                baryon_coupling_range['min'], baryon_coupling_range['max'])
            params['coupling_meson'] = np.random.uniform(
                meson_coupling_range['min'], meson_coupling_range['max'])
            params['phase_shift'] = np.random.uniform(
                phase_shift_range['min'], phase_shift_range['max'])
            
            return params
    
    def create_particles(self, params):
        particles = {}
        
        for name, target in self.config['target_particles'].items():
            if target.get('is_meson', False):
                # Мезон
                particles[name] = MesonModel(
                    composition=target['composition'],
                    params=params,
                    config=self.config
                )
            else:
                # Барион
                particles[name] = BaryonModel(
                    composition=target['composition'],
                    params=params,
                    config=self.config
                )
        
        return particles
    
    def calculate_error(self, particles):
        total_error = 0
        details = {}
        
        for name, particle in particles.items():
            target = self.config['target_particles'][name]
            
            mass = particle.calculate_total_mass()
            charge = particle.calculate_charge()
            spin = particle.calculate_spin()
            
            # Ошибки
            mass_error = abs(mass - target['mass']) / target['mass']
            charge_error = abs(charge - target['charge'])
            spin_error = abs(spin - target['spin'])
            
            # Взвешивание (мезоны важнее сейчас)
            if target.get('is_meson', False):
                weights = {'mass': 5.0, 'charge': 3.0, 'spin': 0.5}
            else:
                weights = {'mass': 2.0, 'charge': 3.0, 'spin': 0.5}
            
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
    
    def save_solution(self, params, error, details, iteration):
        solution = {
            'iteration': iteration,
            'parameters': params,
            'error': error,
            'details': details,
            'temperature': self.temperature,
            'timestamp': time.time()
        }
        
        self.solutions.append(solution)
        
        if self.best_solution is None or error < self.best_solution['error']:
            self.best_solution = solution
            
            print(f"\n{'='*70}")
            print(f"НОВОЕ ЛУЧШЕЕ РЕШЕНИЕ (итерация {iteration})")
            print(f"Общая ошибка: {error:.6f}")
            
            for name in ['proton', 'neutron', 'lambda0', 'pi+', 'k+']:
                if name in details:
                    d = details[name]
                    target = self.config['target_particles'][name]
                    print(f"{name}: {d['mass']:.1f} МэВ (цель {target['mass']}, ошибка {d['mass_error']*100:.1f}%)")
            
            print(f"{'='*70}")
        
        self.solutions.sort(key=lambda x: x['error'])
        max_solutions = self.config['search']['max_solutions']
        if len(self.solutions) > max_solutions:
            self.solutions = self.solutions[:max_solutions]
    
    def run(self):
        print("="*70)
        print("ИНТЕЛЛЕКТУАЛЬНЫЙ ПОИСК ПАРАМЕТРОВ СИНХРОНИЗАЦИИ")
        print("Модель v4.0: Раздельные модели для барионов и мезонов")
        print("="*70)
        
        max_iter = self.config['search']['max_iterations']
        cooling_rate = self.config['search']['cooling_rate']
        
        try:
            while self.iteration < max_iter:
                # Метод генерации
                if self.iteration < 10000 or np.random.random() < 0.3:
                    method = 'random'
                else:
                    method = 'smart'
                
                # Генерация параметров
                params = self.generate_parameters(method)
                
                # Создание частиц
                particles = self.create_particles(params)
                
                # Вычисление ошибки
                error, details = self.calculate_error(particles)
                
                # Принятие решения
                accept = False
                if self.best_solution is None:
                    accept = True
                elif error < self.best_solution['error']:
                    accept = True
                else:
                    prob = np.exp((self.best_solution['error'] - error) / self.temperature)
                    if np.random.random() < prob:
                        accept = True
                
                if accept:
                    self.save_solution(params, error, details, self.iteration)
                
                # Охлаждение
                self.temperature *= cooling_rate
                self.iteration += 1
                
                # Прогресс
                if self.iteration % 1000 == 0:
                    elapsed = time.time() - self.start_time
                    if self.best_solution:
                        proton_mass = self.best_solution['details'].get('proton', {}).get('mass', 0)
                    else:
                        proton_mass = 0
                    
                    print(f"\rИтерация: {self.iteration:,} | "
                          f"Ошибка: {error:.4f} | "
                          f"Протон: {proton_mass:.0f} МэВ | "
                          f"Время: {elapsed:.1f} сек", end='')
        
        except KeyboardInterrupt:
            print("\n\nПоиск прерван пользователем")
        
        finally:
            print(f"\n\n{'='*70}")
            print("ПОИСК ЗАВЕРШЁН")
            print(f"Итераций: {self.iteration:,}")
            print(f"Время: {time.time() - self.start_time:.1f} сек")
            
            if self.best_solution:
                self._print_summary()
    
    def _print_summary(self):
        best = self.best_solution
        details = best['details']
        params = best['parameters']
        
        print(f"\n{'='*70}")
        print("ЛУЧШЕЕ РЕШЕНИЕ:")
        print(f"Общая ошибка: {best['error']:.6f}")
        
        print(f"\nЧАСТИЦЫ:")
        for name in ['proton', 'neutron', 'lambda0', 'pi+', 'k+']:
            if name in details:
                d = details[name]
                target = self.config['target_particles'][name]
                print(f"\n{name.upper()}:")
                print(f"  Масса: {d['mass']:.3f} МэВ (цель: {target['mass']}, ошибка: {d['mass_error']*100:.2f}%)")
                print(f"  Заряд: {d['charge']:.3f} (цель: {target['charge']})")
                print(f"  Спин: {d['spin']} (цель: {target['spin']})")
        
        print(f"\nПАРАМЕТРЫ:")
        for key, value in params.items():
            print(f"  {key}: {value:.6f}")
        
        # Эффективные массы
        print(f"\nЭФФЕКТИВНЫЕ МАССЫ КВАРКОВ:")
        for quark in ['u', 'd', 's']:
            base_mass = self.config['type_properties'][quark]['base_mass']
            effective = base_mass * params[f'amp_{quark}'] * params[f'freq_{quark}'] * 100
            print(f"  {quark}: {effective:.1f} МэВ")
        
        print(f"\nОТНОШЕНИЯ МАСС:")
        m_u = self.config['type_properties']['u']['base_mass'] * params['amp_u'] * params['freq_u']
        m_d = self.config['type_properties']['d']['base_mass'] * params['amp_d'] * params['freq_d']
        m_s = self.config['type_properties']['s']['base_mass'] * params['amp_s'] * params['freq_s']
        print(f"  m_d/m_u: {m_d/m_u:.3f}")
        print(f"  m_s/m_u: {m_s/m_u:.3f}")
        print(f"  m_s/m_d: {m_s/m_d:.3f}")

# ================= ОСНОВНОЙ БЛОК =================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("СИСТЕМА ПОИСКА ПАРАМЕТРОВ СИНХРОНИЗАЦИИ НИТЕЙ")
    print("Версия 4.0: Раздельные модели для барионов и мезонов")
    print("="*70)
    
    search = IntelligentParameterSearchV4(CONFIG)
    search.run()
    
    print("\n" + "="*70)
    print("ГОТОВО! Проверяем модель для барионов и мезонов отдельно.")
    print("="*70)