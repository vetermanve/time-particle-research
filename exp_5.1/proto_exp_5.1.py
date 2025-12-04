"""
Модель синхронизации осциллирующих нитей v5.1
Разные энергии связи для протона и нейтрона
"""

import numpy as np
import json
import time
import os
import sys
from datetime import datetime

# ================= КОНФИГУРАЦИЯ v5.1 =================
CONFIG = {
    # Целевые свойства частиц (в МэВ)
    'target_particles': {
        'proton': {
            'mass': 938.272,
            'charge': 1.0,
            'spin': 0.5,
            'composition': ['u', 'u', 'd'],
            'is_meson': False
        },
        'neutron': {
            'mass': 939.565,
            'charge': 0.0,
            'spin': 0.5,
            'composition': ['u', 'd', 'd'],
            'is_meson': False
        },
        'pi+': {
            'mass': 139.57,
            'charge': 1.0,
            'spin': 0.0,
            'composition': ['u', 'anti_d'],
            'is_meson': True
        }
    },
    
    # Свойства кварков (в МэВ до масштабирования)
    'type_properties': {
        'u': {'charge': 2/3, 'base_mass': 2.3},
        'd': {'charge': -1/3, 'base_mass': 4.8},
        'anti_u': {'charge': -2/3, 'base_mass': 2.3},
        'anti_d': {'charge': 1/3, 'base_mass': 4.8}
    },
    
    # Диапазоны перебора параметров
    'param_ranges': {
        'frequency': {
            'u': {'min': 0.95, 'max': 1.05, 'step': 0.001},
            'd': {'min': 0.90, 'max': 1.00, 'step': 0.001}
        },
        'amplitude': {
            'u': {'min': 1.00, 'max': 1.10, 'step': 0.001},
            'd': {'min': 0.85, 'max': 0.95, 'step': 0.001}
        },
        # РАЗНЫЕ СИЛЫ СВЯЗИ ДЛЯ ПРОТОНА И НЕЙТРОНА!
        'coupling_proton': {
            'min': 0.5,
            'max': 2.0,
            'step': 0.01
        },
        'coupling_neutron': {
            'min': 0.5,
            'max': 2.0,
            'step': 0.01
        },
        'coupling_meson': {
            'min': 4.0,
            'max': 6.0,
            'step': 0.05
        },
        'phase_shift': {
            'min': np.pi * 0.95,
            'max': np.pi * 1.05,
            'step': 0.01
        }
    },
    
    # Параметры поиска
    'search': {
        'max_iterations': 200000,
        'save_interval': 50000,
        'min_error': 0.01,
        'max_solutions': 50,
        'scale_factor': 100.0,
        'temperature': 0.15,
        'cooling_rate': 0.999995
    }
}

# ================= УНИВЕРСАЛЬНАЯ МОДЕЛЬ v5.1 =================
class UniversalParticleModelV51:
    def __init__(self, composition, params, config, particle_name=None):
        self.composition = composition
        self.config = config
        self.particle_name = particle_name
        self.thread_count = len(composition)
        self.is_meson = (self.thread_count == 2)
        
        # Извлекаем параметры
        self.frequencies = []
        self.amplitudes = []
        
        for quark in composition:
            base_type = quark.replace('anti_', '')
            self.frequencies.append(params[f'freq_{base_type}'])
            self.amplitudes.append(params[f'amp_{base_type}'])
        
        # Выбираем правильную силу связи
        if self.is_meson:
            self.coupling = params['coupling_meson']
        elif particle_name == 'proton':
            self.coupling = params['coupling_proton']
        elif particle_name == 'neutron':
            self.coupling = params['coupling_neutron']
        else:
            self.coupling = params.get('coupling_baryon', 1.5)
        
        # Фазы
        if self.is_meson:
            # Для мезонов: кварк и антикварк в противофазе
            self.phases = [0.0, params.get('phase_shift', np.pi)]
        else:
            # Для барионов: оптимизированные фазы
            base_phase = np.random.uniform(0, 2*np.pi)
            if composition == ['u', 'u', 'd']:  # Протон
                self.phases = [base_phase, base_phase, base_phase + np.pi/2]
            elif composition == ['u', 'd', 'd']:  # Нейтрон
                self.phases = [base_phase, base_phase + np.pi/2, base_phase + np.pi/2]
            else:
                self.phases = [base_phase, base_phase + np.pi/2, base_phase + np.pi]
        
        self.type_properties = config['type_properties']
        self.scale_factor = config['search']['scale_factor']
    
    def calculate_charge(self):
        total = sum(self.type_properties[q]['charge'] for q in self.composition)
        return round(total, 10)
    
    def calculate_base_mass(self):
        total = 0
        for i, quark_type in enumerate(self.composition):
            base_mass = self.type_properties[quark_type]['base_mass']
            total += base_mass * self.amplitudes[i] * self.frequencies[i]
        return total
    
    def calculate_synchronization_energy(self):
        # 1. Частотная синхронизация
        freq_coherence = 0
        pairs = 0
        
        for i in range(self.thread_count):
            for j in range(i+1, self.thread_count):
                ratio = self.frequencies[i] / self.frequencies[j]
                simple_ratio = self._find_simple_ratio(ratio, max_denominator=5)
                coherence = 1.0 - abs(ratio - simple_ratio) / simple_ratio
                freq_coherence += max(0, coherence)
                pairs += 1
        
        freq_coherence = freq_coherence / pairs if pairs > 0 else 0.5
        
        # 2. Фазовая синхронизация
        phase_coherence = 0
        for i in range(self.thread_count):
            for j in range(i+1, self.thread_count):
                diff = abs(self.phases[i] - self.phases[j]) % (2*np.pi)
                diff = min(diff, 2*np.pi - diff)
                
                if self.is_meson:
                    phase_coherence += np.cos(diff + np.pi)
                else:
                    phase_coherence += np.cos(diff)
        
        max_pairs = self.thread_count * (self.thread_count - 1) / 2
        phase_coherence = (phase_coherence / max_pairs + 1) / 2 if max_pairs > 0 else 0.5
        
        # 3. Симметрия состава (разная для протона и нейтрона)
        symmetry = 1.0
        if not self.is_meson:
            if self.composition == ['u', 'u', 'd']:  # Протон
                symmetry = 1.1
            elif self.composition == ['u', 'd', 'd']:  # Нейтрон
                symmetry = 1.05  # Меньше для нейтрона
        
        # 4. Дополнительный фактор для нейтрона (учитывает два d-кварка)
        neutron_factor = 1.0
        if self.particle_name == 'neutron':
            # Нейтрон требует меньшей энергии связи из-за двух тяжелых кварков
            neutron_factor = 0.9
        
        # 5. Общая энергия синхронизации
        sync_energy = self.coupling * (0.6 * freq_coherence + 0.4 * phase_coherence) * symmetry * neutron_factor
        
        return sync_energy
    
    def _find_simple_ratio(self, ratio, max_denominator=5):
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
        
        # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ v5.1: ПРОВЕРЯЕМ РЕЗУЛЬТАТЫ
        if self.is_meson:
            # Для мезонов: базовая масса МИНУС энергия связи
            total_mass = base_mass - sync_energy
        else:
            # Для барионов: базовая масса ПЛЮС энергия связи
            total_mass = base_mass + sync_energy
        
        # Масштабируем
        return total_mass * self.scale_factor
    
    def calculate_spin(self):
        return 0.0 if self.is_meson else 0.5

# ================= ИНТЕЛЛЕКТУАЛЬНЫЙ ПОИСК v5.1 =================
class IntelligentParameterSearchV51:
    def __init__(self, config):
        self.config = config
        self.solutions = []
        self.best_solution = None
        self.iteration = 0
        self.start_time = time.time()
        self.temperature = config['search']['temperature']
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = f"particle_search_v51_{timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)
        
        with open(f"{self.result_dir}/config.json", 'w') as f:
            json.dump(config, f, indent=2, default=self._json_serializer)
        
        # Начальные параметры (из v5.0)
        self.current_params = {
            'freq_u': 0.958,
            'freq_d': 0.900,
            'amp_u': 1.050,
            'amp_d': 0.900,
            'coupling_proton': 0.918,
            'coupling_neutron': 0.918,  # Начинаем с одинаковых
            'coupling_meson': 5.000,
            'phase_shift': np.pi
        }
        
        print(f"Директория результатов: {self.result_dir}")
        print("v5.1: Разные энергии связи для протона и нейтрона")
    
    def _json_serializer(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)
    
    def generate_parameters(self, method='smart'):
        ranges = self.config['param_ranges']
        
        if method == 'smart' and self.best_solution and self.iteration > 5000:
            best = self.best_solution['parameters']
            new_params = {}
            temp_factor = max(0.05, self.temperature)
            
            for key in best.keys():
                if 'freq' in key or 'amp' in key:
                    quark = key.split('_')[1]
                    if 'freq' in key:
                        min_val = ranges['frequency'][quark]['min']
                        max_val = ranges['frequency'][quark]['max']
                        std = 0.005 * temp_factor
                    else:
                        min_val = ranges['amplitude'][quark]['min']
                        max_val = ranges['amplitude'][quark]['max']
                        std = 0.003 * temp_factor
                    new_params[key] = np.clip(best[key] + np.random.normal(0, std), min_val, max_val)
                
                elif 'coupling' in key:
                    if key == 'coupling_proton':
                        min_val = ranges['coupling_proton']['min']
                        max_val = ranges['coupling_proton']['max']
                        std = 0.01 * temp_factor
                    elif key == 'coupling_neutron':
                        min_val = ranges['coupling_neutron']['min']
                        max_val = ranges['coupling_neutron']['max']
                        std = 0.01 * temp_factor
                    else:  # coupling_meson
                        min_val = ranges['coupling_meson']['min']
                        max_val = ranges['coupling_meson']['max']
                        std = 0.05 * temp_factor
                    
                    new_params[key] = np.clip(best[key] + np.random.normal(0, std), min_val, max_val)
                
                elif key == 'phase_shift':
                    min_val = ranges['phase_shift']['min']
                    max_val = ranges['phase_shift']['max']
                    std = 0.02 * temp_factor
                    new_params[key] = np.clip(best[key] + np.random.normal(0, std), min_val, max_val)
            
            return new_params
        
        else:
            # Случайная генерация
            params = {}
            
            for quark in ['u', 'd']:
                freq_range = ranges['frequency'][quark]
                amp_range = ranges['amplitude'][quark]
                params[f'freq_{quark}'] = np.random.uniform(freq_range['min'], freq_range['max'])
                params[f'amp_{quark}'] = np.random.uniform(amp_range['min'], amp_range['max'])
            
            params['coupling_proton'] = np.random.uniform(
                ranges['coupling_proton']['min'], ranges['coupling_proton']['max'])
            params['coupling_neutron'] = np.random.uniform(
                ranges['coupling_neutron']['min'], ranges['coupling_neutron']['max'])
            params['coupling_meson'] = np.random.uniform(
                ranges['coupling_meson']['min'], ranges['coupling_meson']['max'])
            params['phase_shift'] = np.random.uniform(
                ranges['phase_shift']['min'], ranges['phase_shift']['max'])
            
            return params
    
    def create_particles(self, params):
        particles = {}
        for name, target in self.config['target_particles'].items():
            particles[name] = UniversalParticleModelV51(
                composition=target['composition'],
                params=params,
                config=self.config,
                particle_name=name
            )
        return particles
    
    def calculate_error(self, particles):
        total_error = 0
        details = {}
        
        # ВЕСА ДЛЯ v5.1 (нейтрон важнее!)
        weights_config = {
            'proton': {'mass': 2.0, 'charge': 5.0, 'spin': 0.5},
            'neutron': {'mass': 8.0, 'charge': 5.0, 'spin': 0.5},  # Увеличили вес массы нейтрона!
            'pi+': {'mass': 2.0, 'charge': 5.0, 'spin': 0.5}
        }
        
        for name, particle in particles.items():
            target = self.config['target_particles'][name]
            weights = weights_config[name]
            
            mass = particle.calculate_total_mass()
            charge = particle.calculate_charge()
            spin = particle.calculate_spin()
            
            mass_error = abs(mass - target['mass']) / target['mass']
            charge_error = abs(charge - target['charge'])
            spin_error = abs(spin - target['spin'])
            
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
            
            for name in ['proton', 'neutron', 'pi+']:
                if name in details:
                    d = details[name]
                    target = self.config['target_particles'][name]
                    mass_err = d['mass_error'] * 100
                    print(f"{name}: {d['mass']:.3f} МэВ (цель {target['mass']}, ошибка {mass_err:.3f}%)")
            
            print(f"{'='*70}")
        
        self.solutions.sort(key=lambda x: x['error'])
        if len(self.solutions) > self.config['search']['max_solutions']:
            self.solutions = self.solutions[:self.config['search']['max_solutions']]
    
    def save_checkpoint(self):
        if self.iteration % self.config['search']['save_interval'] != 0:
            return
        
        checkpoint = {
            'iteration': self.iteration,
            'solutions': self.solutions,
            'best_solution': self.best_solution,
            'temperature': self.temperature,
            'elapsed_time': time.time() - self.start_time
        }
        
        checkpoint_file = f"{self.result_dir}/checkpoint_{self.iteration:08d}.json"
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2, default=self._json_serializer)
            print(f"\n  Контрольная точка: {checkpoint_file}")
        except Exception as e:
            print(f"  Ошибка сохранения: {e}")
    
    def run(self):
        print("="*70)
        print("МОДЕЛЬ СИНХРОНИЗАЦИИ НИТЕЙ v5.1")
        print("Разные энергии связи для протона и нейтрона")
        print("="*70)
        
        max_iter = self.config['search']['max_iterations']
        cooling_rate = self.config['search']['cooling_rate']
        
        try:
            while self.iteration < max_iter:
                if self.iteration < 10000 or np.random.random() < 0.3:
                    method = 'random'
                else:
                    method = 'smart'
                
                params = self.generate_parameters(method)
                particles = self.create_particles(params)
                error, details = self.calculate_error(particles)
                
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
                
                self.temperature *= cooling_rate
                self.iteration += 1
                
                if self.iteration % 1000 == 0:
                    elapsed = time.time() - self.start_time
                    if self.best_solution:
                        best_err = self.best_solution['error']
                        neutron_mass = self.best_solution['details'].get('neutron', {}).get('mass', 0)
                    else:
                        best_err = float('inf')
                        neutron_mass = 0
                    
                    print(f"\rИтерация: {self.iteration:,} | "
                          f"Ошибка: {best_err:.4f} | "
                          f"Нейтрон: {neutron_mass:.0f} МэВ | "
                          f"Время: {elapsed:.1f} сек", end='')
                
                if self.iteration % 50000 == 0:
                    self.save_checkpoint()
        
        except KeyboardInterrupt:
            print("\n\nПоиск прерван")
        
        finally:
            print(f"\n\n{'='*70}")
            print("ПОИСК ЗАВЕРШЁН")
            print(f"Итераций: {self.iteration:,}")
            print(f"Время: {time.time() - self.start_time:.2f} сек")
            
            if self.best_solution:
                self._print_summary()
    
    def _print_summary(self):
        best = self.best_solution
        details = best['details']
        params = best['parameters']
        
        print(f"\n{'='*70}")
        print("ЛУЧШЕЕ РЕШЕНИЕ v5.1:")
        print(f"Общая ошибка: {best['error']:.6f}")
        
        print(f"\nРЕЗУЛЬТАТЫ:")
        print(f"{'Частица':<10} {'Масса':<10} {'Цель':<10} {'Ошибка':<10} {'Заряд':<10} {'Спин':<10}")
        print("-" * 70)
        
        for name in ['proton', 'neutron', 'pi+']:
            if name in details:
                d = details[name]
                target = self.config['target_particles'][name]
                print(f"{name:<10} {d['mass']:<10.1f} {target['mass']:<10.1f} "
                      f"{d['mass_error']*100:<10.3f}% {d['charge']:<10.1f} {d['spin']:<10.1f}")
        
        print(f"\nПАРАМЕТРЫ:")
        for key, value in params.items():
            print(f"  {key}: {value:.6f}")
        
        # Энергии связи
        print(f"\nЭНЕРГИИ СВЯЗИ:")
        particles = self.create_particles(params)
        for name in ['proton', 'neutron', 'pi+']:
            if name in particles:
                p = particles[name]
                base = p.calculate_base_mass()
                sync = p.calculate_synchronization_energy()
                total = p.calculate_total_mass()
                sign = "-" if p.is_meson else "+"
                print(f"  {name}: база={base:.3f}, связь={sync:.3f}, "
                      f"масса=база{sign}связь={total/100:.3f}×100 МэВ")
        
        print(f"\nРАЗНОСТЬ coupling_neutron - coupling_proton: "
              f"{params['coupling_neutron'] - params['coupling_proton']:.3f}")
        
        print(f"\nСОВЕТ: coupling_neutron должен быть МЕНЬШЕ coupling_proton "
              f"для уменьшения массы нейтрона!")

# ================= ОСНОВНОЙ БЛОК =================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("МОДЕЛЬ СИНХРОНИЗАЦИИ НИТЕЙ v5.1")
    print("Ключевое изменение: разные энергии связи для протона и нейтрона")
    print("="*70)
    
    search = IntelligentParameterSearchV51(CONFIG)
    search.run()
    
    print("\n" + "="*70)
    print("ОЖИДАНИЯ v5.1:")
    print("1. Протон: ~938 МэВ (<1% ошибка)")
    print("2. Нейтрон: ~940 МэВ (<2% ошибка)")
    print("3. Пион: ~139 МэВ (<0.1% ошибка)")
    print("4. coupling_neutron < coupling_proton")
    print("="*70)