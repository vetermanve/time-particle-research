"""
Скрипт для поиска параметров синхронизации нитей (модель v4.1)
Исправленные базовые массы кварков
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
        
        # МЕЗОНЫ
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
    
    # Свойства кварков (ПРАВИЛЬНЫЕ массы в МэВ перед масштабированием)
    'type_properties': {
        'u': {'charge': 2/3, 'base_mass': 2.3},   # ~2.3 МэВ
        'd': {'charge': -1/3, 'base_mass': 4.8},  # ~4.8 МэВ
        's': {'charge': -1/3, 'base_mass': 95.0}, # ~95 МэВ
        'anti_u': {'charge': -2/3, 'base_mass': 2.3},
        'anti_d': {'charge': 1/3, 'base_mass': 4.8},
        'anti_s': {'charge': 1/3, 'base_mass': 95.0},
    },
    
    # Диапазоны перебора (оптимизированные)
    'param_ranges': {
        # ОБЩИЕ параметры для всех кварков
        'frequency': {
            'u': {'min': 0.90, 'max': 1.10, 'step': 0.001},
            'd': {'min': 0.90, 'max': 1.10, 'step': 0.001},
            's': {'min': 0.80, 'max': 1.20, 'step': 0.001},
        },
        'amplitude': {
            'u': {'min': 0.95, 'max': 1.05, 'step': 0.001},
            'd': {'min': 0.90, 'max': 1.00, 'step': 0.001},
            's': {'min': 0.90, 'max': 1.10, 'step': 0.001},
        },
        # РАЗНЫЕ силы связи для барионов и мезонов
        'coupling_baryon': {
            'min': 1.0,
            'max': 3.0,
            'step': 0.01
        },
        'coupling_meson': {
            'min': 1.0,
            'max': 5.0,
            'step': 0.05
        },
        'phase_shift': {
            'min': np.pi * 0.8,
            'max': np.pi * 1.2,
            'step': 0.01
        }
    },
    
    # Параметры поиска
    'search': {
        'max_iterations': 200000,
        'save_interval': 50000,
        'min_error': 0.05,
        'max_solutions': 100,
        'scale_factor': 100.0,
        'temperature': 0.2,
        'cooling_rate': 0.99999
    }
}

# ================= МОДЕЛЬ БАРИОНА =================
class BaryonModel:
    def __init__(self, composition, params, config):
        self.composition = composition
        self.config = config
        
        # Параметры для каждого кварка
        self.frequencies = [params['freq_' + q] for q in composition]
        self.amplitudes = [params['amp_' + q] for q in composition]
        
        # Фазы (оптимальные для барионов)
        base_phase = np.random.uniform(0, 2*np.pi)
        if composition == ['u', 'u', 'd']:  # Протон
            self.phases = [base_phase, base_phase, base_phase + np.pi/2]
        elif composition == ['u', 'd', 'd']:  # Нейтрон
            self.phases = [base_phase, base_phase + np.pi/2, base_phase + np.pi/2]
        elif composition == ['u', 'd', 's']:  # Лямбда0
            # Для странного бариона фазы более симметричные
            self.phases = [base_phase, base_phase + 2*np.pi/3, base_phase + 4*np.pi/3]
        else:
            self.phases = [base_phase, base_phase + np.pi/2, base_phase + np.pi]
        
        self.coupling = params['coupling_baryon']
        self.type_properties = config['type_properties']
        self.scale_factor = config['search']['scale_factor']
    
    def calculate_charge(self):
        total = 0
        for q in self.composition:
            total += self.type_properties[q]['charge']
        return round(total, 10)
    
    def calculate_base_mass(self):
        """Базовая масса кварков"""
        total = 0
        for i, q in enumerate(self.composition):
            base_mass = self.type_properties[q]['base_mass']
            total += base_mass * self.amplitudes[i] * self.frequencies[i]
        return total
    
    def calculate_synchronization_energy(self):
        """Энергия связи для барионов"""
        # 1. Частотная синхронизация
        freq_coherence = 0
        pairs = 0
        
        for i in range(3):
            for j in range(i+1, 3):
                ratio = self.frequencies[i] / self.frequencies[j]
                # Идеальная синхронизация при простых отношениях
                simple_ratio = self._find_simple_ratio(ratio, max_denominator=5)
                coherence = 1.0 - abs(ratio - simple_ratio) / simple_ratio
                freq_coherence += max(0, coherence)  # Только положительные
                pairs += 1
        
        freq_coherence = freq_coherence / pairs if pairs > 0 else 0.5
        
        # 2. Фазовая синхронизация
        phase_coherence = 0
        for i in range(3):
            for j in range(i+1, 3):
                diff = abs(self.phases[i] - self.phases[j]) % (2*np.pi)
                diff = min(diff, 2*np.pi - diff)
                phase_coherence += np.cos(diff)
        
        phase_coherence = (phase_coherence / 3 + 1) / 2  # Нормализуем к [0,1]
        
        # 3. Симметрия
        symmetry = 1.0
        comp = self.composition
        if comp == ['u', 'u', 'd'] or comp == ['u', 'd', 'd']:
            symmetry = 1.08  # Нуклоны
        elif 's' in comp:
            if comp.count('s') == 1:
                symmetry = 1.05  # Лямбда, Сигма
            elif comp.count('s') == 2:
                symmetry = 1.03  # Кси
        
        # 4. Общая энергия связи
        energy = self.coupling * (0.6 * freq_coherence + 0.4 * phase_coherence) * symmetry
        return energy
    
    def _find_simple_ratio(self, ratio, max_denominator=5):
        best_error = float('inf')
        best_ratio = ratio
        for den in range(1, max_denominator + 1):
            for num in range(1, max_denominator + 1):
                simple = num / den
                error = abs(ratio - simple)
                if error < best_error:
                    best_error = error
                    best_ratio = simple
        return best_ratio
    
    def calculate_total_mass(self):
        base = self.calculate_base_mass()
        sync = self.calculate_synchronization_energy()
        return (base - sync) * self.scale_factor
    
    def calculate_spin(self):
        return 0.5

# ================= МОДЕЛЬ МЕЗОНА =================
class MesonModel:
    def __init__(self, composition, params, config):
        self.composition = composition
        self.config = config
        
        # Кварк и антикварк
        self.quark = composition[0]
        self.antiquark = composition[1]
        
        # Базовые типы (без anti_)
        quark_type = self.quark.replace('anti_', '')
        antiquark_type = self.antiquark.replace('anti_', '')
        
        # Параметры
        self.freq_q = params['freq_' + quark_type]
        self.freq_aq = params['freq_' + antiquark_type]
        self.amp_q = params['amp_' + quark_type]
        self.amp_aq = params['amp_' + antiquark_type]
        
        self.phase_shift = params.get('phase_shift', np.pi)
        self.coupling = params['coupling_meson']
        
        self.type_properties = config['type_properties']
        self.scale_factor = config['search']['scale_factor']
    
    def calculate_charge(self):
        q_charge = self.type_properties[self.quark]['charge']
        aq_charge = self.type_properties[self.antiquark]['charge']
        return round(q_charge + aq_charge, 10)
    
    def calculate_base_mass(self):
        base_q = self.type_properties[self.quark]['base_mass']
        base_aq = self.type_properties[self.antiquark]['base_mass']
        return (base_q * self.amp_q * self.freq_q + 
                base_aq * self.amp_aq * self.freq_aq)
    
    def calculate_synchronization_energy(self):
        """Энергия связи для мезонов"""
        # 1. Частотная синхронизация (должны быть близки по частоте)
        freq_ratio = self.freq_q / self.freq_aq
        freq_coherence = 1.0 - min(abs(freq_ratio - 1.0), 0.3) / 0.3
        
        # 2. Фазовая синхронизация (должны быть в противофазе)
        # phase_shift должен быть близок к π
        phase_coherence = 1.0 - abs(self.phase_shift - np.pi) / np.pi
        
        # 3. Массовая асимметрия (для разных кварков)
        mass_q = self.type_properties[self.quark]['base_mass'] * self.amp_q * self.freq_q
        mass_aq = self.type_properties[self.antiquark]['base_mass'] * self.amp_aq * self.freq_aq
        mass_ratio = min(mass_q, mass_aq) / max(mass_q, mass_aq)
        
        # 4. Общая энергия (сильно зависит от связи)
        # Для мезонов энергия связи БОЛЬШАЯ, чтобы уменьшить массу
        energy = self.coupling * (0.4 * freq_coherence + 0.6 * phase_coherence) * mass_ratio
        return energy * 2.0  # Усиливаем для мезонов
    
    def calculate_total_mass(self):
        base = self.calculate_base_mass()
        sync = self.calculate_synchronization_energy()
        mass = (base - sync) * self.scale_factor
        return max(mass, 1.0)  # Не отрицательная
    
    def calculate_spin(self):
        return 0.0

# ================= УМНЫЙ ПОИСК =================
class IntelligentSearchV41:
    def __init__(self, config):
        self.config = config
        self.solutions = []
        self.best_solution = None
        self.iteration = 0
        self.start_time = time.time()
        self.temperature = config['search']['temperature']
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = f"particle_search_v41_{timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)
        
        with open(f"{self.result_dir}/config.json", 'w') as f:
            json.dump(config, f, indent=2, default=self._json_serializer)
        
        # Начальные параметры (из v4.0, но с исправленными массами)
        self.current_params = {
            'freq_u': 0.9939,
            'amp_u': 0.9825,
            'freq_d': 0.9000,
            'amp_d': 0.9500,
            'freq_s': 0.95,   # Ближе к u,d
            'amp_s': 0.95,
            'coupling_baryon': 1.76,
            'coupling_meson': 3.0,  # Уменьшили
            'phase_shift': 3.1416
        }
        
        print(f"Директория результатов: {self.result_dir}")
        print(f"Базовые массы кварков:")
        print(f"  u: {config['type_properties']['u']['base_mass']} * 100 = {config['type_properties']['u']['base_mass']*100:.1f} МэВ")
        print(f"  d: {config['type_properties']['d']['base_mass']} * 100 = {config['type_properties']['d']['base_mass']*100:.1f} МэВ")
        print(f"  s: {config['type_properties']['s']['base_mass']} * 100 = {config['type_properties']['s']['base_mass']*100:.1f} МэВ")
    
    def _json_serializer(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)
    
    def generate_parameters(self, method='smart'):
        ranges = self.config['param_ranges']
        
        if method == 'smart' and self.best_solution and self.iteration > 10000:
            # Умная генерация на основе лучшего решения
            best = self.best_solution['parameters']
            new_params = {}
            temp_factor = max(0.1, self.temperature)
            
            for key in best.keys():
                if 'freq' in key:
                    quark = key.split('_')[1]
                    min_val = ranges['frequency'][quark]['min']
                    max_val = ranges['frequency'][quark]['max']
                    std = 0.02 * temp_factor
                    new_params[key] = np.clip(best[key] + np.random.normal(0, std), min_val, max_val)
                
                elif 'amp' in key:
                    quark = key.split('_')[1]
                    min_val = ranges['amplitude'][quark]['min']
                    max_val = ranges['amplitude'][quark]['max']
                    std = 0.01 * temp_factor
                    new_params[key] = np.clip(best[key] + np.random.normal(0, std), min_val, max_val)
                
                elif key == 'coupling_baryon':
                    min_val = ranges['coupling_baryon']['min']
                    max_val = ranges['coupling_baryon']['max']
                    std = 0.05 * temp_factor
                    new_params[key] = np.clip(best[key] + np.random.normal(0, std), min_val, max_val)
                
                elif key == 'coupling_meson':
                    min_val = ranges['coupling_meson']['min']
                    max_val = ranges['coupling_meson']['max']
                    std = 0.1 * temp_factor
                    new_params[key] = np.clip(best[key] + np.random.normal(0, std), min_val, max_val)
                
                elif key == 'phase_shift':
                    min_val = ranges['phase_shift']['min']
                    max_val = ranges['phase_shift']['max']
                    std = 0.1 * temp_factor
                    new_params[key] = np.clip(best[key] + np.random.normal(0, std), min_val, max_val)
            
            return new_params
        
        else:
            # Случайная генерация
            params = {}
            
            for quark in ['u', 'd', 's']:
                freq_range = ranges['frequency'][quark]
                amp_range = ranges['amplitude'][quark]
                
                params[f'freq_{quark}'] = np.random.uniform(freq_range['min'], freq_range['max'])
                params[f'amp_{quark}'] = np.random.uniform(amp_range['min'], amp_range['max'])
            
            params['coupling_baryon'] = np.random.uniform(
                ranges['coupling_baryon']['min'], ranges['coupling_baryon']['max'])
            params['coupling_meson'] = np.random.uniform(
                ranges['coupling_meson']['min'], ranges['coupling_meson']['max'])
            params['phase_shift'] = np.random.uniform(
                ranges['phase_shift']['min'], ranges['phase_shift']['max'])
            
            return params
    
    def create_particles(self, params):
        particles = {}
        
        for name, target in self.config['target_particles'].items():
            if target.get('is_meson', False):
                particles[name] = MesonModel(
                    composition=target['composition'],
                    params=params,
                    config=self.config
                )
            else:
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
            
            mass_error = abs(mass - target['mass']) / target['mass']
            charge_error = abs(charge - target['charge'])
            spin_error = abs(spin - target['spin'])
            
            # Взвешивание ошибок
            weights = {'mass': 3.0, 'charge': 5.0, 'spin': 1.0}
            
            # Особо важны мезоны сейчас
            if name in ['pi+', 'k+']:
                weights['mass'] = 5.0
            
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
                'error': error
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
                    mass_err_percent = d['mass_error'] * 100
                    print(f"{name}: {d['mass']:.1f} МэВ (цель {target['mass']}, ошибка {mass_err_percent:.2f}%)")
            
            print(f"{'='*70}")
        
        # Сортируем и обрезаем
        self.solutions.sort(key=lambda x: x['error'])
        if len(self.solutions) > self.config['search']['max_solutions']:
            self.solutions = self.solutions[:self.config['search']['max_solutions']]
    
    def run(self):
        print("="*70)
        print("ПОИСК ПАРАМЕТРОВ v4.1")
        print("Исправленные базовые массы кварков")
        print("="*70)
        
        max_iter = self.config['search']['max_iterations']
        
        try:
            while self.iteration < max_iter:
                # Выбор метода
                if self.iteration < 5000 or np.random.random() < 0.3:
                    method = 'random'
                else:
                    method = 'smart'
                
                # Генерация параметров
                params = self.generate_parameters(method)
                
                # Создание частиц
                particles = self.create_particles(params)
                
                # Расчет ошибки
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
                self.temperature *= self.config['search']['cooling_rate']
                self.iteration += 1
                
                # Прогресс
                if self.iteration % 1000 == 0:
                    elapsed = time.time() - self.start_time
                    if self.best_solution:
                        best_err = self.best_solution['error']
                        pi_mass = self.best_solution['details'].get('pi+', {}).get('mass', 0)
                    else:
                        best_err = float('inf')
                        pi_mass = 0
                    
                    print(f"\rИтерация: {self.iteration:,} | "
                          f"Лучшая ошибка: {best_err:.4f} | "
                          f"Пион: {pi_mass:.0f} МэВ | "
                          f"Время: {elapsed:.1f} сек", end='')
        
        except KeyboardInterrupt:
            print("\n\nПоиск прерван")
        
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
        
        print(f"\nРЕЗУЛЬТАТЫ:")
        print(f"{'Частица':<10} {'Масса':<10} {'Цель':<10} {'Ошибка':<10} {'Заряд':<10} {'Спин':<10}")
        print("-" * 70)
        
        for name in ['proton', 'neutron', 'lambda0', 'pi+', 'k+']:
            if name in details:
                d = details[name]
                target = self.config['target_particles'][name]
                print(f"{name:<10} {d['mass']:<10.1f} {target['mass']:<10.1f} "
                      f"{d['mass_error']*100:<10.2f}% {d['charge']:<10.1f} {d['spin']:<10.1f}")
        
        print(f"\nПАРАМЕТРЫ:")
        for key, value in params.items():
            print(f"  {key}: {value:.6f}")
        
        # Эффективные массы
        print(f"\nЭФФЕКТИВНЫЕ МАССЫ КВАРКОВ (до синхронизации):")
        for quark in ['u', 'd', 's']:
            base = self.config['type_properties'][quark]['base_mass']
            freq = params[f'freq_{quark}']
            amp = params[f'amp_{quark}']
            effective = base * freq * amp * self.config['search']['scale_factor']
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
    print("МОДЕЛЬ СИНХРОНИЗАЦИИ НИТЕЙ v4.1")
    print("Исправленные массы кварков (в МэВ до масштабирования):")
    print(f"  u: {CONFIG['type_properties']['u']['base_mass']:.1f}")
    print(f"  d: {CONFIG['type_properties']['d']['base_mass']:.1f}")
    print(f"  s: {CONFIG['type_properties']['s']['base_mass']:.1f}")
    print(f"Масштабный коэффициент: {CONFIG['search']['scale_factor']}")
    print("="*70)
    
    search = IntelligentSearchV41(CONFIG)
    search.run()
    
    print("\n" + "="*70)
    print("ГОТОВО! Теперь массы кварков реалистичные.")
    print("="*70)