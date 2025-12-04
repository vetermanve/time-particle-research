"""
Модель синхронизации осциллирующих нитей v5.2
Исправление базовых масс кварков для правильной разности масс
"""

import numpy as np
import json
import time
import os
import sys
from datetime import datetime

# ================= КОНФИГУРАЦИЯ v5.2 =================
CONFIG = {
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
    
    # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: НОВЫЕ МАССЫ КВАРКОВ!
    'type_properties': {
        'u': {'charge': 2/3, 'base_mass': 2.25},    # Было: 2.3
        'd': {'charge': -1/3, 'base_mass': 4.60},   # Было: 4.8 (УМЕНЬШИЛИ!)
        'anti_u': {'charge': -2/3, 'base_mass': 2.25},
        'anti_d': {'charge': 1/3, 'base_mass': 4.60}
    },
    
    'param_ranges': {
        'frequency': {
            'u': {'min': 0.95, 'max': 1.05, 'step': 0.001},
            'd': {'min': 0.90, 'max': 1.00, 'step': 0.001}
        },
        'amplitude': {
            'u': {'min': 1.00, 'max': 1.10, 'step': 0.001},
            'd': {'min': 0.85, 'max': 0.95, 'step': 0.001}
        },
        'coupling_proton': {
            'min': 1.2,
            'max': 1.8,
            'step': 0.01
        },
        'coupling_neutron': {
            'min': 0.3,    # Еще меньше!
            'max': 0.8,
            'step': 0.01
        },
        'coupling_meson': {
            'min': 4.0,
            'max': 5.0,
            'step': 0.05
        },
        'phase_shift': {
            'min': np.pi * 0.95,
            'max': np.pi * 1.05,
            'step': 0.01
        }
    },
    
    'search': {
        'max_iterations': 150000,
        'save_interval': 50000,
        'min_error': 0.005,  # Более строгая цель!
        'max_solutions': 50,
        'scale_factor': 100.0,
        'temperature': 0.1,
        'cooling_rate': 0.999996
    }
}

# ================= МОДЕЛЬ v5.2 =================
class ParticleModelV52:
    def __init__(self, composition, params, config, particle_name=None):
        self.composition = composition
        self.config = config
        self.particle_name = particle_name
        self.is_meson = particle_name == 'pi+'
        self.thread_count = len(composition)
        
        # Параметры кварков
        self.frequencies = []
        self.amplitudes = []
        
        for quark in composition:
            base_type = quark.replace('anti_', '')
            self.frequencies.append(params[f'freq_{base_type}'])
            self.amplitudes.append(params[f'amp_{base_type}'])
        
        # Сила связи
        if self.is_meson:
            self.coupling = params['coupling_meson']
        elif particle_name == 'proton':
            self.coupling = params['coupling_proton']
        else:  # neutron
            self.coupling = params['coupling_neutron']
        
        # Фазы
        if self.is_meson:
            self.phases = [0.0, params.get('phase_shift', np.pi)]
        elif composition == ['u', 'u', 'd']:
            base_phase = np.random.uniform(0, 2*np.pi)
            self.phases = [base_phase, base_phase, base_phase + np.pi/2]
        else:  # neutron
            base_phase = np.random.uniform(0, 2*np.pi)
            self.phases = [base_phase, base_phase + np.pi/2, base_phase + np.pi/2]
        
        self.type_properties = config['type_properties']
        self.scale_factor = config['search']['scale_factor']
    
    def calculate_charge(self):
        total = 0
        for q in self.composition:
            total += self.type_properties[q]['charge']
        return round(total, 10)
    
    def calculate_base_mass(self):
        total = 0
        for i, q in enumerate(self.composition):
            base_mass = self.type_properties[q]['base_mass']
            total += base_mass * self.amplitudes[i] * self.frequencies[i]
        return total
    
    def calculate_synchronization_energy(self):
        # Частотная когерентность
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
        
        # Фазовая когерентность
        phase_coherence = 0
        for i in range(self.thread_count):
            for j in range(i+1, self.thread_count):
                diff = abs(self.phases[i] - self.phases[j]) % (2*np.pi)
                diff = min(diff, 2*np.pi - diff)
                
                if self.is_meson:
                    phase_coherence += np.cos(diff + np.pi)  # Для мезонов
                else:
                    phase_coherence += np.cos(diff)  # Для барионов
        
        max_pairs = self.thread_count * (self.thread_count - 1) / 2
        phase_coherence = (phase_coherence / max_pairs + 1) / 2 if max_pairs > 0 else 0.5
        
        # Симметрия и дополнительные факторы
        symmetry = 1.0
        if self.particle_name == 'proton':
            symmetry = 1.1
        elif self.particle_name == 'neutron':
            symmetry = 0.95  # Меньше для нейтрона
        
        sync_energy = self.coupling * (0.6 * freq_coherence + 0.4 * phase_coherence) * symmetry
        return sync_energy
    
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
        
        if self.is_meson:
            total = base - sync
        else:
            total = base + sync
        
        return total * self.scale_factor
    
    def calculate_spin(self):
        return 0.0 if self.is_meson else 0.5

# ================= ПОИСК v5.2 =================
class IntelligentSearchV52:
    def __init__(self, config):
        self.config = config
        self.solutions = []
        self.best_solution = None
        self.iteration = 0
        self.start_time = time.time()
        self.temperature = config['search']['temperature']
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = f"particle_search_v52_{timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)
        
        with open(f"{self.result_dir}/config.json", 'w') as f:
            json.dump(config, f, indent=2, default=self._json_serializer)
        
        # Начальные параметры из v5.1
        self.current_params = {
            'freq_u': 0.950,
            'freq_d': 0.900,
            'amp_u': 1.000,
            'amp_d': 0.850,
            'coupling_proton': 1.443,
            'coupling_neutron': 0.500,
            'coupling_meson': 4.617,
            'phase_shift': np.pi
        }
        
        print("="*70)
        print("МОДЕЛЬ v5.2: Исправленные массы кварков")
        print(f"u-кварк: {config['type_properties']['u']['base_mass']} (было 2.3)")
        print(f"d-кварк: {config['type_properties']['d']['base_mass']} (было 4.8)")
        print("="*70)
    
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
                        std = 0.003 * temp_factor
                    else:
                        min_val = ranges['amplitude'][quark]['min']
                        max_val = ranges['amplitude'][quark]['max']
                        std = 0.002 * temp_factor
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
                    else:
                        min_val = ranges['coupling_meson']['min']
                        max_val = ranges['coupling_meson']['max']
                        std = 0.03 * temp_factor
                    
                    new_params[key] = np.clip(best[key] + np.random.normal(0, std), min_val, max_val)
                
                elif key == 'phase_shift':
                    min_val = ranges['phase_shift']['min']
                    max_val = ranges['phase_shift']['max']
                    std = 0.01 * temp_factor
                    new_params[key] = np.clip(best[key] + np.random.normal(0, std), min_val, max_val)
            
            return new_params
        
        else:
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
            particles[name] = ParticleModelV52(
                composition=target['composition'],
                params=params,
                config=self.config,
                particle_name=name
            )
        return particles
    
    def calculate_error(self, particles):
        total_error = 0
        details = {}
        
        # ОЧЕНЬ СТРОГИЕ ВЕСА!
        weights = {
            'proton': {'mass': 10.0, 'charge': 5.0, 'spin': 0.5},
            'neutron': {'mass': 20.0, 'charge': 5.0, 'spin': 0.5},  # Удвоили!
            'pi+': {'mass': 5.0, 'charge': 5.0, 'spin': 0.5}
        }
        
        for name, particle in particles.items():
            target = self.config['target_particles'][name]
            w = weights[name]
            
            mass = particle.calculate_total_mass()
            charge = particle.calculate_charge()
            spin = particle.calculate_spin()
            
            mass_error = abs(mass - target['mass']) / target['mass']
            charge_error = abs(charge - target['charge'])
            spin_error = abs(spin - target['spin'])
            
            error = (w['mass'] * mass_error + 
                    w['charge'] * charge_error + 
                    w['spin'] * spin_error)
            
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
    
    def run(self):
        max_iter = self.config['search']['max_iterations']
        cooling_rate = self.config['search']['cooling_rate']
        min_error = self.config['search']['min_error']
        
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
                    solution = {
                        'iteration': self.iteration,
                        'parameters': params,
                        'error': error,
                        'details': details,
                        'temperature': self.temperature
                    }
                    self.solutions.append(solution)
                    
                    if self.best_solution is None or error < self.best_solution['error']:
                        self.best_solution = solution
                        
                        if self.iteration % 5000 == 0 or error < 0.1:
                            print(f"\nИтерация {self.iteration:,}: ошибка {error:.6f}")
                            print(f"  Протон: {details['proton']['mass']:.1f} МэВ (цель 938.3)")
                            print(f"  Нейтрон: {details['neutron']['mass']:.1f} МэВ (цель 939.6)")
                            print(f"  Пион: {details['pi+']['mass']:.1f} МэВ (цель 139.6)")
                
                self.temperature *= cooling_rate
                self.iteration += 1
                
                if self.iteration % 5000 == 0:
                    elapsed = time.time() - self.start_time
                    if self.best_solution:
                        best_err = self.best_solution['error']
                        neutron_mass = self.best_solution['details']['neutron']['mass']
                    else:
                        best_err = float('inf')
                        neutron_mass = 0
                    
                    print(f"\rИтерация: {self.iteration:,} | "
                          f"Лучшая ошибка: {best_err:.4f} | "
                          f"Нейтрон: {neutron_mass:.0f} МэВ | "
                          f"Время: {elapsed:.1f} сек", end='')
                
                if error < min_error:
                    print(f"\n\nДостигнута минимальная ошибка {min_error}!")
                    break
        
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
        print("ЛУЧШЕЕ РЕШЕНИЕ v5.2:")
        print(f"Общая ошибка: {best['error']:.6f}")
        
        print(f"\nРЕЗУЛЬТАТЫ:")
        print(f"{'Частица':<10} {'Масса':<10} {'Цель':<10} {'Ошибка':<12} {'Заряд':<8} {'Спин':<8}")
        print("-" * 70)
        
        for name in ['proton', 'neutron', 'pi+']:
            d = details[name]
            target = self.config['target_particles'][name]
            print(f"{name:<10} {d['mass']:<10.1f} {target['mass']:<10.1f} "
                  f"{d['mass_error']*100:<12.3f}% {d['charge']:<8.1f} {d['spin']:<8.1f}")
        
        print(f"\nПАРАМЕТРЫ:")
        for key, value in params.items():
            print(f"  {key}: {value:.6f}")
        
        # Баланс масс
        print(f"\nБАЛАНС МАСС:")
        particles = self.create_particles(params)
        for name in ['proton', 'neutron', 'pi+']:
            p = particles[name]
            base = p.calculate_base_mass()
            sync = p.calculate_synchronization_energy()
            total = p.calculate_total_mass()
            sign = "-" if name == 'pi+' else "+"
            print(f"  {name}: {base:.3f} {sign} {abs(sync):.3f} = {total/100:.3f} × 100 МэВ")
        
        # Отношение масс кварков
        m_u = params['freq_u'] * params['amp_u'] * 2.25
        m_d = params['freq_d'] * params['amp_d'] * 4.60
        print(f"\nОТНОШЕНИЕ ЭФФЕКТИВНЫХ МАСС d/u: {m_d/m_u:.3f}")
        
        # Расчет разности масс
        p_base = particles['proton'].calculate_base_mass()
        n_base = particles['neutron'].calculate_base_mass()
        p_sync = particles['proton'].calculate_synchronization_energy()
        n_sync = particles['neutron'].calculate_synchronization_energy()
        
        diff_base = n_base - p_base
        diff_sync = n_sync - p_sync
        diff_total = (n_base + n_sync) - (p_base + p_sync)
        
        print(f"\nРАЗНОСТЬ МАСС НЕЙТРОН-ПРОТОН:")
        print(f"  По базовым массам: {diff_base:.3f} (должно быть ~1.575)")
        print(f"  По энергиям связи: {diff_sync:.3f} (должно быть ~-0.282)")
        print(f"  Итоговая разность: {diff_total:.3f} (цель 0.01293)")

# ================= ЗАПУСК =================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("МОДЕЛЬ СИНХРОНИЗАЦИИ НИТЕЙ v5.2")
    print("Ключевое изменение: уменьшена масса d-кварка с 4.8 до 4.6")
    print("Это должно уменьшить базовую массу нейтрона")
    print("="*70)
    
    search = IntelligentSearchV52(CONFIG)
    search.run()
    
    print("\n" + "="*70)
    print("ОЖИДАНИЯ v5.2:")
    print("1. Все частицы: <1% ошибка")
    print("2. Разность масс n-p: ~1.3 МэВ")
    print("3. coupling_neutron ≈ 0.5 (как в v5.1)")
    print("4. coupling_proton ≈ 1.4 (как в v5.1)")
    print("="*70)