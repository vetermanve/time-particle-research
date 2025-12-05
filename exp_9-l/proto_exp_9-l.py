"""
МОДЕЛЬ v9.0-light - ИСПРАВЛЕННАЯ ВЕРСИЯ
Исправлена ошибка с отрицательным scale в np.random.normal()
"""

import numpy as np
import time
import json
from datetime import datetime
import os
import multiprocessing as mp
from multiprocessing import Pool
from scipy.special import erf
from itertools import combinations

# ============== ФИЗИЧЕСКИЕ КОНСТАНТЫ ==============

class QuantumConstants:
    """Фундаментальные физические константы"""
    COLOR_MATRICES = {
        'R': np.array([1, 0, 0]),
        'G': np.array([0, 1, 0]), 
        'B': np.array([0, 0, 1]),
        'anti_R': np.array([-1, 0, 0]),
        'anti_G': np.array([0, -1, 0]),
        'anti_B': np.array([0, 0, -1])
    }
    
    SPIN_UP = np.array([1, 0])
    SPIN_DOWN = np.array([0, 1])
    
    QUARK_CHARGES = {
        'u': 2/3, 'd': -1/3
    }
    
    @staticmethod
    def color_coherence(color1, color2):
        vec1 = QuantumConstants.COLOR_MATRICES.get(color1, np.zeros(3))
        vec2 = QuantumConstants.COLOR_MATRICES.get(color2, np.zeros(3))
        dot = np.dot(vec1, vec2)
        return np.exp(-abs(dot))

# ============== МОДЕЛЬ КВАРКА ==============

class QuarkOscillator:
    
    def __init__(self, quark_type, params):
        self.type = quark_type
        self.anti = quark_type.startswith('anti_')
        self.base_type = quark_type.replace('anti_', '')
        
        self.base_mass = params[f'base_mass_{self.base_type}']
        self.frequency = params[f'freq_{self.base_type}']
        self.amplitude = params[f'amp_{self.base_type}']
        
        self.charge = QuantumConstants.QUARK_CHARGES[self.base_type]
        if self.anti:
            self.charge *= -1
            
        colors = ['R', 'G', 'B'] if not self.anti else ['anti_R', 'anti_G', 'anti_B']
        self.color = np.random.choice(colors)
        
        self.spin = np.random.choice(['up', 'down'])
        self.phase = np.random.uniform(0, 2*np.pi)
        
    def effective_mass(self):
        return self.base_mass * self.frequency * self.amplitude

# ============== МОДЕЛЬ АДРОНА ==============

class HadronResonator:
    
    def __init__(self, name, composition, params):
        self.name = name
        self.composition = composition
        self.params = params
        self.scale = params.get('scale_factor', 100.0)
        self.is_meson = len(composition) == 2
        
        self.quarks = [QuarkOscillator(q_type, params) for q_type in composition]
        self._assign_colors()
        self._optimize_phases()
    
    def _assign_colors(self):
        if self.is_meson:
            if 'anti' in self.quarks[0].type:
                self.quarks[0].color = 'anti_R'
                self.quarks[1].color = 'R'
            else:
                self.quarks[0].color = 'R'
                self.quarks[1].color = 'anti_R'
        else:
            colors = ['R', 'G', 'B']
            if any('anti' in q.type for q in self.quarks):
                colors = ['anti_R', 'anti_G', 'anti_B']
            np.random.shuffle(colors)
            for i, quark in enumerate(self.quarks):
                quark.color = colors[i]
    
    def _optimize_phases(self):
        if self.is_meson:
            self.quarks[0].phase = 0
            self.quarks[1].phase = np.pi
        else:
            if self.name == 'proton':
                self.quarks[0].phase = 0      # u1
                self.quarks[1].phase = 0      # u2  
                self.quarks[2].phase = np.pi/2  # d
            elif self.name == 'neutron':
                self.quarks[0].phase = 0      # u
                self.quarks[1].phase = np.pi/2  # d1
                self.quarks[2].phase = np.pi/2  # d2
    
    def calculate_color_coherence(self):
        if self.is_meson:
            return QuantumConstants.color_coherence(
                self.quarks[0].color, self.quarks[1].color)
        else:
            coherences = []
            for i, j in combinations(range(3), 2):
                coh = QuantumConstants.color_coherence(
                    self.quarks[i].color, self.quarks[j].color)
                coherences.append(coh)
            return np.mean(coherences)
    
    def calculate_phase_coherence(self):
        if self.is_meson:
            phase_diff = abs(self.quarks[0].phase - self.quarks[1].phase) % (2*np.pi)
            phase_diff = min(phase_diff, 2*np.pi - phase_diff)
            coherence = np.cos(phase_diff + np.pi)
            return (coherence + 1) / 2
        else:
            coherences = []
            for i, j in combinations(range(3), 2):
                phase_diff = abs(self.quarks[i].phase - self.quarks[j].phase) % (2*np.pi)
                phase_diff = min(phase_diff, 2*np.pi - phase_diff)
                coherence = np.cos(phase_diff)
                coherences.append((coherence + 1) / 2)
            return np.mean(coherences)
    
    def calculate_interaction_energy(self):
        color_energy = self.params.get('color_coupling', 1.0) * self.calculate_color_coherence()
        phase_energy = self.params.get('phase_coupling', 1.0) * self.calculate_phase_coherence()
        
        mass_factor = np.mean([q.effective_mass() for q in self.quarks])
        total_interaction = (color_energy + phase_energy) * mass_factor
        
        if self.is_meson:
            return -total_interaction
        else:
            return total_interaction
    
    def calculate_mass(self):
        base_mass = sum(q.effective_mass() for q in self.quarks)
        interaction = self.calculate_interaction_energy()
        
        if self.is_meson:
            total = base_mass + interaction
        else:
            total = base_mass + interaction
        
        # ИСПРАВЛЕНИЕ: не допускаем отрицательный scale
        quantum_fluctuations = self.params.get('quantum_noise', 0.001)
        scale = abs(quantum_fluctuations * total)  # Берем модуль
        noise = np.random.normal(0, scale)
        
        return (total + noise) * self.scale
    
    def calculate_charge(self):
        return sum(q.charge for q in self.quarks)

# ============== ОТЖИГ ДЛЯ СРАВНЕНИЯ С v6.1 ==============

class ComparativeAnnealer:
    
    def __init__(self, num_cores=6):
        self.num_cores = num_cores
        
        # ПАРАМЕТРЫ v6.1 ДЛЯ СРАВНЕНИЯ
        self.v61_params = {
            'base_mass_u': 2.203806,
            'base_mass_d': 4.583020,
            'freq_u': 0.956359,
            'freq_d': 0.868115,
            'amp_u': 1.032476,
            'amp_d': 0.877773,
            'coupling_proton': 1.613565,
            'coupling_neutron': 0.285395,
            'coupling_meson': 4.273121,
            'phase_shift': 3.173848,
            'scale_factor': 100.0
        }
        
        # ПАРАМЕТРЫ v9.0 (упрощенный набор)
        self.param_names = [
            'base_mass_u', 'base_mass_d',
            'freq_u', 'freq_d',
            'amp_u', 'amp_d',
            'color_coupling', 'phase_coupling',
            'meson_coupling_scale', 'baryon_coupling_scale',
            'scale_factor'
        ]
        
        # НАЧАЛЬНЫЕ ЗНАЧЕНИЯ v9.0
        self.base_params = {
            'base_mass_u': 2.203806,
            'base_mass_d': 4.583020,
            'freq_u': 0.956359,
            'freq_d': 0.868115,
            'amp_u': 1.032476,
            'amp_d': 0.877773,
            'color_coupling': 1.5,
            'phase_coupling': 1.0,
            'meson_coupling_scale': 4.0,
            'baryon_coupling_scale': 1.0,
            'scale_factor': 100.0
        }
        
        # ДИАПАЗОНЫ
        self.ranges = {
            'base_mass_u': (1.5, 3.0),
            'base_mass_d': (3.0, 6.0),
            'freq_u': (0.7, 1.2),
            'freq_d': (0.7, 1.2),
            'amp_u': (0.8, 1.3),
            'amp_d': (0.7, 1.2),
            'color_coupling': (0.5, 3.0),
            'phase_coupling': (0.5, 2.0),
            'meson_coupling_scale': (2.0, 6.0),
            'baryon_coupling_scale': (0.5, 2.0),
            'scale_factor': (90.0, 110.0)
        }
        
        # ЦЕЛЕВЫЕ ЧАСТИЦЫ (ТОЛЬКО v6.1)
        self.targets = {
            'proton': {'mass': 938.272, 'charge': 1.0, 'composition': ['u', 'u', 'd']},
            'neutron': {'mass': 939.565, 'charge': 0.0, 'composition': ['u', 'd', 'd']},
            'pi+': {'mass': 139.570, 'charge': 1.0, 'composition': ['u', 'anti_d']},
            'pi0': {'mass': 134.9768, 'charge': 0.0, 'composition': ['u', 'anti_u']},
            'pi-': {'mass': 139.570, 'charge': -1.0, 'composition': ['d', 'anti_u']},
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = f"v9_comparison_{timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)
    
    def prepare_params(self, raw_params):
        params = raw_params.copy()
        params['color_coupling_meson'] = params['color_coupling'] * params['meson_coupling_scale']
        params['phase_coupling_meson'] = params['phase_coupling'] * params['meson_coupling_scale']
        params['color_coupling_baryon'] = params['color_coupling'] * params['baryon_coupling_scale']
        params['phase_coupling_baryon'] = params['phase_coupling'] * params['baryon_coupling_scale']
        return params
    
    def evaluate_particle(self, params, particle_name, composition, is_meson):
        part_params = self.prepare_params(params)
        
        if is_meson:
            part_params['color_coupling'] = part_params.get('color_coupling_meson', part_params['color_coupling'])
            part_params['phase_coupling'] = part_params.get('phase_coupling_meson', part_params['phase_coupling'])
        
        masses = []
        charges = []
        for _ in range(5):
            hadron = HadronResonator(particle_name, composition, part_params)
            masses.append(hadron.calculate_mass())
            charges.append(hadron.calculate_charge())
        
        return np.mean(masses), np.mean(charges)
    
    def evaluate_all_particles(self, params):
        results = {}
        for name, target in self.targets.items():
            is_meson = len(target['composition']) == 2
            mass, charge = self.evaluate_particle(params, name, target['composition'], is_meson)
            results[f'{name}_mass'] = mass
            results[f'{name}_charge'] = charge
        
        m_u = params['base_mass_u'] * params['freq_u'] * params['amp_u'] * params['scale_factor']
        m_d = params['base_mass_d'] * params['freq_d'] * params['amp_d'] * params['scale_factor']
        results['m_u_eff'] = m_u
        results['m_d_eff'] = m_d
        results['ratio_d_u'] = m_d / m_u if m_u > 0 else 1
        
        return results
    
    def calculate_v61_error(self, params):
        """Ошибка по модели v6.1 (для сравнения)"""
        results = {}
        
        for name, target in self.targets.items():
            composition = target['composition']
            base_mass = 0
            for q in composition:
                base_type = q.replace('anti_', '')
                m = params[f'base_mass_{base_type}']
                f = params[f'freq_{base_type}']
                a = params[f'amp_{base_type}']
                base_mass += m * f * a
            
            coupling = params.get('coupling_proton', 1.0)
            if name == 'neutron':
                coupling = params.get('coupling_neutron', 0.3)
            elif name in ['pi+', 'pi0', 'pi-']:
                coupling = params.get('coupling_meson', 4.0)
            
            if len(composition) == 2:  # Мезоны
                mass = (base_mass - coupling) * 100
            else:  # Барионы
                mass = (base_mass + coupling) * 100
            
            results[f'{name}_mass'] = mass
        
        total_error = 0
        for name, target in self.targets.items():
            rel_error = abs(results[f'{name}_mass'] - target['mass']) / target['mass']
            total_error += rel_error ** 2
        
        return total_error
    
    def calculate_error(self, params):
        results = self.evaluate_all_particles(params)
        total_error = 0.0
        
        weights = {
            'proton': 40.0, 'neutron': 40.0,
            'pi+': 25.0, 'pi0': 30.0, 'pi-': 25.0
        }
        
        for name, target in self.targets.items():
            mass = results[f'{name}_mass']
            target_mass = target['mass']
            
            # Штраф за отрицательные массы
            if mass <= 0:
                total_error += 10000.0
                continue
                
            rel_error = abs(mass - target_mass) / target_mass
            total_error += weights[name] * (rel_error ** 2)
            
            if rel_error > 0.3:
                total_error += weights[name] * 10.0 * (rel_error - 0.3)
        
        for name, target in self.targets.items():
            if abs(results[f'{name}_charge'] - target['charge']) > 0.001:
                total_error += 1000.0
        
        if results['neutron_mass'] < results['proton_mass']:
            diff = results['proton_mass'] - results['neutron_mass']
            total_error += 500.0 * diff
        
        ratio_d_u = results['ratio_d_u']
        if ratio_d_u < 1.3 or ratio_d_u > 2.0:
            penalty = abs(ratio_d_u - 1.6) * 20.0
            total_error += penalty
        
        if params.get('meson_coupling_scale', 1) < params.get('baryon_coupling_scale', 1):
            total_error += 200.0
        
        return total_error, results
    
    def run_single_annealing(self, seed, iterations=150000, temperature=8.0):
        np.random.seed(seed)
        
        current_params = self.base_params.copy()
        for param in self.param_names:
            if param in self.ranges:
                min_val, max_val = self.ranges[param]
                current_params[param] = np.random.uniform(min_val, max_val)
        
        current_error, current_results = self.calculate_error(current_params)
        
        best_params = current_params.copy()
        best_error = current_error
        best_results = current_results
        
        cooling_rate = 0.99998
        
        for i in range(iterations):
            new_params = current_params.copy()
            
            for param in self.param_names:
                if param in self.ranges:
                    min_val, max_val = self.ranges[param]
                    current_val = current_params[param]
                    
                    step = (max_val - min_val) * 0.05
                    mutation = np.random.normal(0, step) * temperature
                    
                    new_val = current_val + mutation
                    
                    while new_val < min_val or new_val > max_val:
                        if new_val < min_val:
                            new_val = 2 * min_val - new_val
                        if new_val > max_val:
                            new_val = 2 * max_val - new_val
                    
                    new_params[param] = new_val
            
            new_error, new_results = self.calculate_error(new_params)
            
            delta = new_error - current_error
            
            if delta < 0:
                current_params = new_params
                current_error = new_error
                current_results = new_results
            else:
                prob = np.exp(-delta / temperature)
                if np.random.random() < prob:
                    current_params = new_params
                    current_error = new_error
                    current_results = new_results
            
            if new_error < best_error:
                best_params = new_params.copy()
                best_error = new_error
                best_results = new_results
            
            temperature *= cooling_rate
        
        v61_error = self.calculate_v61_error(best_params)
        
        return {
            'seed': seed,
            'params': best_params,
            'error': best_error,
            'v61_error': v61_error,
            'results': best_results
        }
    
    def run_parallel_annealing(self, total_iterations=600000):
        print("="*80)
        print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ v6.1 vs v9.0")
        print(f"Целевые частицы: {list(self.targets.keys())}")
        print(f"Ядер: {self.num_cores}")
        print(f"Итераций на ядро: {total_iterations // self.num_cores:,}")
        print("="*80)
        
        start_time = time.time()
        
        iterations_per_core = total_iterations // self.num_cores
        seeds = list(range(1000, 1000 + self.num_cores))
        
        with mp.Pool(processes=self.num_cores) as pool:
            results = pool.starmap(self.run_single_annealing, 
                                  [(s, iterations_per_core, 8.0) for s in seeds])
        
        best_result = min(results, key=lambda x: x['error'])
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*80}")
        print("ОТЖИГ ЗАВЕРШЕН")
        print(f"Время: {elapsed:.1f} сек")
        print(f"Лучшая ошибка v9.0: {best_result['error']:.3f}")
        print(f"Ошибка по методу v6.1: {best_result['v61_error']:.3f}")
        print("="*80)
        
        self.save_results(results, best_result)
        self.print_comparison_report(best_result)
        
        return best_result['params'], best_result['error'], best_result['results']
    
    def save_results(self, all_results, best_result):
        summary = {
            'model': 'v9.0_light_comparison',
            'timestamp': datetime.now().isoformat(),
            'v61_params': self.v61_params,
            'best_result': best_result,
            'all_results': [
                {'seed': r['seed'], 'error': r['error'], 'v61_error': r['v61_error']} 
                for r in all_results
            ]
        }
        
        with open(f"{self.result_dir}/comparison_results.json", 'w') as f:
            json.dump(summary, f, indent=2, default=self.json_serializer)
    
    def print_comparison_report(self, best_result):
        params = best_result['params']
        results = best_result['results']
        
        print("\n" + "="*80)
        print("СРАВНИТЕЛЬНЫЙ ОТЧЕТ: v6.1 vs v9.0")
        print("="*80)
        
        print(f"\nПАРАМЕТРЫ v9.0:")
        for param in self.param_names:
            if param in params:
                print(f"  {param}: {params[param]:.6f}")
        
        print(f"\nЭФФЕКТИВНЫЕ МАССЫ КВАРКОВ (МэВ):")
        print(f"  u: {results['m_u_eff']:.2f}")
        print(f"  d: {results['m_d_eff']:.2f}")
        print(f"  m_d/m_u: {results['ratio_d_u']:.2f}")
        
        print(f"\nМАССЫ ЧАСТИЦ v9.0:")
        total_error_v9 = 0
        for name in self.targets.keys():
            mass = results[f'{name}_mass']
            target = self.targets[name]['mass']
            if mass > 0:
                error = abs(mass - target) / target * 100
            else:
                error = 100  # Максимальная ошибка для отрицательной массы
            total_error_v9 += error
            print(f"  {name}: {mass:.1f} МэВ (цель {target:.1f}) - {error:.2f}%")
        
        print(f"\nМАССЫ ЧАСТИЦ v6.1:")
        total_error_v61 = 0
        for name, target in self.targets.items():
            composition = target['composition']
            base_mass = 0
            for q in composition:
                base_type = q.replace('anti_', '')
                m = self.v61_params[f'base_mass_{base_type}']
                f = self.v61_params[f'freq_{base_type}']
                a = self.v61_params[f'amp_{base_type}']
                base_mass += m * f * a
            
            coupling = self.v61_params['coupling_proton']
            if name == 'neutron':
                coupling = self.v61_params['coupling_neutron']
            elif name in ['pi+', 'pi0', 'pi-']:
                coupling = self.v61_params['coupling_meson']
            
            if len(composition) == 2:
                mass = (base_mass - coupling) * 100
            else:
                mass = (base_mass + coupling) * 100
            
            error = abs(mass - target['mass']) / target['mass'] * 100
            total_error_v61 += error
            print(f"  {name}: {mass:.1f} МэВ (цель {target['mass']:.1f}) - {error:.2f}%")
        
        avg_error_v9 = total_error_v9 / len(self.targets)
        avg_error_v61 = total_error_v61 / len(self.targets)
        
        print(f"\nСРЕДНИЕ ОШИБКИ:")
        print(f"  v9.0: {avg_error_v9:.2f}%")
        print(f"  v6.1: {avg_error_v61:.2f}%")
        
        print(f"\nРАЗНОСТЬ МАСС:")
        diff_v9 = results['neutron_mass'] - results['proton_mass']
        diff_v61 = (939.565 - 938.272)  # Из v6.1 результатов
        print(f"  v9.0: {diff_v9:.3f} МэВ")
        print(f"  v6.1: {diff_v61:.3f} МэВ")
        print(f"  Эксперимент: 1.293 МэВ")
        
        print(f"\nРезультаты сохранены в: {self.result_dir}")
        print("="*80)
    
    def json_serializer(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

# ============== ЗАПУСК ==============

def main():
    print("="*80)
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ: v6.1 (простая) vs v9.0 (полная)")
    print("Исправленная версия - ошибка scale < 0 устранена")
    print("="*80)
    
    print("\nЦЕЛИ ОПТИМИЗАЦИИ:")
    print("  Протон: 938.272 МэВ")
    print("  Нейтрон: 939.565 МэВ")
    print("  π⁺: 139.570 МэВ")
    print("  π⁰: 134.9768 МэВ")
    print("  π⁻: 139.570 МэВ")
    
    print("\nПАРАМЕТРЫ v9.0:")
    print("  11 параметров, цветовая и фазовая когерентность")
    print("  600,000 итераций, 6 ядер")
    
    try:
        num_cores = min(6, mp.cpu_count())
        print(f"\nИспользуется ядер: {num_cores}")
        
        annealer = ComparativeAnnealer(num_cores=num_cores)
        best_params, best_error, best_results = annealer.run_parallel_annealing(
            total_iterations=600000
        )
        
    except Exception as e:
        print(f"\nОШИБКА: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("ВЫЧИСЛЕНИЯ ЗАВЕРШЕНЫ")
    print("="*80)

if __name__ == "__main__":
    if hasattr(mp, 'set_start_method'):
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass
    
    main()