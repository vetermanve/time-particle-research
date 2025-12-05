"""
МОДЕЛЬ v9.1 - УПРОЩЕННАЯ ВЕРСИЯ БЕЗ СЛОЖНОГО МОНИТОРИНГА
"""

import numpy as np
import time
import json
from datetime import datetime
import os
import multiprocessing as mp
from multiprocessing import Pool, Manager
from scipy.special import erf
from itertools import combinations

# ============== ФИЗИЧЕСКИЕ КОНСТАНТЫ ==============

class QuantumConstants:
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

# ============== ГИБРИДНАЯ МОДЕЛЬ АДРОНА ==============

class HybridHadronResonator:
    
    def __init__(self, name, composition, params):
        self.name = name
        self.composition = composition
        self.params = params
        self.scale = params.get('scale_factor', 100.0)
        self.is_meson = len(composition) == 2
        self.is_neutral_meson = name in ['pi0']
        
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
    
    def calculate_base_interaction_energy(self):
        """Базовая энергия взаимодействия из v9.0 (цвет + фаза)"""
        color_energy = self.params.get('color_coupling', 1.0) * self.calculate_color_coherence()
        phase_energy = self.params.get('phase_coupling', 1.0) * self.calculate_phase_coherence()
        
        mass_factor = np.mean([q.effective_mass() for q in self.quarks])
        base_energy = (color_energy + phase_energy) * mass_factor
        
        if self.is_meson:
            return -base_energy  # Для мезонов уменьшает массу
        else:
            return base_energy   # Для барионов увеличивает массу
    
    def calculate_specific_coupling(self):
        """Специфический коэффициент связи из v6.1"""
        if self.name == 'proton':
            return self.params.get('coupling_proton', 1.0)
        elif self.name == 'neutron':
            return self.params.get('coupling_neutron', 0.3)
        elif self.name in ['pi+', 'pi-']:
            return self.params.get('coupling_meson_charged', 4.0)
        elif self.name == 'pi0':
            return self.params.get('coupling_meson_neutral', 3.5)
        else:
            return 1.0
    
    def calculate_interaction_energy(self):
        """Гибридная энергия взаимодействия: v9.0 × v6.1"""
        base_energy = self.calculate_base_interaction_energy()
        specific_factor = self.calculate_specific_coupling()
        
        return base_energy * specific_factor
    
    def calculate_mass(self):
        base_mass = sum(q.effective_mass() for q in self.quarks)
        interaction = self.calculate_interaction_energy()
        
        total = base_mass + interaction
        
        quantum_fluctuations = self.params.get('quantum_noise', 0.001)
        scale = abs(quantum_fluctuations * total)
        noise = np.random.normal(0, scale)
        
        return (total + noise) * self.scale
    
    def calculate_charge(self):
        return sum(q.charge for q in self.quarks)

# ============== ПРОСТОЙ МОНИТОРИНГ ==============

def save_progress(worker_id, result_dir, iteration, total_iterations, error, temperature):
    """Сохранить прогресс воркера в файл"""
    progress_file = os.path.join(result_dir, f"progress_worker_{worker_id}.json")
    
    progress_data = {
        'worker_id': worker_id,
        'iteration': iteration,
        'total_iterations': total_iterations,
        'progress_percent': (iteration / total_iterations) * 100 if total_iterations > 0 else 0,
        'current_error': error,
        'temperature': temperature,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)

def print_progress_summary(result_dir, num_workers):
    """Вывести суммарный прогресс всех воркеров"""
    total_iterations = 0
    total_completed = 0
    errors = []
    
    for i in range(num_workers):
        progress_file = os.path.join(result_dir, f"progress_worker_{i}.json")
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    data = json.load(f)
                total_iterations += data.get('total_iterations', 0)
                total_completed += data.get('iteration', 0)
                errors.append(data.get('current_error', 0))
            except:
                pass
    
    if total_iterations > 0:
        overall_progress = (total_completed / total_iterations) * 100
        
        print(f"\n{'='*60}")
        print(f"ПРОГРЕСС: {overall_progress:.1f}% ({total_completed:,}/{total_iterations:,} итераций)")
        
        if errors:
            avg_error = np.mean(errors)
            min_error = min(errors)
            print(f"Ошибки: средняя {avg_error:.3f}, лучшая {min_error:.3f}")
        
        print(f"{'='*60}")

# ============== ОТЖИГ ДЛЯ ГИБРИДНОЙ МОДЕЛИ ==============

class HybridAnnealerSimple:
    
    def __init__(self, num_cores=6):
        self.num_cores = num_cores
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = f"v91_simple_{timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)
        
        self.v61_params = {
            'base_mass_u': 2.203806,
            'base_mass_d': 4.583020,
            'freq_u': 0.956359,
            'freq_d': 0.868115,
            'amp_u': 1.032476,
            'amp_d': 0.877773,
            'coupling_proton': 1.613565,
            'coupling_neutron': 0.285395,
            'coupling_meson_charged': 4.273121,
            'coupling_meson_neutral': 3.8,
            'scale_factor': 100.0
        }
        
        self.param_names = [
            'base_mass_u', 'base_mass_d',
            'freq_u', 'freq_d',
            'amp_u', 'amp_d',
            'color_coupling', 'phase_coupling',
            'meson_coupling_scale', 'baryon_coupling_scale',
            'coupling_proton', 'coupling_neutron',
            'coupling_meson_charged', 'coupling_meson_neutral',
            'scale_factor'
        ]
        
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
            'coupling_proton': 1.613565,
            'coupling_neutron': 0.285395,
            'coupling_meson_charged': 4.273121,
            'coupling_meson_neutral': 3.8,
            'scale_factor': 100.0
        }
        
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
            'coupling_proton': (1.0, 2.5),
            'coupling_neutron': (0.1, 0.8),
            'coupling_meson_charged': (3.0, 5.0),
            'coupling_meson_neutral': (2.5, 4.5),
            'scale_factor': (90.0, 110.0)
        }
        
        self.targets = {
            'proton': {'mass': 938.272, 'charge': 1.0, 'composition': ['u', 'u', 'd']},
            'neutron': {'mass': 939.565, 'charge': 0.0, 'composition': ['u', 'd', 'd']},
            'pi+': {'mass': 139.570, 'charge': 1.0, 'composition': ['u', 'anti_d']},
            'pi0': {'mass': 134.9768, 'charge': 0.0, 'composition': ['u', 'anti_u']},
            'pi-': {'mass': 139.570, 'charge': -1.0, 'composition': ['d', 'anti_u']},
        }
    
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
        else:
            part_params['color_coupling'] = part_params.get('color_coupling_baryon', part_params['color_coupling'])
            part_params['phase_coupling'] = part_params.get('phase_coupling_baryon', part_params['phase_coupling'])
        
        masses = []
        charges = []
        for _ in range(10):
            hadron = HybridHadronResonator(particle_name, composition, part_params)
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
        
        results['E_proton'] = params.get('coupling_proton', 1.0)
        results['E_neutron'] = params.get('coupling_neutron', 0.3)
        results['E_meson_charged'] = params.get('coupling_meson_charged', 4.0)
        results['E_meson_neutral'] = params.get('coupling_meson_neutral', 3.5)
        results['ratio_neutron_proton'] = results['E_neutron'] / results['E_proton'] if results['E_proton'] > 0 else 0
        
        return results
    
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
        
        if params.get('coupling_neutron', 0) > params.get('coupling_proton', 1):
            total_error += 300.0
        
        if params.get('coupling_meson_neutral', 0) > params.get('coupling_meson_charged', 4):
            total_error += 200.0
        
        mass_diff = abs((results['neutron_mass'] - results['proton_mass']) - 1.293)
        total_error += 100.0 * mass_diff
        
        return total_error, results
    
    def run_single_annealing(self, worker_id, seed, total_iterations, temperature=8.0):
        """Запуск отжига для одного воркера"""
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
        
        cooling_rate = 0.99997
        update_interval = max(1000, total_iterations // 100)
        
        for i in range(total_iterations):
            new_params = current_params.copy()
            
            for param in self.param_names:
                if param in self.ranges:
                    min_val, max_val = self.ranges[param]
                    current_val = current_params[param]
                    
                    step = (max_val - min_val) * 0.05
                    mutation = np.random.normal(0, step) * temperature
                    
                    if param == 'coupling_neutron' and current_val > 0.5:
                        mutation -= 0.2 * step
                    elif param == 'coupling_proton' and current_val < 1.3:
                        mutation += 0.2 * step
                    elif param == 'coupling_meson_neutral' and current_val > new_params.get('coupling_meson_charged', 4):
                        mutation -= 0.3 * step
                    
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
            
            # Сохраняем прогресс каждые update_interval итераций
            if i % update_interval == 0 or i == total_iterations - 1:
                save_progress(worker_id, self.result_dir, i + 1, 
                            total_iterations, current_error, temperature)
        
        return {
            'worker_id': worker_id,
            'seed': seed,
            'params': best_params,
            'error': best_error,
            'results': best_results
        }
    
    def run_parallel_annealing(self, total_iterations=1200000):
        print("="*80)
        print("ГИБРИДНАЯ МОДЕЛЬ v9.1 - УПРОЩЕННАЯ ВЕРСИЯ")
        print(f"Ядер: {self.num_cores}")
        print(f"Общее количество итераций: {total_iterations:,}")
        print("="*80)
        
        start_time = time.time()
        
        iterations_per_core = total_iterations // self.num_cores
        seeds = list(range(1000, 1000 + self.num_cores))
        
        print(f"\nЗапускаю {self.num_cores} воркеров...")
        print(f"Итераций на воркер: {iterations_per_core:,}")
        print(f"Ожидаемое время: ~{iterations_per_core * 0.0005:.1f} секунд на воркер")
        
        # Запускаем воркеров
        with mp.Pool(processes=self.num_cores) as pool:
            args = [(i, seeds[i], iterations_per_core, 8.0) 
                   for i in range(self.num_cores)]
            
            # Запускаем асинхронно
            async_results = []
            for arg in args:
                async_result = pool.apply_async(self.run_single_annealing, arg)
                async_results.append(async_result)
            
            # Периодически проверяем прогресс
            completed = 0
            while completed < self.num_cores:
                completed = sum(1 for r in async_results if r.ready())
                print_progress_summary(self.result_dir, self.num_cores)
                time.sleep(5)  # Проверяем каждые 5 секунд
            
            # Собираем результаты
            results = [r.get() for r in async_results]
        
        best_result = min(results, key=lambda x: x['error'])
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*80}")
        print("ОТЖИГ ЗАВЕРШЕН")
        print(f"Общее время: {elapsed:.1f} сек")
        print(f"Лучшая ошибка: {best_result['error']:.3f}")
        print("="*80)
        
        self.save_results(results, best_result)
        self.print_final_report(best_result)
        
        return best_result['params'], best_result['error'], best_result['results']
    
    def save_results(self, all_results, best_result):
        summary = {
            'model': 'v9.1_hybrid_simple',
            'timestamp': datetime.now().isoformat(),
            'num_workers': self.num_cores,
            'best_result': best_result,
            'all_workers': all_results,
            'worker_errors': [r['error'] for r in all_results]
        }
        
        with open(os.path.join(self.result_dir, "final_results.json"), 'w') as f:
            json.dump(summary, f, indent=2, default=self.json_serializer)
    
    def print_final_report(self, best_result):
        params = best_result['params']
        results = best_result['results']
        
        print("\n" + "="*80)
        print("ФИНАЛЬНЫЙ ОТЧЕТ v9.1")
        print("="*80)
        
        print(f"\nКЛЮЧЕВЫЕ ПАРАМЕТРЫ:")
        groups = {
            'Массы кварков': ['base_mass_u', 'base_mass_d'],
            'Частоты': ['freq_u', 'freq_d'],
            'Амплитуды': ['amp_u', 'amp_d'],
            'Физические coupling': ['color_coupling', 'phase_coupling'],
            'Специфические coupling': ['coupling_proton', 'coupling_neutron',
                                      'coupling_meson_charged', 'coupling_meson_neutral']
        }
        
        for group_name, param_list in groups.items():
            print(f"  {group_name}:")
            for param in param_list:
                if param in params:
                    print(f"    {param}: {params[param]:.6f}")
        
        print(f"\nЭФФЕКТИВНЫЕ МАССЫ КВАРКОВ (МэВ):")
        print(f"  u: {results['m_u_eff']:.2f}")
        print(f"  d: {results['m_d_eff']:.2f}")
        print(f"  m_d/m_u: {results['ratio_d_u']:.2f}")
        
        print(f"\nЭНЕРГИИ СВЯЗИ:")
        print(f"  Протон: {results['E_proton']:.3f}")
        print(f"  Нейтрон: {results['E_neutron']:.3f}")
        print(f"  Отношение n/p: {results['ratio_neutron_proton']:.3f}")
        
        print(f"\nМАССЫ ЧАСТИЦ:")
        total_error = 0
        for name in self.targets.keys():
            mass = results[f'{name}_mass']
            target = self.targets[name]['mass']
            error = abs(mass - target) / target * 100
            total_error += error
            print(f"  {name}: {mass:.1f} МэВ (цель {target:.1f}) - {error:.2f}%")
        
        avg_error = total_error / len(self.targets)
        print(f"\nСредняя ошибка: {avg_error:.2f}%")
        
        diff = results['neutron_mass'] - results['proton_mass']
        print(f"\nРАЗНОСТЬ МАСС n-p:")
        print(f"  Модель: {diff:.3f} МэВ")
        print(f"  Эксперимент: 1.293 МэВ")
        print(f"  Отклонение: {abs(diff-1.293):.3f} МэВ")
        
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
    print("ГИБРИДНАЯ МОДЕЛЬ v9.1 - УПРОЩЕННАЯ ВЕРСИЯ")
    print("="*80)
    
    print("\nОСОБЕННОСТИ:")
    print("  • Сохраняет физическую глубину v9.0 (цвет, фазы)")
    print("  • Использует прагматизм v6.1 (специфические коэффициенты)")
    print("  • Простой мониторинг через файлы")
    print("  • Проверка прогресса каждые 5 секунд")
    
    print("\nПАРАМЕТРЫ ОПТИМИЗАЦИИ:")
    print("  15 параметров, 6 ядер, 1,200,000 итераций")
    
    try:
        num_cores = min(6, mp.cpu_count())
        print(f"\nИспользуется ядер: {num_cores}")
        
        annealer = HybridAnnealerSimple(num_cores=num_cores)
        best_params, best_error, best_results = annealer.run_parallel_annealing(
            total_iterations=1200000
        )
        
    except KeyboardInterrupt:
        print("\n\nВычисление прервано пользователем")
        print("Промежуточные результаты сохранены")
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