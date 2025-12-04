"""
МОДЕЛЬ v6.5 - Строгая физическая согласованность
Фиксация параметров u и d из v6.1 + настройка s-кварка
"""

import numpy as np
import time
import json
from datetime import datetime
import os

class ParticleModelV65:
    def __init__(self, params, particle_name, composition):
        self.params = params
        self.particle_name = particle_name
        self.composition = composition
        self.is_meson = len(composition) == 2
        
        # ЖЕСТКО ФИКСИРУЕМ параметры u и d из v6.1
        self.base_mass_u = 2.203806  # Фиксировано из v6.1
        self.base_mass_d = 4.583020  # Фиксировано из v6.1
        self.base_mass_s = params.get('base_mass_s', 25.0)  # Настраиваемый
        
        self.freq_u = 0.956359  # Фиксировано
        self.freq_d = 0.868115  # Фиксировано  
        self.freq_s = params.get('freq_s', 0.8)
        
        self.amp_u = 1.032476  # Фиксировано
        self.amp_d = 0.877773  # Фиксировано
        self.amp_s = params.get('amp_s', 0.85)
        
        # Силы связи (настраиваемые)
        self.coupling_proton = params.get('coupling_proton', 1.613565)  # v6.1
        self.coupling_neutron = params.get('coupling_neutron', 0.285395)  # v6.1
        self.coupling_lambda0 = params.get('coupling_lambda0', 1.0)
        self.coupling_meson_light = params.get('coupling_meson_light', 4.273121)  # v6.1
        self.coupling_meson_strange = params.get('coupling_meson_strange', 5.0)
        
        self.phase_shift = 3.173848  # Фиксировано из v6.1
        self.scale = 100.0
        
    def get_quark_params(self, quark):
        """Получить параметры кварка"""
        quark_type = quark.replace('anti_', '')
        
        if quark_type == 'u':
            return self.base_mass_u, self.freq_u, self.amp_u
        elif quark_type == 'd':
            return self.base_mass_d, self.freq_d, self.amp_d
        elif quark_type == 's':
            return self.base_mass_s, self.freq_s, self.amp_s
        else:
            if quark_type == 'u':
                return self.base_mass_u, self.freq_u, self.amp_u
            elif quark_type == 'd':
                return self.base_mass_d, self.freq_d, self.amp_d
            elif quark_type == 's':
                return self.base_mass_s, self.freq_s, self.amp_s
        
        return 1.0, 1.0, 1.0
    
    def calculate_base_mass(self):
        """Расчет базовой массы"""
        total = 0
        for quark in self.composition:
            base_mass, freq, amp = self.get_quark_params(quark)
            total += base_mass * freq * amp
        return total
    
    def calculate_sync_energy(self):
        """Расчет энергии синхронизации"""
        if self.particle_name == 'proton':
            coupling = self.coupling_proton
            phases = [0, 0, np.pi/2]
        elif self.particle_name == 'neutron':
            coupling = self.coupling_neutron
            phases = [0, np.pi/2, np.pi/2]
        elif self.particle_name == 'lambda0':
            coupling = self.coupling_lambda0
            phases = [0, np.pi/2, np.pi]
        elif self.particle_name in ['pi+', 'pi0', 'pi-']:
            coupling = self.coupling_meson_light
            phases = [0, self.phase_shift]
        elif self.particle_name in ['k+', 'k0', 'k-', 'k0_bar']:
            coupling = self.coupling_meson_strange
            phases = [0, self.phase_shift]
        else:
            coupling = self.coupling_meson_light
            phases = [0, self.phase_shift] if self.is_meson else [0, np.pi/2, np.pi/2]
        
        thread_count = len(self.composition)
        
        # Проверка количества фаз
        if len(phases) != thread_count:
            phases = phases[:thread_count]  # Безопасное обрезание
        
        # Частотная когерентность
        freq_coherence = 1.0
        
        # Фазовая когерентность
        if thread_count >= 2:
            phase_coherence_sum = 0
            for i in range(thread_count):
                for j in range(i+1, thread_count):
                    diff = abs(phases[i] - phases[j]) % (2*np.pi)
                    diff = min(diff, 2*np.pi - diff)
                    
                    if self.is_meson:
                        phase_coherence_sum += np.cos(diff + np.pi)
                    else:
                        phase_coherence_sum += np.cos(diff)
            
            max_pairs = thread_count * (thread_count - 1) / 2
            phase_coherence = (phase_coherence_sum / max_pairs + 1) / 2
        else:
            phase_coherence = 0.5
        
        # Симметрия
        symmetry = 1.0
        if not self.is_meson:
            if self.particle_name == 'proton':
                symmetry = 1.1
            elif self.particle_name == 'neutron':
                symmetry = 0.95
            elif self.particle_name == 'lambda0':
                symmetry = 0.9
        
        sync_energy = coupling * (0.6 * freq_coherence + 0.4 * phase_coherence) * symmetry
        return sync_energy
    
    def calculate_mass(self):
        """Расчет массы частицы"""
        base = self.calculate_base_mass()
        sync = self.calculate_sync_energy()
        
        if self.is_meson:
            total = base - sync
        else:
            total = base + sync
        
        return total * self.scale
    
    def calculate_charge(self):
        """Расчет заряда частицы"""
        charges = {
            'u': 2/3, 'd': -1/3, 's': -1/3,
            'anti_u': -2/3, 'anti_d': 1/3, 'anti_s': 1/3
        }
        
        total = 0
        for quark in self.composition:
            total += charges.get(quark, 0)
        
        return round(total, 10)

class StrictAnnealingOptimizerV65:
    def __init__(self):
        # Целевые частицы
        self.target_particles = {
            'proton': {'mass': 938.272, 'charge': 1.0, 'composition': ['u', 'u', 'd']},
            'neutron': {'mass': 939.565, 'charge': 0.0, 'composition': ['u', 'd', 'd']},
            'pi+': {'mass': 139.570, 'charge': 1.0, 'composition': ['u', 'anti_d']},
            'pi0': {'mass': 134.9768, 'charge': 0.0, 'composition': ['u', 'anti_u']},
            'pi-': {'mass': 139.570, 'charge': -1.0, 'composition': ['d', 'anti_u']},
            'k+': {'mass': 493.677, 'charge': 1.0, 'composition': ['u', 'anti_s']},
            'k0': {'mass': 497.611, 'charge': 0.0, 'composition': ['d', 'anti_s']},
            'k-': {'mass': 493.677, 'charge': -1.0, 'composition': ['s', 'anti_u']},
            'k0_bar': {'mass': 497.611, 'charge': 0.0, 'composition': ['s', 'anti_d']},
            'lambda0': {'mass': 1115.683, 'charge': 0.0, 'composition': ['u', 'd', 's']},
        }
        
        # ТОЛЬКО НАСТРАИВАЕМЫЕ параметры
        self.current_params = {
            'base_mass_s': 30.0,  # Должно дать отношение ~13.6 (30/2.2038)
            'freq_s': 0.7,        # Ниже частоты u/d
            'amp_s': 0.8,         # Ниже амплитуды u/d
            
            # Ключевые корректировки:
            'coupling_neutron': 0.35,   # Увеличиваем для нейтрона
            'coupling_lambda0': 0.8,    # Уменьшаем для Λ⁰
            'coupling_meson_strange': 5.2,  # Увеличиваем для каонов
        }
        
        # Диапазоны с ФИЗИЧЕСКИМИ ограничениями
        self.param_ranges = {
            'base_mass_s': (20.0, 60.0),    # Чтобы m_s/m_u = 9-27
            'freq_s': (0.5, 0.9),
            'amp_s': (0.6, 1.0),
            
            'coupling_neutron': (0.30, 0.40),  # Около v6.1 но с поправкой
            'coupling_lambda0': (0.5, 1.5),    # Должен быть меньше протонного
            'coupling_meson_strange': (4.5, 6.0),
        }
        
        # Фиксированные параметры из v6.1 (для расчета ошибок)
        self.fixed_params = {
            'base_mass_u': 2.203806,
            'base_mass_d': 4.583020,
            'freq_u': 0.956359,
            'freq_d': 0.868115,
            'amp_u': 1.032476,
            'amp_d': 0.877773,
            'coupling_proton': 1.613565,
            'coupling_meson_light': 4.273121,
            'phase_shift': 3.173848
        }
        
        self.best_params = None
        self.best_error = float('inf')
        self.best_results = None
        self.history = []
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = f"strict_optimization_v65_{timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)
    
    def combine_params(self, params):
        """Объединение фиксированных и настраиваемых параметров"""
        combined = self.fixed_params.copy()
        combined.update(params)
        return combined
    
    def evaluate_params(self, params):
        """Оценка параметров"""
        combined = self.combine_params(params)
        results = {}
        
        # Создаем модели
        models = {}
        for name, target in self.target_particles.items():
            models[name] = ParticleModelV65(combined, name, target['composition'])
        
        # Рассчитываем массы и заряды
        for name, model in models.items():
            results[f'{name}_mass'] = model.calculate_mass()
            results[f'{name}_charge'] = model.calculate_charge()
        
        # Рассчитываем эффективные массы кварков
        m_u_eff = combined['base_mass_u'] * combined['freq_u'] * combined['amp_u']
        m_d_eff = combined['base_mass_d'] * combined['freq_d'] * combined['amp_d']
        m_s_eff = combined['base_mass_s'] * combined['freq_s'] * combined['amp_s']
        
        results['m_u_eff_mev'] = m_u_eff * 100
        results['m_d_eff_mev'] = m_d_eff * 100
        results['m_s_eff_mev'] = m_s_eff * 100
        results['mass_ratio_d_u'] = m_d_eff / m_u_eff
        results['mass_ratio_s_u'] = m_s_eff / m_u_eff
        
        return results
    
    def calculate_total_error(self, params):
        """Расчет ошибки с ЖЕСТКИМИ ФИЗИЧЕСКИМИ ограничениями"""
        results = self.evaluate_params(params)
        total_error = 0.0
        
        # 1. КРИТИЧЕСКИ ВАЖНЫЕ ОШИБКИ МАСС
        critical_errors = {
            'proton': 20.0,
            'neutron': 50.0,   # УВЕЛИЧЕННЫЙ вес
            'lambda0': 40.0,   # УВЕЛИЧЕННЫЙ вес
            'k+': 15.0,
            'k0': 15.0,
        }
        
        for name, weight in critical_errors.items():
            target = self.target_particles[name]['mass']
            calculated = results[f'{name}_mass']
            abs_error = abs(calculated - target)
            rel_error = abs_error / target if target > 0 else 1.0
            
            # Комбинированная ошибка с усилением больших отклонений
            mass_error = weight * (abs_error + rel_error * 10)
            
            # ДОПОЛНИТЕЛЬНЫЙ ШТРАФ за большие отклонения
            if rel_error > 0.01:  # >1%
                mass_error *= (1 + rel_error * 10)
            
            total_error += mass_error
        
        # 2. Остальные частицы (меньший вес)
        other_particles = ['pi+', 'pi0', 'pi-', 'k-', 'k0_bar']
        for name in other_particles:
            target = self.target_particles[name]['mass']
            calculated = results[f'{name}_mass']
            abs_error = abs(calculated - target)
            total_error += abs_error * 2.0
        
        # 3. ОШИБКИ ЗАРЯДОВ (СТРОГИЕ)
        for name, target in self.target_particles.items():
            calculated = results[f'{name}_charge']
            if abs(calculated - target['charge']) > 0.001:
                total_error += 10000.0
        
        # 4. ФИЗИЧЕСКИЕ ОГРАНИЧЕНИЯ (ГИГАНТСКИЕ ШТРАФЫ)
        penalties = 0.0
        
        # a) Отношение m_s/m_u должно быть 20-30
        mass_ratio_s_u = results['mass_ratio_s_u']
        if mass_ratio_s_u < 15 or mass_ratio_s_u > 35:
            penalties += 5000.0 * abs(mass_ratio_s_u - 25)
        
        # b) coupling_neutron должен быть МЕНЬШЕ coupling_proton
        if params['coupling_neutron'] >= self.fixed_params['coupling_proton']:
            penalties += 10000.0
        
        # c) Масса нейтрона должна быть БОЛЬШЕ массы протона
        neutron_mass = results['neutron_mass']
        proton_mass = results['proton_mass']
        target_diff = 1.293  # Нейтрон тяжелее протона на 1.293 МэВ
        actual_diff = neutron_mass - proton_mass
        
        if actual_diff < 0:  # Если нейтрон легче протона
            penalties += 5000.0 * abs(actual_diff)
        elif abs(actual_diff - target_diff) > 0.5:  # Неправильная разность
            penalties += 1000.0 * abs(actual_diff - target_diff)
        
        # d) Масса Λ⁰ должна быть больше массы протона, но меньше 1200 МэВ
        lambda_mass = results['lambda0_mass']
        if lambda_mass < 1100 or lambda_mass > 1150:
            penalties += 1000.0 * abs(lambda_mass - 1115.683)
        
        # e) Массы каонов должны быть близки к целям
        kplus_mass = results['k+_mass']
        k0_mass = results['k0_mass']
        if abs(kplus_mass - 493.677) > 10:
            penalties += 500.0 * abs(kplus_mass - 493.677)
        if abs(k0_mass - 497.611) > 10:
            penalties += 500.0 * abs(k0_mass - 497.611)
        
        total_error += penalties
        
        return total_error, results
    
    def mutate_params(self, params, temperature):
        """Умная мутация с учетом физических ограничений"""
        new_params = params.copy()
        
        for key in params.keys():
            if key in self.param_ranges:
                min_val, max_val = self.param_ranges[key]
                current = params[key]
                
                # Адаптивный шаг
                range_width = max_val - min_val
                
                # Разные скорости для разных параметров
                if key == 'base_mass_s':
                    step_factor = 0.04
                elif 'coupling' in key:
                    step_factor = 0.02
                else:
                    step_factor = 0.015
                
                step = range_width * step_factor * temperature * np.random.randn()
                new_val = current + step
                
                # Притяжение к физически разумным значениям
                if key == 'base_mass_s':
                    # Стремление к ~25-30
                    attraction = 0.1 * (27.5 - new_val)  # Притяжение к 27.5
                    new_val += attraction * temperature
                elif key == 'coupling_neutron':
                    # Должно быть меньше coupling_proton (1.6136)
                    if new_val > 1.6:
                        new_val = 1.6 - abs(new_val - 1.6) * 0.5
                
                # Ограничение диапазоном
                new_val = max(min_val, min(max_val, new_val))
                new_params[key] = new_val
        
        return new_params
    
    def run_optimization(self, iterations=1000000, initial_temp=3.0,
                        cooling_rate=0.999995, save_interval=100000):
        """Запуск оптимизации"""
        print("="*80)
        print("СТРОГАЯ ОПТИМИЗАЦИЯ v6.5")
        print("Фиксированы параметры u и d из v6.1")
        print(f"Итераций: {iterations:,}")
        print("="*80)
        
        current_params = self.current_params.copy()
        current_error, current_results = self.calculate_total_error(current_params)
        
        self.best_params = current_params.copy()
        self.best_error = current_error
        self.best_results = current_results
        
        temperature = initial_temp
        start_time = time.time()
        
        stats = {'accepts': 0, 'improves': 0, 'rejects': 0}
        
        for i in range(iterations):
            # Генерация нового решения
            if i < 200000:
                new_params = self.mutate_params(current_params, temperature * 1.5)
            else:
                new_params = self.mutate_params(current_params, temperature)
            
            # Оценка
            new_error, new_results = self.calculate_total_error(new_params)
            
            # Критерий принятия
            delta_error = new_error - current_error
            
            if delta_error < 0:
                current_params = new_params
                current_error = new_error
                current_results = new_results
                stats['accepts'] += 1
                stats['improves'] += 1
            else:
                probability = np.exp(-delta_error / temperature)
                if np.random.random() < probability:
                    current_params = new_params
                    current_error = new_error
                    current_results = new_results
                    stats['accepts'] += 1
                else:
                    stats['rejects'] += 1
            
            # Обновление лучшего
            if new_error < self.best_error:
                self.best_params = new_params.copy()
                self.best_error = new_error
                self.best_results = new_results
                
                self.history.append({
                    'iteration': i,
                    'error': self.best_error,
                    'params': self.best_params.copy(),
                    'results': self.best_results.copy(),
                    'temperature': temperature
                })
            
            # Охлаждение
            temperature *= cooling_rate
            
            # Вывод прогресса
            if i % 50000 == 0:
                elapsed = time.time() - start_time
                progress = (i / iterations) * 100
                
                neutron_err = abs(self.best_results.get('neutron_mass', 0) - 939.565)
                lambda_err = abs(self.best_results.get('lambda0_mass', 0) - 1115.683)
                ratio_s_u = self.best_results.get('mass_ratio_s_u', 0)
                
                print(f"\rИтерация {i:,}/{iterations:,} ({progress:.1f}%) | "
                      f"Ошибка: {self.best_error:.1f} | "
                      f"Нейтрон: {neutron_err:.1f} МэВ | "
                      f"Λ⁰: {lambda_err:.1f} МэВ | "
                      f"m_s/m_u: {ratio_s_u:.2f} | "
                      f"Время: {elapsed:.0f}с", end='', flush=True)
            
            # Сохранение
            if i % save_interval == 0 and i > 0:
                self.save_checkpoint(i)
        
        # Финальные результаты
        elapsed = time.time() - start_time
        print(f"\n\n{'='*80}")
        print("ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
        print(f"Время: {elapsed:.1f} сек, Итераций: {iterations:,}")
        print(f"Лучшая ошибка: {self.best_error:.1f}")
        print(f"Улучшений: {stats['improves']}")
        
        self.save_final_results()
        self.print_detailed_report()
        
        return self.best_params, self.best_error, self.best_results
    
    def save_checkpoint(self, iteration):
        """Сохранение контрольной точки"""
        checkpoint = {
            'iteration': iteration,
            'error': self.best_error,
            'params': self.best_params,
            'results': self.best_results,
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f"{self.result_dir}/checkpoint_{iteration:08d}.json"
        with open(filename, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=self.json_serializer)
    
    def save_final_results(self):
        """Сохранение результатов"""
        results = {
            'optimization_info': {
                'best_error': self.best_error,
                'iterations': len(self.history),
                'timestamp': datetime.now().isoformat()
            },
            'fixed_parameters': self.fixed_params,
            'optimized_parameters': self.best_params,
            'results': self.best_results
        }
        
        with open(f"{self.result_dir}/final_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=self.json_serializer)
        
        self.save_text_report()
    
    def save_text_report(self):
        """Сохранение текстового отчета"""
        filename = f"{self.result_dir}/FINAL_REPORT.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ФИНАЛЬНЫЙ ОТЧЕТ МОДЕЛИ v6.5\n")
            f.write("="*80 + "\n\n")
            
            f.write("ФИКСИРОВАННЫЕ ПАРАМЕТРЫ (из v6.1):\n")
            for key, value in self.fixed_params.items():
                f.write(f"  {key}: {value:.6f}\n")
            
            f.write("\nОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ:\n")
            for key, value in self.best_params.items():
                f.write(f"  {key}: {value:.6f}\n")
            
            f.write("\nРЕЗУЛЬТАТЫ:\n")
            f.write(f"{'Частица':<10} {'Масса (МэВ)':<12} {'Цель (МэВ)':<12} {'Ошибка (%)':<10}\n")
            f.write("-"*80 + "\n")
            
            for name in self.target_particles.keys():
                target = self.target_particles[name]['mass']
                mass = self.best_results.get(f'{name}_mass', 0)
                error_pct = abs(mass - target) / target * 100 if target > 0 else 0
                f.write(f"{name:<10} {mass:<12.3f} {target:<12.3f} {error_pct:<10.3f}\n")
    
    def print_detailed_report(self):
        """Вывод детального отчета"""
        print(f"\n{'='*80}")
        print("ДЕТАЛЬНЫЙ ОТЧЕТ v6.5")
        print("="*80)
        
        print(f"\nФИКСИРОВАННЫЕ ПАРАМЕТРЫ u и d:")
        print(f"  base_mass_u: {self.fixed_params['base_mass_u']:.6f}")
        print(f"  base_mass_d: {self.fixed_params['base_mass_d']:.6f}")
        print(f"  freq_u: {self.fixed_params['freq_u']:.6f}, freq_d: {self.fixed_params['freq_d']:.6f}")
        print(f"  amp_u: {self.fixed_params['amp_u']:.6f}, amp_d: {self.fixed_params['amp_d']:.6f}")
        
        print(f"\nОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value:.6f}")
        
        print(f"\nКЛЮЧЕВЫЕ РЕЗУЛЬТАТЫ:")
        key_particles = ['proton', 'neutron', 'pi+', 'k+', 'lambda0']
        for name in key_particles:
            target = self.target_particles[name]['mass']
            mass = self.best_results.get(f'{name}_mass', 0)
            error_pct = abs(mass - target) / target * 100
            print(f"  {name}: {mass:.3f} МэВ (цель {target:.3f}) - ошибка {error_pct:.3f}%")
        
        print(f"\nФИЗИЧЕСКИЕ ПАРАМЕТРЫ КВАРКОВ:")
        print(f"  u-кварк: {self.best_results.get('m_u_eff_mev', 0):.2f} МэВ")
        print(f"  d-кварк: {self.best_results.get('m_d_eff_mev', 0):.2f} МэВ")
        print(f"  s-кварк: {self.best_results.get('m_s_eff_mev', 0):.2f} МэВ")
        print(f"  Отношение m_d/m_u: {self.best_results.get('mass_ratio_d_u', 0):.3f}")
        print(f"  Отношение m_s/m_u: {self.best_results.get('mass_ratio_s_u', 0):.3f}")
        
        print(f"\nCoupling ПАРАМЕТРЫ:")
        print(f"  coupling_proton: {self.fixed_params['coupling_proton']:.3f} (фиксировано)")
        print(f"  coupling_neutron: {self.best_params['coupling_neutron']:.3f}")
        print(f"  coupling_lambda0: {self.best_params['coupling_lambda0']:.3f}")
        print(f"  Отношение neutron/proton: {self.best_params['coupling_neutron']/self.fixed_params['coupling_proton']:.3f}")
        
        print(f"\nРезультаты сохранены в: {self.result_dir}")
        print("="*80)
    
    def json_serializer(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

# ================= ЗАПУСК =================
if __name__ == "__main__":
    print("="*80)
    print("МОДЕЛЬ v6.5 - СТРОГАЯ ФИЗИЧЕСКАЯ СОГЛАСОВАННОСТЬ")
    print("Фиксация параметров u и d из v6.1")
    print("Настройка только s-кварка и coupling параметров")
    print("="*80)
    
    print("\nФИКСИРОВАННЫЕ ПАРАМЕТРЫ ИЗ v6.1:")
    print(f"  u-кварк: base_mass=2.203806, freq=0.956359, amp=1.032476")
    print(f"  d-кварк: base_mass=4.583020, freq=0.868115, amp=0.877773")
    print(f"  coupling_proton: 1.613565")
    print(f"  coupling_meson_light: 4.273121")
    
    print("\nНАСТРАИВАЕМЫЕ ПАРАМЕТРЫ:")
    print(f"  s-кварк: base_mass, freq, amp")
    print(f"  coupling_neutron, coupling_lambda0, coupling_meson_strange")
    
    print("\nЦЕЛЕВЫЕ ФИЗИЧЕСКИЕ ОГРАНИЧЕНИЯ:")
    print(f"  1. m_s/m_u = 20-30")
    print(f"  2. Нейтрон тяжелее протона на 1.293 МэВ")
    print(f"  3. coupling_neutron < coupling_proton")
    print(f"  4. Масса Λ⁰ = 1115.7 ± 15 МэВ")
    print("="*80)
    
    optimizer = StrictAnnealingOptimizerV65()
    
    try:
        best_params, best_error, best_results = optimizer.run_optimization(
            iterations=1000000,
            initial_temp=3.0,
            cooling_rate=0.999995,
            save_interval=100000
        )
        
    except KeyboardInterrupt:
        print("\n\nОптимизация прервана.")
        if optimizer.best_params:
            print(f"\nЛучшие параметры (ошибка: {optimizer.best_error:.1f}):")
            for key, value in optimizer.best_params.items():
                print(f"  {key}: {value:.6f}")
    
    print("\n" + "="*80)
    print("ЭКСПЕРИМЕНТ ЗАВЕРШЕН")
    print("="*80)