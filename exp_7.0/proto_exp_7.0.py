"""
МОДЕЛЬ v7.0 - ЭКСПОНЕНЦИАЛЬНАЯ ПАРАДИГМА
Единая мультипликативная модель синхронизации
"""

import numpy as np
import time
import json
from datetime import datetime
import os
import sys

class ParticleModelV70:
    def __init__(self, params, particle_name, composition):
        self.params = params
        self.particle_name = particle_name
        self.composition = composition
        self.is_meson = len(composition) == 2
        self.scale = 100.0  # Масштабный коэффициент
        
        # Базовые параметры кварков
        self.base_mass_u = params.get('base_mass_u', 2.0)
        self.base_mass_d = params.get('base_mass_d', 3.0)
        self.base_mass_s = params.get('base_mass_s', 60.0)  # Значительно увеличено
        
        # Частоты (внутренние коды колебаний)
        self.freq_u = params.get('freq_u', 1.0)
        self.freq_d = params.get('freq_d', 1.0)
        self.freq_s = params.get('freq_s', 0.5)
        
        # Амплитуды (энергия колебаний)
        self.amp_u = params.get('amp_u', 1.0)
        self.amp_d = params.get('amp_d', 1.0)
        self.amp_s = params.get('amp_s', 0.8)
        
        # Параметры синхронизации
        self.coupling_meson = params.get('coupling_meson', -2.0)  # Отрицательный для мезонов
        self.coupling_baryon = params.get('coupling_baryon', 0.1)   # Положительный для барионов
        
        # Фазовый сдвиг для мезонов
        self.phase_shift = params.get('phase_shift', np.pi)
        
        # Коэффициенты для coherence
        self.coherence_weights = params.get('coherence_weights', [0.6, 0.4])  # [частотная, фазовая]
        
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
            return 1.0, 1.0, 1.0
    
    def calculate_base_mass(self):
        """Расчет базовой массы (без синхронизации)"""
        total = 0.0
        for quark in self.composition:
            base_mass, freq, amp = self.get_quark_params(quark)
            total += base_mass * freq * amp
        return total
    
    def calculate_coherence(self):
        """Расчет меры когерентности колебаний (0-1)"""
        if self.is_meson:
            # Для мезонов: два кварка
            phases = [0, self.phase_shift]
            freqs = []
            for quark in self.composition:
                _, freq, _ = self.get_quark_params(quark)
                freqs.append(freq)
        else:
            # Для барионов: три кварка
            if self.particle_name == 'proton':
                phases = [0, 0, np.pi/2]  # u, u, d
            elif self.particle_name == 'neutron':
                phases = [0, np.pi/2, np.pi/2]  # u, d, d
            elif self.particle_name == 'lambda0':
                phases = [0, np.pi/2, np.pi]  # u, d, s
            else:
                phases = [0, np.pi/4, np.pi/2]  # по умолчанию
            
            freqs = []
            for quark in self.composition:
                _, freq, _ = self.get_quark_params(quark)
                freqs.append(freq)
        
        # 1. Частотная когерентность (близость частот)
        if len(freqs) > 1:
            freq_std = np.std(freqs)
            freq_coherence = 1.0 / (1.0 + freq_std)  # 1 при одинаковых частотах
        else:
            freq_coherence = 1.0
        
        # 2. Фазовая когерентность
        if len(phases) >= 2:
            phase_coherence_sum = 0.0
            pair_count = 0
            
            for i in range(len(phases)):
                for j in range(i+1, len(phases)):
                    phase_diff = abs(phases[i] - phases[j]) % (2*np.pi)
                    phase_diff = min(phase_diff, 2*np.pi - phase_diff)
                    
                    if self.is_meson:
                        # Для мезонов: идеальная синхронизация при разности π
                        phase_coherence_sum += np.cos(phase_diff + np.pi)
                    else:
                        # Для барионов: идеальная синхронизация при разности 0 или 2π/3
                        phase_coherence_sum += np.cos(phase_diff)
                    
                    pair_count += 1
            
            phase_coherence = (phase_coherence_sum / pair_count + 1.0) / 2.0
        else:
            phase_coherence = 0.5
        
        # 3. Итоговая когерентность
        total_coherence = (self.coherence_weights[0] * freq_coherence + 
                          self.coherence_weights[1] * phase_coherence)
        
        # Ограничиваем диапазон
        return np.clip(total_coherence, 0.0, 1.0)
    
    def calculate_sync_factor(self):
        """Расчет фактора синхронизации (экспоненциальный)"""
        coherence = self.calculate_coherence()
        
        if self.is_meson:
            # Для мезонов: exp(отрицательный × когерентность) < 1
            sync_factor = np.exp(self.coupling_meson * coherence)
        else:
            # Для барионов: exp(положительный × когерентность) > 1
            sync_factor = np.exp(self.coupling_baryon * coherence)
        
        return sync_factor
    
    def calculate_mass(self):
        """Расчет массы частицы"""
        base_mass = self.calculate_base_mass()
        sync_factor = self.calculate_sync_factor()
        
        return base_mass * sync_factor * self.scale
    
    def calculate_charge(self):
        """Расчет заряда частицы"""
        charges = {
            'u': 2/3, 'd': -1/3, 's': -1/3,
            'anti_u': -2/3, 'anti_d': 1/3, 'anti_s': 1/3
        }
        
        total = 0.0
        for quark in self.composition:
            total += charges.get(quark, 0.0)
        
        return round(total, 10)

class ExponentialAnnealingOptimizerV70:
    def __init__(self):
        # Целевые частицы и их экспериментальные массы
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
        
        # Начальные параметры (основаны на физических соображениях)
        self.current_params = {
            # Базовые массы кварков (в условных единицах)
            'base_mass_u': 2.0,
            'base_mass_d': 3.0,
            'base_mass_s': 60.0,  # s-кварк значительно тяжелее
            
            # Частоты колебаний (близки к 1)
            'freq_u': 1.0,
            'freq_d': 1.0,
            'freq_s': 0.5,  # s-кварк имеет другую частоту
            
            # Амплитуды (энергия колебаний)
            'amp_u': 1.0,
            'amp_d': 1.0,
            'amp_s': 0.8,
            
            # Параметры синхронизации
            'coupling_meson': -2.0,   # Отрицательный для мезонов
            'coupling_baryon': 0.1,    # Положительный для барионов
            
            # Фазовый сдвиг для мезонов
            'phase_shift': np.pi,
            
            # Веса для когерентности
            'coherence_weights': [0.6, 0.4]  # [частотная, фазовая]
        }
        
        # Диапазоны параметров (широкие)
        self.param_ranges = {
            'base_mass_u': (1.0, 4.0),
            'base_mass_d': (2.0, 6.0),
            'base_mass_s': (30.0, 150.0),  # Очень широкий для s-кварка
            
            'freq_u': (0.5, 1.5),
            'freq_d': (0.5, 1.5),
            'freq_s': (0.2, 1.0),
            
            'amp_u': (0.5, 1.5),
            'amp_d': (0.5, 1.5),
            'amp_s': (0.3, 1.2),
            
            'coupling_meson': (-5.0, -0.1),    # Обязательно отрицательный
            'coupling_baryon': (0.01, 2.0),    # Обязательно положительный
            
            'phase_shift': (2.5, 3.5),  # Около π
            
            'coherence_weights': [(0.3, 0.7), (0.3, 0.7)]  # Оба в диапазоне 0.3-0.7
        }
        
        self.best_params = None
        self.best_error = float('inf')
        self.best_results = None
        self.history = []
        
        # Создаем директорию для результатов
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = f"exponential_model_v70_{timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)
    
    def evaluate_params(self, params):
        """Оценка параметров для всех частиц"""
        results = {}
        
        # Создаем модели для всех частиц
        models = {}
        for name, target in self.target_particles.items():
            models[name] = ParticleModelV70(params, name, target['composition'])
        
        # Рассчитываем массы и заряды
        for name, model in models.items():
            results[f'{name}_mass'] = model.calculate_mass()
            results[f'{name}_charge'] = model.calculate_charge()
            
            # Также сохраняем базовую массу и фактор синхронизации
            results[f'{name}_base_mass'] = model.calculate_base_mass() * model.scale
            results[f'{name}_sync_factor'] = model.calculate_sync_factor()
            results[f'{name}_coherence'] = model.calculate_coherence()
        
        # Рассчитываем эффективные массы кварков (в МэВ)
        m_u_eff = params['base_mass_u'] * params['freq_u'] * params['amp_u'] * 100
        m_d_eff = params['base_mass_d'] * params['freq_d'] * params['amp_d'] * 100
        m_s_eff = params['base_mass_s'] * params['freq_s'] * params['amp_s'] * 100
        
        results['m_u_eff_mev'] = m_u_eff
        results['m_d_eff_mev'] = m_d_eff
        results['m_s_eff_mev'] = m_s_eff
        results['mass_ratio_d_u'] = m_d_eff / m_u_eff if m_u_eff > 0 else 1.0
        results['mass_ratio_s_u'] = m_s_eff / m_u_eff if m_u_eff > 0 else 1.0
        
        # Ключевые факторы синхронизации
        for name in ['proton', 'neutron', 'pi+', 'k+', 'lambda0']:
            results[f'sync_factor_{name}'] = results.get(f'{name}_sync_factor', 1.0)
            results[f'coherence_{name}'] = results.get(f'{name}_coherence', 0.5)
        
        return results
    
    def calculate_total_error(self, params):
        """Умная функция ошибки с учетом особенностей экспоненциальной модели"""
        results = self.evaluate_params(params)
        total_error = 0.0
        
        # 1. ОШИБКИ МАСС (основной вклад)
        for name, target in self.target_particles.items():
            target_mass = target['mass']
            calculated_mass = results[f'{name}_mass']
            
            # Относительная ошибка
            rel_error = abs(calculated_mass - target_mass) / target_mass
            
            # Веса в зависимости от важности частицы
            if name in ['proton', 'neutron']:
                weight = 50.0  # Высокий вес для нуклонов
            elif name in ['k+', 'k0', 'lambda0']:
                weight = 30.0  # Средний вес для странных частиц
            elif name in ['pi+', 'pi0', 'pi-']:
                weight = 20.0  # Легкие мезоны
            else:
                weight = 10.0
            
            # Квадратичная ошибка
            mass_error = weight * rel_error ** 2
            total_error += mass_error
            
            # Дополнительный штраф за большие отклонения
            if rel_error > 0.2:  # >20%
                total_error += weight * 5.0 * (rel_error - 0.2)
        
        # 2. ОШИБКИ ЗАРЯДОВ (очень строго)
        for name, target in self.target_particles.items():
            target_charge = target['charge']
            calculated_charge = results[f'{name}_charge']
            if abs(calculated_charge - target_charge) > 0.001:
                total_error += 1000.0  # Большой штраф
        
        # 3. ФИЗИЧЕСКИЕ ОГРАНИЧЕНИЯ
        
        # a) coupling_meson должен быть отрицательным
        if params['coupling_meson'] >= 0:
            total_error += 1000.0
        
        # b) coupling_baryon должен быть положительным
        if params['coupling_baryon'] <= 0:
            total_error += 1000.0
        
        # c) Нейтрон должен быть тяжелее протона
        n_mass = results.get('neutron_mass', 0)
        p_mass = results.get('proton_mass', 0)
        if n_mass < p_mass:
            total_error += 500.0 * (p_mass - n_mass)
        
        # d) Отношение m_s/m_u должно быть 20-40
        mass_ratio_s_u = results.get('mass_ratio_s_u', 1.0)
        if mass_ratio_s_u < 15 or mass_ratio_s_u > 45:
            penalty = abs(mass_ratio_s_u - 30) * 10.0
            total_error += penalty
        
        # e) Факторы синхронизации должны быть разумными
        sync_pi = results.get('sync_factor_pi+', 1.0)
        sync_proton = results.get('sync_factor_proton', 1.0)
        
        # Для мезонов sync_factor < 1 (уменьшение массы)
        if sync_pi >= 1.0:
            total_error += 100.0 * (sync_pi - 0.9)
        
        # Для барионов sync_factor может быть >1 или <1 в зависимости от базы
        # Но проверим: если база протона > целевой массы, то sync_factor должен быть <1
        proton_base = results.get('proton_base_mass', 0)
        if proton_base > 938.272 and sync_proton >= 1.0:
            total_error += 100.0 * sync_proton
        
        # f) Штраф за отрицательные массы
        for name in self.target_particles.keys():
            if results.get(f'{name}_mass', 0) <= 0:
                total_error += 10000.0
        
        # g) Когерентность должна быть в разумных пределах (0.2-0.95)
        for name in ['proton', 'neutron', 'pi+', 'k+']:
            coherence = results.get(f'coherence_{name}', 0.5)
            if coherence < 0.1 or coherence > 0.99:
                penalty = max(0.1 - coherence, coherence - 0.99, 0) * 100.0
                total_error += penalty
        
        return total_error, results
    
    def mutate_params(self, params, temperature, iteration, max_iterations):
        """Адаптивная мутация параметров"""
        new_params = params.copy()
        
        # Определяем фазу оптимизации
        phase = iteration / max_iterations
        
        for key in params.keys():
            if key in self.param_ranges:
                min_val, max_val = self.param_ranges[key]
                current = params[key]
                
                # Разные стратегии для разных типов параметров
                if key == 'coherence_weights':
                    # Для весов нужна особая обработка
                    weight_min, weight_max = self.param_ranges[key][0]
                    new_weights = []
                    for w in current:
                        if phase < 0.5:
                            mutation = np.random.normal(0, 0.1) * temperature
                        else:
                            mutation = np.random.normal(0, 0.02) * temperature
                        
                        new_w = w + mutation
                        new_w = max(weight_min, min(weight_max, new_w))
                        new_weights.append(new_w)
                    
                    # Нормализуем, чтобы сумма была 1.0
                    total = sum(new_weights)
                    if total > 0:
                        new_weights = [w/total for w in new_weights]
                    new_params[key] = new_weights
                    
                else:
                    # Для обычных параметров
                    if phase < 0.3:
                        # Фаза 1: широкий поиск
                        step = (max_val - min_val) * 0.15
                    elif phase < 0.7:
                        # Фаза 2: средний поиск
                        step = (max_val - min_val) * 0.05
                    else:
                        # Фаза 3: тонкая настройка
                        step = (max_val - min_val) * 0.01
                    
                    # Гауссова мутация с температурой
                    mutation = np.random.normal(0, step) * temperature
                    
                    # Для coupling параметров - направленная мутация
                    if key == 'coupling_meson' and current > -0.5:
                        # Притяжение к отрицательным значениям
                        mutation -= 0.1 * (current + 0.5)
                    
                    if key == 'coupling_baryon' and current < 0.5:
                        # Притяжение к положительным значениям
                        mutation += 0.1 * (0.5 - current)
                    
                    if key == 'base_mass_s':
                        # Притяжение к значениям, дающим m_s/m_u ≈ 30
                        if 'results' in locals() and iteration > 100000:
                            current_ratio = results.get('mass_ratio_s_u', 1.0)
                            if current_ratio < 25:
                                mutation += 0.1 * (25 - current_ratio)
                            elif current_ratio > 35:
                                mutation -= 0.1 * (current_ratio - 35)
                    
                    new_val = current + mutation
                    
                    # Ограничение диапазона с отражением
                    while new_val < min_val or new_val > max_val:
                        if new_val < min_val:
                            new_val = 2 * min_val - new_val
                        if new_val > max_val:
                            new_val = 2 * max_val - new_val
                    
                    new_params[key] = new_val
        
        return new_params
    
    def run_optimization(self, iterations=2000000, initial_temp=8.0,
                        cooling_rate=0.999995, save_interval=100000):
        """Запуск оптимизации методом отжига"""
        print("="*80)
        print("ЭКСПОНЕНЦИАЛЬНАЯ МОДЕЛЬ v7.0 - ОПТИМИЗАЦИЯ")
        print("Мультипликативная парадигма синхронизации")
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
            try:
                # Генерация нового решения
                new_params = self.mutate_params(current_params, temperature, i, iterations)
                
                # Оценка нового решения
                new_error, new_results = self.calculate_total_error(new_params)
                
                # Критерий принятия
                delta_error = new_error - current_error
                
                if delta_error < 0:
                    # Улучшение - всегда принимаем
                    current_params = new_params
                    current_error = new_error
                    current_results = new_results
                    stats['accepts'] += 1
                    stats['improves'] += 1
                else:
                    # Ухудшение - принимаем с вероятностью exp(-ΔE/T)
                    probability = np.exp(-delta_error / temperature)
                    if np.random.random() < probability:
                        current_params = new_params
                        current_error = new_error
                        current_results = new_results
                        stats['accepts'] += 1
                    else:
                        stats['rejects'] += 1
                
                # Обновление лучшего решения
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
                    
                    # Ключевые метрики
                    proton_err = abs(self.best_results.get('proton_mass', 0) - 938.272) / 938.272 * 100
                    neutron_err = abs(self.best_results.get('neutron_mass', 0) - 939.565) / 939.565 * 100
                    pi_err = abs(self.best_results.get('pi+_mass', 0) - 139.570) / 139.570 * 100
                    k_err = abs(self.best_results.get('k+_mass', 0) - 493.677) / 493.677 * 100
                    
                    ratio_s_u = self.best_results.get('mass_ratio_s_u', 0)
                    
                    print(f"\rИтерация {i:,}/{iterations:,} ({progress:.1f}%) | "
                          f"Ошибка: {self.best_error:.1f} | "
                          f"p:{proton_err:.1f}% n:{neutron_err:.1f}% π:{pi_err:.1f}% K:{k_err:.1f}% | "
                          f"m_s/m_u: {ratio_s_u:.1f} | "
                          f"Темп: {temperature:.3f}", end='', flush=True)
                
                # Сохранение контрольной точки
                if i % save_interval == 0 and i > 0:
                    self.save_checkpoint(i)
                    
            except Exception as e:
                # В случае ошибки продолжаем со старыми параметрами
                continue
        
        # Финальные результаты
        elapsed = time.time() - start_time
        print(f"\n\n{'='*80}")
        print("ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
        print(f"Время: {elapsed:.1f} сек, Итераций: {iterations:,}")
        print(f"Лучшая ошибка: {self.best_error:.3f}")
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
        """Сохранение финальных результатов"""
        results = {
            'model_version': 'v7.0_exponential',
            'optimization_info': {
                'best_error': self.best_error,
                'total_iterations': len(self.history),
                'timestamp': datetime.now().isoformat()
            },
            'parameters': self.best_params,
            'results': self.best_results,
            'history_summary': self.history[-100:] if self.history else []
        }
        
        with open(f"{self.result_dir}/final_results_v70.json", 'w') as f:
            json.dump(results, f, indent=2, default=self.json_serializer)
        
        self.save_text_report()
    
    def save_text_report(self):
        """Сохранение текстового отчета"""
        filename = f"{self.result_dir}/REPORT_v70.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ЭКСПОНЕНЦИАЛЬНАЯ МОДЕЛЬ v7.0 - ФИНАЛЬНЫЙ ОТЧЕТ\n")
            f.write("="*80 + "\n\n")
            
            f.write("ФИЗИЧЕСКИЕ ПРИНЦИПЫ МОДЕЛИ:\n")
            f.write("  Масса = База × exp(Связь × Когерентность) × 100\n")
            f.write("  - Для мезонов: Связь < 0 (уменьшение массы)\n")
            f.write("  - Для барионов: Связь > 0 (увеличение массы)\n\n")
            
            f.write("ПАРАМЕТРЫ МОДЕЛИ:\n")
            for key, value in self.best_params.items():
                if key == 'coherence_weights':
                    f.write(f"  {key}: [{value[0]:.4f}, {value[1]:.4f}]\n")
                else:
                    f.write(f"  {key}: {value:.6f}\n")
            
            f.write("\nМАССЫ ЧАСТИЦ:\n")
            f.write(f"{'Частица':<10} {'Масса (МэВ)':<12} {'Цель':<12} {'Ошибка (%)':<10} {'База':<12} {'Фактор':<10}\n")
            f.write("-"*80 + "\n")
            
            key_particles = ['proton', 'neutron', 'pi+', 'k+', 'k0', 'lambda0']
            for name in key_particles:
                target = self.target_particles[name]['mass']
                mass = self.best_results.get(f'{name}_mass', 0)
                base = self.best_results.get(f'{name}_base_mass', 0)
                factor = self.best_results.get(f'{name}_sync_factor', 1)
                error_pct = abs(mass - target) / target * 100 if target > 0 else 0
                f.write(f"{name:<10} {mass:<12.3f} {target:<12.3f} {error_pct:<10.3f} {base:<12.3f} {factor:<10.3f}\n")
    
    def print_detailed_report(self):
        """Вывод детального отчета в консоль"""
        print(f"\n{'='*80}")
        print("ЭКСПОНЕНЦИАЛЬНАЯ МОДЕЛЬ v7.0 - ДЕТАЛЬНЫЙ ОТЧЕТ")
        print("="*80)
        
        print(f"\nПАРАМЕТРЫ СИНХРОНИЗАЦИИ:")
        print(f"  coupling_meson: {self.best_params.get('coupling_meson', 0):.4f} (должен быть < 0)")
        print(f"  coupling_baryon: {self.best_params.get('coupling_baryon', 0):.4f} (должен быть > 0)")
        print(f"  phase_shift: {self.best_params.get('phase_shift', 0):.4f} (≈π = {np.pi:.4f})")
        
        print(f"\nЭФФЕКТИВНЫЕ МАССЫ КВАРКОВ (МэВ):")
        print(f"  u: {self.best_results.get('m_u_eff_mev', 0):.2f}")
        print(f"  d: {self.best_results.get('m_d_eff_mev', 0):.2f}")
        print(f"  s: {self.best_results.get('m_s_eff_mev', 0):.2f}")
        print(f"  m_d/m_u: {self.best_results.get('mass_ratio_d_u', 0):.3f}")
        print(f"  m_s/m_u: {self.best_results.get('mass_ratio_s_u', 0):.3f}")
        
        print(f"\nКЛЮЧЕВЫЕ ЧАСТИЦЫ (ошибки в %):")
        key_particles = ['proton', 'neutron', 'pi+', 'k+', 'lambda0']
        for name in key_particles:
            target = self.target_particles[name]['mass']
            mass = self.best_results.get(f'{name}_mass', 0)
            error_pct = abs(mass - target) / target * 100
            coherence = self.best_results.get(f'coherence_{name}', 0)
            sync_factor = self.best_results.get(f'sync_factor_{name}', 0)
            print(f"  {name}: {mass:.3f} МэВ (цель {target:.3f}) - {error_pct:.3f}%")
            print(f"     Когерентность: {coherence:.4f}, Фактор: {sync_factor:.4f}")
        
        print(f"\nПРОВЕРКА ФИЗИЧЕСКИХ ОГРАНИЧЕНИЙ:")
        n_mass = self.best_results.get('neutron_mass', 0)
        p_mass = self.best_results.get('proton_mass', 0)
        print(f"  Нейтрон тяжелее протона: {n_mass > p_mass} (разность: {n_mass-p_mass:.3f} МэВ)")
        
        c_meson = self.best_params.get('coupling_meson', 0)
        c_baryon = self.best_params.get('coupling_baryon', 0)
        print(f"  coupling_meson < 0: {c_meson < 0} ({c_meson:.4f})")
        print(f"  coupling_baryon > 0: {c_baryon > 0} ({c_baryon:.4f})")
        
        mass_ratio = self.best_results.get('mass_ratio_s_u', 0)
        print(f"  m_s/m_u в диапазоне 20-40: {20 <= mass_ratio <= 40} ({mass_ratio:.2f})")
        
        print(f"\nРезультаты сохранены в: {self.result_dir}")
        print("="*80)
    
    def json_serializer(self, obj):
        """Сериализатор для JSON"""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [float(x) if isinstance(x, (np.integer, np.floating)) else x for x in obj]
        return str(obj)

# ================= ЗАПУСК ОПТИМИЗАЦИИ =================
if __name__ == "__main__":
    print("="*80)
    print("ЭКСПОНЕНЦИАЛЬНАЯ МОДЕЛЬ v7.0 - ЗАПУСК")
    print("="*80)
    
    print("\nОСОБЕННОСТИ НОВОЙ МОДЕЛИ:")
    print("  1. Мультипликативная парадигма: M = base × exp(coupling × coherence) × 100")
    print("  2. coupling_meson < 0 (уменьшает массу мезонов)")
    print("  3. coupling_baryon > 0 (увеличивает массу барионов)")
    print("  4. Единая формула для всех частиц")
    
    print("\nЦЕЛЕВЫЕ ДИАПАЗОНЫ ПАРАМЕТРОВ:")
    print(f"  base_mass_s: 30-150 (для m_s/m_u ≈ 25-35)")
    print(f"  coupling_meson: -5.0 до -0.1 (обязательно отрицательный)")
    print(f"  coupling_baryon: 0.01 до 2.0 (обязательно положительный)")
    
    print("\nОЖИДАЕМЫЕ РЕЗУЛЬТАТЫ:")
    print("  1. Все coupling параметры с правильным знаком")
    print("  2. m_s/m_u ≈ 25-35")
    print("  3. Нуклоны: ошибка < 1%")
    print("  4. Странные частицы: ошибка < 5%")
    print("="*80)
    
    # Создаем и запускаем оптимизатор
    optimizer = ExponentialAnnealingOptimizerV70()
    
    try:
        print("\nНачинаю оптимизацию...")
        best_params, best_error, best_results = optimizer.run_optimization(
            iterations=2000000,      # 2 миллиона итераций
            initial_temp=8.0,       # Начальная температура
            cooling_rate=0.999995,  # Скорость охлаждения
            save_interval=100000    # Сохранять каждые 100к итераций
        )
        
    except KeyboardInterrupt:
        print("\n\nОптимизация прервана пользователем.")
        if optimizer.best_params:
            print(f"\nЛучшие найденные параметры (ошибка: {optimizer.best_error:.1f}):")
            for key, value in optimizer.best_params.items():
                if key == 'coherence_weights':
                    print(f"  {key}: [{value[0]:.4f}, {value[1]:.4f}]")
                else:
                    print(f"  {key}: {value:.6f}")
    
    except Exception as e:
        print(f"\nКритическая ошибка: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n" + "="*80)
        print("РАБОТА ЗАВЕРШЕНА")
        print("="*80)