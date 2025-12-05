import numpy as np
import json
from datetime import datetime
import os

class EnhancedTopologicalModelV16:
    """Улучшенная модель с электромагнитными поправками и смешиванием"""
    
    def __init__(self):
        self.scale = 100.0
        
        # Целевые массы
        self.targets = {
            'proton': 938.272,
            'neutron': 939.565,
            'pi+': 139.570,
            'pi0': 134.9768,
            'pi-': 139.570,
            'K+': 493.677,
            'K0': 497.611,
            'K-': 493.677,
            'eta': 547.862,
            'Lambda0': 1115.683,
        }
        
        # Состав
        self.composition = {
            'proton': ['u', 'u', 'd'],
            'neutron': ['u', 'd', 'd'],
            'pi+': ['u', 'anti_d'],
            'pi0': ['mixed_pi0'],  # uū + dđ смесь
            'pi-': ['d', 'anti_u'],
            'K+': ['u', 'anti_s'],
            'K0': ['d', 'anti_s'],
            'K-': ['s', 'anti_u'],
            'eta': ['mixed_eta'],   # сложное смешивание
            'Lambda0': ['u', 'd', 's'],
        }
        
        # Начальные параметры из v15.0
        self.params = {
            # Эффективные массы
            'm_u': 2.47,
            'm_d': 3.2225,
            'm_s': 8.0164,
            
            # Базовые coupling
            'C_baryon': 1.0,      # Для протона
            'C_meson': 3.2341,    # Для пионов
            
            # Топологические факторы
            'f_neutron': 0.67,    # Нейтрон относительно протона
            'f_kaon': 1.2,        # Для каонов
            'f_eta': 1.1,         # Для η
            'f_lambda': 0.8,      # Для Λ⁰
            
            # Смешивание
            'mix_pi0': 1.1,       # π⁰ mixing (>1 делает π⁰ легче π⁺)
            'mix_eta': 0.9,       # η mixing
            
            # Электромагнитные поправки (в единицах модели)
            'EM_pion': 0.025,     # ~2.5 МэВ для заряженных пионов
            'EM_kaon': 0.02,      # ~2 МэВ для заряженных каонов
            'EM_proton': 0.05,    # ~5 МэВ для протона
        }
    
    def calculate_base_mass(self, particle):
        """Базовая масса"""
        comp = self.composition[particle]
        
        if particle == 'pi0':
            # π⁰ = (uū + dđ)/√2 упрощённо
            return (2*self.params['m_u'] + 2*self.params['m_d']) / 2.0
        
        if particle == 'eta':
            # η = (uū + dđ + sš)/√3, но с весом
            return (2*self.params['m_u'] + 2*self.params['m_d'] + 
                    self.params['m_s']) / 3.0
        
        total = 0
        for quark in comp:
            if quark in ['u', 'anti_u']:
                total += self.params['m_u']
            elif quark in ['d', 'anti_d']:
                total += self.params['m_d']
            elif quark in ['s', 'anti_s']:
                total += self.params['m_s']
        
        return total
    
    def calculate_coupling(self, particle):
        """Вычисление coupling с топологическими факторами"""
        if particle == 'proton':
            return self.params['C_baryon']
        
        if particle == 'neutron':
            return self.params['C_baryon'] * self.params['f_neutron']
        
        if particle in ['pi+', 'pi-']:
            return self.params['C_meson']
        
        if particle == 'pi0':
            return self.params['C_meson'] * self.params['mix_pi0']
        
        if particle in ['K+', 'K-']:
            return self.params['C_meson'] * self.params['f_kaon']
        
        if particle == 'K0':
            return self.params['C_meson'] * self.params['f_kaon']
        
        if particle == 'eta':
            return self.params['C_meson'] * self.params['f_eta'] * self.params['mix_eta']
        
        if particle == 'Lambda0':
            return self.params['C_baryon'] * self.params['f_lambda']
        
        return 1.0
    
    def calculate_em_correction(self, particle):
        """Электромагнитная поправка"""
        if particle == 'proton':
            return self.params['EM_proton'] * self.scale
        
        if particle in ['pi+', 'pi-']:
            return self.params['EM_pion'] * self.scale
        
        if particle in ['K+', 'K-']:
            return self.params['EM_kaon'] * self.scale
        
        return 0.0
    
    def calculate_mass(self, particle):
        """Расчёт массы с EM поправками"""
        base = self.calculate_base_mass(particle)
        coupling = self.calculate_coupling(particle)
        em_corr = self.calculate_em_correction(particle)
        
        # Барионы: +coupling, мезоны: -coupling
        if particle in ['proton', 'neutron', 'Lambda0']:
            mass = (base + coupling) * self.scale + em_corr
        else:
            mass = (base - coupling) * self.scale + em_corr
        
        return mass
    
    def error_function(self, params_array):
        """Функция ошибки с улучшенными штрафами"""
        param_names = [
            'm_u', 'm_d', 'm_s',
            'C_baryon', 'C_meson',
            'f_neutron', 'f_kaon', 'f_eta', 'f_lambda',
            'mix_pi0', 'mix_eta',
            'EM_pion', 'EM_kaon', 'EM_proton'
        ]
        
        for i, name in enumerate(param_names):
            self.params[name] = params_array[i]
        
        total_error = 0.0
        
        # Ошибки масс
        for particle, target in self.targets.items():
            mass = self.calculate_mass(particle)
            
            if mass <= 0:
                total_error += 10000.0
                continue
            
            rel_error = (mass - target) / target
            total_error += rel_error ** 2
        
        # Физические штрафы
        
        # 1. m_s > m_d > m_u
        if self.params['m_s'] <= self.params['m_d']:
            total_error += 50.0
        
        if self.params['m_d'] <= self.params['m_u']:
            total_error += 50.0
        
        # 2. Разность масс n-p ≈ 1.293 МэВ
        mass_n = self.calculate_mass('neutron')
        mass_p = self.calculate_mass('proton')
        diff_np = mass_n - mass_p
        diff_error = abs(diff_np - 1.293) / 1.293
        total_error += 20.0 * diff_error
        
        # 3. Разность масс π⁺-π⁰ ≈ 4.6 МэВ
        mass_pi_plus = self.calculate_mass('pi+')
        mass_pi_zero = self.calculate_mass('pi0')
        diff_pi = mass_pi_plus - mass_pi_zero
        if diff_pi <= 0:  # π⁺ должен быть тяжелее π⁰
            total_error += 100.0
        else:
            diff_error_pi = abs(diff_pi - 4.6) / 4.6
            total_error += 20.0 * diff_error_pi
        
        # 4. K⁰ должен быть тяжелее K⁺ на ~4 МэВ
        mass_K0 = self.calculate_mass('K0')
        mass_Kplus = self.calculate_mass('K+')
        diff_K = mass_K0 - mass_Kplus
        if diff_K <= 0:
            total_error += 50.0
        else:
            diff_error_K = abs(diff_K - 4.0) / 4.0
            total_error += 10.0 * diff_error_K
        
        # 5. coupling нейтрона < coupling протона
        if self.params['f_neutron'] >= 1.0:
            total_error += 100.0
        
        return total_error
    
    def run_optimization(self, iterations=500000):
        """Оптимизация методом случайного поиска с отжигом"""
        print("Запуск оптимизации v16.0...")
        
        # Начальные значения из v15.0
        initial_params = [
            # m_u, m_d, m_s
            2.47, 3.2225, 8.0164,
            # C_baryon, C_meson
            1.0, 3.2341,
            # f_neutron, f_kaon, f_eta, f_lambda
            0.67, 1.2, 1.1, 0.8,
            # mix_pi0, mix_eta
            1.1, 0.9,
            # EM_pion, EM_kaon, EM_proton
            0.025, 0.02, 0.05
        ]
        
        # Границы
        bounds = [
            # m_u, m_d, m_s
            (2.0, 3.0), (2.5, 4.0), (7.0, 12.0),
            # C_baryon, C_meson
            (0.5, 2.0), (2.0, 5.0),
            # f_neutron, f_kaon, f_eta, f_lambda
            (0.1, 0.8), (0.8, 1.5), (0.8, 1.5), (0.5, 1.2),
            # mix_pi0, mix_eta
            (1.05, 1.3), (0.8, 1.2),
            # EM_pion, EM_kaon, EM_proton
            (0.01, 0.05), (0.01, 0.04), (0.02, 0.08)
        ]
        
        best_params = initial_params
        best_error = self.error_function(initial_params)
        
        temperature = 1.0
        cooling_rate = 0.99999
        
        for i in range(iterations):
            # Генерация новых параметров
            new_params = []
            for j, (min_val, max_val) in enumerate(bounds):
                current = best_params[j]
                step = (max_val - min_val) * 0.1 * temperature
                mutation = np.random.uniform(-step, step)
                new_val = current + mutation
                
                # Ограничение границами
                new_val = max(min_val, min(max_val, new_val))
                new_params.append(new_val)
            
            # Оценка
            new_error = self.error_function(new_params)
            
            # Принятие решения
            if new_error < best_error:
                best_error = new_error
                best_params = new_params
            else:
                delta = new_error - best_error
                prob = np.exp(-delta / temperature)
                if np.random.random() < prob:
                    best_params = new_params
            
            # Охлаждение
            temperature *= cooling_rate
            
            if i % 50000 == 0:
                print(f"Iteration {i}: error = {best_error:.6f}, temp = {temperature:.4f}")
        
        return best_params, best_error

def main():
    print("="*80)
    print("УЛУЧШЕННАЯ ТОПОЛОГИЧЕСКАЯ МОДЕЛЬ v16.0")
    print("С ЭЛЕКТРОМАГНИТНЫМИ ПОПРАВКАМИ И СМЕШИВАНИЕМ")
    print("="*80)
    
    model = EnhancedTopologicalModelV16()
    
    # Запускаем оптимизацию
    best_params, best_error = model.run_optimization(iterations=200000)
    
    # Обновляем параметры
    param_names = [
        'm_u', 'm_d', 'm_s',
        'C_baryon', 'C_meson',
        'f_neutron', 'f_kaon', 'f_eta', 'f_lambda',
        'mix_pi0', 'mix_eta',
        'EM_pion', 'EM_kaon', 'EM_proton'
    ]
    
    for i, name in enumerate(param_names):
        model.params[name] = best_params[i]
    
    print(f"\nОптимизация завершена. Лучшая ошибка: {best_error:.6f}")
    
    print("\n" + "="*80)
    print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ v16.0")
    print("="*80)
    
    print("\nПАРАМЕТРЫ МОДЕЛИ:")
    for name, value in model.params.items():
        print(f"  {name}: {value:.6f}")
    
    print(f"\nЭФФЕКТИВНЫЕ МАССЫ КВАРКОВ (МэВ):")
    print(f"  u: {model.params['m_u']*100:.1f} МэВ")
    print(f"  d: {model.params['m_d']*100:.1f} МэВ") 
    print(f"  s: {model.params['m_s']*100:.1f} МэВ")
    print(f"  Отношение m_d/m_u: {model.params['m_d']/model.params['m_u']:.3f}")
    print(f"  Отношение m_s/m_u: {model.params['m_s']/model.params['m_u']:.3f}")
    
    print(f"\nМАССЫ ЧАСТИЦ:")
    print(f"{'Частица':<10} {'Расчёт':<10} {'Цель':<10} {'Ошибка %':<10}")
    print("-"*50)
    
    total_error = 0
    for particle in model.targets:
        mass = model.calculate_mass(particle)
        target = model.targets[particle]
        error_pct = abs(mass - target) / target * 100
        total_error += error_pct
        
        status = "✓" if error_pct < 1.0 else "⚠" if error_pct < 5.0 else "✗"
        print(f"{status} {particle:<8} {mass:<10.3f} {target:<10.3f} {error_pct:<10.3f}")
    
    avg_error = total_error / len(model.targets)
    print(f"\nСредняя ошибка: {avg_error:.2f}%")
    
    # Ключевые физические проверки
    print(f"\nФИЗИЧЕСКИЕ ПРОВЕРКИ:")
    
    # 1. Разность масс n-p
    diff_np = model.calculate_mass('neutron') - model.calculate_mass('proton')
    print(f"  Разность масс n-p: {diff_np:.3f} МэВ (цель 1.293 МэВ)")
    
    # 2. Разность масс π⁺-π⁰
    diff_pi = model.calculate_mass('pi+') - model.calculate_mass('pi0')
    print(f"  Разность масс π⁺-π⁰: {diff_pi:.3f} МэВ (цель 4.6 МэВ)")
    
    # 3. Разность масс K⁰-K⁺
    diff_K = model.calculate_mass('K0') - model.calculate_mass('K+')
    print(f"  Разность масс K⁰-K⁺: {diff_K:.3f} МэВ (ожидается ~4 МэВ)")
    
    # 4. Отношение coupling нейтрон/протон
    ratio_coupling = model.params['f_neutron']
    print(f"  Отношение coupling нейтрон/протон: {ratio_coupling:.3f}")
    
    # Сохранение результатов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"enhanced_model_v16_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    results = {
        'model': 'v16.0_enhanced_topological',
        'timestamp': datetime.now().isoformat(),
        'error': float(best_error),
        'parameters': model.params,
        'masses': {p: float(model.calculate_mass(p)) for p in model.targets},
        'physical_checks': {
            'mass_diff_np': float(diff_np),
            'mass_diff_pi': float(diff_pi),
            'mass_diff_K': float(diff_K),
            'coupling_ratio_np': float(ratio_coupling)
        }
    }
    
    with open(f"{results_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nРезультаты сохранены в: {results_dir}")
    print("="*80)

if __name__ == "__main__":
    main()