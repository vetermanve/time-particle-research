"""
V11.0: Физически обоснованная модель легких адронов
Основана на реалистичном потенциале КХД
"""

import numpy as np
from scipy.optimize import minimize
import json
from datetime import datetime

class QCDRealisticModelV11:
    """
    Реалистичная модель адронов с минимальным числом параметров
    Основана на КХД-подобном потенциале
    """
    
    def __init__(self):
        # ФИКСИРОВАННЫЕ параметры (из экспериментов)
        self.m_u = 2.16  # МэВ (u-кварк)
        self.m_d = 4.67  # МэВ (d-кварк)
        
        # Целевые массы (МэВ) - только u/d адроны
        self.targets = {
            # Мезоны (спин 0)
            'pi+': {'mass': 139.57039, 'quarks': ['u', 'anti_d'], 'spin': 0},
            'pi0': {'mass': 134.9768,  'quarks': ['mix'], 'spin': 0},
            'pi-': {'mass': 139.57039, 'quarks': ['d', 'anti_u'], 'spin': 0},
            
            # Мезоны (спин 1)  
            'rho+': {'mass': 775.26, 'quarks': ['u', 'anti_d'], 'spin': 1},
            'rho0': {'mass': 775.26, 'quarks': ['mix'], 'spin': 1},
            'rho-': {'mass': 775.26, 'quarks': ['d', 'anti_u'], 'spin': 1},
            'omega': {'mass': 782.65, 'quarks': ['mix'], 'spin': 1},
            
            # Барионы (спин 1/2)
            'proton': {'mass': 938.2720813, 'quarks': ['u', 'u', 'd'], 'spin': 0.5},
            'neutron': {'mass': 939.5654133, 'quarks': ['u', 'd', 'd'], 'spin': 0.5},
            
            # Барионы (спин 3/2)
            'delta++': {'mass': 1232.0, 'quarks': ['u', 'u', 'u'], 'spin': 1.5},
            'delta+':  {'mass': 1232.0, 'quarks': ['u', 'u', 'd'], 'spin': 1.5},
            'delta0':  {'mass': 1232.0, 'quarks': ['u', 'd', 'd'], 'spin': 1.5},
            'delta-':  {'mass': 1232.0, 'quarks': ['d', 'd', 'd'], 'spin': 1.5},
        }
        
        # ВСЕГО 4 свободных параметра!
        # 1. α_s - константа сильной связи
        # 2. σ - параметр линейного потенциала (стринг-тензия)
        # 3. κ - спин-спиновое взаимодействие
        # 4. δ - поправка на изоспин (разность u-d)
        
        self.params = {
            'alpha_s': 0.3,      # Константа сильной связи (безразмерная)
            'sigma': 0.18,       # Стринг-тензия (ГэВ²) ~ (0.18 ГэВ)²
            'kappa': 0.05,       # Спин-спиновое взаимодействие
            'delta': 0.001,      # Изоспиновая поправка
        }
    
    def quark_mass(self, flavor):
        """Масса кварка с поправкой на изоспин"""
        if flavor == 'u':
            return self.m_u * (1 + self.params['delta'])
        elif flavor == 'd':
            return self.m_d * (1 - self.params['delta'])
        elif flavor == 'anti_u':
            return self.m_u * (1 + self.params['delta'])
        elif flavor == 'anti_d':
            return self.m_d * (1 - self.params['delta'])
        else:
            return 0.0
    
    def calculate_meson_mass(self, q1, q2, spin):
        """
        Масса мезона по формуле КХД
        M = m1 + m2 + V_potential + V_spin
        """
        m1 = self.quark_mass(q1)
        m2 = self.quark_mass(q2)
        
        # Сумма масс кварков (пренебрежимо мала)
        mass_sum = m1 + m2
        
        # Приведенная масса
        mu = (m1 * m2) / (m1 + m2) if (m1 + m2) > 0 else 0
        
        # Кулоновская часть (цветовой потенциал)
        # V_coul = -4/3 * α_s / r, оценка для основного состояния
        # Используем боровскую формулу: E = -μ * (α_s)^2 / 2
        V_coulomb = -0.5 * mu * (self.params['alpha_s'] ** 2) * 1000  # в МэВ
        
        # Линейный потенциал (конфайнмент)
        # V_linear = σ * r, оценка из соотношения: r ~ 1/√σ
        # Для основного состояния: <V_linear> ~ σ^(2/3) * μ^(-1/3)
        sigma_gev = self.params['sigma']  # в ГэВ²
        sigma_mev = sigma_gev * 1e6  # в МэВ²
        V_linear = (sigma_mev ** (2/3)) * (mu ** (-1/3))
        
        # Спин-спиновое взаимодействие
        # V_spin = (8π/3) * (α_s/m1*m2) * S1·S2 * δ(r)
        # Для мезонов: S1·S2 = [s(s+1) - 3/2]/4
        s = spin
        spin_factor = (s * (s + 1) - 1.5) / 4.0
        
        # Упрощенная формула
        V_spin = self.params['kappa'] * spin_factor * 1000 / (m1 * m2)
        
        # Итоговая масса
        total = mass_sum + V_coulomb + V_linear + V_spin
        
        return total
    
    def calculate_baryon_mass(self, quarks, spin):
        """
        Масса бариона по модели КХД
        Упрощенный подход: учитываем парные взаимодействия
        """
        # Сумма масс кварков
        mass_sum = sum(self.quark_mass(q) for q in quarks)
        
        # Приближение: барион как система трех взаимодействующих кварков
        # Используем модель гармонического осциллятора
        
        # Эффективная масса системы
        # Для симметричной конфигурации
        if len(set(quarks)) == 1:  # Все кварки одинаковы (Δ⁺⁺, Δ⁻)
            m_eff = self.quark_mass(quarks[0])
        else:
            # Средняя масса
            m_eff = mass_sum / 3.0
        
        # Энергия конфайнмента (основной вклад!)
        # В барионах энергия связи примерно 99% массы
        sigma_mev = self.params['sigma'] * 1e6  # в МэВ²
        
        # Масштаб энергии: √σ ~ 400-500 МэВ
        V_confinement = 3 * np.sqrt(sigma_mev)  # три струны
        
        # Спин-спиновое взаимодействие
        s = spin
        # Для барионов: сумма попарных спин-спиновых взаимодействий
        V_spin = self.params['kappa'] * s * (s + 1) * 100
        
        # Кулоновская поправка (мала)
        V_coulomb = -self.params['alpha_s'] * m_eff * 10
        
        total = mass_sum + V_confinement + V_spin + V_coulomb
        
        return total
    
    def calculate_all_masses(self):
        """Рассчитать массы всех частиц"""
        results = {}
        
        for name, target in self.targets.items():
            quarks = target['quarks']
            spin = target['spin']
            
            if len(quarks) == 2:  # Мезон
                if 'mix' in quarks:
                    # Для нейтральных частиц берем среднее
                    mass_plus = self.calculate_meson_mass('u', 'anti_d', spin)
                    mass_minus = self.calculate_meson_mass('d', 'anti_u', spin)
                    mass = (mass_plus + mass_minus) / 2
                else:
                    mass = self.calculate_meson_mass(quarks[0], quarks[1], spin)
            else:  # Барион
                mass = self.calculate_baryon_mass(quarks, spin)
            
            results[name] = mass
        
        return results
    
    def error_function(self, params_array):
        """Функция ошибки для оптимизации"""
        # Обновляем параметры
        self.params['alpha_s'] = params_array[0]
        self.params['sigma'] = params_array[1]
        self.params['kappa'] = params_array[2]
        self.params['delta'] = params_array[3]
        
        # Рассчитываем массы
        calculated = self.calculate_all_masses()
        
        # Суммарная ошибка
        total_error = 0
        
        for name, target in self.targets.items():
            target_mass = target['mass']
            calc_mass = calculated[name]
            
            # Относительная ошибка с весом
            rel_error = abs(calc_mass - target_mass) / target_mass
            
            # Критически важные частицы имеют больший вес
            weights = {
                'proton': 3.0,
                'neutron': 3.0,
                'pi+': 2.0,
                'rho+': 2.0,
                'delta++': 1.0,
                'others': 0.5
            }
            
            weight = weights.get(name, weights['others'])
            total_error += rel_error * weight
        
        # Штраф за нефизичные параметры
        if params_array[0] < 0.1 or params_array[0] > 0.5:  # alpha_s
            total_error += 10.0
        if params_array[1] < 0.1 or params_array[1] > 0.3:  # sigma
            total_error += 10.0
        
        return total_error
    
    def optimize(self):
        """Оптимизация параметров"""
        print("\n" + "="*80)
        print("ЭКСПЕРИМЕНТ v11.0: Оптимизация реалистичной модели")
        print("="*80)
        
        # Начальные параметры (физически разумные)
        initial_params = [
            0.3,   # alpha_s
            0.18,  # sigma (ГэВ²)
            0.05,  # kappa
            0.001  # delta
        ]
        
        # Границы параметров
        bounds = [
            (0.2, 0.4),    # alpha_s
            (0.15, 0.22),  # sigma
            (0.01, 0.1),   # kappa
            (-0.01, 0.01)  # delta
        ]
        
        # Оптимизация
        result = minimize(
            self.error_function,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
        
        # Обновляем параметры
        self.params['alpha_s'] = result.x[0]
        self.params['sigma'] = result.x[1]
        self.params['kappa'] = result.x[2]
        self.params['delta'] = result.x[3]
        
        return result
    
    def print_results(self):
        """Вывод результатов"""
        calculated = self.calculate_all_masses()
        
        print("\n" + "="*80)
        print("РЕЗУЛЬТАТЫ v11.0")
        print("="*80)
        
        print(f"\nПАРАМЕТРЫ МОДЕЛИ (4 параметра):")
        print(f"  α_s (константа сильной связи): {self.params['alpha_s']:.6f}")
        print(f"  σ (стринг-тензия, ГэВ²): {self.params['sigma']:.6f}")
        print(f"  κ (спин-спиновое): {self.params['kappa']:.6f}")
        print(f"  δ (изоспиновая поправка): {self.params['delta']:.6f}")
        
        print(f"\nМАССЫ ЧАСТИЦ:")
        print(f"{'Частица':<12} {'Рассч. (МэВ)':<15} {'Эксп. (МэВ)':<15} {'Ошибка (%)':<12}")
        print("-"*80)
        
        categories = {
            'Мезоны (спин 0)': ['pi+', 'pi0', 'pi-'],
            'Мезоны (спин 1)': ['rho+', 'rho0', 'rho-', 'omega'],
            'Барионы (спин 1/2)': ['proton', 'neutron'],
            'Барионы (спин 3/2)': ['delta++', 'delta+', 'delta0', 'delta-']
        }
        
        for category, particles in categories.items():
            print(f"\n{category}:")
            for name in particles:
                if name in calculated:
                    calc = calculated[name]
                    target = self.targets[name]['mass']
                    error_pct = abs(calc - target) / target * 100
                    print(f"  {name:<10} {calc:<15.3f} {target:<15.3f} {error_pct:<12.3f}")
        
        # Важные соотношения
        print(f"\nКЛЮЧЕВЫЕ СООТНОШЕНИЯ:")
        
        # Разность масс нейтрон-протон
        m_n = calculated['neutron']
        m_p = calculated['proton']
        diff = m_n - m_p
        target_diff = 1.293332
        print(f"  Δm(n-p): {diff:.3f} МэВ (цель: {target_diff:.3f}, ошибка: {abs(diff-target_diff):.3f} МэВ)")
        
        # Отношение масс ρ/π
        m_rho = calculated['rho+']
        m_pi = calculated['pi+']
        print(f"  m(ρ)/m(π): {m_rho/m_pi:.3f} (цель: {775.26/139.57:.3f})")
        
        # Отношение масс Δ/p
        m_delta = calculated['delta++']
        print(f"  m(Δ)/m(p): {m_delta/m_p:.3f} (цель: {1232/938:.3f})")
        
        # Эффективные параметры
        print(f"\nЭФФЕКТИВНЫЕ ПАРАМЕТРЫ:")
        sigma_mev = self.params['sigma'] * 1000  # в МэВ
        confinement_scale = np.sqrt(sigma_mev)
        print(f"  Масштаб конфайнмента: {confinement_scale:.1f} МэВ")
        print(f"  Длина струны (оценка): {1/confinement_scale:.3f} фм")
        
        return calculated
    
    def save_results(self):
        """Сохранение результатов"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"v11_results_{timestamp}.json"
        
        results = {
            'model': 'v11.0_realistic_qcd',
            'timestamp': timestamp,
            'parameters': self.params,
            'quark_masses': {'m_u': self.m_u, 'm_d': self.m_d},
            'calculated_masses': self.calculate_all_masses(),
            'target_masses': {k: v['mass'] for k, v in self.targets.items()}
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=self._json_serializer)
        
        print(f"\nРезультаты сохранены в {filename}")
        return filename
    
    def _json_serializer(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)


# ================= ЗАПУСК ЭКСПЕРИМЕНТА =================
if __name__ == "__main__":
    print("🚀 ЗАПУСК ЭКСПЕРИМЕНТА v11.0")
    print("Реалистичная модель КХД для легких адронов")
    print("Только u/d кварки, 4 свободных параметра")
    
    model = QCDRealisticModelV11()
    
    # Оптимизация
    result = model.optimize()
    
    # Вывод результатов
    if result.success:
        print(f"\n✅ Оптимизация успешна!")
        print(f"Итераций: {result.nit}")
        print(f"Функция ошибки: {result.fun:.6f}")
        
        calculated = model.print_results()
        model.save_results()
        
        # Анализ точности
        errors = []
        for name in model.targets:
            target = model.targets[name]['mass']
            calc = calculated[name]
            rel_error = abs(calc - target) / target
            errors.append(rel_error)
        
        print(f"\n📊 СТАТИСТИКА ТОЧНОСТИ:")
        print(f"Средняя ошибка: {np.mean(errors)*100:.2f}%")
        print(f"Максимальная ошибка: {np.max(errors)*100:.2f}%")
        print(f"Стандартное отклонение: {np.std(errors)*100:.2f}%")
        
        print(f"\n🎯 КРИТЕРИИ УСПЕХА:")
        print(f"1. Протон < 5% ошибки: {'✅' if abs(calculated['proton']-938.27)/938.27 < 0.05 else '❌'}")
        print(f"2. π⁺ < 10% ошибки: {'✅' if abs(calculated['pi+']-139.57)/139.57 < 0.1 else '❌'}")
        print(f"3. ρ⁺ < 20% ошибки: {'✅' if abs(calculated['rho+']-775.26)/775.26 < 0.2 else '❌'}")
        print(f"4. Δ⁺⁺ < 30% ошибки: {'✅' if abs(calculated['delta++']-1232)/1232 < 0.3 else '❌'}")
        
    else:
        print(f"\n❌ Оптимизация не удалась: {result.message}")