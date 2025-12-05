"""
v11.1.2: Фундаментальная модель мезонов (аналитическое приближение)
Основа: Потенциал Корнелла в приближении гармонического осциллятора
Цель: Получить физически корректные масштабы масс и соотношение m(ρ)/m(π)
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import json
from datetime import datetime

class CornellMesonModelV112:
    """
    Упрощённая аналитическая модель для оценки масс мезонов
    Использует приближение гармонического осциллятора для потенциала Корнелла
    """
    
    def __init__(self):
        # ----------------------------
        # ФИЗИЧЕСКИЕ КОНСТАНТЫ
        # ----------------------------
        self.hbar_c = 197.3269804  # ћc в МэВ·фм
        
        # Текущие массы кварков (МэВ) - ФИКСИРОВАНЫ
        self.m_u = 2.16    # u-кварк
        self.m_d = 4.67    # d-кварк
        self.m_ud = (self.m_u + self.m_d) / 2  # средняя масса для u/d мезонов
        
        # Целевые массы (МэВ) - ТОЛЬКО u/d мезоны
        self.target_masses = {
            'pi+': {'mass': 139.570, 'spin': 0, 'quarks': ['u', 'anti_d']},
            'rho+': {'mass': 775.260, 'spin': 1, 'quarks': ['u', 'anti_d']},
            'pi0': {'mass': 134.977, 'spin': 0, 'quarks': ['mix']},
            'rho0': {'mass': 775.260, 'spin': 1, 'quarks': ['mix']}
        }
        
        # ----------------------------
        # ПАРАМЕТРЫ МОДЕЛИ (3 свободных параметра)
        # ----------------------------
        # Начальные значения из физических соображений
        self.params = {
            'sigma_linear': 950.0,   # Стринг-тензия в МэВ/фм (√σ ≈ 440 МэВ)
            'a': 130.0,              # Кулоновский параметр в МэВ·фм
            'kappa_ss': 40000.0      # Константа спин-спинового взаимодействия в МэВ·фм³
        }
        
        # Диапазоны для оптимизации (физически разумные)
        self.param_bounds = {
            'sigma_linear': (800.0, 1200.0),   # Соответствует √σ от 400 до 500 МэВ
            'a': (100.0, 200.0),              # α_s от 0.5 до 1.0
            'kappa_ss': (20000.0, 60000.0)    # Из литературы
        }
        
        # Для хранения результатов
        self.best_params = None
        self.best_results = None
        self.optimization_history = []
        
    def quark_mass(self, flavor):
        """Масса кварка с учётом античастицы"""
        if flavor in ['u', 'anti_u']:
            return self.m_u
        elif flavor in ['d', 'anti_d']:
            return self.m_d
        else:
            return self.m_ud
    
    def calculate_meson_properties(self, quark1, quark2, spin, params=None):
        """
        Основная функция: расчёт свойств мезона в приближении гармонического осциллятора
        
        Возвращает: массу, энергию связи, |ψ(0)|²
        """
        if params is None:
            params = self.params
        
        sigma = params['sigma_linear']  # МэВ/фм
        a = params['a']                 # МэВ·фм
        kappa_ss = params['kappa_ss']   # МэВ·фм³
        
        # Массы кварков
        m1 = self.quark_mass(quark1)
        m2 = self.quark_mass(quark2)
        
        # Приведённая масса системы (МэВ)
        mu = (m1 * m2) / (m1 + m2)
        
        # -----------------------------------------------------------------
        # 1. ОСНОВНАЯ ЭНЕРГИЯ СВЯЗИ (гармонический осциллятор + кулон)
        # -----------------------------------------------------------------
        
        # Характерная частота осциллятора из линейного потенциала V(r) = σ·r
        # Для радиального уравнения: ω = sqrt(2σ / μ)
        omega = np.sqrt(2.0 * sigma / mu)  # в МэВ
        
        # Энергия нулевых колебаний 3D гармонического осциллятора
        # E_zero = (3/2)ћω, но ћ=1 в наших единицах (МэВ)
        E_zero_point = 1.5 * omega
        
        # Оценка среднего расстояния между кварками
        # Для основного состояния осциллятора: <r> ≈ 1/√(μω)
        r_avg = 1.0 / np.sqrt(mu * omega)  # в фм
        
        # Кулоновская энергия (отрицательная - притяжение)
        E_coulomb = -a / r_avg
        
        # Основная энергия связи (пока без спина)
        E_binding_main = E_zero_point + E_coulomb
        
        # -----------------------------------------------------------------
        # 2. СПИН-СПИНОВОЕ ВЗАИМОДЕЙСТВИЕ
        # -----------------------------------------------------------------
        
        # |ψ(0)|² - вероятность найти кварки в одной точке
        # Для основного состояния 3D осциллятора: |ψ(0)|² = (μω/π)^(3/2)
        psi0_squared = (mu * omega / np.pi) ** 1.5  # в фм^(-3)
        
        # Спиновый фактор для мезонов
        if spin == 0:  # Псевдоскалярные мезоны (π)
            spin_factor = -3.0 / 8.0  # = -0.375
        else:  # Векторные мезоны (ρ)
            spin_factor = 1.0 / 8.0   # = +0.125
        
        # Энергия спин-спинового взаимодействия
        E_spin = kappa_ss * spin_factor * psi0_squared
        
        # -----------------------------------------------------------------
        # 3. ИТОГОВАЯ ЭНЕРГИЯ СВЯЗИ И МАССА
        # -----------------------------------------------------------------
        
        # Полная энергия связи (должна быть отрицательной!)
        E_total = E_binding_main + E_spin
        
        # Масса мезона: M = m1 + m2 + E_binding
        M_meson = m1 + m2 + E_total
        
        # Дополнительные параметры для анализа
        analysis = {
            'mu': mu,
            'omega': omega,
            'r_avg': r_avg,
            'E_zero': E_zero_point,
            'E_coulomb': E_coulomb,
            'E_spin': E_spin,
            'psi0_squared': psi0_squared
        }
        
        return M_meson, E_total, analysis
    
    def calculate_all_masses(self, params=None):
        """Рассчитать массы всех мезонов"""
        if params is None:
            params = self.params
        
        results = {}
        
        for name, info in self.target_masses.items():
            quarks = info['quarks']
            spin = info['spin']
            
            # Для нейтральных мезонов берём среднее
            if 'mix' in quarks:
                M1, E1, a1 = self.calculate_meson_properties('u', 'anti_d', spin, params)
                M2, E2, a2 = self.calculate_meson_properties('d', 'anti_u', spin, params)
                M_meson = (M1 + M2) / 2
                E_bind = (E1 + E2) / 2
                analysis = {'average_of': [a1, a2]}
            else:
                M_meson, E_bind, analysis = self.calculate_meson_properties(
                    quarks[0], quarks[1], spin, params
                )
            
            results[name] = {
                'mass': M_meson,
                'binding_energy': E_bind,
                'target': info['mass'],
                'analysis': analysis
            }
        
        return results
    
    def error_function(self, params_array):
        """Функция ошибки для оптимизации (фокусируемся на π⁺ и ρ⁺)"""
        # Преобразуем массив в словарь
        params = {
            'sigma_linear': params_array[0],
            'a': params_array[1],
            'kappa_ss': params_array[2]
        }
        
        # Рассчитываем массы
        results = self.calculate_all_masses(params)
        
        # Основная цель: соотношение m(ρ)/m(π) ≈ 5.555
        M_pi = results['pi+']['mass']
        M_rho = results['rho+']['mass']
        
        if M_pi <= 0 or M_rho <= 0:
            return 1e10  # Большая ошибка при нефизичных массах
        
        # Рассчитываем соотношение
        ratio = M_rho / M_pi
        target_ratio = 775.260 / 139.570  # ≈ 5.555
        
        # Ошибка соотношения (главный критерий!)
        ratio_error = abs(ratio - target_ratio) / target_ratio
        
        # Также учитываем абсолютные массы (но с меньшим весом)
        mass_errors = []
        for name in ['pi+', 'rho+']:
            target = self.target_masses[name]['mass']
            calc = results[name]['mass']
            rel_error = abs(calc - target) / target
            mass_errors.append(rel_error)
        
        avg_mass_error = np.mean(mass_errors)
        
        # Штраф за нефизичные энергии связи
        E_pi = results['pi+']['binding_energy']
        E_rho = results['rho+']['binding_energy']
        
        penalty = 0.0
        if E_pi > 0 or E_rho > 0:  # Энергия связи должна быть отрицательной!
            penalty += 1000.0
        
        # Общая ошибка
        total_error = ratio_error * 10.0 + avg_mass_error + penalty
        
        # Сохраняем историю
        self.optimization_history.append({
            'params': params.copy(),
            'ratio': ratio,
            'error': total_error,
            'M_pi': M_pi,
            'M_rho': M_rho
        })
        
        return total_error
    
    def optimize_parameters(self):
        """Оптимизация параметров модели"""
        print("\n" + "="*80)
        print("v11.1.2: Оптимизация параметров (метод Нелдера-Мида)")
        print("Цель: m(ρ)/m(π) ≈ 5.555")
        print("="*80)
        
        # Начальные параметры
        x0 = [
            self.params['sigma_linear'],
            self.params['a'],
            self.params['kappa_ss']
        ]
        
        # Границы
        bounds = [
            self.param_bounds['sigma_linear'],
            self.param_bounds['a'],
            self.param_bounds['kappa_ss']
        ]
        
        # Оптимизация методом Нелдера-Мида
        result = minimize(
            self.error_function,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 500,
                'xatol': 1e-4,
                'fatol': 1e-4,
                'disp': True
            }
        )
        
        if result.success:
            print(f"✅ Оптимизация успешна за {result.nit} итераций")
            
            # Обновляем параметры
            self.params.update({
                'sigma_linear': result.x[0],
                'a': result.x[1],
                'kappa_ss': result.x[2]
            })
            
            self.best_params = self.params.copy()
            self.best_results = self.calculate_all_masses(self.params)
            
        else:
            print(f"⚠️ Оптимизация завершилась: {result.message}")
            print("Используем начальные параметры")
            self.best_params = self.params.copy()
            self.best_results = self.calculate_all_masses(self.params)
        
        return result
    
    def print_detailed_results(self):
        """Подробный вывод результатов"""
        if self.best_results is None:
            self.best_results = self.calculate_all_masses(self.params)
        
        print("\n" + "="*80)
        print("РЕЗУЛЬТАТЫ v11.1.2")
        print("="*80)
        
        print(f"\nОПТИМАЛЬНЫЕ ПАРАМЕТРЫ:")
        print(f"  σ (стринг-тензия): {self.params['sigma_linear']:.1f} МэВ/фм")
        print(f"  √σ (масштаб конфайнмента): {np.sqrt(self.params['sigma_linear']*self.hbar_c):.0f} МэВ")
        print(f"  a (кулоновский параметр): {self.params['a']:.1f} МэВ·фм")
        print(f"  α_eff = a/(ћc): {self.params['a']/self.hbar_c:.3f}")
        print(f"  κ_ss (спин-спин): {self.params['kappa_ss']:.0f} МэВ·фм³")
        
        print(f"\n{'Мезон':<10} {'Масса (МэВ)':<15} {'Эксп. (МэВ)':<15} {'Ошибка %':<12} {'E_связи (МэВ)':<15}")
        print("-"*80)
        
        for name in ['pi+', 'rho+', 'pi0', 'rho0']:
            if name in self.best_results:
                res = self.best_results[name]
                M_calc = res['mass']
                M_target = res['target']
                E_bind = res['binding_energy']
                
                error_pct = abs(M_calc - M_target) / M_target * 100
                
                print(f"{name:<10} {M_calc:<15.1f} {M_target:<15.1f} "
                      f"{error_pct:<12.1f} {E_bind:<15.1f}")
        
        # Ключевое соотношение
        M_pi = self.best_results['pi+']['mass']
        M_rho = self.best_results['rho+']['mass']
        ratio = M_rho / M_pi
        target_ratio = 775.260 / 139.570
        
        print(f"\nКЛЮЧЕВЫЕ СООТНОШЕНИЯ:")
        print(f"  m(ρ⁺)/m(π⁺): {ratio:.3f}")
        print(f"  Целевое: {target_ratio:.3f}")
        print(f"  Ошибка: {abs(ratio-target_ratio)/target_ratio*100:.1f}%")
        
        # Анализ энергий связи
        print(f"\nАНАЛИЗ ЭНЕРГИЙ СВЯЗИ (π⁺):")
        analysis = self.best_results['pi+']['analysis']
        if 'average_of' not in analysis:
            print(f"  Приведённая масса μ: {analysis['mu']:.3f} МэВ")
            print(f"  Частота осциллятора ω: {analysis['omega']:.1f} МэВ")
            print(f"  Среднее расстояние <r>: {analysis['r_avg']:.3f} фм")
            print(f"  Энергия нулевых колебаний: {analysis['E_zero']:.1f} МэВ")
            print(f"  Кулоновская энергия: {analysis['E_coulomb']:.1f} МэВ")
            print(f"  Спин-спин энергия: {analysis['E_spin']:.1f} МэВ")
            print(f"  |ψ(0)|²: {analysis['psi0_squared']:.2e} фм⁻³")
        
        # Проверка физической осмысленности
        print(f"\nФИЗИЧЕСКАЯ ОСМЫСЛЕННОСТЬ:")
        E_pi = self.best_results['pi+']['binding_energy']
        E_rho = self.best_results['rho+']['binding_energy']
        
        checks = [
            (E_pi < 0, f"Энергия связи π⁺ отрицательна ({E_pi:.1f} МэВ)"),
            (E_rho < 0, f"Энергия связи ρ⁺ отрицательна ({E_rho:.1f} МэВ)"),
            (abs(E_pi) > 100, f"Энергия связи π⁺ значительна (>100 МэВ)"),
            (M_pi > 0 and M_pi < 500, f"Масса π⁺ в разумных пределах ({M_pi:.1f} МэВ)"),
            (M_rho > 500 and M_rho < 1000, f"Масса ρ⁺ в разумных пределах ({M_rho:.1f} МэВ)")
        ]
        
        for check, message in checks:
            print(f"  {'✅' if check else '❌'} {message}")
    
    def plot_results(self):
        """Визуализация результатов"""
        if self.best_results is None:
            self.best_results = self.calculate_all_masses(self.params)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Сравнение масс
        ax1 = axes[0, 0]
        names = ['π⁺', 'ρ⁺']
        calc_masses = [self.best_results['pi+']['mass'], 
                      self.best_results['rho+']['mass']]
        target_masses = [self.target_masses['pi+']['mass'],
                        self.target_masses['rho+']['mass']]
        
        x = np.arange(len(names))
        width = 0.35
        
        ax1.bar(x - width/2, calc_masses, width, label='v11.1.2', color='skyblue', alpha=0.8)
        ax1.bar(x + width/2, target_masses, width, label='Эксперимент', color='lightcoral', alpha=0.8)
        
        ax1.set_xlabel('Мезон')
        ax1.set_ylabel('Масса (МэВ)')
        ax1.set_title('Сравнение расчётных и экспериментальных масс')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Добавляем значения
        for i, (calc, target) in enumerate(zip(calc_masses, target_masses)):
            ax1.text(i - width/2, calc + 20, f'{calc:.0f}', ha='center', va='bottom')
            ax1.text(i + width/2, target + 20, f'{target:.0f}', ha='center', va='bottom')
        
        # 2. Соотношение масс
        ax2 = axes[0, 1]
        ratio_calc = calc_masses[1] / calc_masses[0]
        ratio_target = target_masses[1] / target_masses[0]
        
        ax2.bar(['v11.1.2', 'Цель'], [ratio_calc, ratio_target], 
                color=['skyblue', 'lightcoral'], alpha=0.8)
        ax2.set_ylabel('m(ρ⁺)/m(π⁺)')
        ax2.set_title('Соотношение масс')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Потенциал Корнелла
        ax3 = axes[1, 0]
        r = np.linspace(0.05, 2.0, 200)
        sigma = self.params['sigma_linear']
        a = self.params['a']
        
        V_coulomb = -a / r
        V_linear = sigma * r
        V_total = V_coulomb + V_linear
        
        ax3.plot(r, V_coulomb, 'b--', alpha=0.7, label='Кулоновский (-a/r)')
        ax3.plot(r, V_linear, 'g--', alpha=0.7, label='Линейный (σ·r)')
        ax3.plot(r, V_total, 'r-', linewidth=2, label='Корнелл (-a/r + σ·r)')
        
        ax3.set_xlabel('Расстояние r (фм)')
        ax3.set_ylabel('Потенциал V(r) (МэВ)')
        ax3.set_title('Потенциал Корнелла')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Энергии связи
        ax4 = axes[1, 1]
        energies = ['Нулевые\nколебания', 'Кулоновское\nпритяжение', 'Спин-спин', 'Итоговая']
        
        # Для π⁺
        analysis = self.best_results['pi+']['analysis']
        if 'average_of' not in analysis:
            E_vals = [
                analysis['E_zero'],
                analysis['E_coulomb'],
                analysis['E_spin'],
                self.best_results['pi+']['binding_energy']
            ]
            
            colors = ['orange', 'blue', 'green', 'red']
            ax4.bar(energies, E_vals, color=colors, alpha=0.7)
            ax4.set_ylabel('Энергия (МэВ)')
            ax4.set_title('Вклады в энергию связи π⁺')
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Линия нуля
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig('v11_1_2_results.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def save_results(self):
        """Сохранение результатов в файл"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"v11_1_2_results_{timestamp}.json"
        
        results = {
            'model_version': '11.1.2',
            'description': 'Аналитическая модель мезонов на основе приближения гармонического осциллятора для потенциала Корнелла',
            'timestamp': timestamp,
            'physical_constants': {
                'hbar_c_MeV_fm': self.hbar_c,
                'm_u_MeV': self.m_u,
                'm_d_MeV': self.m_d
            },
            'optimized_parameters': self.params,
            'calculated_masses': {},
            'analysis': {}
        }
        
        # Сохраняем массы
        for name, res in self.best_results.items():
            results['calculated_masses'][name] = {
                'mass_MeV': res['mass'],
                'binding_energy_MeV': res['binding_energy'],
                'target_mass_MeV': res['target']
            }
        
        # Сохраняем анализ для π⁺
        if 'pi+' in self.best_results and 'average_of' not in self.best_results['pi+']['analysis']:
            analysis = self.best_results['pi+']['analysis']
            results['analysis']['pion'] = {
                'reduced_mass_MeV': analysis['mu'],
                'oscillator_frequency_MeV': analysis['omega'],
                'average_distance_fm': analysis['r_avg'],
                'zero_point_energy_MeV': analysis['E_zero'],
                'coulomb_energy_MeV': analysis['E_coulomb'],
                'spin_spin_energy_MeV': analysis['E_spin'],
                'psi0_squared_fm-3': analysis['psi0_squared']
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=self._json_serializer)
        
        print(f"\n💾 Результаты сохранены в {filename}")
        return filename
    
    def _json_serializer(self, obj):
        """Сериализатор для JSON"""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    def run_optimization_pipeline(self):
        """Полный пайплайн оптимизации и анализа"""
        print("\n" + "="*80)
        print("🚀 ЗАПУСК v11.1.2: АНАЛИТИЧЕСКАЯ МОДЕЛЬ МЕЗОНОВ")
        print("="*80)
        
        # 1. Быстрая оценка с начальными параметрами
        print("\n1. БЫСТРАЯ ОЦЕНКА (начальные параметры):")
        initial_results = self.calculate_all_masses()
        M_pi_init = initial_results['pi+']['mass']
        M_rho_init = initial_results['rho+']['mass']
        print(f"   π⁺: {M_pi_init:.1f} МэВ, ρ⁺: {M_rho_init:.1f} МэВ")
        print(f"   Соотношение: {M_rho_init/M_pi_init:.3f}")
        
        # 2. Оптимизация параметров
        print("\n2. ОПТИМИЗАЦИЯ ПАРАМЕТРОВ...")
        opt_result = self.optimize_parameters()
        
        # 3. Детальный анализ результатов
        print("\n3. АНАЛИЗ РЕЗУЛЬТАТОВ:")
        self.print_detailed_results()
        
        # 4. Визуализация
        print("\n4. ВИЗУАЛИЗАЦИЯ...")
        self.plot_results()
        
        # 5. Сохранение
        print("\n5. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ...")
        saved_file = self.save_results()
        
        # 6. Итоговая оценка
        print("\n" + "="*80)
        print("ИТОГОВАЯ ОЦЕНКА v11.1.2")
        print("="*80)
        
        # Ключевые метрики
        M_pi = self.best_results['pi+']['mass']
        M_rho = self.best_results['rho+']['mass']
        E_pi = self.best_results['pi+']['binding_energy']
        ratio = M_rho / M_pi
        
        metrics = [
            ("Энергия связи π⁺ отрицательна", E_pi < 0),
            ("|E_связи(π⁺)| > 100 МэВ", abs(E_pi) > 100),
            ("Масса π⁺ < 500 МэВ", M_pi < 500),
            ("Масса ρ⁺ > 500 МэВ", M_rho > 500),
            ("Соотношение m(ρ)/m(π) > 1.5", ratio > 1.5),
            ("Ошибка соотношения < 50%", abs(ratio - 5.555)/5.555 < 0.5)
        ]
        
        passed = sum(1 for _, condition in metrics if condition)
        total = len(metrics)
        
        print(f"\nПройдено критериев: {passed}/{total}")
        for desc, condition in metrics:
            print(f"  {'✅' if condition else '❌'} {desc}")
        
        print(f"\n🎯 Основная цель (m(ρ)/m(π) ≈ 5.555):")
        print(f"   Получено: {ratio:.3f}")
        print(f"   Ошибка: {abs(ratio-5.555)/5.555*100:.1f}%")
        
        print(f"\n📈 Следующий шаг (v11.2):")
        print(f"   • Использовать эти параметры как начальные для численного решения")
        print(f"   • Добавить барионы (протон, нейтрон)")
        print(f"   • Уточнить спин-спиновое взаимодействие")
        
        return self.best_params, self.best_results


# ================= ЗАПУСК =================
if __name__ == "__main__":
    # Создаём и запускаем модель
    model = CornellMesonModelV112()
    
    try:
        # Запускаем полный пайплайн
        best_params, best_results = model.run_optimization_pipeline()
        
        # Краткий итог
        print(f"\n{'='*60}")
        print("v11.1.2 УСПЕШНО ЗАВЕРШЕН!")
        print(f"{'='*60}")
        print(f"Масса π⁺: {best_results['pi+']['mass']:.1f} МэВ")
        print(f"Масса ρ⁺: {best_results['rho+']['mass']:.1f} МэВ")
        print(f"Соотношение: {best_results['rho+']['mass']/best_results['pi+']['mass']:.3f}")
        print(f"Параметры сохранены для v11.2")
        
    except Exception as e:
        print(f"\n❌ Ошибка выполнения: {e}")
        import traceback
        traceback.print_exc()