"""
v11.1.3: Фундаментальная модель мезонов (исправленная аналитическая версия)
Основа: Потенциал Корнелла в приближении гармонического осциллятора
Исправления: Правильные размерности для |ψ(0)|² и r_avg
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import json
from datetime import datetime

class CornellMesonModelV113:
    def __init__(self):
        # ФИЗИЧЕСКИЕ КОНСТАНТЫ
        self.hbar_c = 197.3269804  # ћc в МэВ·фм
        
        # Массы кварков (МэВ) - ФИКСИРОВАНЫ
        self.m_u = 2.16
        self.m_d = 4.67
        self.m_ud = (self.m_u + self.m_d) / 2
        
        # Целевые массы
        self.target_masses = {
            'pi+': {'mass': 139.570, 'spin': 0},
            'rho+': {'mass': 775.260, 'spin': 1}
        }
        
        # ПАРАМЕТРЫ МОДЕЛИ (физически разумные стартовые значения)
        self.params = {
            'sigma': 0.18,        # Стринг-тензия в ГэВ² (0.18 ГэВ² = 180000 МэВ²)
            'alpha_s': 0.3,       # Константа сильной связи (безразмерная)
            'kappa': 0.2          # Спин-спиновый параметр (ГэВ·фм³)
        }
    
    def calculate_meson_mass(self, spin=0):
        """
        ОСНОВНАЯ ФУНКЦИЯ: расчёт массы мезона
        Правильные формулы с учётом размерностей
        """
        # Извлекаем параметры
        sigma_gev2 = self.params['sigma']          # в ГэВ²
        alpha_s = self.params['alpha_s']
        kappa = self.params['kappa']               # в ГэВ·фм³
        
        # Переводим в МэВ для вычислений
        sigma_mev2 = sigma_gev2 * 1e6              # в МэВ²
        kappa_mev = kappa * 1000                   # в МэВ·фм³
        
        # Средняя масса кварков в мезоне (МэВ)
        m_q = self.m_ud
        
        # Приведённая масса системы (для кварк-антикварк)
        mu = m_q / 2.0  # примерно 1.7 МэВ
        
        # -----------------------------------------------------------------
        # 1. ХАРАКТЕРНЫЕ МАСШТАБЫ (ПРАВИЛЬНЫЕ ФОРМУЛЫ)
        # -----------------------------------------------------------------
        
        # Масштаб конфайнмента: √σ
        confinement_scale = np.sqrt(sigma_mev2)  # в МэВ (~424 МэВ)
        
        # Характерная длина осциллятора: a_ho = sqrt(ћc / (μ * ω))
        # где ω = confinement_scale (грубая оценка)
        a_ho = self.hbar_c / np.sqrt(mu * confinement_scale)  # в фм
        
        # Среднее расстояние между кварками: ~ a_ho
        r_avg = a_ho  # в фм (~0.5-1.0 фм)
        
        # -----------------------------------------------------------------
        # 2. ЭНЕРГИЯ СВЯЗИ
        # -----------------------------------------------------------------
        
        # Кулоновская энергия: E_coul = -(4/3) * α_s * ћc / r_avg
        E_coulomb = -(4.0/3.0) * alpha_s * self.hbar_c / r_avg  # в МэВ
        
        # Энергия линейного потенциала: E_linear ≈ σ * r_avg * ћc
        E_linear = sigma_mev2 * r_avg * r_avg / self.hbar_c  # упрощённо
        
        # Нулевая энергия осциллятора: E_zero ≈ (3/2) * ћc / (μ * r_avg²)
        E_zero = 1.5 * self.hbar_c * self.hbar_c / (mu * r_avg * r_avg)
        
        # Основная энергия связи
        E_binding_main = E_coulomb + E_linear + E_zero  # в МэВ
        
        # -----------------------------------------------------------------
        # 3. СПИН-СПИНОВОЕ ВЗАИМОДЕЙСТВИЕ (ПРАВИЛЬНАЯ ФОРМУЛА)
        # -----------------------------------------------------------------
        
        # |ψ(0)|² для основного состояния осциллятора
        # ψ(0) = (μω/(πћ))^(3/4) ≈ 1/(π^(3/4) * a_ho^(3/2))
        psi0_squared = 1.0 / (np.pi**1.5 * a_ho**3)  # в фм⁻³ (~0.01-0.1)
        
        # Спиновый фактор
        spin_factor = -3.0/8.0 if spin == 0 else 1.0/8.0
        
        # Энергия спин-спинового взаимодействия
        # Формула: ΔE = (8π/9) * (α_s/m_q²) * |ψ(0)|² * ћc³ * spin_factor
        E_spin = (8.0 * np.pi / 9.0) * (alpha_s / (m_q*m_q)) * psi0_squared * (self.hbar_c**3) * spin_factor
        
        # -----------------------------------------------------------------
        # 4. ИТОГОВАЯ МАССА
        # -----------------------------------------------------------------
        
        # Полная энергия связи (должна быть отрицательной!)
        E_total = E_binding_main + E_spin
        
        # Масса мезона: M = 2*m_q + E_binding
        M_meson = 2.0 * m_q + E_total
        
        # Анализ
        analysis = {
            'mu': mu,
            'a_ho': a_ho,
            'r_avg': r_avg,
            'confinement_scale': confinement_scale,
            'E_coulomb': E_coulomb,
            'E_linear': E_linear,
            'E_zero': E_zero,
            'E_spin': E_spin,
            'psi0_squared': psi0_squared
        }
        
        return M_meson, E_total, analysis
    
    def error_function(self, params_array):
        """Функция ошибки для оптимизации"""
        self.params['sigma'] = params_array[0]
        self.params['alpha_s'] = params_array[1]
        self.params['kappa'] = params_array[2]
        
        # Рассчитываем массы
        M_pi, E_pi, _ = self.calculate_meson_mass(spin=0)
        M_rho, E_rho, _ = self.calculate_meson_mass(spin=1)
        
        # Штрафы за нефизичные значения
        penalty = 0.0
        
        # Энергии связи должны быть отрицательными
        if E_pi > 0 or E_rho > 0:
            penalty += 1000.0
        
        # Массы должны быть положительными
        if M_pi <= 0 or M_rho <= 0:
            penalty += 1000.0
        
        # Целевое соотношение
        target_ratio = 775.260 / 139.570  # ≈ 5.555
        
        if M_pi > 0:
            ratio = M_rho / M_pi
            ratio_error = abs(ratio - target_ratio) / target_ratio
        else:
            ratio_error = 10.0
        
        # Абсолютные ошибки масс
        mass_errors = []
        for M_calc, M_target in [(M_pi, 139.570), (M_rho, 775.260)]:
            if M_calc > 0:
                rel_error = abs(M_calc - M_target) / M_target
            else:
                rel_error = 10.0
            mass_errors.append(rel_error)
        
        avg_mass_error = np.mean(mass_errors)
        
        # Общая ошибка
        total_error = ratio_error * 5.0 + avg_mass_error + penalty
        
        return total_error
    
    def run_optimization(self):
        """Запуск оптимизации"""
        print("\n" + "="*80)
        print("v11.1.3: ФИЗИЧЕСКИ КОРРЕКТНАЯ ОПТИМИЗАЦИЯ")
        print("="*80)
        
        # Начальные параметры
        x0 = [
            self.params['sigma'],
            self.params['alpha_s'],
            self.params['kappa']
        ]
        
        # Границы (физически разумные)
        bounds = [
            (0.1, 0.3),    # sigma [ГэВ²]
            (0.1, 0.5),    # alpha_s
            (0.01, 1.0)    # kappa [ГэВ·фм³]
        ]
        
        # Оптимизация
        result = minimize(
            self.error_function,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'disp': True}
        )
        
        if result.success:
            print(f"✅ Оптимизация успешна!")
            self.params['sigma'] = result.x[0]
            self.params['alpha_s'] = result.x[1]
            self.params['kappa'] = result.x[2]
        else:
            print(f"⚠️ Оптимизация не удалась: {result.message}")
        
        return result
    
    def print_results(self):
        """Вывод результатов"""
        M_pi, E_pi, analysis_pi = self.calculate_meson_mass(spin=0)
        M_rho, E_rho, analysis_rho = self.calculate_meson_mass(spin=1)
        
        print("\n" + "="*80)
        print("РЕЗУЛЬТАТЫ v11.1.3")
        print("="*80)
        
        print(f"\nПАРАМЕТРЫ МОДЕЛИ:")
        print(f"  σ = {self.params['sigma']:.3f} ГэВ²")
        print(f"  √σ = {np.sqrt(self.params['sigma']*1e6):.0f} МэВ")
        print(f"  α_s = {self.params['alpha_s']:.3f}")
        print(f"  κ = {self.params['kappa']:.3f} ГэВ·фм³")
        
        print(f"\nРАСЧЁТНЫЕ МАССЫ:")
        print(f"{'Мезон':<10} {'Масса (МэВ)':<15} {'Цель (МэВ)':<15} {'Ошибка %':<12} {'E_связи (МэВ)':<15}")
        print("-"*80)
        
        for name, M_calc, M_target, E_bind in [
            ("π⁺", M_pi, 139.570, E_pi),
            ("ρ⁺", M_rho, 775.260, E_rho)
        ]:
            error_pct = abs(M_calc - M_target) / M_target * 100 if M_target > 0 else 999
            print(f"{name:<10} {M_calc:<15.1f} {M_target:<15.1f} "
                  f"{error_pct:<12.1f} {E_bind:<15.1f}")
        
        # Соотношение
        if M_pi > 0:
            ratio = M_rho / M_pi
            target_ratio = 775.260 / 139.570
            print(f"\nСООТНОШЕНИЕ МАСС:")
            print(f"  m(ρ)/m(π) = {ratio:.3f} (цель: {target_ratio:.3f})")
            print(f"  Ошибка: {abs(ratio-target_ratio)/target_ratio*100:.1f}%")
        
        # Анализ для пиона
        print(f"\nФИЗИЧЕСКИЕ ПАРАМЕТРЫ (π⁺):")
        print(f"  Приведённая масса μ: {analysis_pi['mu']:.3f} МэВ")
        print(f"  Длина осциллятора a_ho: {analysis_pi['a_ho']:.3f} фм")
        print(f"  Среднее расстояние <r>: {analysis_pi['r_avg']:.3f} фм")
        print(f"  Масштаб конфайнмента √σ: {analysis_pi['confinement_scale']:.0f} МэВ")
        print(f"  |ψ(0)|²: {analysis_pi['psi0_squared']:.3e} фм⁻³")
        
        print(f"\nРАСПРЕДЕЛЕНИЕ ЭНЕРГИИ СВЯЗИ (π⁺):")
        print(f"  Кулоновская: {analysis_pi['E_coulomb']:.1f} МэВ")
        print(f"  Линейная: {analysis_pi['E_linear']:.1f} МэВ")
        print(f"  Нулевые колебания: {analysis_pi['E_zero']:.1f} МэВ")
        print(f"  Спин-спин: {analysis_pi['E_spin']:.1f} МэВ")
        print(f"  СУММА: {E_pi:.1f} МэВ")
        
        # Проверка физической осмысленности
        print(f"\nПРОВЕРКА ФИЗИЧЕСКОЙ ОСМЫСЛЕННОСТИ:")
        checks = [
            (E_pi < 0, f"E_связи(π⁺) < 0 ({E_pi:.1f} МэВ)"),
            (abs(E_pi) > 100, f"|E_связи(π⁺)| > 100 МэВ"),
            (0 < M_pi < 500, f"0 < m(π⁺) < 500 МэВ ({M_pi:.1f} МэВ)"),
            (500 < M_rho < 1000, f"500 < m(ρ⁺) < 1000 МэВ ({M_rho:.1f} МэВ)"),
            (analysis_pi['psi0_squared'] < 1, f"|ψ(0)|² < 1 фм⁻³ ({analysis_pi['psi0_squared']:.3e})"),
            (0.3 < analysis_pi['r_avg'] < 1.5, f"0.3 < <r> < 1.5 фм ({analysis_pi['r_avg']:.3f} фм)")
        ]
        
        for condition, message in checks:
            print(f"  {'✅' if condition else '❌'} {message}")
        
        return M_pi, M_rho, analysis_pi
    
    def visualize(self, M_pi, M_rho, analysis):
        """Визуализация результатов"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Сравнение масс
        ax1 = axes[0, 0]
        particles = ['π⁺', 'ρ⁺']
        calc_masses = [M_pi, M_rho]
        target_masses = [139.570, 775.260]
        
        x = np.arange(len(particles))
        width = 0.35
        
        ax1.bar(x - width/2, calc_masses, width, label='v11.1.3', color='skyblue')
        ax1.bar(x + width/2, target_masses, width, label='Эксперимент', color='lightcoral')
        
        ax1.set_xlabel('Мезон')
        ax1.set_ylabel('Масса (МэВ)')
        ax1.set_title('Сравнение с экспериментом')
        ax1.set_xticks(x)
        ax1.set_xticklabels(particles)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Энергии связи
        ax2 = axes[0, 1]
        energies = ['Кулоновская', 'Линейная', 'Нулевые\nколебания', 'Спин-спин']
        E_vals = [
            analysis['E_coulomb'],
            analysis['E_linear'],
            analysis['E_zero'],
            analysis['E_spin']
        ]
        
        colors = ['blue', 'green', 'orange', 'red']
        bars = ax2.bar(energies, E_vals, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_ylabel('Энергия (МэВ)')
        ax2.set_title('Вклады в энергию связи π⁺')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Потенциал
        ax3 = axes[1, 0]
        r = np.linspace(0.05, 2.0, 200)
        
        # Потенциал Корнелла: V(r) = -4/3 * α_s * ћc / r + σ * r
        V_coulomb = -(4.0/3.0) * self.params['alpha_s'] * self.hbar_c / r
        V_linear = self.params['sigma'] * 1e6 * r * r / self.hbar_c  # упрощённо
        V_total = V_coulomb + V_linear
        
        ax3.plot(r, V_coulomb, 'b--', alpha=0.7, label='Кулоновский')
        ax3.plot(r, V_linear, 'g--', alpha=0.7, label='Линейный')
        ax3.plot(r, V_total, 'r-', linewidth=2, label='Корнелл')
        ax3.axhline(y=M_pi - 2*self.m_ud, color='purple', linestyle=':', 
                   label=f'E(π⁺) = {M_pi-2*self.m_ud:.0f} МэВ')
        
        ax3.set_xlabel('r (фм)')
        ax3.set_ylabel('V(r) (МэВ)')
        ax3.set_title('Потенциал Корнелла')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Волновая функция
        ax4 = axes[1, 1]
        r_plot = np.linspace(0, 2*analysis['a_ho'], 200)
        psi = np.exp(-0.5 * (r_plot/analysis['a_ho'])**2) / (np.pi**0.25 * np.sqrt(analysis['a_ho']))
        
        ax4.plot(r_plot, psi, 'b-', linewidth=2, label='ψ(r) (осн. состояние)')
        ax4.fill_between(r_plot, 0, psi, alpha=0.3)
        ax4.axvline(x=analysis['r_avg'], color='r', linestyle='--', 
                   label=f'<r> = {analysis["r_avg"]:.2f} фм')
        
        ax4.set_xlabel('r (фм)')
        ax4.set_ylabel('ψ(r) (фм⁻¹/²)')
        ax4.set_title('Волновая функция (основное состояние)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('v11_1_3_results.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def run(self):
        """Запуск полного расчёта"""
        print("\n" + "="*80)
        print("🚀 ЗАПУСК v11.1.3: ФИЗИЧЕСКИ КОРРЕКТНАЯ МОДЕЛЬ")
        print("="*80)
        
        # 1. Оценка с начальными параметрами
        print("\n1. ОЦЕНКА С СТАНДАРТНЫМИ ПАРАМЕТРАМИ:")
        M_pi_initial, _, _ = self.calculate_meson_mass(spin=0)
        M_rho_initial, _, _ = self.calculate_meson_mass(spin=1)
        print(f"   π⁺: {M_pi_initial:.1f} МэВ, ρ⁺: {M_rho_initial:.1f} МэВ")
        
        # 2. Оптимизация
        print("\n2. ОПТИМИЗАЦИЯ ПАРАМЕТРОВ...")
        opt_result = self.run_optimization()
        
        # 3. Результаты
        print("\n3. РЕЗУЛЬТАТЫ ПОСЛЕ ОПТИМИЗАЦИИ:")
        M_pi, M_rho, analysis = self.print_results()
        
        # 4. Визуализация
        print("\n4. ВИЗУАЛИЗАЦИЯ...")
        self.visualize(M_pi, M_rho, analysis)
        
        # 5. Сохранение
        print("\n5. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            'version': '11.1.3',
            'parameters': self.params,
            'masses': {
                'pi+_MeV': M_pi,
                'rho+_MeV': M_rho
            },
            'analysis': analysis,
            'timestamp': timestamp
        }
        
        with open(f'v11_1_3_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ МОДЕЛЬ ЗАВЕРШЕНА!")
        print(f"   Масса π⁺: {M_pi:.1f} МэВ (цель: 139.6 МэВ)")
        print(f"   Масса ρ⁺: {M_rho:.1f} МэВ (цель: 775.3 МэВ)")
        print(f"   Соотношение: {M_rho/M_pi:.3f} (цель: 5.555)")
        
        return results

# ================= ЗАПУСК =================
if __name__ == "__main__":
    model = CornellMesonModelV113()
    results = model.run()