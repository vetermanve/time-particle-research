"""
v11.1.4: Квантово-механическая модель мезонов
Основа: Аналитическое приближение для потенциала Корнелла
"""

import numpy as np
from scipy.optimize import minimize

class QMMesonModelV114:
    def __init__(self):
        # Константы
        self.hbar_c = 197.3269804  # МэВ·фм
        
        # Массы кварков (МэВ)
        self.m_u = 2.16
        self.m_d = 4.67
        self.m_ud = (self.m_u + self.m_d) / 2  # ~3.4 МэВ
        
        # Цели
        self.target_pi = 139.570
        self.target_rho = 775.260
        
        # Параметры (будут оптимизированы)
        self.params = {
            'sigma': 0.18,     # ГэВ² (стринг-тензия)
            'alpha_s': 0.3,    # Константа сильной связи
            'kappa': 0.02      # Спин-спиновый параметр (ГэВ·фм³)
        }
    
    def calculate_meson_mass(self, spin=0):
        """Квантово-механическая оценка массы мезона"""
        # Извлекаем параметры
        sigma_gev2 = self.params['sigma']      # ГэВ²
        alpha_s = self.params['alpha_s']
        kappa = self.params['kappa']           # ГэВ·фм³
        
        # Переводим в МэВ
        sigma = sigma_gev2 * 1e6               # МэВ²
        kappa_mev = kappa * 1000               # МэВ·фм³
        
        # Приведённая масса системы кварк-антикварк
        mu = self.m_ud / 2.0                   # ~1.7 МэВ
        
        # ------------------------------------------------------------
        # 1. КУЛОНОВСКАЯ ЧАСТЬ (водородоподобная)
        # E_coul = - (4/3) * α_s² * μ / 2
        E_coulomb = -(4.0/3.0) * alpha_s**2 * mu / 2.0
        
        # Боровский радиус для системы
        a0 = self.hbar_c / (alpha_s * mu)      # фм
        
        # ------------------------------------------------------------
        # 2. ЛИНЕЙНАЯ ЧАСТЬ (гармонический осциллятор)
        # Характерная частота: ω = √(σ/μ)
        omega = np.sqrt(sigma / mu)            # МэВ
        
        # Энергия нулевых колебаний 3D осциллятора
        E_oscillator = 1.5 * omega             # МэВ
        
        # Характерная длина осциллятора
        a_ho = np.sqrt(self.hbar_c / (mu * omega))  # фм
        
        # ------------------------------------------------------------
        # 3. СПИН-СПИНОВАЯ ПОПРАВКА
        # |ψ(0)|² для кулоновской + осцилляторной волновой функции
        # Приближение: ψ(0)² ≈ 1/(π * a_eff^3), где a_eff = min(a0, a_ho)
        a_eff = min(a0, a_ho)
        psi0_squared = 1.0 / (np.pi * a_eff**3)  # фм⁻³
        
        # Спиновый фактор
        spin_factor = -3.0/8.0 if spin == 0 else 1.0/8.0
        
        # Энергия спин-спинового взаимодействия
        # ΔE = (8π/9) * α_s * |ψ(0)|² * (ћc)³ / (m1*m2) * spin_factor
        E_spin = (8.0 * np.pi / 9.0) * alpha_s * psi0_squared
        E_spin *= (self.hbar_c**3) / (self.m_ud**2)
        E_spin *= spin_factor
        
        # ------------------------------------------------------------
        # 4. ИТОГОВАЯ ЭНЕРГИЯ СВЯЗИ И МАССА
        # E_binding = E_coulomb + E_oscillator + E_spin
        E_binding = E_coulomb + E_oscillator + E_spin
        
        # Масса мезона: M = 2*m_q + E_binding
        M_meson = 2.0 * self.m_ud + E_binding
        
        # Анализ
        analysis = {
            'mu': mu,
            'a0': a0,
            'a_ho': a_ho,
            'omega': omega,
            'E_coulomb': E_coulomb,
            'E_oscillator': E_oscillator,
            'E_spin': E_spin,
            'psi0_squared': psi0_squared
        }
        
        return M_meson, E_binding, analysis
    
    def error_function(self, params_array):
        """Функция ошибки"""
        self.params['sigma'] = params_array[0]
        self.params['alpha_s'] = params_array[1]
        self.params['kappa'] = params_array[2]
        
        # Рассчитываем массы
        M_pi, E_pi, _ = self.calculate_meson_mass(spin=0)
        M_rho, E_rho, _ = self.calculate_meson_mass(spin=1)
        
        # Основная цель: соотношение масс
        target_ratio = self.target_rho / self.target_pi
        
        if M_pi > 0:
            ratio = M_rho / M_pi
            ratio_error = abs(ratio - target_ratio) / target_ratio
        else:
            ratio_error = 10.0
        
        # Абсолютные ошибки
        mass_error_pi = abs(M_pi - self.target_pi) / self.target_pi
        mass_error_rho = abs(M_rho - self.target_rho) / self.target_rho
        
        # Штрафы
        penalty = 0.0
        if E_pi > 0 or E_rho > 0:  # Энергии связи должны быть отрицательными
            penalty += 100.0
        if M_pi <= 0 or M_rho <= 0:
            penalty += 100.0
        
        # Общая ошибка
        total_error = ratio_error + (mass_error_pi + mass_error_rho)/2.0 + penalty
        
        return total_error
    
    def run(self):
        """Запуск модели"""
        print("\n" + "="*80)
        print("v11.1.4: КВАНТОВО-МЕХАНИЧЕСКАЯ МОДЕЛЬ МЕЗОНОВ")
        print("="*80)
        
        # Начальные параметры (физически разумные)
        x0 = [0.18, 0.3, 0.02]
        
        # Границы
        bounds = [
            (0.1, 0.3),    # sigma [ГэВ²]
            (0.1, 0.5),    # alpha_s
            (0.001, 0.1)   # kappa [ГэВ·фм³]
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
            print("✅ Оптимизация успешна!")
            self.params['sigma'] = result.x[0]
            self.params['alpha_s'] = result.x[1]
            self.params['kappa'] = result.x[2]
        
        # Результаты
        M_pi, E_pi, analysis_pi = self.calculate_meson_mass(spin=0)
        M_rho, E_rho, analysis_rho = self.calculate_meson_mass(spin=1)
        
        print(f"\n{'='*80}")
        print("РЕЗУЛЬТАТЫ v11.1.4")
        print(f"{'='*80}")
        
        print(f"\nПАРАМЕТРЫ:")
        print(f"  σ = {self.params['sigma']:.3f} ГэВ²")
        print(f"  √σ = {np.sqrt(self.params['sigma']*1e6):.0f} МэВ")
        print(f"  α_s = {self.params['alpha_s']:.3f}")
        print(f"  κ = {self.params['kappa']:.3f} ГэВ·фм³")
        
        print(f"\nМАССЫ:")
        print(f"  π⁺: {M_pi:.1f} МэВ (цель: {self.target_pi:.1f})")
        print(f"  ρ⁺: {M_rho:.1f} МэВ (цель: {self.target_rho:.1f})")
        
        if M_pi > 0:
            ratio = M_rho / M_pi
            print(f"\nСООТНОШЕНИЕ: {ratio:.3f} (цель: {self.target_rho/self.target_pi:.3f})")
        
        print(f"\nАНАЛИЗ ДЛЯ π⁺:")
        print(f"  Приведённая масса μ: {analysis_pi['mu']:.3f} МэВ")
        print(f"  Боровский радиус a0: {analysis_pi['a0']:.2f} фм")
        print(f"  Длина осциллятора a_ho: {analysis_pi['a_ho']:.2f} фм")
        print(f"  |ψ(0)|²: {analysis_pi['psi0_squared']:.3e} фм⁻³")
        print(f"  E_кулон: {analysis_pi['E_coulomb']:.1f} МэВ")
        print(f"  E_осциллятор: {analysis_pi['E_oscillator']:.1f} МэВ")
        print(f"  E_спин-спин: {analysis_pi['E_spin']:.1f} МэВ")
        print(f"  E_сумма: {E_pi:.1f} МэВ")
        
        # Проверка физической осмысленности
        print(f"\nПРОВЕРКА:")
        checks = [
            (E_pi < 0, f"E_связи(π⁺) < 0"),
            (abs(E_pi) > 100, f"|E_связи| > 100 МэВ"),
            (0 < M_pi < 500, f"m(π⁺) в разумных пределах"),
            (analysis_pi['a_ho'] < 2.0, f"a_ho < 2.0 фм (получено {analysis_pi['a_ho']:.2f} фм)"),
            (abs(analysis_pi['E_oscillator']) < 1000, f"E_осциллятор < 1000 МэВ")
        ]
        
        for check, msg in checks:
            print(f"  {'✅' if check else '❌'} {msg}")
        
        return M_pi, M_rho

# Запуск
if __name__ == "__main__":
    model = QMMesonModelV114()
    M_pi, M_rho = model.run()