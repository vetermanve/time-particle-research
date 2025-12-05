"""
v11.1.5: Модель мезонов с составляющими массами кварков
Физическая основа: Кварки в адронах имеют эффективные массы ~300 МэВ
"""

import numpy as np
from scipy.optimize import minimize

class ConstituentMesonModelV115:
    def __init__(self):
        # КОНСТИТУЕНТНЫЕ МАССЫ (МэВ) - подлежат оптимизации!
        # Вместо 2-5 МэВ, реально в адронах ~300 МэВ
        self.M_u = 300.0  # составляющая масса u-кварка
        self.M_d = 300.0  # составляющая масса d-кварка
        
        # Параметры взаимодействия (физически разумные)
        self.params = {
            'sigma': 0.18,      # ГэВ² (0.18 ГэВ² → √σ ≈ 424 МэВ)
            'alpha_s': 0.3,     # константа связи
            'spin_factor': 0.1  # безразмерный спин-спиновый параметр
        }
        
        # Целевые массы
        self.target = {
            'pi+': 139.570,
            'rho+': 775.260,
            'pi0': 134.977,
            'rho0': 775.260
        }
    
    def calculate_meson_mass(self, spin=0):
        """Основной расчёт с составляющими массами"""
        sigma = self.params['sigma'] * 1e6  # в МэВ²
        alpha_s = self.params['alpha_s']
        
        # Приведённая масса системы кварк-антикварк
        # Для u-anti-d: M_u и M_d
        mu = (self.M_u * self.M_d) / (self.M_u + self.M_d)
        
        # ------------------------------------------------------------
        # 1. ХАРАКТЕРНЫЕ МАСШТАБЫ (реалистичные!)
        # ------------------------------------------------------------
        
        # Масштаб конфайнмента
        confinement_scale = np.sqrt(sigma)  # ~424 МэВ
        
        # Длина осциллятора из линейного потенциала
        # ω = √(σ/μ) - частота осциллятора
        omega = np.sqrt(sigma / mu)  # МэВ
        
        a_ho = np.sqrt(self.hbar_c / (mu * omega))  # фм (ћc=197 МэВ·фм)
        
        # ------------------------------------------------------------
        # 2. ЭНЕРГИЯ СВЯЗИ (квантово-механическая оценка)
        # ------------------------------------------------------------
        
        # Кулоновская энергия: E_coul = -(4/3)α_s * (ћc) / r_eff
        r_eff = a_ho  # эффективное расстояние
        E_coulomb = -(4.0/3.0) * alpha_s * self.hbar_c / r_eff
        
        # Энергия нулевых колебаний 3D осциллятора
        E_zero = 1.5 * omega
        
        # Основная энергия связи
        E_binding = E_coulomb + E_zero  # должна быть ОТРИЦАТЕЛЬНОЙ!
        
        # ------------------------------------------------------------
        # 3. СПИН-СПИНОВОЕ ВЗАИМОДЕЙСТВИЕ
        # ------------------------------------------------------------
        
        # |ψ(0)|² для осциллятора
        psi0_squared = 1.0 / (np.pi**1.5 * a_ho**3)  # фм⁻³
        
        # Спиновый фактор для мезонов
        spin_factor = -3.0/8.0 if spin == 0 else 1.0/8.0
        
        # Энергия спин-спинового взаимодействия
        # ΔE = K * spin_factor * |ψ(0)|² * (ћc)³ / (M_u * M_d)
        K = self.params['spin_factor'] * 1e6  # МэВ·фм³
        E_spin = K * spin_factor * psi0_squared * (self.hbar_c**3) / (self.M_u * self.M_d)
        
        # ------------------------------------------------------------
        # 4. ИТОГОВАЯ МАССА
        # ------------------------------------------------------------
        
        # Масса мезона: M = M_u + M_d + E_binding + E_spin
        M_meson = self.M_u + self.M_d + E_binding + E_spin
        
        # Анализ
        analysis = {
            'mu': mu,
            'a_ho': a_ho,
            'omega': omega,
            'E_coulomb': E_coulomb,
            'E_zero': E_zero,
            'E_spin': E_spin,
            'psi0_squared': psi0_squared
        }
        
        return M_meson, E_binding + E_spin, analysis
    
    def error_function(self, params_array):
        """Функция ошибки с оптимизацией составляющих масс"""
        # Параметры: [M_u, M_d, sigma, alpha_s, spin_factor]
        self.M_u = params_array[0]
        self.M_d = params_array[1]
        self.params['sigma'] = params_array[2]
        self.params['alpha_s'] = params_array[3]
        self.params['spin_factor'] = params_array[4]
        
        # Рассчитываем массы
        M_pi, E_pi, _ = self.calculate_meson_mass(spin=0)
        M_rho, E_rho, _ = self.calculate_meson_mass(spin=1)
        
        # Ошибки масс
        error_pi = abs(M_pi - self.target['pi+']) / self.target['pi+']
        error_rho = abs(M_rho - self.target['rho+']) / self.target['rho+']
        
        # Соотношение масс
        if M_pi > 0:
            ratio = M_rho / M_pi
            target_ratio = self.target['rho+'] / self.target['pi+']
            ratio_error = abs(ratio - target_ratio) / target_ratio
        else:
            ratio_error = 10.0
        
        # Штрафы за нефизичные значения
        penalty = 0.0
        if E_pi > 0 or E_rho > 0:  # Энергии связи должны быть отрицательными
            penalty += 100.0
        if M_pi <= 0 or M_rho <= 0:
            penalty += 100.0
        if self.M_u <= 0 or self.M_d <= 0:
            penalty += 100.0
        
        # Общая ошибка
        total_error = error_pi + error_rho + ratio_error + penalty
        
        return total_error
    
    def run(self):
        """Запуск модели"""
        self.hbar_c = 197.3269804  # МэВ·фм
        
        print("\n" + "="*80)
        print("v11.1.5: МОДЕЛЬ С СОСТАВЛЯЮЩИМИ МАССАМИ КВАРКОВ")
        print("="*80)
        
        # Начальные параметры (физически разумные)
        # [M_u, M_d, sigma, alpha_s, spin_factor]
        x0 = [300.0, 300.0, 0.18, 0.3, 0.1]
        
        # Границы
        bounds = [
            (200.0, 400.0),   # M_u (МэВ)
            (200.0, 400.0),   # M_d (МэВ)
            (0.1, 0.3),       # sigma (ГэВ²)
            (0.1, 0.5),       # alpha_s
            (0.01, 1.0)       # spin_factor
        ]
        
        # Оптимизация
        result = minimize(
            self.error_function,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'disp': True, 'ftol': 1e-8}
        )
        
        if result.success:
            print("✅ Оптимизация успешна!")
            self.M_u, self.M_d = result.x[0], result.x[1]
            self.params['sigma'] = result.x[2]
            self.params['alpha_s'] = result.x[3]
            self.params['spin_factor'] = result.x[4]
        
        # Финальные результаты
        M_pi, E_pi, analysis_pi = self.calculate_meson_mass(spin=0)
        M_rho, E_rho, analysis_rho = self.calculate_meson_mass(spin=1)
        
        print(f"\n{'='*80}")
        print("РЕЗУЛЬТАТЫ v11.1.5")
        print(f"{'='*80}")
        
        print(f"\nОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ:")
        print(f"  M_u (составляющая) = {self.M_u:.1f} МэВ")
        print(f"  M_d (составляющая) = {self.M_d:.1f} МэВ")
        print(f"  σ = {self.params['sigma']:.3f} ГэВ²")
        print(f"  √σ = {np.sqrt(self.params['sigma']*1e6):.0f} МэВ")
        print(f"  α_s = {self.params['alpha_s']:.3f}")
        print(f"  spin_factor = {self.params['spin_factor']:.3f}")
        
        print(f"\nРАСЧЁТНЫЕ МАССЫ:")
        print(f"  π⁺: {M_pi:.1f} МэВ (цель: {self.target['pi+']:.1f})")
        print(f"  ρ⁺: {M_rho:.1f} МэВ (цель: {self.target['rho+']:.1f})")
        
        if M_pi > 0:
            ratio = M_rho / M_pi
            target_ratio = self.target['rho+'] / self.target['pi+']
            print(f"\nСООТНОШЕНИЕ МАСС:")
            print(f"  m(ρ)/m(π) = {ratio:.3f} (цель: {target_ratio:.3f})")
            print(f"  Ошибка: {abs(ratio-target_ratio)/target_ratio*100:.1f}%")
        
        print(f"\nФИЗИЧЕСКИЕ ПАРАМЕТРЫ (π⁺):")
        print(f"  Приведённая масса μ: {analysis_pi['mu']:.1f} МэВ")
        print(f"  Длина осциллятора a_ho: {analysis_pi['a_ho']:.3f} фм")
        print(f"  Частота ω: {analysis_pi['omega']:.1f} МэВ")
        print(f"  |ψ(0)|²: {analysis_pi['psi0_squared']:.3e} фм⁻³")
        print(f"  Кулоновская энергия: {analysis_pi['E_coulomb']:.1f} МэВ")
        print(f"  Энергия нулевых колебаний: {analysis_pi['E_zero']:.1f} МэВ")
        print(f"  Спин-спин энергия: {analysis_pi['E_spin']:.1f} МэВ")
        print(f"  Полная энергия связи: {E_pi:.1f} МэВ")
        
        # Проверка физической осмысленности
        print(f"\nПРОВЕРКА ФИЗИЧЕСКОЙ ОСМЫСЛЕННОСТИ:")
        checks = [
            (E_pi < 0, f"E_связи(π⁺) < 0 ({E_pi:.1f} МэВ)"),
            (abs(E_pi) > 50, f"|E_связи| > 50 МэВ"),
            (100 < M_pi < 500, f"100 < m(π⁺) < 500 МэВ ({M_pi:.1f} МэВ)"),
            (500 < M_rho < 1000, f"500 < m(ρ⁺) < 1000 МэВ ({M_rho:.1f} МэВ)"),
            (0.3 < analysis_pi['a_ho'] < 1.5, f"0.3 < a_ho < 1.5 фм ({analysis_pi['a_ho']:.3f} фм)"),
            (analysis_pi['psi0_squared'] < 1.0, f"|ψ(0)|² < 1 фм⁻³ ({analysis_pi['psi0_squared']:.3e})")
        ]
        
        passed = 0
        for check, msg in checks:
            if check:
                print(f"  ✅ {msg}")
                passed += 1
            else:
                print(f"  ❌ {msg}")
        
        print(f"\nПройдено проверок: {passed}/{len(checks)}")
        
        # Оценка успеха
        print(f"\n{'='*80}")
        print("ОЦЕНКА УСПЕХА МОДЕЛИ:")
        
        if passed >= 4 and abs(M_pi - self.target['pi+']) < 100 and abs(M_rho - self.target['rho+']) < 200:
            print("✅ МОДЕЛЬ РАБОТАЕТ! Получены физически разумные значения.")
            print(f"   Теперь можно переходить к v11.2 для добавления других частиц.")
        else:
            print("⚠️ МОДЕЛЬ ТРЕБУЕТ ДОРАБОТКИ.")
            print(f"   Основные проблемы: {'слишком большие массы' if M_pi > 500 else 'слишком малые массы' if M_pi < 100 else 'проблемы с энергией связи'}")
        
        return result.success

# Запуск
if __name__ == "__main__":
    model = ConstituentMesonModelV115()
    success = model.run()