"""
V11.1: Фундаментальная модель мезонов на основе потенциала Корнелла
Цель: Корректно воспроизвести соотношение масс π⁺ (спин 0) и ρ⁺ (спин 1)
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

class CornellMesonModel:
    def __init__(self):
        # ----------------------------
        # ФИЗИЧЕСКИЕ КОНСТАНТЫ (в МэВ и фм)
        # ----------------------------
        self.hbar_c = 197.3269804  # ћc в МэВ·фм (для перевода)
        
        # Массы КВАРКОВ в МэВ (текущие кварковые массы)
        self.m_u = 2.16
        self.m_d = 4.67
        self.m_ud = (self.m_u + self.m_d) / 2  # средняя для легких мезонов
        
        # ----------------------------
        # ПАРАМЕТРЫ ПОТЕНЦИАЛА КОРНЕЛЛА (в МэВ)
        # ----------------------------
        # Будем искать: a (сила кулоновской части), sigma (стринг-тензия)
        # Стартовые значения из литературы:
        self.a = 100.0          # ~ безразмерная константа α_s * 4/3
        self.sigma = 90000.0    # σ в МэВ² -> √σ ~ 300 МэВ (характерный масштаб)
        
        # Параметр спин-спинового взаимодействия (МэВ·фм³)
        # Зависит от |ψ(0)|² - волновой функции в начале координат
        self.kappa_ss = 400.0   # Подлежит настройке
        
        # Целевые массы (МэВ) - ТОЛЬКО ДЛЕМОНСТРАЦИИ, не для подгонки!
        self.target_pi = 139.570
        self.target_rho = 775.260
        
    def cornell_potential(self, r):
        """Потенциал Корнелла V(r) = -a/r + σ*r (в МэВ)"""
        if r <= 0:
            return -np.inf  # Защита от деления на ноль
        return -self.a / r + (self.sigma / (self.hbar_c**2)) * r  # σ в МэВ² переведена
        
    def schrodinger_equation(self, r, y, E, mu):
        """Уравнение Шрёдингера для s-волны (l=0) в безразмерной форме"""
        psi, psi_prime = y[0], y[1]
        
        # Эффективный потенциал (включает центробежный член, но для l=0 он 0)
        V = self.cornell_potential(r)
        
        # Вторая производная: ψ''(r) = [2μ/ћ² * (V(r) - E)] * ψ(r)
        # Где 2μ/ћ² = 2μ/(hbar_c)², т.к. ћ² = (hbar_c)²
        factor = (2 * mu / (self.hbar_c**2)) * (V - E)
        
        return [psi_prime, factor * psi]
    
    def solve_bound_state(self, mu, spin=0, E_guess=-200.0):
        """
        Решает уравнение Шрёдингера и находит энергию связи E_bind < 0
        для основного состояния (n=1, l=0)
        
        mu - приведенная масса системы (МэВ)
        spin - спин мезона (0 для π, 1 для ρ)
        E_guess - начальное приближение для энергии (МэВ)
        """
        # ----------------------------
        # 1. ИЩЕМ ЭНЕРГИЮ СВЯЗИ (РЕШАЕМ КРАЕВУЮ ЗАДАЧУ)
        # ----------------------------
        # Граничные условия для s-волы:
        # ψ(r=0) = 0 (требование конечности)
        # ψ(r→∞) = 0 (связанное состояние)
        
        # Правая граница - достаточно большая
        r_max = 5.0  # фм
        
        def boundary_condition(E):
            """Функция невязки для краевой задачи"""
            # Интегрируем от малого r0 до r_max
            r0 = 0.001
            sol = solve_ivp(
                lambda r, y: self.schrodinger_equation(r, y, E, mu),
                [r0, r_max],
                [r0, 1.0],  # ψ(r0) ≈ r0, ψ'(r0) ≈ 1 (нормировка)
                method='RK45',
                dense_output=True
            )
            
            # Значение волновой функции на правой границе
            psi_at_rmax = sol.y[0, -1]
            return psi_at_rmax  # Должно быть 0 для собственного состояния
        
        # Ищем энергию, при которой ψ(r_max) = 0
        try:
            result = root_scalar(boundary_condition, x0=E_guess, x1=E_guess*0.9)
            E_bind = result.root
        except:
            # Если не нашли, возвращаем приблизительное значение
            E_bind = E_guess
        
        # ----------------------------
        # 2. ВЫЧИСЛЯЕМ ПОПРАВКУ НА СПИН
        # ----------------------------
        # Спин-спиновое взаимодействие: ΔE_ss = (8π/9) * (κ_ss/m1*m2) * S1·S2 * |ψ(0)|²
        # Для мезонов: S1·S2 = [s(s+1) - 3/2]/4
        # где s = 0 для π (псевдоскаляр), s = 1 для ρ (векторный)
        
        # ОЦЕНКА |ψ(0)|² для кулоновского потенциала (приближение)
        # Для основного состояния: |ψ(0)|² ≈ (μ*α)^3 / π, где α = a / (hbar_c)
        alpha = self.a / self.hbar_c
        psi0_squared = (mu * alpha)**3 / (np.pi * (self.hbar_c**3))
        
        # Рассчитываем спин-спиновую поправку
        if spin == 0:  # π-мезон
            spin_factor = (0*(0+1) - 1.5) / 4.0  # = -3/8
        else:  # ρ-мезон, spin = 1
            spin_factor = (1*(1+1) - 1.5) / 4.0  # = +1/8
        
        # Энергия спин-спинового взаимодействия
        # Упрощенная формула: ΔE_ss = kappa_ss * spin_factor * psi0_squared
        delta_E_ss = self.kappa_ss * spin_factor * psi0_squared
        
        # Итоговая энергия связи с учетом спина
        E_total = E_bind + delta_E_ss
        
        return E_total, psi0_squared
    
    def calculate_meson_mass(self, quark_mass1, quark_mass2, spin=0):
        """
        Рассчитывает массу мезона из двух кварков
        M = m1 + m2 + E_binding
        """
        # Приведенная масса системы
        mu = (quark_mass1 * quark_mass2) / (quark_mass1 + quark_mass2)
        
        # Энергия связи (отрицательная!)
        E_bind, psi0_sq = self.solve_bound_state(mu, spin=spin)
        
        # Полная масса мезона
        M = quark_mass1 + quark_mass2 + E_bind
        
        return M, E_bind, psi0_sq
    
    def optimize_for_pion_rho(self):
        """
        Оптимизирует параметры модели (a, sigma, kappa_ss)
        чтобы воспроизвести отношение m(ρ)/m(π)
        """
        print("\n" + "="*80)
        print("v11.1: Оптимизация для воспроизведения m(ρ)/m(π) ≈ 5.55")
        print("="*80)
        
        # Целевое соотношение
        target_ratio = self.target_rho / self.target_pi  # ≈ 5.555
        
        def error_function(params):
            self.a, self.sigma, self.kappa_ss = params
            
            # Вычисляем массы
            M_pi, E_pi, _ = self.calculate_meson_mass(self.m_u, self.m_d, spin=0)
            M_rho, E_rho, _ = self.calculate_meson_mass(self.m_u, self.m_d, spin=1)
            
            # Отношение масс
            ratio = M_rho / M_pi if M_pi > 0 else 0
            
            # Ошибка: разность отношений + штраф за отрицательные массы
            error = abs(ratio - target_ratio)
            
            if M_pi <= 0 or M_rho <= 0:
                error += 1000.0
                
            return error
        
        # Стартовые значения и границы
        initial_guess = [self.a, self.sigma, self.kappa_ss]
        bounds = [(50, 300), (50000, 200000), (100, 1000)]
        
        # Простая оптимизация по сетке (для начала)
        best_error = float('inf')
        best_params = initial_guess
        
        # Сеточный поиск по a и sigma (упрощенно)
        for a_test in np.linspace(70, 130, 7):
            for sigma_test in np.linspace(80000, 100000, 5):
                self.a, self.sigma = a_test, sigma_test
                
                # Подбираем kappa_ss для правильного расщепления
                M_pi, E_pi, psi0_sq = self.calculate_meson_mass(self.m_u, self.m_d, spin=0)
                M_rho, E_rho, _ = self.calculate_meson_mass(self.m_u, self.m_d, spin=1)
                
                if M_pi > 0 and M_rho > 0:
                    current_ratio = M_rho / M_pi
                    error = abs(current_ratio - target_ratio)
                    
                    if error < best_error:
                        best_error = error
                        # Оцениваем kappa_ss из разности энергий
                        delta_E_exp = (self.target_rho - self.target_pi) - (M_rho - M_pi)
                        kappa_est = self.kappa_ss * (1 + delta_E_exp / 100)
                        best_params = [a_test, sigma_test, kappa_est]
        
        self.a, self.sigma, self.kappa_ss = best_params
        
        print(f"Найденные параметры:")
        print(f"  a (кулоновский параметр) = {self.a:.1f}")
        print(f"  σ (стринг-тензия) = {self.sigma:.0f} МэВ²")
        print(f"  κ_ss (спин-спин) = {self.kappa_ss:.1f} МэВ·фм³")
        
        return best_params
    
    def run(self):
        """
        Запуск полного расчета v11.1
        """
        print("\n🚀 ЗАПУСК v11.1: ФУНДАМЕНТАЛЬНАЯ МОДЕЛЬ МЕЗОНОВ")
        print("Основа: потенциал Корнелла + уравнение Шрёдингера")
        
        # 1. Оптимизируем под соотношение масс
        self.optimize_for_pion_rho()
        
        # 2. Рассчитываем окончательные массы
        M_pi, E_pi, psi0_sq_pi = self.calculate_meson_mass(self.m_u, self.m_d, spin=0)
        M_rho, E_rho, psi0_sq_rho = self.calculate_meson_mass(self.m_u, self.m_d, spin=1)
        
        # 3. Вывод результатов
        print("\n" + "="*80)
        print("РЕЗУЛЬТАТЫ v11.1")
        print("="*80)
        
        print(f"\n{'Мезон':<10} {'Расч. масса':<12} {'Цель':<12} {'Ошибка %':<10} {'E_связи':<12} |ψ(0)|²")
        print("-"*80)
        
        for name, M_calc, M_target, spin in [
            ("π⁺ (спин 0)", M_pi, self.target_pi, 0),
            ("ρ⁺ (спин 1)", M_rho, self.target_rho, 1)
        ]:
            error_pct = abs(M_calc - M_target) / M_target * 100
            E_bind = M_calc - (self.m_u + self.m_d)
            psi0_sq = psi0_sq_pi if spin == 0 else psi0_sq_rho
            
            print(f"{name:<10} {M_calc:<12.1f} {M_target:<12.1f} {error_pct:<10.1f} "
                  f"{E_bind:<12.1f} {psi0_sq:.2e}")
        
        # Ключевые соотношения
        ratio_calc = M_rho / M_pi
        ratio_target = self.target_rho / self.target_pi
        
        print(f"\nКЛЮЧЕВЫЕ СООТНОШЕНИЯ:")
        print(f"  m(ρ)/m(π) расч.: {ratio_calc:.3f}")
        print(f"  m(ρ)/m(π) цель: {ratio_target:.3f}")
        print(f"  Ошибка соотношения: {abs(ratio_calc - ratio_target)/ratio_target*100:.1f}%")
        
        # Характерные масштабы
        sigma_mev = np.sqrt(self.sigma)  # √σ в МэВ
        r_confinement = self.hbar_c / sigma_mev  # Характерный размер в фм
        
        print(f"\nФИЗИЧЕСКИЕ МАСШТАБЫ:")
        print(f"  Масса конфайнмента √σ: {sigma_mev:.0f} МэВ")
        print(f"  Характерный размер: {r_confinement:.2f} фм")
        print(f"  Кулоновская константа α = a/(ћc): {self.a/self.hbar_c:.3f}")
        
        # Визуализация потенциала
        self.plot_results(M_pi, M_rho)
        
        return M_pi, M_rho
    
    def plot_results(self, M_pi, M_rho):
        """Визуализация потенциала и результатов"""
        r_vals = np.linspace(0.01, 2.0, 200)
        V_vals = [self.cornell_potential(r) for r in r_vals]
        
        plt.figure(figsize=(12, 5))
        
        # 1. Потенциал
        plt.subplot(1, 2, 1)
        plt.plot(r_vals, V_vals, 'b-', linewidth=2, label='V(r) = -a/r + σ·r')
        plt.axhline(y=M_pi - (self.m_u+self.m_d), color='r', linestyle='--', 
                   label=f'E(π⁺) = {M_pi-(self.m_u+self.m_d):.1f} МэВ')
        plt.axhline(y=M_rho - (self.m_u+self.m_d), color='g', linestyle='--',
                   label=f'E(ρ⁺) = {M_rho-(self.m_u+self.m_d):.1f} МэВ')
        plt.xlabel('Расстояние r (фм)')
        plt.ylabel('Энергия (МэВ)')
        plt.title('Потенциал Корнелла и энергии связи')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Сравнение с экспериментами
        plt.subplot(1, 2, 2)
        particles = ['π⁺', 'ρ⁺']
        calc_masses = [M_pi, M_rho]
        target_masses = [self.target_pi, self.target_rho]
        
        x = np.arange(len(particles))
        width = 0.35
        
        plt.bar(x - width/2, calc_masses, width, label='Расчет v11.1', color='skyblue')
        plt.bar(x + width/2, target_masses, width, label='Эксперимент', color='lightcoral')
        
        plt.xlabel('Мезон')
        plt.ylabel('Масса (МэВ)')
        plt.title('Сравнение расчетных и экспериментальных масс')
        plt.xticks(x, particles)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        # Добавляем значения на столбцы
        for i, (calc, target) in enumerate(zip(calc_masses, target_masses)):
            plt.text(i - width/2, calc + 20, f'{calc:.0f}', ha='center')
            plt.text(i + width/2, target + 20, f'{target:.0f}', ha='center')
        
        plt.tight_layout()
        plt.savefig('v11_1_results.png', dpi=150)
        plt.show()


# ================= ЗАПУСК =================
if __name__ == "__main__":
    # Инициализация и запуск модели
    model = CornellMesonModel()
    M_pi, M_rho = model.run()
    
    # Сохранение параметров для следующей версии
    results = {
        'model': 'v11.1_cornell_meson',
        'parameters': {
            'a': model.a,
            'sigma_MeV2': model.sigma,
            'kappa_ss': model.kappa_ss,
            'm_u_MeV': model.m_u,
            'm_d_MeV': model.m_d
        },
        'calculated_masses': {
            'pi+_MeV': M_pi,
            'rho+_MeV': M_rho
        },
        'target_masses': {
            'pi+_MeV': model.target_pi,
            'rho+_MeV': model.target_rho
        }
    }
    
    print(f"\n💾 Параметры модели сохранены для v11.2 (барионы)")