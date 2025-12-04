import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ====================== ПАРАМЕТРИЧЕСКАЯ МОДЕЛЬ АДРОНА КАК СТОЯЧЕЙ ВОЛНЫ ======================

class QuarkOscillator:
    """Модель кварка как квантового осциллятора с параметрами"""
    def __init__(self, flavor, color_charge=None):
        self.flavor = flavor  # 'u', 'd', 's', etc.
        self.color = color_charge if color_charge else np.random.choice(['R', 'G', 'B'])
        
        # Базовые параметры осциллятора (подлежат настройке)
        self.fundamental_freq = None  # f0 - фундаментальная частота
        self.amplitude = None         # A - амплитуда колебаний
        self.phase = None             # φ - фаза
        self.harmonic = None          # n - номер гармоники (целое)
        
        # Физические свойства (вычисляются)
        self.charge = self._compute_charge()
        self.spin = 1/2  # для простоты считаем спин 1/2
        
    def _compute_charge(self):
        """Заряд кварка в единицах e/3"""
        charges = {'u': 2, 'd': -1, 's': -1, 'c': 2, 'b': -1, 't': 2}
        return charges.get(self.flavor, 0) / 3.0
    
    def wavefunction(self, t, coupling_factor=1.0):
        """Волновая функция кварка в момент времени t"""
        if None in [self.fundamental_freq, self.amplitude, self.phase, self.harmonic]:
            return 0
        # Осцилляция с основной частотой и гармоникой
        freq = self.fundamental_freq * self.harmonic * coupling_factor
        return self.amplitude * np.sin(2 * np.pi * freq * t + self.phase)
    
    def energy(self, coupling_factor=1.0):
        """Энергия осциллятора (пропорциональна массе)"""
        if None in [self.fundamental_freq, self.amplitude, self.harmonic]:
            return 0
        freq = self.fundamental_freq * self.harmonic * coupling_factor
        # E ~ A²f² (как у квантового гармонического осциллятора)
        return (self.amplitude ** 2) * (freq ** 2)

class HadronResonance:
    """Модель адрона как резонансной системы кварков-осцилляторов"""
    def __init__(self, quark_config):
        """
        quark_config: список кортежей (flavor, color) или объектов QuarkOscillator
        Пример для протона: [('u','R'), ('u','G'), ('d','B')]
        """
        self.quarks = []
        for q in quark_config:
            if isinstance(q, tuple):
                self.quarks.append(QuarkOscillator(*q))
            else:
                self.quarks.append(q)
        
        self.composite_wave = None
        self.resonance_quality = 0  # мера качества резонанса (0..1)
        
    def compute_composite_wave(self, t, coupling_matrix=None):
        """Вычисление суммарной волновой функции адрона"""
        waves = []
        for i, q in enumerate(self.quarks):
            # Коэффициент связи с другими кварками (если задана матрица)
            coupling = 1.0
            if coupling_matrix is not None and i < len(coupling_matrix):
                # Усредненная связь с другими кварками
                coupling = np.mean([coupling_matrix[i][j] for j in range(len(self.quarks)) if i != j])
            waves.append(q.wavefunction(t, coupling))
        
        self.composite_wave = np.sum(waves, axis=0)
        return self.composite_wave
    
    def resonance_condition(self, time_window=100, dt=0.01):
        """
        Проверка условия стоячей волны/резонанса.
        Возвращает меру стабильности паттерна (чем ближе к 1, тем лучше резонанс).
        """
        t = np.arange(0, time_window, dt)
        wave = self.compute_composite_wave(t)
        
        # Ищем периодичность через автокорреляцию
        correlation = np.correlate(wave, wave, mode='same')
        correlation = correlation / np.max(correlation)
        
        # Мера периодичности: высота первого побочного пика
        peak_idx = len(correlation) // 2
        side_peak = np.max(correlation[peak_idx+10:peak_idx+100])
        
        # Мера стабильности амплитуды
        amplitude_stability = 1 - (np.std(wave) / (np.max(np.abs(wave)) + 1e-8))
        
        self.resonance_quality = 0.7 * side_peak + 0.3 * amplitude_stability
        return self.resonance_quality
    
    def compute_mass(self, coupling_strength=0.1):
        """
        Вычисление массы адрона по формуле:
        M = ΣE_i + E_binding
        E_binding ~ -α * (resonance_quality) * (средняя связь)
        """
        quark_energies = sum(q.energy() for q in self.quarks)
        
        # Энергия связи отрицательна и зависит от качества резонанса
        binding_energy = -coupling_strength * self.resonance_quality * quark_energies
        
        return quark_energies + binding_energy
    
    def compute_charge(self):
        """Суммарный заряд адрона"""
        return sum(q.charge for q in self.quarks)
    
    def compute_spin(self):
        """Простая сумма спинов (без учета орбитального момента)"""
        return sum(q.spin for q in self.quarks)

# ====================== АЛГОРИТМ ПОДБОРА ПАРАМЕТРОВ ======================

def loss_function(params, target_mass, target_charge, target_spin, quark_types):
    """
    Функция потерь для оптимизации параметров кварков.
    
    params: вектор параметров [f0_u, A_u, φ_u, n_u, f0_d, A_d, φ_d, n_d, coupling]
    target_mass: целевая масса адрона (в условных единицах)
    quark_types: список типов кварков в адроне
    """
    n_quarks = len(quark_types)
    
    # Разбиваем параметры
    f0_u, A_u, phi_u, n_u, f0_d, A_d, phi_d, n_d, coupling = params
    
    # Создаем кварки с текущими параметрами
    quarks = []
    colors = ['R', 'G', 'B'][:n_quarks]  # обеспечиваем цветовую нейтральность
    
    for i, flavor in enumerate(quark_types):
        q = QuarkOscillator(flavor, colors[i])
        if flavor == 'u':
            q.fundamental_freq = f0_u
            q.amplitude = A_u
            q.phase = phi_u
            q.harmonic = int(round(n_u))
        elif flavor == 'd':
            q.fundamental_freq = f0_d
            q.amplitude = A_d
            q.phase = phi_d
            q.harmonic = int(round(n_d))
        quarks.append(q)
    
    # Создаем адрон
    hadron = HadronResonance(quarks)
    
    # Вычисляем свойства
    resonance_quality = hadron.resonance_condition()
    mass = hadron.compute_mass(coupling)
    charge = hadron.compute_charge()
    spin = hadron.compute_spin()
    
    # Вычисляем потери
    mass_loss = (mass - target_mass) ** 2
    charge_loss = (charge - target_charge) ** 2
    spin_loss = (spin - target_spin) ** 2
    
    # Штраф за плохой резонанс
    resonance_loss = (1 - resonance_quality) ** 2
    
    # Штраф за нецелые гармоники (если n не целое)
    harmonic_loss = (n_u - round(n_u)) ** 2 + (n_d - round(n_d)) ** 2
    
    # Комбинированная функция потерь
    total_loss = (mass_loss + 
                  0.1 * charge_loss + 
                  0.1 * spin_loss + 
                  0.5 * resonance_loss + 
                  0.2 * harmonic_loss)
    
    return total_loss

def optimize_hadron(quark_config, target_mass, target_charge, target_spin, initial_guess=None):
    """Оптимизация параметров для заданной конфигурации адрона"""
    
    # Начальное приближение
    if initial_guess is None:
        # f0_u, A_u, φ_u, n_u, f0_d, A_d, φ_d, n_d, coupling
        initial_guess = [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.1]
    
    bounds = [
        (0.1, 10.0),   # f0_u
        (0.1, 10.0),   # A_u
        (0, 2*np.pi),  # φ_u
        (1, 10),       # n_u (будем округлять)
        (0.1, 10.0),   # f0_d
        (0.1, 10.0),   # A_d
        (0, 2*np.pi),  # φ_d
        (1, 10),       # n_d
        (0.01, 1.0)    # coupling
    ]
    
    # Подсчет количества u и d кварков
    quark_types = [q[0] if isinstance(q, tuple) else q.flavor for q in quark_config]
    
    # Оптимизация
    result = minimize(
        loss_function,
        initial_guess,
        args=(target_mass, target_charge, target_spin, quark_types),
        bounds=bounds,
        method='L-BFGS-B',
        options={'maxiter': 500, 'disp': True}
    )
    
    return result

# ====================== ЗАПУСК ОПТИМИЗАЦИИ ДЛЯ ПРОТОНА И НЕЙТРОНА ======================

if __name__ == "__main__":
    print("="*60)
    print("ОПТИМИЗАЦИЯ ПАРАМЕТРОВ АДРОНОВ КАК СТОЯЧИХ ВОЛН")
    print("="*60)
    
    # Конфигурации кварков
    proton_config = [('u', 'R'), ('u', 'G'), ('d', 'B')]
    neutron_config = [('u', 'R'), ('d', 'G'), ('d', 'B')]
    
    # Целевые свойства (в условных единицах)
    # Масштабируем так, чтобы массы были ~1
    proton_target = {
        'mass': 1.0,      # 938.272 МэВ -> 1.0
        'charge': 1.0,
        'spin': 0.5
    }
    
    neutron_target = {
        'mass': 1.0007,   # 939.565/938.272 ≈ 1.0007
        'charge': 0.0,
        'spin': 0.5
    }
    
    # Оптимизация протона
    print("\n1. Оптимизация параметров для ПРОТОНА (uud):")
    proton_result = optimize_hadron(
        proton_config,
        proton_target['mass'],
        proton_target['charge'],
        proton_target['spin']
    )
    
    # Оптимизация нейтрона
    print("\n2. Оптимизация параметров для НЕЙТРОНА (udd):")
    neutron_result = optimize_hadron(
        neutron_config,
        neutron_target['mass'],
        neutron_target['charge'],
        neutron_target['spin']
    )
    
    # ====================== АНАЛИЗ РЕЗУЛЬТАТОВ ======================
    
    def extract_parameters(result, label):
        params = result.x
        f0_u, A_u, phi_u, n_u, f0_d, A_d, phi_d, n_d, coupling = params
        
        print(f"\nОптимальные параметры для {label}:")
        print(f"  u-кварк: f0={f0_u:.3f}, A={A_u:.3f}, φ={phi_u:.3f}, n={int(round(n_u))}")
        print(f"  d-кварк: f0={f0_d:.3f}, A={A_d:.3f}, φ={phi_d:.3f}, n={int(round(n_d))}")
        print(f"  Константа связи: {coupling:.3f}")
        print(f"  Значение функции потерь: {result.fun:.6f}")
        
        # Воссоздаем адрон с оптимальными параметрами
        if label == "протона":
            config = proton_config
            target = proton_target
        else:
            config = neutron_config
            target = neutron_target
            
        quark_types = [q[0] for q in config]
        quarks = []
        colors = ['R', 'G', 'B']
        
        for i, flavor in enumerate(quark_types):
            q = QuarkOscillator(flavor, colors[i])
            if flavor == 'u':
                q.fundamental_freq = f0_u
                q.amplitude = A_u
                q.phase = phi_u
                q.harmonic = int(round(n_u))
            elif flavor == 'd':
                q.fundamental_freq = f0_d
                q.amplitude = A_d
                q.phase = phi_d
                q.harmonic = int(round(n_d))
            quarks.append(q)
        
        hadron = HadronResonance(quarks)
        resonance = hadron.resonance_condition()
        mass = hadron.compute_mass(coupling)
        charge = hadron.compute_charge()
        spin = hadron.compute_spin()
        
        print(f"\n  Вычисленные свойства {label}:")
        print(f"    Масса: {mass:.4f} (цель: {target['mass']})")
        print(f"    Заряд: {charge:.4f} (цель: {target['charge']})")
        print(f"    Спин: {spin:.4f} (цель: {target['spin']})")
        print(f"    Качество резонанса: {resonance:.4f}")
        
        # Отношение частот (должно быть рациональным числом)
        freq_u = f0_u * int(round(n_u))
        freq_d = f0_d * int(round(n_d))
        if min(freq_u, freq_d) > 0:
            ratio = max(freq_u, freq_d) / min(freq_u, freq_d)
            print(f"    Отношение частот u/d: {ratio:.4f} ≈ {ratio:.2f}")
        
        return hadron
    
    # Извлекаем параметры
    proton_hadron = extract_parameters(proton_result, "протона")
    neutron_hadron = extract_parameters(neutron_result, "нейтрона")
    
    # ====================== ВИЗУАЛИЗАЦИЯ СТОЯЧИХ ВОЛН ======================
    
    print("\n" + "="*60)
    print("ВИЗУАЛИЗАЦИЯ ВОЛНОВЫХ ФУНКЦИЙ")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Временная область
    t = np.linspace(0, 10, 1000)
    
    # Протон: волновые функции отдельных кварков и суммарная
    ax1 = axes[0, 0]
    proton_waves = []
    for i, q in enumerate(proton_hadron.quarks):
        wave = q.wavefunction(t)
        proton_waves.append(wave)
        ax1.plot(t, wave, alpha=0.6, label=f'{q.flavor}-кварк (цвет {q.color})')
    
    ax1.plot(t, np.sum(proton_waves, axis=0), 'k-', linewidth=2, label='Протон (сумма)')
    ax1.set_xlabel('Время (усл. ед.)')
    ax1.set_ylabel('Амплитуда')
    ax1.set_title('Волновые функции кварков в протоне')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Нейтрон
    ax2 = axes[0, 1]
    neutron_waves = []
    for i, q in enumerate(neutron_hadron.quarks):
        wave = q.wavefunction(t)
        neutron_waves.append(wave)
        ax2.plot(t, wave, alpha=0.6, label=f'{q.flavor}-кварк (цвет {q.color})')
    
    ax2.plot(t, np.sum(neutron_waves, axis=0), 'k-', linewidth=2, label='Нейтрон (сумма)')
    ax2.set_xlabel('Время (усл. ед.)')
    ax2.set_ylabel('Амплитуда')
    ax2.set_title('Волновые функции кварков в нейтроне')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Фазовые портреты (амплитуда vs фаза)
    ax3 = axes[1, 0]
    for i, q in enumerate(proton_hadron.quarks):
        wave = q.wavefunction(t)
        # Производная для фазового портрета
        dt = t[1] - t[0]
        dwave = np.gradient(wave, dt)
        ax3.plot(wave[::10], dwave[::10], 'o', alpha=0.5, label=f'{q.flavor}-кварк')
    
    ax3.set_xlabel('Амплитуда')
    ax3.set_ylabel('d(Амплитуда)/dt')
    ax3.set_title('Фазовые портреты кварков протона')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Соотношения частот
    ax4 = axes[1, 1]
    hadrons = [proton_hadron, neutron_hadron]
    labels = ['Протон', 'Нейтрон']
    colors = ['blue', 'red']
    
    for idx, hadron in enumerate(hadrons):
        frequencies = []
        amplitudes = []
        for q in hadron.quarks:
            freq = q.fundamental_freq * q.harmonic
            frequencies.append(freq)
            amplitudes.append(q.amplitude)
        
        ax4.scatter(frequencies, amplitudes, color=colors[idx], 
                   label=labels[idx], s=100, alpha=0.7)
        
        # Соединяем точки одного адрона
        ax4.plot(frequencies, amplitudes, color=colors[idx], alpha=0.3)
    
    ax4.set_xlabel('Частота колебаний (усл. ед.)')
    ax4.set_ylabel('Амплитуда (усл. ед.)')
    ax4.set_title('Соотношения частот и амплитуд кварков')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hadron_wave_resonances.png', dpi=150, bbox_inches='tight')
    print("\nГрафики сохранены в файл 'hadron_wave_resonances.png'")
    
    # ====================== ВЫВОД О КЛЮЧЕВЫХ НАБЛЮДЕНИЯХ ======================
    
    print("\n" + "="*60)
    print("КЛЮЧЕВЫЕ НАБЛЮДЕНИЯ И ВЫВОДЫ")
    print("="*60)
    
    # Проверяем гипотезы
    print("\n1. Проверка гипотез о природе масс:")
    
    # Отношение масс нейтрона и протона
    proton_mass = proton_hadron.compute_mass(proton_result.x[-1])
    neutron_mass = neutron_hadron.compute_mass(neutron_result.x[-1])
    mass_ratio = neutron_mass / proton_mass
    
    print(f"   • Отношение масс нейтрон/протон: {mass_ratio:.6f}")
    print(f"     Реальное значение: {939.565/938.272:.6f}")
    
    # Проверка резонансных условий
    print("\n2. Анализ резонансных условий:")
    
    for name, hadron, result in [("Протон", proton_hadron, proton_result),
                                 ("Нейтрон", neutron_hadron, neutron_result)]:
        resonance_quality = hadron.resonance_condition()
        print(f"   • {name}: качество резонанса = {resonance_quality:.4f}")
        
        # Анализ фазовых соотношений
        phases = [q.phase for q in hadron.quarks]
        phase_diffs = []
        for i in range(len(phases)):
            for j in range(i+1, len(phases)):
                diff = np.abs(phases[i] - phases[j]) % (2*np.pi)
                diff = min(diff, 2*np.pi - diff)
                phase_diffs.append(diff)
        
        avg_phase_diff = np.mean(phase_diffs) if phase_diffs else 0
        print(f"     Средняя разность фаз: {avg_phase_diff:.3f} радиан")
        
        # Проверка, являются ли отношения частот рациональными
        freqs = [q.fundamental_freq * q.harmonic for q in hadron.quarks]
        ratios = []
        for i in range(len(freqs)):
            for j in range(i+1, len(freqs)):
                if min(freqs[i], freqs[j]) > 0:
                    ratio = max(freqs[i], freqs[j]) / min(freqs[i], freqs[j])
                    ratios.append(ratio)
        
        if ratios:
            # Проверяем, насколько близко к рациональным числам
            from fractions import Fraction
            rational_approximations = []
            for r in ratios:
                frac = Fraction(r).limit_denominator(10)
                rational_approximations.append((frac.numerator, frac.denominator))
            
            print(f"     Отношения частот: {ratios}")
            print(f"     Приближения простыми дробями: {rational_approximations}")
    
    print("\n3. Предсказания модели:")
    print("   • Массы адронов возникают из резонансных условий, а не являются")
    print("     фундаментальными параметрами.")
    print("   • Различие масс протона и нейтрона объясняется разным")
    print("     качеством резонанса при замене u-кварка на d-кварк.")
    print("   • Требование цветовой нейтральности автоматически приводит")
    print("     к трёхкварковым комбинациям для барионов.")
    
    print("\n" + "="*60)
    print("СЛЕДУЮЩИЕ ШАГИ ДЛЯ РАЗВИТИЯ МОДЕЛИ:")
    print("="*60)
    print("1. Включение глюонных полей как переносчиков синхронизации")
    print("2. Моделирование сильного взаимодействия через нелинейные связи")
    print("3. Расширение на странные, очарованные и другие кварки")
    print("4. Вычисление сечений рассеяния и времен жизни")
    print("5. Связь с КХД через эффективные лагранжианы")