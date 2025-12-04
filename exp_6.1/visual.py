import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec

# Параметры модели v6.1
params = {
    'base_mass_u': 2.203806,
    'base_mass_d': 4.583020,
    'freq_u': 0.956359,
    'freq_d': 0.868115,
    'amp_u': 1.032476,
    'amp_d': 0.877773,
    'coupling_proton': 1.613565,
    'coupling_neutron': 0.285395,
    'coupling_meson': 4.273121,
    'phase_shift': 3.173848,
    'scale_factor': 100.0
}

class QuarkWavefunction:
    """Класс для расчета и визуализации волновых функций кварков"""
    
    def __init__(self, params):
        self.params = params
        
    def quark_oscillation(self, x, quark_type, phase=0.0):
        """Волновая функция осциллирующего кварка"""
        if quark_type == 'u':
            freq = params['freq_u']
            amp = params['amp_u']
        elif quark_type == 'd':
            freq = params['freq_d']
            amp = params['amp_d']
        elif quark_type == 'anti_d':
            freq = params['freq_d']
            amp = params['amp_d']
        else:
            freq = 1.0
            amp = 1.0
        
        # Колебательная функция
        return amp * np.sin(2 * np.pi * freq * x + phase)
    
    def quark_potential(self, x, quark_type):
        """Потенциальная функция кварка (форма нити)"""
        if quark_type == 'u':
            base_mass = params['base_mass_u']
            amp = params['amp_u']
        elif quark_type == 'd':
            base_mass = params['base_mass_d']
            amp = params['amp_d']
        else:
            base_mass = 1.0
            amp = 1.0
        
        # Форма нити (гауссово распределение + осцилляции)
        center = 0.5  # Центр частицы
        width = 0.15  # Ширина распределения
        
        # Форма нити
        gaussian = base_mass * np.exp(-((x - center) ** 2) / (2 * width ** 2))
        
        # Осцилляции внутри нити
        oscillation = amp * np.sin(8 * np.pi * x)
        
        return gaussian * (1 + 0.2 * oscillation)
    
    def calculate_total_wavefunction(self, composition, phases):
        """Расчет общей волновой функции частицы"""
        x = np.linspace(0, 1, 1000)
        total_wf = np.zeros_like(x)
        
        wf_dict = {}
        for i, (quark, phase) in enumerate(zip(composition, phases)):
            wf = self.quark_oscillation(x, quark, phase)
            wf_dict[f'{quark}_{i+1}'] = wf
            total_wf += wf
        
        # Добавляем затухание на краях (конфайнмент)
        confinement = np.exp(-10 * (x - 0.5) ** 2) + np.exp(-10 * (x - 0.5) ** 2)
        total_wf *= confinement
        
        return x, total_wf, wf_dict
    
    def calculate_3d_structure(self, composition, phases):
        """Расчет 3D структуры частицы"""
        # Создаем 2D сетку
        x = np.linspace(-1, 1, 100)
        y = np.linspace(-1, 1, 100)
        X, Y = np.meshgrid(x, y)
        
        # Расстояние от центра
        R = np.sqrt(X**2 + Y**2)
        
        # Форма частицы
        particle_shape = np.zeros_like(R)
        
        # Распределение кварков
        quark_positions = []
        n_quarks = len(composition)
        
        # Располагаем кварки по окружности
        for i in range(n_quarks):
            angle = 2 * np.pi * i / n_quarks
            qx = 0.3 * np.cos(angle)
            qy = 0.3 * np.sin(angle)
            quark_positions.append((qx, qy, composition[i]))
            
            # Волновая функция кварка в пространстве
            dist = np.sqrt((X - qx)**2 + (Y - qy)**2)
            if composition[i] == 'u':
                amp = params['amp_u']
                freq = params['freq_u']
            else:
                amp = params['amp_d']
                freq = params['freq_d']
            
            # Колебания кварка в пространстве
            phase = phases[i]
            quark_wf = amp * np.exp(-dist**2 / 0.05) * np.sin(freq * 10 * dist + phase)
            particle_shape += quark_wf
        
        # Общая форма частицы (конфайнмент)
        confinement = np.exp(-R**2 / 0.3)
        particle_shape *= confinement
        
        return X, Y, particle_shape, quark_positions
    
    def visualize_proton(self):
        """Визуализация протона"""
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('ПРОТОН: Структура волновых функций (uud)', fontsize=16, fontweight='bold')
        
        # 1. Осцилляции кварков во времени
        gs1 = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs1[0, 0])
        x = np.linspace(0, 4*np.pi, 1000)
        
        # u-кварки (фаза 0)
        u1_wf = self.quark_oscillation(x, 'u', 0)
        u2_wf = self.quark_oscillation(x, 'u', 0)
        # d-кварк (фаза π/2)
        d_wf = self.quark_oscillation(x, 'd', np.pi/2)
        
        ax1.plot(x, u1_wf, 'r-', linewidth=2, alpha=0.7, label='u-кварк 1 (фаза=0)')
        ax1.plot(x, u2_wf, 'r--', linewidth=2, alpha=0.7, label='u-кварк 2 (фаза=0)')
        ax1.plot(x, d_wf, 'b-', linewidth=2, alpha=0.7, label='d-кварк (фаза=π/2)')
        
        total_wf = u1_wf + u2_wf + d_wf
        ax1.plot(x, total_wf, 'k-', linewidth=3, alpha=0.9, label='Суммарная волновая функция')
        
        ax1.set_xlabel('Время/Координата', fontsize=12)
        ax1.set_ylabel('Амплитуда', fontsize=12)
        ax1.set_title('Осцилляции кварков протона', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Спектр частот
        ax2 = fig.add_subplot(gs1[0, 1])
        frequencies = [params['freq_u'], params['freq_u'], params['freq_d']]
        amplitudes = [params['amp_u'], params['amp_u'], params['amp_d']]
        colors = ['red', 'red', 'blue']
        labels = ['u1', 'u2', 'd']
        
        bars = ax2.bar(range(3), amplitudes, color=colors, alpha=0.7)
        for i, (bar, freq) in enumerate(zip(bars, frequencies)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'f={freq:.3f}', ha='center', va='bottom')
        
        ax2.set_xlabel('Кварк', fontsize=12)
        ax2.set_ylabel('Амплитуда', fontsize=12)
        ax2.set_title('Спектр частот и амплитуд', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(3))
        ax2.set_xticklabels(labels)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. 2D профиль протона
        ax3 = fig.add_subplot(gs1[1, 0])
        x_profile = np.linspace(0, 1, 500)
        
        u1_profile = self.quark_potential(x_profile, 'u')
        u2_profile = self.quark_potential(x_profile, 'u')
        d_profile = self.quark_potential(x_profile, 'd')
        
        ax3.plot(x_profile, u1_profile, 'r-', alpha=0.6, label='u-кварк 1')
        ax3.plot(x_profile, u2_profile, 'r--', alpha=0.6, label='u-кварк 2')
        ax3.plot(x_profile, d_profile, 'b-', alpha=0.6, label='d-кварк')
        
        total_profile = u1_profile + u2_profile + d_profile
        ax3.plot(x_profile, total_profile, 'k-', linewidth=2, alpha=0.8, label='Протон')
        
        ax3.fill_between(x_profile, total_profile, alpha=0.2, color='gray')
        
        ax3.set_xlabel('Координата', fontsize=12)
        ax3.set_ylabel('Потенциал/Плотность', fontsize=12)
        ax3.set_title('Пространственный профиль', fontsize=14, fontweight='bold')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # 4. Фазовая диаграмма
        ax4 = fig.add_subplot(gs1[1, 1])
        phases = [0, 0, np.pi/2]
        
        # Когерентность
        coherence = np.zeros(100)
        for i in range(100):
            time = i * 0.1
            u1 = np.sin(params['freq_u'] * time + 0)
            u2 = np.sin(params['freq_u'] * time + 0)
            d = np.sin(params['freq_d'] * time + np.pi/2)
            coherence[i] = (u1 * u2 + u1 * d + u2 * d) / 3
        
        ax4.plot(np.linspace(0, 10, 100), coherence, 'g-', linewidth=2)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax4.fill_between(np.linspace(0, 10, 100), coherence, 0, 
                        where=(coherence > 0), color='green', alpha=0.3, label='Положительная когерентность')
        ax4.fill_between(np.linspace(0, 10, 100), coherence, 0,
                        where=(coherence < 0), color='red', alpha=0.3, label='Отрицательная когерентность')
        
        ax4.set_xlabel('Время', fontsize=12)
        ax4.set_ylabel('Когерентность', fontsize=12)
        ax4.set_title('Фазовая когерентность', fontsize=14, fontweight='bold')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        
        # 5. 3D структура протона
        ax5 = fig.add_subplot(235, projection='3d')
        X, Y, Z, quark_pos = self.calculate_3d_structure(['u', 'u', 'd'], [0, 0, np.pi/2])
        
        surf = ax5.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8, 
                              linewidth=0, antialiased=True)
        
        # Отметим положение кварков
        for pos in quark_pos:
            ax5.scatter(pos[0], pos[1], 0.5, s=100, color='r' if pos[2]=='u' else 'b', 
                       marker='o', edgecolors='k', linewidth=1.5)
        
        ax5.set_xlabel('X', fontsize=10)
        ax5.set_ylabel('Y', fontsize=10)
        ax5.set_zlabel('Плотность', fontsize=10)
        ax5.set_title('3D структура протона', fontsize=14, fontweight='bold')
        
        # 6. Энергетическая диаграмма
        ax6 = fig.add_subplot(236)
        
        # Энергетические уровни
        components = ['2×u-кварка', 'd-кварк', 'Базовая масса', 'Энергия синхронизации']
        energies = [
            2 * params['base_mass_u'] * params['freq_u'] * params['amp_u'],
            params['base_mass_d'] * params['freq_d'] * params['amp_d'],
            0,  # разделитель
            params['coupling_proton']
        ]
        
        colors = ['orange', 'blue', 'gray', 'green']
        
        y_pos = np.arange(len(components))
        bars = ax6.barh(y_pos, energies, color=colors, alpha=0.7)
        
        for i, (bar, energy) in enumerate(zip(bars, energies)):
            width = bar.get_width()
            ax6.text(width + 0.05, bar.get_y() + bar.get_height()/2.,
                    f'{energy:.3f}', ha='left', va='center')
        
        ax6.set_yticks(y_pos)
        ax6.set_yticklabels(components)
        ax6.set_xlabel('Энергия (в единицах модели)', fontsize=12)
        ax6.set_title('Энергетическая диаграмма протона', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_neutron(self):
        """Визуализация нейтрона"""
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('НЕЙТРОН: Структура волновых функций (udd)', fontsize=16, fontweight='bold')
        
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Осцилляции кварков
        ax1 = fig.add_subplot(gs[0, 0])
        x = np.linspace(0, 4*np.pi, 1000)
        
        u_wf = self.quark_oscillation(x, 'u', 0)
        d1_wf = self.quark_oscillation(x, 'd', np.pi/2)
        d2_wf = self.quark_oscillation(x, 'd', np.pi/2)
        
        ax1.plot(x, u_wf, 'r-', linewidth=2, alpha=0.7, label='u-кварк (фаза=0)')
        ax1.plot(x, d1_wf, 'b-', linewidth=2, alpha=0.7, label='d-кварк 1 (фаза=π/2)')
        ax1.plot(x, d2_wf, 'b--', linewidth=2, alpha=0.7, label='d-кварк 2 (фаза=π/2)')
        
        total_wf = u_wf + d1_wf + d2_wf
        ax1.plot(x, total_wf, 'k-', linewidth=3, alpha=0.9, label='Суммарная волновая функция')
        
        ax1.set_xlabel('Время/Координата', fontsize=12)
        ax1.set_ylabel('Амплитуда', fontsize=12)
        ax1.set_title('Осцилляции кварков нейтрона', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Сравнение с протоном
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Протон
        x_comp = np.linspace(0, 2*np.pi, 500)
        proton_wf = (2*self.quark_oscillation(x_comp, 'u', 0) + 
                    self.quark_oscillation(x_comp, 'd', np.pi/2))
        neutron_wf = (self.quark_oscillation(x_comp, 'u', 0) + 
                     2*self.quark_oscillation(x_comp, 'd', np.pi/2))
        
        ax2.plot(x_comp, proton_wf, 'r-', linewidth=2, alpha=0.7, label='Протон (uud)')
        ax2.plot(x_comp, neutron_wf, 'b-', linewidth=2, alpha=0.7, label='Нейтрон (udd)')
        
        ax2.set_xlabel('Время/Координата', fontsize=12)
        ax2.set_ylabel('Амплитуда', fontsize=12)
        ax2.set_title('Сравнение протона и нейтрона', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # 3. Фазовая диаграмма
        ax3 = fig.add_subplot(gs[0, 2], projection='polar')
        
        phases = [0, np.pi/2, np.pi/2]  # u, d1, d2
        radii = [params['amp_u'], params['amp_d'], params['amp_d']]
        colors = ['red', 'blue', 'blue']
        
        for phase, radius, color in zip(phases, radii, colors):
            ax3.plot([0, phase], [0, radius], color=color, linewidth=3, alpha=0.7)
            ax3.scatter(phase, radius, s=100, color=color, edgecolors='k', linewidth=1.5)
        
        ax3.set_title('Фазовая диаграмма нейтрона', fontsize=14, fontweight='bold', pad=20)
        ax3.grid(True)
        
        # 4. Пространственное распределение
        ax4 = fig.add_subplot(gs[1, 0])
        x_profile = np.linspace(0, 1, 500)
        
        u_profile = self.quark_potential(x_profile, 'u')
        d1_profile = self.quark_potential(x_profile, 'd')
        d2_profile = self.quark_potential(x_profile, 'd')
        
        ax4.plot(x_profile, u_profile, 'r-', alpha=0.6, label='u-кварк')
        ax4.plot(x_profile, d1_profile, 'b-', alpha=0.6, label='d-кварк 1')
        ax4.plot(x_profile, d2_profile, 'b--', alpha=0.6, label='d-кварк 2')
        
        total_profile = u_profile + d1_profile + d2_profile
        ax4.plot(x_profile, total_profile, 'k-', linewidth=2, alpha=0.8, label='Нейтрон')
        
        ax4.fill_between(x_profile, total_profile, alpha=0.2, color='gray')
        
        ax4.set_xlabel('Координата', fontsize=12)
        ax4.set_ylabel('Потенциал/Плотность', fontsize=12)
        ax4.set_title('Пространственный профиль', fontsize=14, fontweight='bold')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        
        # 5. 3D структура
        ax5 = fig.add_subplot(gs[1, 1], projection='3d')
        X, Y, Z, quark_pos = self.calculate_3d_structure(['u', 'd', 'd'], [0, np.pi/2, np.pi/2])
        
        surf = ax5.plot_surface(X, Y, Z, cmap=cm.plasma, alpha=0.8, 
                              linewidth=0, antialiased=True)
        
        for pos in quark_pos:
            ax5.scatter(pos[0], pos[1], 0.5, s=100, color='r' if pos[2]=='u' else 'b', 
                       marker='o', edgecolors='k', linewidth=1.5)
        
        ax5.set_xlabel('X', fontsize=10)
        ax5.set_ylabel('Y', fontsize=10)
        ax5.set_zlabel('Плотность', fontsize=10)
        ax5.set_title('3D структура нейтрона', fontsize=14, fontweight='bold')
        
        # 6. Энергетическая диаграмма
        ax6 = fig.add_subplot(gs[1, 2])
        
        components = ['u-кварк', '2×d-кварка', 'Базовая масса', 'Энергия синхронизации']
        energies = [
            params['base_mass_u'] * params['freq_u'] * params['amp_u'],
            2 * params['base_mass_d'] * params['freq_d'] * params['amp_d'],
            0,  # разделитель
            params['coupling_neutron']
        ]
        
        colors = ['red', 'blue', 'gray', 'green']
        
        y_pos = np.arange(len(components))
        bars = ax6.barh(y_pos, energies, color=colors, alpha=0.7)
        
        for i, (bar, energy) in enumerate(zip(bars, energies)):
            width = bar.get_width()
            ax6.text(width + 0.05, bar.get_y() + bar.get_height()/2.,
                    f'{energy:.3f}', ha='left', va='center')
        
        ax6.set_yticks(y_pos)
        ax6.set_yticklabels(components)
        ax6.set_xlabel('Энергия (в единицах модели)', fontsize=12)
        ax6.set_title('Энергетическая диаграмма нейтрона', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_pion(self):
        """Визуализация пиона"""
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('ПИОН π⁺: Структура волновых функций (u-anti-d)', fontsize=16, fontweight='bold')
        
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Осцилляции кварков
        ax1 = fig.add_subplot(gs[0, 0])
        x = np.linspace(0, 4*np.pi, 1000)
        
        u_wf = self.quark_oscillation(x, 'u', 0)
        anti_d_wf = -self.quark_oscillation(x, 'anti_d', params['phase_shift'])
        
        ax1.plot(x, u_wf, 'r-', linewidth=2, alpha=0.7, label='u-кварк (фаза=0)')
        ax1.plot(x, anti_d_wf, 'b-', linewidth=2, alpha=0.7, label='anti-d кварк (фаза=π)')
        
        total_wf = u_wf + anti_d_wf
        ax1.plot(x, total_wf, 'k-', linewidth=3, alpha=0.9, label='Суммарная волновая функция')
        
        ax1.set_xlabel('Время/Координата', fontsize=12)
        ax1.set_ylabel('Амплитуда', fontsize=12)
        ax1.set_title('Осцилляции кварков пиона', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Противофазность
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Увеличим масштаб
        x_detail = np.linspace(0, 2*np.pi, 500)
        u_detail = self.quark_oscillation(x_detail, 'u', 0)
        anti_d_detail = -self.quark_oscillation(x_detail, 'anti_d', params['phase_shift'])
        
        ax2.plot(x_detail, u_detail, 'r-', linewidth=2, alpha=0.7)
        ax2.plot(x_detail, anti_d_detail, 'b-', linewidth=2, alpha=0.7)
        
        # Заполнение между кривыми
        ax2.fill_between(x_detail, u_detail, anti_d_detail, 
                        where=(u_detail > anti_d_detail), 
                        color='red', alpha=0.2, label='Область u > anti-d')
        ax2.fill_between(x_detail, u_detail, anti_d_detail,
                        where=(u_detail < anti_d_detail),
                        color='blue', alpha=0.2, label='Область anti-d > u')
        
        ax2.set_xlabel('Время/Координата', fontsize=12)
        ax2.set_ylabel('Амплитуда', fontsize=12)
        ax2.set_title('Противофазность кварков в пионе', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # 3. Фазовая диаграмма
        ax3 = fig.add_subplot(gs[0, 2], projection='polar')
        
        phases = [0, params['phase_shift']]
        radii = [params['amp_u'], params['amp_d']]
        colors = ['red', 'blue']
        
        for phase, radius, color in zip(phases, radii, colors):
            ax3.plot([0, phase], [0, radius], color=color, linewidth=3, alpha=0.7)
            ax3.scatter(phase, radius, s=100, color=color, edgecolors='k', linewidth=1.5)
        
        ax3.set_title('Фазовая диаграмма пиона', fontsize=14, fontweight='bold', pad=20)
        ax3.grid(True)
        
        # 4. Пространственное распределение
        ax4 = fig.add_subplot(gs[1, 0])
        x_profile = np.linspace(0, 1, 500)
        
        u_profile = self.quark_potential(x_profile, 'u')
        anti_d_profile = self.quark_potential(x_profile, 'd')
        
        ax4.plot(x_profile, u_profile, 'r-', alpha=0.6, label='u-кварк')
        ax4.plot(x_profile, anti_d_profile, 'b-', alpha=0.6, label='anti-d кварк')
        
        total_profile = u_profile + anti_d_profile
        ax4.plot(x_profile, total_profile, 'k-', linewidth=2, alpha=0.8, label='Пион')
        
        ax4.fill_between(x_profile, total_profile, alpha=0.2, color='purple')
        
        ax4.set_xlabel('Координата', fontsize=12)
        ax4.set_ylabel('Потенциал/Плотность', fontsize=12)
        ax4.set_title('Пространственный профиль', fontsize=14, fontweight='bold')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        
        # 5. 3D структура
        ax5 = fig.add_subplot(gs[1, 1], projection='3d')
        X, Y, Z, quark_pos = self.calculate_3d_structure(['u', 'anti_d'], [0, params['phase_shift']])
        
        surf = ax5.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.8, 
                              linewidth=0, antialiased=True)
        
        for pos in quark_pos:
            ax5.scatter(pos[0], pos[1], 0.5, s=100, color='r' if pos[2]=='u' else 'b', 
                       marker='o', edgecolors='k', linewidth=1.5)
        
        ax5.set_xlabel('X', fontsize=10)
        ax5.set_ylabel('Y', fontsize=10)
        ax5.set_zlabel('Плотность', fontsize=10)
        ax5.set_title('3D структура пиона', fontsize=14, fontweight='bold')
        
        # 6. Энергетическая диаграмма
        ax6 = fig.add_subplot(gs[1, 2])
        
        components = ['u-кварк', 'anti-d кварк', 'Базовая масса', 'Энергия синхронизации (вычитается)']
        energies = [
            params['base_mass_u'] * params['freq_u'] * params['amp_u'],
            params['base_mass_d'] * params['freq_d'] * params['amp_d'],
            0,  # разделитель
            -params['coupling_meson']  # Отрицательная для мезонов
        ]
        
        colors = ['red', 'blue', 'gray', 'green']
        
        y_pos = np.arange(len(components))
        bars = ax6.barh(y_pos, energies, color=colors, alpha=0.7)
        
        for i, (bar, energy) in enumerate(zip(bars, energies)):
            width = bar.get_width()
            sign = '+' if energy >= 0 else ''
            ax6.text(width + (0.05 if energy >= 0 else -0.3), 
                    bar.get_y() + bar.get_height()/2.,
                    f'{sign}{energy:.3f}', ha='left' if energy >= 0 else 'right', va='center')
        
        ax6.set_yticks(y_pos)
        ax6.set_yticklabels(components)
        ax6.set_xlabel('Энергия (в единицах модели)', fontsize=12)
        ax6.set_title('Энергетическая диаграмма пиона', fontsize=14, fontweight='bold')
        ax6.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax6.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_comparison(self):
        """Сравнительная визуализация всех частиц"""
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle('СРАВНЕНИЕ ВОЛНОВЫХ ФУНКЦИЙ АДРОНОВ', fontsize=18, fontweight='bold')
        
        # 1. Сравнение суммарных волновых функций
        ax1 = fig.add_subplot(2, 3, 1)
        x = np.linspace(0, 3*np.pi, 1000)
        
        # Протон
        proton_wf = (2*self.quark_oscillation(x, 'u', 0) + 
                    self.quark_oscillation(x, 'd', np.pi/2))
        # Нейтрон
        neutron_wf = (self.quark_oscillation(x, 'u', 0) + 
                     2*self.quark_oscillation(x, 'd', np.pi/2))
        # Пион
        pion_wf = (self.quark_oscillation(x, 'u', 0) - 
                  self.quark_oscillation(x, 'd', params['phase_shift']))
        
        ax1.plot(x, proton_wf, 'r-', linewidth=2, label='Протон (uud)')
        ax1.plot(x, neutron_wf, 'b-', linewidth=2, label='Нейтрон (udd)')
        ax1.plot(x, pion_wf, 'g-', linewidth=2, label='Пион π⁺ (u-anti-d)')
        
        ax1.set_xlabel('Время/Координата', fontsize=12)
        ax1.set_ylabel('Амплитуда', fontsize=12)
        ax1.set_title('Суммарные волновые функции', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Спектр частот
        ax2 = fig.add_subplot(2, 3, 2)
        
        particles = ['Протон', 'Нейтрон', 'Пион']
        u_counts = [2, 1, 1]  # Количество u-кварков
        d_counts = [1, 2, 1]  # Количество d-кварков
        
        total_freq = []
        for u_count, d_count in zip(u_counts, d_counts):
            total_freq.append(u_count * params['freq_u'] + d_count * params['freq_d'])
        
        colors = ['red', 'blue', 'green']
        bars = ax2.bar(particles, total_freq, color=colors, alpha=0.7)
        
        for bar, freq in zip(bars, total_freq):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{freq:.3f}', ha='center', va='bottom')
        
        ax2.set_ylabel('Суммарная частота', fontsize=12)
        ax2.set_title('Спектр частот адронов', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Сравнение амплитуд
        ax3 = fig.add_subplot(2, 3, 3)
        
        total_amp = []
        for u_count, d_count in zip(u_counts, d_counts):
            total_amp.append(u_count * params['amp_u'] + d_count * params['amp_d'])
        
        bars = ax3.bar(particles, total_amp, color=colors, alpha=0.7)
        
        for bar, amp in zip(bars, total_amp):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{amp:.3f}', ha='center', va='bottom')
        
        ax3.set_ylabel('Суммарная амплитуда', fontsize=12)
        ax3.set_title('Суммарные амплитуды', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Энергии связи
        ax4 = fig.add_subplot(2, 3, 4)
        
        sync_energies = [
            params['coupling_proton'],
            params['coupling_neutron'],
            -params['coupling_meson']  # Отрицательная для пиона
        ]
        
        bars = ax4.bar(particles, sync_energies, color=colors, alpha=0.7)
        
        for bar, energy in zip(bars, sync_energies):
            height = bar.get_height()
            sign = '+' if energy >= 0 else ''
            ax4.text(bar.get_x() + bar.get_width()/2., 
                    height + (0.1 if energy >= 0 else -0.12),
                    f'{sign}{energy:.3f}', ha='center', va='bottom' if energy >= 0 else 'top')
        
        ax4.set_ylabel('Энергия синхронизации', fontsize=12)
        ax4.set_title('Энергии связи адронов', fontsize=14, fontweight='bold')
        ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Отношения coupling
        ax5 = fig.add_subplot(2, 3, 5)
        
        couplings = [
            params['coupling_proton'],
            params['coupling_neutron'],
            params['coupling_meson']
        ]
        
        bars = ax5.bar(particles, couplings, color=colors, alpha=0.7)
        
        for bar, coupling in zip(bars, couplings):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{coupling:.3f}', ha='center', va='bottom')
        
        ax5.set_ylabel('Coupling параметр', fontsize=12)
        ax5.set_title('Силы связи частиц', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Фазовая когерентность
        ax6 = fig.add_subplot(2, 3, 6)
        
        # Расчет когерентности
        coherences = []
        
        # Протон: [0, 0, π/2]
        proton_coh = (np.cos(0-0) + np.cos(0-np.pi/2) + np.cos(0-np.pi/2)) / 3
        # Нейтрон: [0, π/2, π/2]
        neutron_coh = (np.cos(0-np.pi/2) + np.cos(0-np.pi/2) + np.cos(np.pi/2-np.pi/2)) / 3
        # Пион: [0, π]
        pion_coh = np.cos(0-params['phase_shift'])
        
        coherences = [proton_coh, neutron_coh, pion_coh]
        
        bars = ax6.bar(particles, coherences, color=colors, alpha=0.7)
        
        for bar, coh in zip(bars, coherences):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{coh:.3f}', ha='center', va='bottom')
        
        ax6.set_ylabel('Фазовая когерентность', fontsize=12)
        ax6.set_title('Когерентность колебаний', fontsize=14, fontweight='bold')
        ax6.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()

# Запуск визуализации
if __name__ == "__main__":
    print("="*80)
    print("ВИЗУАЛИЗАЦИЯ ВОЛНОВЫХ ФУНКЦИЙ КВАРКОВ В АДРОНАХ")
    print("Параметры модели v6.1")
    print("="*80)
    
    # Создаем визуализатор
    visualizer = QuarkWavefunction(params)
    
    # Визуализация всех частиц
    print("\n1. Визуализация протона...")
    visualizer.visualize_proton()
    
    print("\n2. Визуализация нейтрона...")
    visualizer.visualize_neutron()
    
    print("\n3. Визуализация пиона...")
    visualizer.visualize_pion()
    
    print("\n4. Сравнительная визуализация...")
    visualizer.visualize_comparison()
    
    print("\n" + "="*80)
    print("ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА")
    print("="*80)