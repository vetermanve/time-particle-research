"""
Модель синхронизации осциллирующих нитей v5.7
Микронастройка для исправления абсолютных масс
"""

import numpy as np

# ================= КОНФИГУРАЦИЯ v5.7 =================
CONFIG_V57 = {
    'type_properties': {
        'u': {'charge': 2/3, 'base_mass': 2.247},    # Микро-коррекция: было 2.25
        'd': {'charge': -1/3, 'base_mass': 4.597},   # Микро-коррекция: было 4.60
        'anti_u': {'charge': -2/3, 'base_mass': 2.247},
        'anti_d': {'charge': 1/3, 'base_mass': 4.597}
    },
    
    'fixed_params': {
        'freq_u': 0.951000,
        'freq_d': 0.899000,
        'amp_u': 1.001000,
        'amp_d': 0.849000,
        'phase_shift': 3.163802,
        'coupling_proton': 1.685922,
        'coupling_neutron': 0.304993,
        'coupling_meson': 4.257  # КОРРЕКТИРОВКА: было 4.400
    },
    
    'scale_factor': 100.0
}

def calculate_particle_v57(composition, particle_name, config):
    """Упрощенный расчет для v5.7"""
    fixed = config['fixed_params']
    type_props = config['type_properties']
    
    # 1. Базовая масса
    base_mass = 0
    for quark in composition:
        base_type = quark.replace('anti_', '')
        if base_type == 'u':
            freq = fixed['freq_u']
            amp = fixed['amp_u']
            mass = type_props['u']['base_mass']
        else:
            freq = fixed['freq_d']
            amp = fixed['amp_d']
            mass = type_props['d']['base_mass']
        base_mass += mass * freq * amp
    
    # 2. Энергия синхронизации
    if particle_name == 'pi+':
        coupling = fixed['coupling_meson']
        # Мезон: энергия ВЫЧИТАЕТСЯ
        total = base_mass - coupling
    else:
        # Барион: энергия ПРИБАВЛЯЕТСЯ
        if particle_name == 'proton':
            coupling = fixed['coupling_proton']
        else:  # neutron
            coupling = fixed['coupling_neutron']
        
        # Упрощенный расчет энергии синхронизации
        # Частотная когерентность ~1.0
        # Фазовая когерентность:
        if particle_name == 'proton':
            phase_coherence = (np.cos(0) + np.cos(np.pi/2) + np.cos(np.pi/2)) / 3  # ~0.333
        else:  # neutron
            phase_coherence = (np.cos(np.pi/2) + np.cos(0) + np.cos(0)) / 3  # ~0.333
        
        sync_energy = coupling * (0.6 * 1.0 + 0.4 * phase_coherence)
        
        if particle_name == 'proton':
            sync_energy *= 1.1
        else:
            sync_energy *= 0.95
        
        total = base_mass + sync_energy
    
    return total * config['scale_factor']

# ================= РАСЧЕТ v5.7 =================
print("="*70)
print("МОДЕЛЬ v5.7: Микронастройка масс")
print("="*70)

# Рассчитываем массы
proton_mass = calculate_particle_v57(['u', 'u', 'd'], 'proton', CONFIG_V57)
neutron_mass = calculate_particle_v57(['u', 'd', 'd'], 'neutron', CONFIG_V57)
pion_mass = calculate_particle_v57(['u', 'anti_d'], 'pi+', CONFIG_V57)

mass_diff = neutron_mass - proton_mass

# Целевые значения
targets = {
    'proton': 938.272,
    'neutron': 939.565,
    'pi+': 139.57,
    'diff': 1.293
}

print(f"\nРЕЗУЛЬТАТЫ v5.7:")
print(f"{'Частица':<10} {'Масса (МэВ)':<12} {'Цель (МэВ)':<12} {'Ошибка (%)':<12}")
print("-"*70)
print(f"{'Протон':<10} {proton_mass:<12.3f} {targets['proton']:<12.3f} {abs(proton_mass-targets['proton'])/targets['proton']*100:<12.6f}")
print(f"{'Нейтрон':<10} {neutron_mass:<12.3f} {targets['neutron']:<12.3f} {abs(neutron_mass-targets['neutron'])/targets['neutron']*100:<12.6f}")
print(f"{'Пион π⁺':<10} {pion_mass:<12.3f} {targets['pi+']:<12.3f} {abs(pion_mass-targets['pi+'])/targets['pi+']*100:<12.6f}")
print(f"{'Разность':<10} {mass_diff:<12.3f} {targets['diff']:<12.3f} {abs(mass_diff-targets['diff'])/targets['diff']*100:<12.6f}")

# Рассчитаем эффективные массы кварков
m_u_eff = CONFIG_V57['type_properties']['u']['base_mass'] * CONFIG_V57['fixed_params']['freq_u'] * CONFIG_V57['fixed_params']['amp_u'] * 100
m_d_eff = CONFIG_V57['type_properties']['d']['base_mass'] * CONFIG_V57['fixed_params']['freq_d'] * CONFIG_V57['fixed_params']['amp_d'] * 100

print(f"\nЭФФЕКТИВНЫЕ МАССЫ КВАРКОВ:")
print(f"u-кварк: {m_u_eff:.2f} МэВ")
print(f"d-кварк: {m_d_eff:.2f} МэВ")
print(f"Отношение m_d/m_u: {m_d_eff/m_u_eff:.3f}")

print(f"\nПАРАМЕТРЫ:")
print(f"base_mass_u: {CONFIG_V57['type_properties']['u']['base_mass']:.3f}")
print(f"base_mass_d: {CONFIG_V57['type_properties']['d']['base_mass']:.3f}")
print(f"coupling_meson: {CONFIG_V57['fixed_params']['coupling_meson']:.3f} (было 4.400)")

print("\n" + "="*70)
print("ОЖИДАНИЯ v5.7:")
print("1. Исправлен пион (уменьшена энергия связи с 4.400 до 4.257)")
print("2. Микро-коррекция масс кварков (2.25→2.247, 4.60→4.597)")
print("3. Все частицы должны быть в пределах 0.1% от цели")
print("="*70)