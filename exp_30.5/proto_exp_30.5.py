"""
ЭКСПЕРИМЕНТ 30.5 - ИСПРАВЛЕННАЯ ВЕРСИЯ
Ключевые исправления:
1. Правильные дробные заряды кварков
2. Правильная массовая шкала
3. Правильная классификация адронов
4. Оптимизированный алгоритм поиска
"""

import numpy as np
import random
import math
import time
from collections import Counter

class PreonV5:
    """ПРЕОН с правильными зарядами"""
    
    def __init__(self, preon_id: int):
        self.id = preon_id
        
        # Три типа преонов с разной вероятностью:
        # A: +1/3 (для создания u-кварков) - 40%
        # B: -1/3 (для d-кварков) - 40%
        # C: 0 (заполнитель) - 20%
        
        rand = random.random()
        if rand < 0.4:
            self.charge = 1/3  # Тип A
            self.color = random.choice(['R', 'G', 'B'])
        elif rand < 0.8:
            self.charge = -1/3  # Тип B
            self.color = random.choice(['R', 'G', 'B'])
        else:
            self.charge = 0  # Тип C
            self.color = 'neutral'
        
        # Резонансный код
        self.code = [random.random() for _ in range(3)]
        
        # Базовая масса
        self.base_mass = random.uniform(0.8, 1.2)
        
        # Фаза
        self.phase = random.uniform(0, 2 * math.pi)

class QuarkV5:
    """КВАРК с правильными свойствами"""
    
    def __init__(self, preon_triplet):
        self.preons = preon_triplet
        
        # Заряд = сумма зарядов преонов
        self.charge = sum(p.charge for p in preon_triplet)
        
        # Определяем тип кварка
        if abs(self.charge - 2/3) < 0.01:
            self.type = 'u'  # u-кварк
        elif abs(self.charge + 1/3) < 0.01:
            self.type = 'd'  # d-кварк
        else:
            self.type = 'other'
        
        # Цвет кварка
        colors = [p.color for p in preon_triplet if p.color != 'neutral']
        if colors:
            self.color = max(set(colors), key=colors.count)
        else:
            self.color = random.choice(['R', 'G', 'B'])
        
        # Масса кварка
        self.mass = sum(p.base_mass for p in preon_triplet) * 10
    
    def __repr__(self):
        return f"Quark[{self.type}, charge={self.charge:.3f}, mass={self.mass:.1f}]"

class HadronV5:
    """АДРОН с правильной классификацией"""
    
    def __init__(self, quark_triplet):
        self.quarks = quark_triplet
        
        # Проверка цветовой нейтральности
        colors = [q.color for q in quark_triplet]
        self.is_color_neutral = (len(set(colors)) >= 3)  # R, G, B все разные
        
        if self.is_color_neutral:
            # Заряд адрона
            self.charge = sum(q.charge for q in quark_triplet)
            
            # Масса адрона
            base_mass = sum(q.mass for q in quark_triplet)
            self.mass = base_mass * 3  # Энергия связи
            
            # Определяем тип
            self.type = self._determine_type()
        else:
            self.charge = 0
            self.mass = 0
            self.type = "invalid"
    
    def _determine_type(self):
        """Определяем тип адрона"""
        rounded_charge = round(self.charge)
        
        # Состав кварков
        quark_types = [q.type for q in self.quarks]
        type_counter = Counter(quark_types)
        
        # Протон: uud
        if (abs(rounded_charge - 1.0) < 0.1 and 
            type_counter.get('u', 0) == 2 and 
            type_counter.get('d', 0) == 1):
            return "proton"
        
        # Нейтрон: udd  
        elif (abs(rounded_charge - 0.0) < 0.1 and
              type_counter.get('u', 0) == 1 and
              type_counter.get('d', 0) == 2):
            return "neutron"
        
        # Заряженные барионы
        elif abs(rounded_charge - 1.0) < 0.1:
            return "charged_baryon"
        
        # Нейтральные барионы
        elif abs(rounded_charge - 0.0) < 0.1:
            return "neutral_baryon"
        
        # Отрицательные барионы
        elif abs(rounded_charge + 1.0) < 0.1:
            return "negative_baryon"
        
        else:
            return "exotic_baryon"
    
    def __repr__(self):
        if self.is_color_neutral:
            return f"Hadron[{self.type}, charge={self.charge:.1f}, mass={self.mass:.1f}]"
        else:
            return "Hadron[INVALID]"

def experiment_30_5():
    """Запуск исправленного эксперимента"""
    print("=" * 80)
    print("🚀 ЭКСПЕРИМЕНТ 30.5 - ИСПРАВЛЕННАЯ ВЕРСИЯ")
    print("=" * 80)
    
    # Параметры
    NUM_PREONS = 500
    NUM_CHECKS = 20000
    
    # 1. Создаем преоны
    print(f"Создание {NUM_PREONS} преонов...")
    preons = [PreonV5(i) for i in range(NUM_PREONS)]
    
    # 2. Ищем кварки (тройки преонов)
    print(f"Поиск кварков...")
    quarks = []
    used_preons = set()
    
    # Простой алгоритм: ищем тройки с подходящими зарядами
    for i in range(NUM_CHECKS):
        # Выбираем случайную тройку
        idx1, idx2, idx3 = random.sample(range(len(preons)), 3)
        
        # Пропускаем если преоны уже используются
        if idx1 in used_preons or idx2 in used_preons or idx3 in used_preons:
            continue
        
        # Создаем кварк из тройки преонов
        triplet = [preons[idx1], preons[idx2], preons[idx3]]
        quark = QuarkV5(triplet)
        
        # Принимаем только u и d кварки
        if quark.type in ['u', 'd']:
            quarks.append(quark)
            used_preons.update([idx1, idx2, idx3])
            
            if len(quarks) >= 100:  # Ограничим 100 кварками
                break
    
    print(f"Найдено {len(quarks)} кварков")
    
    # Статистика кварков
    quark_types = [q.type for q in quarks]
    type_counts = Counter(quark_types)
    print(f"  u-кварки: {type_counts.get('u', 0)}")
    print(f"  d-кварки: {type_counts.get('d', 0)}")
    print(f"  другие: {type_counts.get('other', 0)}")
    
    # 3. Строим адроны из кварков
    print("\nПостроение адронов...")
    hadrons = []
    
    # Группируем кварки по цвету
    quarks_by_color = {'R': [], 'G': [], 'B': []}
    for quark in quarks:
        if quark.color in quarks_by_color:
            quarks_by_color[quark.color].append(quark)
    
    # Строим цветонейтральные комбинации
    max_combinations = 100
    combinations_found = 0
    
    for r_q in quarks_by_color['R'][:10]:
        for g_q in quarks_by_color['G'][:10]:
            for b_q in quarks_by_color['B'][:10]:
                if combinations_found >= max_combinations:
                    break
                
                # Проверяем, что кварки разные
                if (r_q is g_q or r_q is b_q or g_q is b_q):
                    continue
                
                hadron = HadronV5([r_q, g_q, b_q])
                if hadron.is_color_neutral:
                    hadrons.append(hadron)
                    combinations_found += 1
    
    print(f"Построено {len(hadrons)} адронов")
    
    # 4. Анализ результатов
    print("\n📊 АНАЛИЗ РЕЗУЛЬТАТОВ:")
    
    hadron_types = [h.type for h in hadrons]
    type_counts = Counter(hadron_types)
    
    for hadron_type, count in type_counts.items():
        print(f"  {hadron_type}: {count}")
    
    # Найденные протоны и нейтроны
    protons = [h for h in hadrons if h.type == "proton"]
    neutrons = [h for h in hadrons if h.type == "neutron"]
    
    if protons:
        print(f"\n🎯 НАЙДЕНО ПРОТОНОВ: {len(protons)}")
        for i, p in enumerate(protons[:3], 1):
            print(f"  {i}. Масса: {p.mass:.1f} (цель: 938.3 МэВ)")
            print(f"     Заряд: {p.charge:.1f}")
    
    if neutrons:
        print(f"\n🎯 НАЙДЕНО НЕЙТРОНОВ: {len(neutrons)}")
        for i, n in enumerate(neutrons[:3], 1):
            print(f"  {i}. Масса: {n.mass:.1f} (цель: 939.6 МэВ)")
            print(f"     Заряд: {n.charge:.1f}")
    
    # Средние массы
    if hadrons:
        masses = [h.mass for h in hadrons if h.is_color_neutral]
        charges = [h.charge for h in hadrons if h.is_color_neutral]
        
        print(f"\n📈 СРЕДНИЕ ПОКАЗАТЕЛИ:")
        print(f"  Средняя масса адрона: {np.mean(masses):.1f} МэВ")
        print(f"  Средний заряд адрона: {np.mean(charges):.2f}")
    
    return hadrons

# Запуск эксперимента
if __name__ == "__main__":
    hadrons = experiment_30_5()