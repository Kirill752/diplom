import numpy as np
import pyvista as pv
from typing import List, Tuple, Dict
import time

import pickle
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Point:
    def __init__(self, x: float, y: float, z: float = 0):
        self.x = x
        self.y = y
        self.z = z
    
    def GetCoords(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)
    
    def __repr__(self):
        return f"Point({self.x}, {self.y}, {self.z})"

class Box:
    """Класс для создания параллелепипеда"""
    
    def __init__(self, x: float, y: float, z: float, dx: float, dy: float, dz: float, mesh_size: float = None):
        self.x = x
        self.y = y
        self.z = z
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.mesh_size = mesh_size
    
    def GetCenter(self) -> Point:
        return Point(self.x + self.dx/2, self.y + self.dy/2, self.z + self.dz/2)
    
    def GetOrigin(self) -> Point:
        """Возвращает начальную точку"""
        return Point(self.x, self.y, self.z)
    
    def GetDimensions(self) -> Tuple[float, float, float]:
        """Возвращает размеры параллелепипеда"""
        return (self.dx, self.dy, self.dz)

    def GetCoords(self) -> Tuple[float, float, float, float, float, float]:
        """Возвращает координаты (x, y, z, dx, dy, dz)"""
        return (self.x, self.y, self.z, self.dx, self.dy, self.dz)
    
    def GetBounds(self) -> List[float]:
        """Возвращает границы в формате [x_min, x_max, y_min, y_max, z_min, z_max]"""
        return [self.x, self.x + self.dx, self.y, self.y + self.dy, self.z, self.z + self.dz]
    
    def __repr__(self):
        return f"Box(x={self.x}, y={self.y}, z={self.z}, dx={self.dx}, dy={self.dy}, dz={self.dz})"

class Envelope:
    """Класс для создания огибающего электрода вокруг узкой части наномостика"""
    
    def __init__(self, bridge_center_x: float, bridge_center_y: float, bridge_center_z: float,
                 grip_width: float, grip_height: float, envelope_gap: float, 
                 envelope_thickness: float, envelope_length: float,
                 mesh_size: float):
        """
        Args:
            bridge_center_x, y, z: центр узкой части наномостика
            grip_width: ширина узкой части наномостика
            grip_height: высота узкой части наномостика  
            envelope_gap: зазор между электродом и наномостиком
            envelope_thickness: толщина слоя
            envelope_length: длина слоя
            mesh_size: размер сетки
        """
        self.bridge_center_x = bridge_center_x
        self.bridge_center_y = bridge_center_y
        self.bridge_center_z = bridge_center_z
        self.grip_width = grip_width
        self.grip_height = grip_height
        self.envelope_gap = envelope_gap
        self.envelope_length = envelope_length
        self.envelope_thickness = envelope_thickness
        self.mesh_size = mesh_size
        
        self.volumes = []
        
    def GetVolumes(self):
        return self.volumes

    def create_c_shaped(self):
        # Убираем зазор - электрод плотно прилегает к наномостику
        width = self.grip_width + 2 * self.envelope_thickness  # убираем envelope_gap
        height = self.grip_height  # убираем envelope_gap

        # Левая вертикальная часть
        self.volumes.append(Box(
            self.bridge_center_x - self.envelope_length/2,
            self.bridge_center_y - width/2,
            self.bridge_center_z,
            self.envelope_length,
            self.envelope_thickness,
            height + self.envelope_thickness,
            self.mesh_size
        ))

        # Верхняя горизонтальная часть
        self.volumes.append(Box(
            self.bridge_center_x - self.envelope_length/2,
            self.bridge_center_y - self.grip_width/2,
            self.bridge_center_z + height,
            self.envelope_length,
            self.grip_width,
            self.envelope_thickness,
            self.mesh_size
        ))

        # Правая вертикальная часть
        self.volumes.append(Box(
            self.bridge_center_x - self.envelope_length/2,
            self.bridge_center_y + width/2 - self.envelope_thickness,
            self.bridge_center_z,
            self.envelope_length,
            self.envelope_thickness,
            height + self.envelope_thickness,
            self.mesh_size
        ))


class NanoBridge:
    """Класс для создания наномостика в форме собачей кости"""
    
    def __init__(self, grip_length=10, grip_width=5, grip_height=5,
                 end_length=5, end_width=10, mesh_size=0.2):
        self.grip_length = grip_length
        self.grip_width = grip_width
        self.grip_height = grip_height
        self.end_length = end_length
        self.end_width = end_width
        self.mesh_size = mesh_size
        
        self.boxes = []
        self.oxide_boxes = []
    
    def create_nano_bridge_geometry(self):
        """Создает геометрию наномостика из трех Box"""
        gl, gw, gh, el, ew, ms = self.grip_length, self.grip_width, self.grip_height, self.end_length, self.end_width, self.mesh_size

        left_box = Box(
            -gl/2 - el, -ew/2, 0,
            el, ew, gh,
            ms
        )
        
        center_box = Box(
            -gl/2, -gw/2, 0,
            gl, gw, gh,
            ms
        )
        
        right_box = Box(
            gl/2, -ew/2, 0,
            el, ew, gh,
            ms
        )
        
        self.boxes = [left_box, center_box, right_box]
        return self.boxes

    def create_oxide(self):
        (left, center, right) = self.get_boxes()
        leftCenter = left.GetCenter().GetCoords()
        centerCenter = center.GetCenter().GetCoords()
        rightCenter = right.GetCenter().GetCoords()
        
        # Увеличим толщину оксида для лучшей визуализации
        oxide_thickness = 1.0  # было 0.3
        
        ox1 = Envelope(
            bridge_center_x=centerCenter[0], bridge_center_y=centerCenter[1], bridge_center_z=0,
            grip_width=self.grip_width, grip_height=self.grip_height,
            envelope_gap=0, envelope_thickness=oxide_thickness, 
            envelope_length=self.grip_length,
            mesh_size=0.4
        )
        ox1.create_c_shaped()  
        
        ox2 = Envelope(
            bridge_center_x=leftCenter[0], bridge_center_y=leftCenter[1], bridge_center_z=0,
            grip_width=self.end_width, grip_height=self.grip_height,
            envelope_gap=0, envelope_thickness=oxide_thickness,
            envelope_length=self.end_length,
            mesh_size=0.4
        )
        ox2.create_c_shaped()  
        
        ox3 = Envelope(
            bridge_center_x=rightCenter[0], bridge_center_y=rightCenter[1], bridge_center_z=0,
            grip_width=self.end_width, grip_height=self.grip_height,
            envelope_gap=0, envelope_thickness=oxide_thickness,
            envelope_length=self.end_length,
            mesh_size=0.4
        )
        ox3.create_c_shaped()

        # Добавляем все оксидные боксы в один список
        self.oxide_boxes = ox1.GetVolumes() + ox2.GetVolumes() + ox3.GetVolumes()
        print(f"Создано оксидных боксов: {len(self.oxide_boxes)}")
    
    def get_boxes(self) -> List[Box]:
        return self.boxes

    def get_oxide_boxes(self) -> List[Box]:
        return self.oxide_boxes
    
    def get_total_bounds(self) -> List[float]:
        """Возвращает общие границы наномостика"""
        all_boxes = self.boxes + self.oxide_boxes
        all_bounds = [box.GetBounds() for box in all_boxes]
        
        bounds_array = np.array(all_bounds)
        return [
            bounds_array[:, 0].min(),  # x_min
            bounds_array[:, 1].max(),  # x_max
            bounds_array[:, 2].min(),  # y_min
            bounds_array[:, 3].max(),  # y_max
            bounds_array[:, 4].min(),  # z_min
            bounds_array[:, 5].max()   # z_max
        ]

class Substrate:
    """Класс для создания подложки"""
    
    def __init__(self, thickness=5, padding=10, mesh_size=1.0):
        self.thickness = thickness
        self.padding = padding
        self.mesh_size = mesh_size
        self.box = None
    
    def create_substrate(self, nano_bounds):
        """Создает подложку на основе границ наномостика"""
        x_min, x_max, y_min, y_max, z_min, z_max = nano_bounds
        
        # Подложка располагается под наномостиком
        substrate_z = z_min - self.thickness
        
        # Подложка расширяется за пределы наномостика
        substrate_x_min = x_min - self.padding
        substrate_x_max = x_max + self.padding
        substrate_y_min = y_min - self.padding
        substrate_y_max = y_max + self.padding
        
        substrate_dx = substrate_x_max - substrate_x_min
        substrate_dy = substrate_y_max - substrate_y_min
        
        self.box = Box(
            substrate_x_min, substrate_y_min, substrate_z,
            substrate_dx, substrate_dy, self.thickness,
            self.mesh_size
        )
        
        return self.box

class AirEnvironment:
    """Класс для создания окружающей среды (воздуха)"""
    
    def __init__(self, padding=15, height_above=20, height_below=10, mesh_size=1.5):
        self.padding = padding
        self.height_above = height_above
        self.height_below = height_below
        self.mesh_size = mesh_size
        self.box = None
    
    def create_air_environment(self, nano_bounds):
        """Создает окружающую среду на основе границ наномостика"""
        x_min, x_max, y_min, y_max, z_min, z_max = nano_bounds
        
        # Окружающая среда охватывает все элементы
        air_x_min = x_min - self.padding
        air_x_max = x_max + self.padding
        air_y_min = y_min - self.padding
        air_y_max = y_max + self.padding
        air_z_min = z_min - self.height_below
        air_z_max = z_max + self.height_above
        
        air_dx = air_x_max - air_x_min
        air_dy = air_y_max - air_y_min
        air_dz = air_z_max - air_z_min
        
        self.box = Box(
            air_x_min, air_y_min, air_z_min,
            air_dx, air_dy, air_dz,
            self.mesh_size
        )
        
        return self.box

class GateElectrode:
    """Класс для создания затворного электрода"""
    
    def __init__(self, center_point: Point, width=15, height=8, length=5, thickness=2, mesh_size=0.6):
        self.center = center_point
        self.width = width
        self.height = height
        self.length = length
        self.thickness = thickness
        self.mesh_size = mesh_size
        self.boxes = []
    
    def create_gate_electrode(self):
        """Создает затворный электрод в форме буквы C"""
        # Создаем оболочку для электрода БЕЗ зазора
        gate = Envelope(
            bridge_center_x=self.center.x,
            bridge_center_y=self.center.y, 
            bridge_center_z=0,
            grip_width=self.width,
            grip_height=self.height,
            envelope_gap=0,  # НУЛЕВОЙ зазор - электрод плотно прилегает
            envelope_thickness=self.thickness,
            envelope_length=self.length,
            mesh_size=self.mesh_size
        )
        
        gate.create_c_shaped()
        self.boxes = gate.GetVolumes()

        mainPart = gate.GetVolumes()
        
        # Улучшаем дополнительные части электрода
        leftHand = Box(
            mainPart[0].GetOrigin().x, 
            mainPart[0].GetOrigin().y, 
            mainPart[0].GetOrigin().z,
            mainPart[0].GetDimensions()[0], 
            -self.thickness * 3, 
            self.thickness * 2,
            self.mesh_size
        )
        
        rightHand = Box(
            mainPart[2].GetOrigin().x, 
            mainPart[2].GetOrigin().y, 
            mainPart[2].GetOrigin().z,
            mainPart[2].GetDimensions()[0], 
            self.thickness * 3, 
            self.thickness * 2,
            self.mesh_size
        )
        
        self.boxes.extend([leftHand, rightHand])
        
        print(f"Создано частей электрода: {len(self.boxes)}")
        for i, box in enumerate(self.boxes):
            print(f"  Часть {i}: {box.GetBounds()}")
        
        return self.boxes

class CompleteNanoSystem:
    """Класс для создания полной системы: наномостик + подложка + воздух + электрод"""
    
    def __init__(self):
        self.nano_bridge = None
        self.substrate = None
        self.air_environment = None
        self.gate_electrode = None
        self.all_components = []
    
    def create_complete_system(self):
        """Создает полную систему"""
        # 1. Создаем наномостик
        self.nano_bridge = NanoBridge(
            grip_length=20,
            grip_width=8,
            grip_height=6,
            end_length=8,
            end_width=12,
            mesh_size=0.2
        )
        
        self.nano_bridge.create_nano_bridge_geometry()
        self.nano_bridge.create_oxide()
        
        # 2. Создаем подложку
        nano_bounds = self.nano_bridge.get_total_bounds()
        self.substrate = Substrate(thickness=5, padding=10, mesh_size=1.0)
        substrate_box = self.substrate.create_substrate(nano_bounds)
        
        # 3. Создаем окружающую среду (воздух)
        # Обновляем границы с учетом подложки
        all_bounds_so_far = nano_bounds + substrate_box.GetBounds()
        bounds_array = np.array([all_bounds_so_far[:6], all_bounds_so_far[6:]])
        total_bounds = [
            bounds_array[:, 0].min(), bounds_array[:, 1].max(),
            bounds_array[:, 2].min(), bounds_array[:, 3].max(), 
            bounds_array[:, 4].min(), bounds_array[:, 5].max()
        ]
        
        self.air_environment = AirEnvironment(padding=15, height_above=20, height_below=10, mesh_size=1.5)
        air_box = self.air_environment.create_air_environment(total_bounds)
        
        # 4. Создаем затворный электрод
        (_, center_box, _) = self.nano_bridge.get_boxes()
        center_point = center_box.GetCenter()
        
        self.gate_electrode = GateElectrode(
            center_point=center_point,
            width=10,   # Теперь ширина равна ширине центральной части наномостика
            height=7,  # Высота равна высоте наномостика
            length=8, # Длина равна длине центральной части
            thickness=2,
            mesh_size=0.6
        )
        gate_boxes = self.gate_electrode.create_gate_electrode()
        
        # Собираем все компоненты
        self.all_components = {
            'nano_bridge': self.nano_bridge.get_boxes(),
            'oxide': self.nano_bridge.get_oxide_boxes(),
            'substrate': [substrate_box],
            'air': [air_box],
            'gate': gate_boxes
        }
        
        return self.all_components

class ElectricFieldSolver:
    """Класс для решения уравнения Лапласа и расчета электрического поля"""
    
    def __init__(self, nano_system: CompleteNanoSystem, grid_resolution=50):
        self.nano_system = nano_system
        self.grid_resolution = grid_resolution
        self.potential = None
        self.electric_field = None
        self.grid = None
        self.mask = None
        # Добавляем относительные диэлектрические проницаемости
        self.dielectric_constants = {
            'air': 1.0,
            'oxide': 3.9,  # SiO2
            'nano_bridge': 11.7,  # Кремний
            'substrate': 11.7,  # Кремниевая подложка
            'gate': 1e10  # Металл (очень большое значение)
        }
        
    def create_computational_grid(self):
        """Создает вычислительную сетку для решения уравнения Лапласа"""
        air_box = self.nano_system.air_environment.box
        air_bounds = air_box.GetBounds()
        
        # Создаем регулярную сетку
        x = np.linspace(air_bounds[0], air_bounds[1], self.grid_resolution)
        y = np.linspace(air_bounds[2], air_bounds[3], self.grid_resolution)
        z = np.linspace(air_bounds[4], air_bounds[5], self.grid_resolution)
        
        self.grid = np.meshgrid(x, y, z, indexing='ij')
        return self.grid
    
    def get_dielectric_constant_at_point(self, point):
        """Возвращает диэлектрическую проницаемость в точке"""
        # Проверяем материалы в порядке приоритета (металлы -> диэлектрики -> воздух)
        if self.is_point_in_material(point, 'gate'):
            return self.dielectric_constants['gate']  # Металл
        elif self.is_point_in_material(point, 'substrate'):
            return self.dielectric_constants['substrate']  # Подложка (проводящая)
        elif self.is_point_in_material(point, 'nano_bridge'):
            return self.dielectric_constants['nano_bridge']  # Наномостик (полупроводник)
        elif self.is_point_in_material(point, 'oxide'):
            return self.dielectric_constants['oxide']  # Оксид
        else:
            return self.dielectric_constants['air']  # Воздух
    
    def create_boundary_mask(self, gate_potential=10.0):
        """Создает маску граничных условий - ТОЛЬКО для проводников"""
        X, Y, Z = self.grid
        self.mask = np.zeros(X.shape, dtype=bool)
        boundary_values = np.zeros(X.shape)
        
        print("Создание маски граничных условий...")
        
        # Счетчики для отладки
        gate_points = 0
        ground_points = 0
        
        # ТОЛЬКО проводящие материалы имеют фиксированный потенциал
        # Подложка - земля (0V) - проводник
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    point = (X[i,j,k], Y[i,j,k], Z[i,j,k])
                    
                    # Подложка - земля (0V) - ПРОВОДНИК
                    if self.is_point_in_material(point, 'substrate'):
                        self.mask[i,j,k] = True
                        boundary_values[i,j,k] = 0.0
                        ground_points += 1
        
        # Электрод - заданный потенциал (10V) - ПРОВОДНИК
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    point = (X[i,j,k], Y[i,j,k], Z[i,j,k])
                    
                    # Электрод - заданный потенциал - ПРОВОДНИК
                    if self.is_point_in_material(point, 'gate'):
                        self.mask[i,j,k] = True
                        boundary_values[i,j,k] = gate_potential
                        gate_points += 1
        
        print(f"Точек на электроде (проводник): {gate_points:,}")
        print(f"Точек на земле (проводник): {ground_points:,}")
        print("Наномостик и оксид - ДИЭЛЕКТРИКИ (без фиксированного потенциала)")
        
        # Проверяем, что электрод имеет правильный потенциал
        gate_indices = np.where(self.mask & (boundary_values == gate_potential))
        if len(gate_indices[0]) > 0:
            print(f"Потенциал электрода установлен правильно: {gate_potential}V")
        else:
            print("ВНИМАНИЕ: Не найдено точек с потенциалом электрода!")
            
        return boundary_values
    
    def is_point_in_material(self, point, material_type):
        """Проверяет, находится ли точка внутри указанного материала"""
        x, y, z = point
        
        components = self.nano_system.all_components
        if material_type in components:
            for box in components[material_type]:
                bounds = box.GetBounds()
                # Используем более точную проверку с учетом знаков размеров
                x_min = min(bounds[0], bounds[1])
                x_max = max(bounds[0], bounds[1])
                y_min = min(bounds[2], bounds[3])
                y_max = max(bounds[2], bounds[3])
                z_min = min(bounds[4], bounds[5])
                z_max = max(bounds[4], bounds[5])
                
                if (x_min <= x <= x_max and 
                    y_min <= y <= y_max and 
                    z_min <= z <= z_max):
                    return True
        return False
    
    def solve_laplace_sor(self, gate_potential=10.0, omega=1.8, max_iter=1000, tolerance=1e-1):
        """Решает уравнение Лапласа с использованием SOR метода с учетом диэлектриков"""
        print("Создание вычислительной сетки...")
        self.create_computational_grid()
        X, Y, Z = self.grid
        
        print(f"Размер сетки: {X.shape}")
        
        # Создаем маску граничных условий (только для проводников)
        boundary_values = self.create_boundary_mask(gate_potential)
        
        # Инициализация потенциала
        self.potential = boundary_values.copy()
        
        print("Решение уравнения Лапласа методом SOR с учетом диэлектриков...")
        start_time = time.time()
        
        # SOR метод с учетом диэлектриков
        for iteration in range(max_iter):
            max_change = 0.0
            
            # Обновление потенциала во внутренних точках
            for i in range(1, X.shape[0]-1):
                for j in range(1, X.shape[1]-1):
                    for k in range(1, X.shape[2]-1):
                        # Пропускаем точки с фиксированными граничными условиями (проводники)
                        if self.mask[i,j,k]:
                            continue
                            
                        # Получаем диэлектрические проницаемости в соседних точках
                        eps_center = self.get_dielectric_constant_at_point((X[i,j,k], Y[i,j,k], Z[i,j,k]))
                        eps_x_prev = self.get_dielectric_constant_at_point((X[i-1,j,k], Y[i-1,j,k], Z[i-1,j,k]))
                        eps_x_next = self.get_dielectric_constant_at_point((X[i+1,j,k], Y[i+1,j,k], Z[i+1,j,k]))
                        eps_y_prev = self.get_dielectric_constant_at_point((X[i,j-1,k], Y[i,j-1,k], Z[i,j-1,k]))
                        eps_y_next = self.get_dielectric_constant_at_point((X[i,j+1,k], Y[i,j+1,k], Z[i,j+1,k]))
                        eps_z_prev = self.get_dielectric_constant_at_point((X[i,j,k-1], Y[i,j,k-1], Z[i,j,k-1]))
                        eps_z_next = self.get_dielectric_constant_at_point((X[i,j,k+1], Y[i,j,k+1], Z[i,j,k+1]))
                        
                        # Усредненные диэлектрические проницаемости на гранях
                        eps_x_avg_prev = (eps_center + eps_x_prev) / 2
                        eps_x_avg_next = (eps_center + eps_x_next) / 2
                        eps_y_avg_prev = (eps_center + eps_y_prev) / 2
                        eps_y_avg_next = (eps_center + eps_y_next) / 2
                        eps_z_avg_prev = (eps_center + eps_z_prev) / 2
                        eps_z_avg_next = (eps_center + eps_z_next) / 2
                        
                        # Вычисляем новое значение с учетом диэлектриков
                        numerator = (eps_x_avg_prev * self.potential[i-1,j,k] + 
                                   eps_x_avg_next * self.potential[i+1,j,k] +
                                   eps_y_avg_prev * self.potential[i,j-1,k] + 
                                   eps_y_avg_next * self.potential[i,j+1,k] +
                                   eps_z_avg_prev * self.potential[i,j,k-1] + 
                                   eps_z_avg_next * self.potential[i,j,k+1])
                        
                        denominator = (eps_x_avg_prev + eps_x_avg_next +
                                     eps_y_avg_prev + eps_y_avg_next +
                                     eps_z_avg_prev + eps_z_avg_next)
                        
                        if denominator > 0:
                            new_value = numerator / denominator
                            
                            # Применяем релаксацию (SOR)
                            change = omega * (new_value - self.potential[i,j,k])
                            self.potential[i,j,k] += change
                            
                            # Отслеживаем максимальное изменение
                            max_change = max(max_change, abs(change))
            
            # Проверка сходимости каждые 50 итераций
            if iteration % 10 == 0:
                print(f"Итерация {iteration}, максимальное изменение: {max_change:.2e}")
            
            if max_change < tolerance:
                end_time = time.time()
                print(f"Сходимость достигнута на итерации {iteration+1}")
                print(f"Время решения: {end_time - start_time:.2f} секунд")
                break
        
        if iteration == max_iter - 1:
            end_time = time.time()
            print(f"Достигнуто максимальное число итераций {max_iter}")
            print(f"Время решения: {end_time - start_time:.2f} секунд")
        
        print("Расчет электрического поля...")
        # Расчет электрического поля E = -∇φ
        self.calculate_electric_field()
        
        return self.potential
    
    def calculate_electric_field(self):
        """Вычисляет электрическое поле как градиент потенциала"""
        X, Y, Z = self.grid
        
        # Вычисление градиента
        Ex = -np.gradient(self.potential, axis=0)
        Ey = -np.gradient(self.potential, axis=1)
        Ez = -np.gradient(self.potential, axis=2)
        
        self.electric_field = (Ex, Ey, Ez)
        return self.electric_field
    
    def visualize_nanobridge_cross_sections(self):
        """Визуализация срезов распределения поля внутри наномостика"""
        if self.potential is None:
            print("Сначала необходимо решить уравнение Лапласа!")
            return
        
        X, Y, Z = self.grid
        
        # Создаем PyVista структурированную сетку
        grid = pv.StructuredGrid(X, Y, Z)
        grid.point_data["potential"] = self.potential.ravel(order='F')
        
        # Вычисляем модуль электрического поля
        Ex, Ey, Ez = self.electric_field
        E_magnitude = np.sqrt(Ex**2 + Ey**2 + Ez**2)
        grid.point_data["E_magnitude"] = E_magnitude.ravel(order='F')
        
        # Получаем границы наномостика для точных срезов
        nano_bridge = self.nano_system.nano_bridge
        center_box = nano_bridge.get_boxes()[1]  # Центральная часть
        center_bounds = center_box.GetBounds()
        
        # Координаты для срезов
        center_x = (center_bounds[0] + center_bounds[1]) / 2
        center_y = (center_bounds[2] + center_bounds[3]) / 2  
        center_z = (center_bounds[4] + center_bounds[5]) / 2
        
        # Создаем plotter с несколькими видами
        plotter = pv.Plotter(shape=(2, 2))
        
        # 1. Срез по XZ плоскости через центр наномостика (Y = center_y)
        plotter.subplot(0, 0)
        slice_y = grid.slice(normal='y', origin=[0, center_y, 0])
        plotter.add_mesh(slice_y, scalars="potential", cmap='coolwarm', 
                        scalar_bar_args={'title': "Потенциал, V"})
        plotter.add_title(f"Срез XZ (Y={center_y:.1f}) - Потенциал")
        
        # 2. Срез по XY плоскости через середину высоты наномостика (Z = center_z)
        plotter.subplot(0, 1)
        slice_z = grid.slice(normal='z', origin=[0, 0, center_z])
        plotter.add_mesh(slice_z, scalars="potential", cmap='coolwarm',
                        scalar_bar_args={'title': "Потенциал, V"})
        plotter.add_title(f"Срез XY (Z={center_z:.1f}) - Потенциал")
        
        # 3. Срез по XZ плоскости - напряженность поля
        plotter.subplot(1, 0)
        plotter.add_mesh(slice_y, scalars="E_magnitude", cmap='hot',
                        scalar_bar_args={'title': "|E|, V/м"})
        plotter.add_title(f"Срез XZ (Y={center_y:.1f}) - Напряженность")
        
        # 4. Срез через наномостик с изоповерхностями
        plotter.subplot(1, 1)
        
        # Создаем ограничивающую рамку вокруг наномостика
        nano_bounds = nano_bridge.get_total_bounds()
        clip_box = pv.Box(bounds=nano_bounds)
        
        # Вырезаем область наномостика
        nano_region = grid.clip_box(clip_box, invert=False)
        
        if nano_region.n_points > 0:
            # Добавляем изоповерхности внутри наномостика
            contours = nano_region.contour([1, 3, 5, 7, 9])  # Изоповерхности от 1V до 9V
            plotter.add_mesh(contours, cmap='viridis', opacity=0.7,
                           scalar_bar_args={'title': "Потенциал, V"})
            plotter.add_title("3D изоповерхности в наномостике")
        
        plotter.show()
    
    def visualize_dielectric_properties(self):
        """Визуализация распределения диэлектрических проницаемостей"""
        if self.grid is None:
            print("Сначала необходимо создать сетку!")
            return
        
        X, Y, Z = self.grid
        
        # Создаем PyVista структурированную сетку
        grid = pv.StructuredGrid(X, Y, Z)
        
        # Создаем маску диэлектрических проницаемостей
        eps_grid = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    point = (X[i,j,k], Y[i,j,k], Z[i,j,k])
                    eps_grid[i,j,k] = self.get_dielectric_constant_at_point(point)
        
        grid.point_data["dielectric_constant"] = eps_grid.ravel(order='F')
        
        plotter = pv.Plotter()
        
        # Срез диэлектрических проницаемостей
        slice_y = grid.slice(normal='y')
        plotter.add_mesh(slice_y, scalars="dielectric_constant", cmap='tab10',
                        scalar_bar_args={'title': "Отн. диэл. проницаемость"})
        
        plotter.add_axes()
        plotter.add_title("Распределение диэлектрических проницаемостей")
        plotter.show()
    
    def visualize_electric_field_with_geometry(self, visualizer):
        """Визуализация электрического поля вместе с геометрией системы"""
        if self.potential is None:
            print("Сначала необходимо решить уравнение Лапласа!")
            return
        
        X, Y, Z = self.grid
        Ex, Ey, Ez = self.electric_field
        
        # Создаем PyVista структурированную сетку
        grid = pv.StructuredGrid(X, Y, Z)
        
        # Добавляем данные о потенциале и поле
        grid.point_data["potential"] = self.potential.ravel(order='F')
        
        # Вычисляем модуль электрического поля
        E_magnitude = np.sqrt(Ex**2 + Ey**2 + Ez**2)
        grid.point_data["E_magnitude"] = E_magnitude.ravel(order='F')
        
        # Создаем plotter
        plotter = pv.Plotter()
        
        # 1. Сначала добавляем геометрию системы с прозрачностью
        if not visualizer.pyvista_objects:
            visualizer.convert_to_pyvista()
            
        for obj in visualizer.pyvista_objects:
            if obj['type'] == 'air':
                continue  # Пропускаем воздух
            
            # Разная прозрачность для разных материалов
            if obj['type'] == 'gate':
                opacity = 0.7  # Электрод - более прозрачный
            elif obj['type'] == 'nano_bridge':
                opacity = 0.8  # Наномостик - менее прозрачный
            else:
                opacity = 0.6  # Остальные материалы
            
            plotter.add_mesh(obj['mesh'], 
                           color=obj['color'], 
                           opacity=opacity,
                           edge_color='black',
                           line_width=1,
                           show_edges=True,
                           label=obj['name'])
        
        # 2. Добавляем срез потенциала через всю систему (включая материалы)
        slice_y0 = grid.slice(normal='y')
        plotter.add_mesh(slice_y0, scalars="potential", cmap='coolwarm', 
                        opacity=0.6, label="Срез потенциала (Y=0)")
        
        plotter.add_legend(bcolor='white', face='r', loc='upper right')
        plotter.add_axes()
        plotter.add_title("Электрическое поле в системе с диэлектриками")
        plotter.show()

class NanoSystemVisualizer:
    """Класс для визуализации полной системы"""
    
    def __init__(self, nano_system: CompleteNanoSystem):
        self.nano_system = nano_system
        self.pyvista_objects = []
    
    def convert_to_pyvista(self):
        """Конвертирует все компоненты системы в PyVista объекты"""
        self.pyvista_objects = []
        
        components = self.nano_system.all_components
        colors = {
            'nano_bridge': ['red', 'gold', 'blue'],
            'oxide': ['darkgreen'] * len(components['oxide']),
            'substrate': ['gray'],
            'air': ['lightblue'],
            'gate': ['purple'] * len(components['gate'])
        }
        
        names = {
            'nano_bridge': ['Left Bridge', 'Center Bridge', 'Right Bridge'],
            'oxide': ['Oxide'] * len(components['oxide']),
            'substrate': ['Substrate'],
            'air': ['Air Environment'],
            'gate': ['Gate Electrode'] * len(components['gate'])
        }
        
        # Конвертируем каждый компонент
        component_idx = 0
        for comp_type, boxes in components.items():
            for i, box in enumerate(boxes):
                x, y, z, dx, dy, dz = box.GetCoords()
                
                # Корректируем границы для отрицательных размеров
                x_min = min(x, x + dx)
                x_max = max(x, x + dx)
                y_min = min(y, y + dy)
                y_max = max(y, y + dy)
                z_min = min(z, z + dz)
                z_max = max(z, z + dz)
                
                pv_box = pv.Box(bounds=[
                    x_min, x_max,
                    y_min, y_max,  
                    z_min, z_max
                ])
                
                n_points = pv_box.n_points
                pv_box.point_data["component"] = np.full(n_points, component_idx)
                
                color = colors[comp_type][i] if comp_type in colors else 'white'
                name = names[comp_type][i] if comp_type in names else f'{comp_type}_{i}'
                
                self.pyvista_objects.append({
                    'mesh': pv_box,
                    'name': name,
                    'color': color,
                    'type': comp_type
                })
                component_idx += 1
        
        print(f"Всего объектов для визуализации: {len(self.pyvista_objects)}")
        return self.pyvista_objects
    
    def visualize_complete_system(self):
        """Визуализация полной системы"""
        if not self.pyvista_objects:
            self.convert_to_pyvista()
        
        plotter = pv.Plotter()
        
        for obj in self.pyvista_objects:
            opacity = 0.3 if obj['type'] == 'air' else 0.8
            show_edges = obj['type'] != 'air'
            
            plotter.add_mesh(obj['mesh'], 
                           color=obj['color'], 
                           opacity=opacity,
                           edge_color='black' if show_edges else None,
                           line_width=1,
                           show_edges=show_edges,
                           label=obj['name'])
        
        plotter.add_legend()
        plotter.add_axes()
        plotter.add_title("Полная система: Наномостик + Подложка + Воздух + Электрод")
        plotter.show()

class ElectricFieldSolver:
    """Класс для решения уравнения Лапласа и расчета электрического поля"""
    
    def __init__(self, nano_system: CompleteNanoSystem, grid_resolution=50):
        self.nano_system = nano_system
        self.grid_resolution = grid_resolution
        self.potential = None
        self.electric_field = None
        self.grid = None
        self.mask = None
        # Добавляем относительные диэлектрические проницаемости
        self.dielectric_constants = {
            'air': 1.0,
            'oxide': 3.9,  # SiO2
            'nano_bridge': 11.7,  # Кремний
            'substrate': 11.7,  # Кремниевая подложка
            'gate': 1e10  # Металл (очень большое значение)
        }
        
    def save_results(self, filename="electric_field_results.pkl"):
        """Сохраняет результаты расчета в файл"""
        if self.potential is None:
            print("Нет данных для сохранения! Сначала выполните расчет.")
            return
            
        data_to_save = {
            'potential': self.potential,
            'electric_field': self.electric_field,
            'grid': self.grid,
            'mask': self.mask,
            'dielectric_constants': self.dielectric_constants,
            'grid_resolution': self.grid_resolution,
            'nano_bounds': self.nano_system.nano_bridge.get_total_bounds(),
            'timestamp': time.time()
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data_to_save, f)
        
        print(f"Результаты сохранены в файл: {filename}")
        print(f"Размер массива потенциала: {self.potential.shape}")
        print(f"Размер сетки: {self.grid_resolution}^3 = {self.grid_resolution**3} точек")
        
    def load_results(self, filename="electric_field_results.pkl"):
        """Загружает результаты расчета из файла"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            self.potential = data['potential']
            self.electric_field = data['electric_field']
            self.grid = data['grid']
            self.mask = data['mask']
            self.dielectric_constants = data.get('dielectric_constants', self.dielectric_constants)
            self.grid_resolution = data.get('grid_resolution', self.grid_resolution)
            
            print(f"Результаты загружены из файла: {filename}")
            print(f"Размер массива потенциала: {self.potential.shape}")
            print(f"Дата расчета: {time.ctime(data['timestamp'])}")
            
            return True
            
        except FileNotFoundError:
            print(f"Файл {filename} не найден!")
            return False
        except Exception as e:
            print(f"Ошибка при загрузке файла: {e}")
            return False

    def create_computational_grid(self):
        """Создает вычислительную сетку для решения уравнения Лапласа"""
        air_box = self.nano_system.air_environment.box
        air_bounds = air_box.GetBounds()
        
        # Создаем регулярную сетку
        x = np.linspace(air_bounds[0], air_bounds[1], self.grid_resolution)
        y = np.linspace(air_bounds[2], air_bounds[3], self.grid_resolution)
        z = np.linspace(air_bounds[4], air_bounds[5], self.grid_resolution)
        
        self.grid = np.meshgrid(x, y, z, indexing='ij')
        return self.grid
    
    def get_dielectric_constant_at_point(self, point):
        """Возвращает диэлектрическую проницаемость в точке"""
        # Проверяем материалы в порядке приоритета (металлы -> диэлектрики -> воздух)
        if self.is_point_in_material(point, 'gate'):
            return self.dielectric_constants['gate']  # Металл
        elif self.is_point_in_material(point, 'substrate'):
            return self.dielectric_constants['substrate']  # Подложка (проводящая)
        elif self.is_point_in_material(point, 'nano_bridge'):
            return self.dielectric_constants['nano_bridge']  # Наномостик (полупроводник)
        elif self.is_point_in_material(point, 'oxide'):
            return self.dielectric_constants['oxide']  # Оксид
        else:
            return self.dielectric_constants['air']  # Воздух

    def set_dielectric_constants(self, new_constants: Dict):
        """Устанавливает новые значения диэлектрических проницаемостей"""
        for material, value in new_constants.items():
            if material in self.dielectric_constants:
                self.dielectric_constants[material] = value
                print(f"Диэлектрическая проницаемость {material} установлена в {value}")
    
    def create_boundary_mask(self, gate_potential=10.0):
        """Создает маску граничных условий - ТОЛЬКО проводники и границы воздуха"""
        X, Y, Z = self.grid
        self.mask = np.zeros(X.shape, dtype=bool)
        boundary_values = np.zeros(X.shape)
        
        print("Создание маски граничных условий...")
        print("ПОДЛОЖКА - ДИЭЛЕКТРИК (без фиксированного потенциала)")
        
        # Счетчики для отладки
        gate_points = 0
        boundary_points = 0
        
        # 1. Границы воздушной среды - земля (0V)
        air_box = self.nano_system.air_environment.box
        air_bounds = air_box.GetBounds()
        
        tolerance = 1e-6  # Погрешность для сравнения координат
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    x, y, z = X[i,j,k], Y[i,j,k], Z[i,j,k]
                    
                    # Проверяем, находится ли точка на границе воздушной среды
                    on_boundary = (
                        abs(x - air_bounds[0]) < tolerance or 
                        abs(x - air_bounds[1]) < tolerance or
                        abs(y - air_bounds[2]) < tolerance or 
                        abs(y - air_bounds[3]) < tolerance or
                        abs(z - air_bounds[4]) < tolerance or 
                        abs(z - air_bounds[5]) < tolerance
                    )
                    
                    if on_boundary:
                        self.mask[i,j,k] = True
                        boundary_values[i,j,k] = 0.0
                        boundary_points += 1
        
        # 2. Электрод - заданный потенциал (10V) - ПРОВОДНИК
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    point = (X[i,j,k], Y[i,j,k], Z[i,j,k])
                    
                    # Электрод - заданный потенциал - ПРОВОДНИК
                    if self.is_point_in_material(point, 'gate'):
                        self.mask[i,j,k] = True
                        boundary_values[i,j,k] = gate_potential
                        gate_points += 1
        
        print(f"Точек на электроде (проводник): {gate_points:,}")
        print(f"Точек на границах воздуха (земля): {boundary_points:,}")
        print("Подложка, наномостик и оксид - ДИЭЛЕКТРИКИ (без фиксированного потенциала)")
        
        # Проверяем, что электрод имеет правильный потенциал
        gate_indices = np.where(self.mask & (boundary_values == gate_potential))
        if len(gate_indices[0]) > 0:
            print(f"✓ Потенциал электрода установлен правильно: {gate_potential}V")
        else:
            print("✗ ВНИМАНИЕ: Не найдено точек с потенциалом электрода!")
            
        return boundary_values
    
    def is_point_in_material(self, point, material_type):
        """Проверяет, находится ли точка внутри указанного материала"""
        x, y, z = point
        
        components = self.nano_system.all_components
        if material_type in components:
            for box in components[material_type]:
                bounds = box.GetBounds()
                # Используем более точную проверку с учетом знаков размеров
                x_min = min(bounds[0], bounds[1])
                x_max = max(bounds[0], bounds[1])
                y_min = min(bounds[2], bounds[3])
                y_max = max(bounds[2], bounds[3])
                z_min = min(bounds[4], bounds[5])
                z_max = max(bounds[4], bounds[5])
                
                if (x_min <= x <= x_max and 
                    y_min <= y <= y_max and 
                    z_min <= z <= z_max):
                    return True
        return False
    
    def solve_laplace_sor(self, gate_potential=10.0, omega=1.8, max_iter=1000, tolerance=5e-1):
        """Решает уравнение Лапласа с использованием SOR метода с учетом диэлектриков"""
        print("Создание вычислительной сетки...")
        self.create_computational_grid()
        X, Y, Z = self.grid
        
        print(f"Размер сетки: {X.shape}")
        
        # Создаем маску граничных условий (только для проводников)
        boundary_values = self.create_boundary_mask(gate_potential)
        
        # Инициализация потенциала
        self.potential = boundary_values.copy()
        
        print("Решение уравнения Лапласа методом SOR с учетом диэлектриков...")
        start_time = time.time()
        
        # SOR метод с учетом диэлектриков
        for iteration in range(max_iter):
            max_change = 0.0
            
            # Обновление потенциала во внутренних точках
            for i in range(1, X.shape[0]-1):
                for j in range(1, X.shape[1]-1):
                    for k in range(1, X.shape[2]-1):
                        # Пропускаем точки с фиксированными граничными условиями (проводники)
                        if self.mask[i,j,k]:
                            continue
                            
                        # Получаем диэлектрические проницаемости в соседних точках
                        eps_center = self.get_dielectric_constant_at_point((X[i,j,k], Y[i,j,k], Z[i,j,k]))
                        eps_x_prev = self.get_dielectric_constant_at_point((X[i-1,j,k], Y[i-1,j,k], Z[i-1,j,k]))
                        eps_x_next = self.get_dielectric_constant_at_point((X[i+1,j,k], Y[i+1,j,k], Z[i+1,j,k]))
                        eps_y_prev = self.get_dielectric_constant_at_point((X[i,j-1,k], Y[i,j-1,k], Z[i,j-1,k]))
                        eps_y_next = self.get_dielectric_constant_at_point((X[i,j+1,k], Y[i,j+1,k], Z[i,j+1,k]))
                        eps_z_prev = self.get_dielectric_constant_at_point((X[i,j,k-1], Y[i,j,k-1], Z[i,j,k-1]))
                        eps_z_next = self.get_dielectric_constant_at_point((X[i,j,k+1], Y[i,j,k+1], Z[i,j,k+1]))
                        
                        # Усредненные диэлектрические проницаемости на гранях
                        eps_x_avg_prev = (eps_center + eps_x_prev) / 2
                        eps_x_avg_next = (eps_center + eps_x_next) / 2
                        eps_y_avg_prev = (eps_center + eps_y_prev) / 2
                        eps_y_avg_next = (eps_center + eps_y_next) / 2
                        eps_z_avg_prev = (eps_center + eps_z_prev) / 2
                        eps_z_avg_next = (eps_center + eps_z_next) / 2
                        
                        # Вычисляем новое значение с учетом диэлектриков
                        numerator = (eps_x_avg_prev * self.potential[i-1,j,k] + 
                                   eps_x_avg_next * self.potential[i+1,j,k] +
                                   eps_y_avg_prev * self.potential[i,j-1,k] + 
                                   eps_y_avg_next * self.potential[i,j+1,k] +
                                   eps_z_avg_prev * self.potential[i,j,k-1] + 
                                   eps_z_avg_next * self.potential[i,j,k+1])
                        
                        denominator = (eps_x_avg_prev + eps_x_avg_next +
                                     eps_y_avg_prev + eps_y_avg_next +
                                     eps_z_avg_prev + eps_z_avg_next)
                        
                        if denominator > 0:
                            new_value = numerator / denominator
                            
                            # Применяем релаксацию (SOR)
                            change = omega * (new_value - self.potential[i,j,k])
                            self.potential[i,j,k] += change
                            
                            # Отслеживаем максимальное изменение
                            max_change = max(max_change, abs(change))
            
            # Проверка сходимости каждые 50 итераций
            if iteration % 10 == 0:
                print(f"Итерация {iteration}, максимальное изменение: {max_change:.2e}")
            
            if max_change < tolerance:
                end_time = time.time()
                print(f"Сходимость достигнута на итерации {iteration+1}")
                print(f"Время решения: {end_time - start_time:.2f} секунд")
                break
        
        if iteration == max_iter - 1:
            end_time = time.time()
            print(f"Достигнуто максимальное число итераций {max_iter}")
            print(f"Время решения: {end_time - start_time:.2f} секунд")
        
        print("Расчет электрического поля...")
        # Расчет электрического поля E = -∇φ
        self.calculate_electric_field()
        
        # Автоматически сохраняем результаты после расчета
        self.save_results()
        
        return self.potential
    
    def calculate_electric_field(self):
        """Вычисляет электрическое поле как градиент потенциала"""
        X, Y, Z = self.grid
        
        # Вычисление градиента
        Ex = -np.gradient(self.potential, axis=0)
        Ey = -np.gradient(self.potential, axis=1)
        Ez = -np.gradient(self.potential, axis=2)
        
        self.electric_field = (Ex, Ey, Ez)
        return self.electric_field

    def quick_visualize_from_file(self, filename="electric_field_results.pkl"):
        """Быстрая визуализация из сохраненного файла"""
        if not self.load_results(filename):
            return
        
        print("Быстрая визуализация из файла...")
        self.visualize_nanobridge_cross_sections()
    
    def visualize_nanobridge_cross_sections(self):
        """Визуализация срезов распределения поля внутри наномостика"""
        if self.potential is None:
            print("Сначала необходимо решить уравнение Лапласа!")
            return
        
        X, Y, Z = self.grid
        
        # Создаем PyVista структурированную сетку
        grid = pv.StructuredGrid(X, Y, Z)
        grid.point_data["potential"] = self.potential.ravel(order='F')
        
        # Вычисляем модуль электрического поля
        Ex, Ey, Ez = self.electric_field
        E_magnitude = np.sqrt(Ex**2 + Ey**2 + Ez**2)
        grid.point_data["E_magnitude"] = E_magnitude.ravel(order='F')
        
        # Получаем границы наномостика для точных срезов
        nano_bridge = self.nano_system.nano_bridge
        center_box = nano_bridge.get_boxes()[1]  # Центральная часть
        center_bounds = center_box.GetBounds()
        
        # Координаты для срезов
        center_x = (center_bounds[0] + center_bounds[1]) / 2
        center_y = (center_bounds[2] + center_bounds[3]) / 2  
        center_z = (center_bounds[4] + center_bounds[5]) / 2
        
        # Создаем plotter с несколькими видами
        plotter = pv.Plotter(shape=(2, 2))
        
        # 1. Срез по XZ плоскости через центр наномостика (Y = center_y)
        plotter.subplot(0, 0)
        slice_y = grid.slice(normal='y', origin=[0, center_y, 0])
        plotter.add_mesh(slice_y, scalars="potential", cmap='coolwarm', 
                        scalar_bar_args={'title': "Потенциал, V"})
        plotter.add_title(f"Срез XZ (Y={center_y:.1f}) - Потенциал")
        
        # 2. Срез по XY плоскости через середину высоты наномостика (Z = center_z)
        plotter.subplot(0, 1)
        slice_z = grid.slice(normal='z', origin=[0, 0, center_z])
        plotter.add_mesh(slice_z, scalars="potential", cmap='coolwarm',
                        scalar_bar_args={'title': "Потенциал, V"})
        plotter.add_title(f"Срез XY (Z={center_z:.1f}) - Потенциал")
        
        # 3. Срез по XZ плоскости - напряженность поля
        plotter.subplot(1, 0)
        plotter.add_mesh(slice_y, scalars="E_magnitude", cmap='hot',
                        scalar_bar_args={'title': "|E|, V/м"})
        plotter.add_title(f"Срез XZ (Y={center_y:.1f}) - Напряженность")
        
        # 4. Срез через наномостик с изоповерхностями
        plotter.subplot(1, 1)
        
        # Создаем ограничивающую рамку вокруг наномостика
        nano_bounds = nano_bridge.get_total_bounds()
        clip_box = pv.Box(bounds=nano_bounds)
        
        # Вырезаем область наномостика
        nano_region = grid.clip_box(clip_box, invert=False)
        
        if nano_region.n_points > 0:
            # Добавляем изоповерхности внутри наномостика
            contours = nano_region.contour([1, 3, 5, 7, 9])  # Изоповерхности от 1V до 9V
            plotter.add_mesh(contours, cmap='viridis', opacity=0.7,
                           scalar_bar_args={'title': "Потенциал, V"})
            plotter.add_title("3D изоповерхности в наномостике")
        
        plotter.show()

class ElectrodePotentialVisualizer:
    """Класс для специализированной визуализации распределения потенциала внутри электрода"""
    
    def __init__(self, field_solver, nano_system=None):
        self.field_solver = field_solver
        self.nano_system = nano_system
        self.electrode_points = None
        self.electrode_potentials = None
        self.electrode_coords = None
    
    def is_point_in_electrode(self, point):
        """Проверяет, находится ли точка внутри электрода (упрощенная версия для работы без полной системы)"""
        if self.nano_system:
            # Используем полную систему если доступна
            return self.field_solver.is_point_in_material(point, 'gate')
        else:
            # Упрощенная проверка по координатам (предполагаем стандартную геометрию)
            x, y, z = point
            # Примерные границы электрода из стандартной конфигурации
            if (-15 <= x <= 15 and 
                -8 <= y <= 8 and 
                0 <= z <= 10):
                return True
            return False
    
    def extract_electrode_data(self):
        """Извлекает данные о потенциале внутри электрода"""
        if self.field_solver.potential is None:
            print("Сначала необходимо загрузить или рассчитать потенциал!")
            return False
        
        X, Y, Z = self.field_solver.grid
        potential = self.field_solver.potential
        
        # Собираем все точки, находящиеся внутри электрода
        electrode_points = []
        electrode_potentials = []
        electrode_coords = []
        
        print("Извлечение данных электрода...")
        total_points = X.shape[0] * X.shape[1] * X.shape[2]
        count = 0
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    point = (X[i,j,k], Y[i,j,k], Z[i,j,k])
                    if self.is_point_in_electrode(point):
                        electrode_points.append(point)
                        electrode_potentials.append(potential[i,j,k])
                        electrode_coords.append((i, j, k))  # индексы сетки
                        count += 1
        
        self.electrode_points = np.array(electrode_points)
        self.electrode_potentials = np.array(electrode_potentials)
        self.electrode_coords = np.array(electrode_coords)
        
        print(f"Найдено точек внутри электрода: {count}")
        if len(electrode_potentials) > 0:
            print(f"Диапазон потенциалов: {self.electrode_potentials.min():.3f} - {self.electrode_potentials.max():.3f} V")
        else:
            print("ВНИМАНИЕ: Не найдено точек внутри электрода!")
        
        return len(electrode_points) > 0
    
    def quick_electrode_analysis_from_file(self, filename="electric_field_results.pkl"):
        """Быстрый анализ электрода непосредственно из файла"""
        print("=" * 60)
        print("БЫСТРАЯ ВИЗУАЛИЗАЦИЯ ЭЛЕКТРОДА ИЗ ФАЙЛА")
        print("=" * 60)
        
        # Загружаем результаты
        if not self.field_solver.load_results(filename):
            return False
        
        # Извлекаем данные электрода
        if not self.extract_electrode_data():
            print("Не удалось извлечь данные электрода!")
            return False
        
        # Выполняем визуализацию
        print("\n1. Статистический анализ:")
        self.print_electrode_statistics()
        
        print("\n2. 3D визуализация с matplotlib:")
        self.plot_electrode_potential_3d_matplotlib()
        
        print("\n3. Интерактивная 3D визуализация с PyVista:")
        self.plot_electrode_potential_3d_pyvista()
        
        return True
    
    def plot_electrode_potential_3d_matplotlib(self):
        """3D визуализация с использованием matplotlib"""
        if not self.extract_electrode_data():
            return
        
        points = self.electrode_points
        potentials = self.electrode_potentials
        
        fig = plt.figure(figsize=(15, 10))
        
        # 1. 3D scatter plot
        ax1 = fig.add_subplot(221, projection='3d')
        scatter = ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                             c=potentials, cmap='viridis', s=20, alpha=0.7)
        ax1.set_xlabel('X (мкм)')
        ax1.set_ylabel('Y (мкм)')
        ax1.set_zlabel('Z (мкм)')
        ax1.set_title('3D распределение потенциала в электроде')
        plt.colorbar(scatter, ax=ax1, label='Потенциал (V)')
        
        # 2. Проекция на XY плоскость
        ax2 = fig.add_subplot(222)
        scatter2 = ax2.scatter(points[:, 0], points[:, 1], c=potentials, 
                              cmap='viridis', s=15, alpha=0.7)
        ax2.set_xlabel('X (мкм)')
        ax2.set_ylabel('Y (мкм)')
        ax2.set_title('Проекция XY')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Потенциал (V)')
        
        # 3. Проекция на XZ плоскость
        ax3 = fig.add_subplot(223)
        scatter3 = ax3.scatter(points[:, 0], points[:, 2], c=potentials, 
                              cmap='viridis', s=15, alpha=0.7)
        ax3.set_xlabel('X (мкм)')
        ax3.set_ylabel('Z (мкм)')
        ax3.set_title('Проекция XZ')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=ax3, label='Потенциал (V)')
        
        # 4. Гистограмма распределения потенциалов
        ax4 = fig.add_subplot(224)
        ax4.hist(potentials, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.set_xlabel('Потенциал (V)')
        ax4.set_ylabel('Количество точек')
        ax4.set_title('Распределение потенциалов в электроде')
        ax4.grid(True, alpha=0.3)
        
        # Добавляем статистику на гистограмму
        mean_pot = np.mean(potentials)
        std_pot = np.std(potentials)
        ax4.axvline(mean_pot, color='red', linestyle='--', label=f'Среднее: {mean_pot:.3f} V')
        ax4.axvline(mean_pot + std_pot, color='orange', linestyle='--', alpha=0.7, label=f'±σ: {std_pot:.3f} V')
        ax4.axvline(mean_pot - std_pot, color='orange', linestyle='--', alpha=0.7)
        ax4.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_electrode_potential_3d_pyvista(self):
        """3D визуализация с использованием PyVista с сеткой"""
        if not self.extract_electrode_data():
            return
        
        points = self.electrode_points
        potentials = self.electrode_potentials
        
        # Создаем облако точек PyVista
        point_cloud = pv.PolyData(points)
        point_cloud["potential"] = potentials
        
        # Находим границы электрода
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        
        print(f"Границы электрода:")
        print(f"  X: {x_min:.3f} - {x_max:.3f}")
        print(f"  Y: {y_min:.3f} - {y_max:.3f}") 
        print(f"  Z: {z_min:.3f} - {z_max:.3f}")
        
        # Создаем plotter с несколькими видами
        plotter = pv.Plotter(shape=(2, 2))
        
        # 1. 3D облако точек с потенциалом
        plotter.subplot(0, 0)
        plotter.add_mesh(point_cloud, scalars="potential", cmap='viridis', 
                        point_size=10, opacity=0.8, render_points_as_spheres=True,
                        scalar_bar_args={'title': "Потенциал, V"})
        plotter.add_axes()
        plotter.add_title("3D распределение потенциала\n(облако точек)")
        
        # 2. Объемная визуализация с изоповерхностями
        plotter.subplot(0, 1)
        
        # Создаем ограничивающий бокс для электрода
        electrode_bounds = [x_min, x_max, y_min, y_max, z_min, z_max]
        electrode_box = pv.Box(bounds=electrode_bounds)
        
        # Создаем структурированную сетку для всего пространства
        X, Y, Z = self.field_solver.grid
        grid = pv.StructuredGrid(X, Y, Z)
        grid["potential"] = self.field_solver.potential.ravel(order='F')
        
        # Вырезаем область электрода
        electrode_region = grid.clip_box(electrode_box, invert=False)
        
        if electrode_region.n_points > 0:
            # Добавляем изоповерхности внутри электрода
            contours = electrode_region.contour()
            plotter.add_mesh(contours, cmap='coolwarm', opacity=0.7,
                           scalar_bar_args={'title': "Потенциал, V"})
            plotter.add_mesh(electrode_box, color='white', opacity=0.1, 
                           show_edges=True, line_width=2)
            plotter.add_title("Изоповерхности потенциала\nв объеме электрода")
        
        # 3. Срез через центр электрода
        plotter.subplot(1, 0)
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        center_z = (z_min + z_max) / 2
        
        # Срез по YZ плоскости
        slice_x = grid.slice(normal='x', origin=[center_x, 0, 0])
        plotter.add_mesh(slice_x, scalars="potential", cmap='hot', opacity=0.8,
                        scalar_bar_args={'title': "Потенциал, V"})
        plotter.add_title(f"Срез YZ (X={center_x:.2f})")
        
        # 4. Проволочная сетка электрода с цветом по потенциалу
        plotter.subplot(1, 1)
        
        # Создаем поверхность из облака точек
        if len(points) > 100:  # Только если достаточно точек
            try:
                surface = point_cloud.reconstruct_surface()
                plotter.add_mesh(surface, scalars="potential", cmap='plasma', 
                               show_edges=True, line_width=1, opacity=0.8,
                               scalar_bar_args={'title': "Потенциал, V"})
                plotter.add_title("Поверхность электрода\nс сеткой")
            except:
                plotter.add_mesh(point_cloud, scalars="potential", cmap='plasma',
                               point_size=8, render_points_as_spheres=True,
                               scalar_bar_args={'title': "Потенциал, V"})
                plotter.add_title("Облако точек электрода")
        else:
            plotter.add_mesh(point_cloud, scalars="potential", cmap='plasma',
                           point_size=8, render_points_as_spheres=True,
                           scalar_bar_args={'title': "Потенциал, V"})
            plotter.add_title("Облако точек электрода")
        
        plotter.show()
        
        # Детальный анализ
        self.detailed_electrode_analysis()
    
    def detailed_electrode_analysis(self):
        """Детальный анализ распределения потенциала в электроде"""
        if self.electrode_potentials is None:
            return
        
        potentials = self.electrode_potentials
        
        print("\n" + "="*50)
        print("ДЕТАЛЬНЫЙ АНАЛИЗ РАСПРЕДЕЛЕНИЯ ПОТЕНЦИАЛА В ЭЛЕКТРОДЕ")
        print("="*50)
        
        # Основная статистика
        print(f"Общее количество точек: {len(potentials):,}")
        print(f"Средний потенциал: {np.mean(potentials):.6f} V")
        print(f"Стандартное отклонение: {np.std(potentials):.6f} V")
        print(f"Медиана: {np.median(potentials):.6f} V")
        print(f"Минимальный потенциал: {np.min(potentials):.6f} V")
        print(f"Максимальный потенциал: {np.max(potentials):.6f} V")
        print(f"Размах: {np.ptp(potentials):.6f} V")
        
        # Анализ однородности
        if np.mean(potentials) > 0:
            variation_coefficient = np.std(potentials) / np.mean(potentials) * 100
            print(f"Коэффициент вариации: {variation_coefficient:.4f} %")
        
        # Проверка на идеальный проводник
        expected_potential = 10.0  # ожидаемый потенциал электрода
        max_deviation = np.max(np.abs(potentials - expected_potential))
        print(f"Максимальное отклонение от {expected_potential}V: {max_deviation:.6f} V")
        
        # Процент точек с отклонением менее 1%
        tolerance_1percent = expected_potential * 0.01
        points_within_1percent = np.sum(np.abs(potentials - expected_potential) <= tolerance_1percent)
        percentage_1percent = points_within_1percent / len(potentials) * 100
        print(f"Точек в пределах ±1% от {expected_potential}V: {percentage_1percent:.2f}%")
    
    def print_electrode_statistics(self):
        """Вывод статистики по распределению потенциала"""
        if self.electrode_potentials is None:
            return
            
        potentials = self.electrode_potentials
        
        print("\nСТАТИСТИКА РАСПРЕДЕЛЕНИЯ ПОТЕНЦИАЛА В ЭЛЕКТРОДЕ:")
        print(f"Количество точек: {len(potentials):,}")
        print(f"Диапазон: [{potentials.min():.6f}, {potentials.max():.6f}] V")
        print(f"Среднее: {np.mean(potentials):.6f} V ± {np.std(potentials):.6f} V")
        print(f"Медиана: {np.median(potentials):.6f} V")
        
        # Квантили
        if len(potentials) > 0:
            quantiles = np.quantile(potentials, [0.1, 0.25, 0.75, 0.9])
            print(f"10%-квантиль: {quantiles[0]:.6f} V")
            print(f"25%-квантиль: {quantiles[1]:.6f} V") 
            print(f"75%-квантиль: {quantiles[2]:.6f} V")
            print(f"90%-квантиль: {quantiles[3]:.6f} V")
            
            # Однородность
            if np.mean(potentials) > 0:
                cv = np.std(potentials) / np.mean(potentials) * 100
                print(f"Однородность (CV): {cv:.4f}%")

# Основная функция с улучшенным управлением
def main():
    print("=" * 60)
    print("СИСТЕМА АНАЛИЗА НАНОМОСТИКА С ЭЛЕКТРОДОМ")
    print("=" * 60)
    
    # Проверяем наличие сохраненных результатов
    saved_file = "electric_field_results.pkl"
    
    if os.path.exists(saved_file):
        print(f"✓ Найден сохраненный файл: {saved_file}")
        print("\nВыберите опцию:")
        print("1 - Быстрая визуализация электрода из файла")
        print("2 - Полный расчет с визуализацией всей системы")
        print("3 - Только визуализация геометрии системы")
        
        choice = input("\nВведите номер опции (1-3): ").strip()
        
        if choice == "1":
            # БЫСТРАЯ ВИЗУАЛИЗАЦИЯ ЭЛЕКТРОДА ИЗ ФАЙЛА
            print("\n" + "="*50)
            print("ЗАГРУЗКА И ВИЗУАЛИЗАЦИЯ ЭЛЕКТРОДА ИЗ ФАЙЛА")
            print("="*50)
            
            # Создаем систему для геометрии (опционально)
            nano_system = CompleteNanoSystem()
            nano_system.create_complete_system()
            
            # Создаем solver и визуализатор
            field_solver = ElectricFieldSolver(nano_system)
            electrode_viz = ElectrodePotentialVisualizer(field_solver, nano_system)
            
            # Запускаем быструю визуализацию
            electrode_viz.quick_electrode_analysis_from_file(saved_file)
            return
            
        elif choice == "3":
            # ТОЛЬКО ВИЗУАЛИЗАЦИЯ ГЕОМЕТРИИ
            print("\nСоздание и визуализация геометрии системы...")
            nano_system = CompleteNanoSystem()
            components = nano_system.create_complete_system()
            
            visualizer = NanoSystemVisualizer(nano_system)
            visualizer.visualize_complete_system()
            return
    
    # ПОЛНЫЙ РАСЧЕТ (опция 2 или если файла нет)
    print("\nСоздание полной системы наномостика...")
    
    # Создаем полную систему
    nano_system = CompleteNanoSystem()
    components = nano_system.create_complete_system()
    
    print(f"Компоненты системы:")
    for comp_type, boxes in components.items():
        print(f"  {comp_type}: {len(boxes)} боксов")
    
    # Визуализируем геометрию
    visualizer = NanoSystemVisualizer(nano_system)
    
    print("\n1. Визуализация полной системы:")
    visualizer.visualize_complete_system()
    
    # Расчет электрического поля
    print("\n2. Расчет электрического поля с учетом диэлектриков...")
    field_solver = ElectricFieldSolver(nano_system, grid_resolution=50)
    
    # Выполняем расчет
    print("Выполнение нового расчета...")
    field_solver.solve_laplace_sor(gate_potential=10.0, max_iter=1000, tolerance=5e-1)
    
    # Визуализация электрода после расчета
    print("\n3. Визуализация распределения потенциала в электроде:")
    electrode_viz = ElectrodePotentialVisualizer(field_solver, nano_system)
    
    print("3.1. Статистический анализ:")
    electrode_viz.extract_electrode_data()
    electrode_viz.print_electrode_statistics()
    
    print("3.2. 3D визуализация с matplotlib:")
    electrode_viz.plot_electrode_potential_3d_matplotlib()
    
    print("3.3. Интерактивная 3D визуализация с PyVista:")
    electrode_viz.plot_electrode_potential_3d_pyvista()
    
    print("\n4. Дополнительная визуализация системы:")
    field_solver.visualize_nanobridge_cross_sections()

# Функция для быстрой визуализации только электрода
def quick_electrode_visualization(filename="electric_field_results.pkl"):
    """Быстрая визуализация только распределения потенциала в электроде"""
    print("=" * 60)
    print("БЫСТРАЯ ВИЗУАЛИЗАЦИЯ ЭЛЕКТРОДА")
    print("=" * 60)
    
    # Создаем solver без системы (для экономии времени)
    field_solver = ElectricFieldSolver()
    
    # Создаем визуализатор
    electrode_viz = ElectrodePotentialVisualizer(field_solver)
    
    # Запускаем анализ
    success = electrode_viz.quick_electrode_analysis_from_file(filename)
    
    if not success:
        print("Не удалось выполнить визуализацию электрода!")
        print("Убедитесь, что файл с результатами существует и содержит корректные данные.")

class NanowirePotentialAnalyzer:
    """Специализированный анализатор потенциальной ямы в нанопроводе"""
    
    def __init__(self, field_solver, nano_system):
        self.field_solver = field_solver
        self.nano_system = nano_system
        self.nanowire_points = None
        self.nanowire_potentials = None
        
    def extract_nanowire_data(self):
        """Извлекает данные о потенциале внутри нанопровода"""
        if self.field_solver.potential is None:
            print("Сначала необходимо загрузить или рассчитать потенциал!")
            return False
        
        X, Y, Z = self.field_solver.grid
        potential = self.field_solver.potential
        
        nanowire_points = []
        nanowire_potentials = []
        
        print("Извлечение данных нанопровода...")
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    point = (X[i,j,k], Y[i,j,k], Z[i,j,k])
                    if self.field_solver.is_point_in_material(point, 'nano_bridge'):
                        nanowire_points.append(point)
                        nanowire_potentials.append(potential[i,j,k])
        
        self.nanowire_points = np.array(nanowire_points)
        self.nanowire_potentials = np.array(nanowire_potentials)
        
        print(f"Найдено точек в нанопроводе: {len(nanowire_points)}")
        if len(nanowire_potentials) > 0:
            print(f"Диапазон потенциалов: {self.nanowire_potentials.min():.3f} - {self.nanowire_potentials.max():.3f} V")
        
        return len(nanowire_points) > 0
    
    def plot_potential_profile_along_nanowire(self):
        """Строит профиль потенциала вдоль нанопровода"""
        if not self.extract_nanowire_data():
            return
        
        points = self.nanowire_points
        potentials = self.nanowire_potentials
        
        # Сортируем точки по координате X (вдоль нанопровода)
        sorted_indices = np.argsort(points[:, 0])
        sorted_x = points[sorted_indices, 0]
        sorted_potentials = potentials[sorted_indices]
        
        # Сглаживаем для лучшей визуализации
        from scipy.ndimage import gaussian_filter1d
        smoothed_potentials = gaussian_filter1d(sorted_potentials, sigma=5)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 1. Профиль потенциала вдоль X
        ax1.plot(sorted_x, sorted_potentials, 'b.', alpha=0.3, label='Исходные точки')
        ax1.plot(sorted_x, smoothed_potentials, 'r-', linewidth=2, label='Сглаженный профиль')
        ax1.set_xlabel('Координата X (мкм)')
        ax1.set_ylabel('Потенциал (V)')
        ax1.set_title('Профиль потенциала вдоль нанопровода')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Размечаем области нанопровода
        nano_bridge = self.nano_system.nano_bridge
        boxes = nano_bridge.get_boxes()
        
        colors = ['lightcoral', 'lightgreen', 'lightblue']
        labels = ['Левая часть', 'Центральная часть', 'Правая часть']
        
        for i, box in enumerate(boxes):
            bounds = box.GetBounds()
            x_center = (bounds[0] + bounds[1]) / 2
            width = bounds[1] - bounds[0]
            
            ax1.axvspan(bounds[0], bounds[1], alpha=0.2, color=colors[i], label=labels[i])
            ax1.axvline(x=bounds[0], color=colors[i], linestyle='--', alpha=0.7)
            ax1.axvline(x=bounds[1], color=colors[i], linestyle='--', alpha=0.7)
            ax1.text(x_center, ax1.get_ylim()[0] + 0.1, labels[i], 
                    ha='center', va='bottom', fontsize=10, color=colors[i])
        
        ax1.legend()
        
        # 2. Гистограмма распределения потенциалов в центральной части
        center_box = boxes[1]  # Центральная часть
        center_bounds = center_box.GetBounds()
        
        # Выбираем точки в центральной части
        center_mask = (points[:, 0] >= center_bounds[0]) & (points[:, 0] <= center_bounds[1])
        center_potentials = potentials[center_mask]
        
        if len(center_potentials) > 0:
            ax2.hist(center_potentials, bins=30, alpha=0.7, color='green', edgecolor='black')
            ax2.set_xlabel('Потенциал (V)')
            ax2.set_ylabel('Количество точек')
            ax2.set_title('Распределение потенциалов в центральной части нанопровода')
            ax2.grid(True, alpha=0.3)
            
            # Статистика
            mean_pot = np.mean(center_potentials)
            std_pot = np.std(center_potentials)
            min_pot = np.min(center_potentials)
            max_pot = np.max(center_potentials)
            
            ax2.axvline(mean_pot, color='red', linestyle='--', linewidth=2, 
                       label=f'Среднее: {mean_pot:.3f} V')
            ax2.axvline(mean_pot - std_pot, color='orange', linestyle='--', alpha=0.7,
                       label=f'±σ: {std_pot:.3f} V')
            ax2.axvline(mean_pot + std_pot, color='orange', linestyle='--', alpha=0.7)
            
            # Показываем глубину потенциальной ямы
            gate_potential = 10.0  # Потенциал затвора
            potential_well_depth = gate_potential - min_pot
            ax2.axvline(min_pot, color='blue', linestyle=':', linewidth=2,
                       label=f'Мин: {min_pot:.3f} V\nГлубина ямы: {potential_well_depth:.3f} V')
            
            ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Анализ потенциальной ямы
        self.analyze_potential_well()
    
    def analyze_potential_well(self):
        """Детальный анализ потенциальной ямы"""
        if self.nanowire_potentials is None:
            return
        
        potentials = self.nanowire_potentials
        points = self.nanowire_points
        
        print("\n" + "="*60)
        print("АНАЛИЗ ПОТЕНЦИАЛЬНОЙ ЯМЫ В НАНОПРОВОДЕ")
        print("="*60)
        
        # Находим минимальный потенциал и его местоположение
        min_potential_idx = np.argmin(potentials)
        min_potential = potentials[min_potential_idx]
        min_location = points[min_potential_idx]
        
        print(f"Минимальный потенциал: {min_potential:.6f} V")
        print(f"Местоположение минимума: X={min_location[0]:.3f}, Y={min_location[1]:.3f}, Z={min_location[2]:.3f}")
        
        # Глубина потенциальной ямы относительно затвора
        gate_potential = 10.0
        well_depth = gate_potential - min_potential
        print(f"Глубина потенциальной ямы: {well_depth:.6f} V")
        
        # Анализ в центральной части
        center_box = self.nano_system.nano_bridge.get_boxes()[1]
        center_bounds = center_box.GetBounds()
        
        center_mask = (points[:, 0] >= center_bounds[0]) & (points[:, 0] <= center_bounds[1])
        center_potentials = potentials[center_mask]
        center_points = points[center_mask]
        
        if len(center_potentials) > 0:
            center_min = np.min(center_potentials)
            center_max = np.max(center_potentials)
            center_avg = np.mean(center_potentials)
            
            print(f"\nВ центральной части нанопровода:")
            print(f"  Минимальный потенциал: {center_min:.6f} V")
            print(f"  Максимальный потенциал: {center_max:.6f} V") 
            print(f"  Средний потенциал: {center_avg:.6f} V")
            print(f"  Размах: {center_max - center_min:.6f} V")
            
            # Находим градиент в центральной части
            if len(center_points) > 1:
                sorted_center_indices = np.argsort(center_points[:, 0])
                sorted_center_x = center_points[sorted_center_indices, 0]
                sorted_center_pot = center_potentials[sorted_center_indices]
                
                # Вычисляем градиент
                gradient = np.gradient(sorted_center_pot, sorted_center_x)
                max_gradient = np.max(np.abs(gradient))
                print(f"  Максимальный градиент: {max_gradient:.6f} V/мкм")
        
        # Сравнение с краевыми частями
        left_box = self.nano_system.nano_bridge.get_boxes()[0]
        right_box = self.nano_system.nano_bridge.get_boxes()[2]
        
        left_mask = (points[:, 0] >= left_box.GetBounds()[0]) & (points[:, 0] <= left_box.GetBounds()[1])
        right_mask = (points[:, 0] >= right_box.GetBounds()[0]) & (points[:, 0] <= right_box.GetBounds()[1])
        
        left_avg = np.mean(potentials[left_mask]) if np.any(left_mask) else 0
        right_avg = np.mean(potentials[right_mask]) if np.any(right_mask) else 0
        
        print(f"\nСравнение частей нанопровода:")
        print(f"  Левая часть (средний): {left_avg:.6f} V")
        print(f"  Центральная часть (средний): {center_avg:.6f} V") 
        print(f"  Правая часть (средний): {right_avg:.6f} V")
        
        # Определяем наличие потенциальной ямы
        if center_avg < left_avg and center_avg < right_avg:
            print(f"\n✓ ОБНАРУЖЕНА ПОТЕНЦИАЛЬНАЯ ЯМА!")
            print(f"  Центральная часть имеет более низкий потенциал, чем края")
            barrier_height = min(left_avg, right_avg) - center_avg
            print(f"  Высота барьера: {barrier_height:.6f} V")
        else:
            print(f"\n✗ Потенциальная яма не обнаружена")
            print(f"  Центральная часть имеет более высокий потенциал, чем края")
    
    def visualize_3d_potential_well(self):
        """3D визуализация потенциальной ямы в нанопроводе"""
        if not self.extract_nanowire_data():
            return
        
        points = self.nanowire_points
        potentials = self.nanowire_potentials
        
        # Создаем облако точек PyVista
        point_cloud = pv.PolyData(points)
        point_cloud["potential"] = potentials
        
        # Получаем границы нанопровода
        nano_bridge = self.nano_system.nano_bridge
        nanowire_bounds = nano_bridge.get_total_bounds()
        
        # Создаем структурированную сетку для всего пространства
        X, Y, Z = self.field_solver.grid
        grid = pv.StructuredGrid(X, Y, Z)
        grid["potential"] = self.field_solver.potential.ravel(order='F')
        
        # Вырезаем область нанопровода
        nanowire_box = pv.Box(bounds=nanowire_bounds)
        nanowire_region = grid.clip_box(nanowire_box, invert=False)
        
        plotter = pv.Plotter(shape=(2, 2))
        
        # 1. 3D облако точек нанопровода
        plotter.subplot(0, 0)
        plotter.add_mesh(point_cloud, scalars="potential", cmap='coolwarm', 
                        point_size=8, opacity=0.8, render_points_as_spheres=True,
                        scalar_bar_args={'title': "Потенциал, V"})
        plotter.add_mesh(nanowire_box, color='white', opacity=0.1, 
                       show_edges=True, line_width=2)
        plotter.add_axes()
        plotter.add_title("3D распределение потенциала\nв нанопроводе")
        
        # 2. Изоповерхности внутри нанопровода
        plotter.subplot(0, 1)
        if nanowire_region.n_points > 0:
            # Создаем изоповерхности для визуализации ямы
            contours = nanowire_region.contour(isosurfaces=10)
            plotter.add_mesh(contours, cmap='viridis', opacity=0.8,
                           scalar_bar_args={'title': "Потенциал, V"})
            plotter.add_title("Изоповерхности потенциала\nв нанопроводе")
        
        # 3. Срез через центр нанопровода (XZ плоскость)
        plotter.subplot(1, 0)
        center_y = (nanowire_bounds[2] + nanowire_bounds[3]) / 2
        slice_y = grid.slice(normal='y', origin=[0, center_y, 0])
        
        # Обрезаем срез до области нанопровода
        slice_clipped = slice_y.clip_box(nanowire_box, invert=False)
        plotter.add_mesh(slice_clipped, scalars="potential", cmap='hot', 
                        scalar_bar_args={'title': "Потенциал, V"})
        plotter.add_title(f"Срез XZ через центр (Y={center_y:.1f})")
        
        # 4. Срез через минимальный потенциал
        plotter.subplot(1, 1)
        min_potential_idx = np.argmin(potentials)
        min_location = points[min_potential_idx]
        
        slice_x = grid.slice(normal='x', origin=[min_location[0], 0, 0])
        slice_x_clipped = slice_x.clip_box(nanowire_box, invert=False)
        plotter.add_mesh(slice_x_clipped, scalars="potential", cmap='plasma',
                        scalar_bar_args={'title': "Потенциал, V"})
        plotter.add_title(f"Срез YZ через минимум (X={min_location[0]:.1f})")
        
        # Добавляем маркер минимальной точки
        min_point = pv.PolyData([min_location])
        plotter.add_mesh(min_point, color='red', point_size=20, 
                       render_points_as_spheres=True, label='Минимум потенциала')
        
        plotter.show()
    
    def plot_potential_contour_map(self):
        """2D контурная карта потенциала в плоскости нанопровода"""
        if self.field_solver.potential is None:
            return
        
        X, Y, Z = self.field_solver.grid
        potential = self.field_solver.potential
        
        # Находим срез через середину высоты нанопровода
        nano_bridge = self.nano_system.nano_bridge
        center_box = nano_bridge.get_boxes()[1]
        center_bounds = center_box.GetBounds()
        center_z = (center_bounds[4] + center_bounds[5]) / 2
        
        # Находим ближайший индекс по Z
        z_idx = np.argmin(np.abs(Z[0,0,:] - center_z))
        
        # Создаем 2D срез
        potential_slice = potential[:, :, z_idx]
        X_slice = X[:, :, z_idx]
        Y_slice = Y[:, :, z_idx]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Контурная карта
        contours = ax1.contour(X_slice, Y_slice, potential_slice, levels=20, colors='black', alpha=0.6)
        ax1.clabel(contours, inline=True, fontsize=8)
        im = ax1.contourf(X_slice, Y_slice, potential_slice, levels=50, cmap='viridis')
        plt.colorbar(im, ax=ax1, label='Потенциал (V)')
        
        # Отмечаем область нанопровода
        for box in nano_bridge.get_boxes():
            bounds = box.GetBounds()
            rect = plt.Rectangle((bounds[0], bounds[2]), bounds[1]-bounds[0], bounds[3]-bounds[2],
                               fill=False, edgecolor='red', linewidth=2, label='Нанопровод')
            ax1.add_patch(rect)
        
        ax1.set_xlabel('X (мкм)')
        ax1.set_ylabel('Y (мкм)')
        ax1.set_title('Контурная карта потенциала в плоскости нанопровода')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # 2. 3D поверхность потенциала
        from mpl_toolkits.mplot3d import Axes3D
        ax2 = fig.add_subplot(122, projection='3d')
        
        # Уменьшаем разрешение для лучшей визуализации
        stride = max(1, X_slice.shape[0] // 50)
        X_plot = X_slice[::stride, ::stride]
        Y_plot = Y_slice[::stride, ::stride]
        Z_plot = potential_slice[::stride, ::stride]
        
        surf = ax2.plot_surface(X_plot, Y_plot, Z_plot, cmap='coolwarm', 
                               alpha=0.8, linewidth=0, antialiased=True)
        
        ax2.set_xlabel('X (мкм)')
        ax2.set_ylabel('Y (мкм)')
        ax2.set_zlabel('Потенциал (V)')
        ax2.set_title('3D поверхность потенциала')
        
        plt.colorbar(surf, ax=ax2, label='Потенциал (V)', shrink=0.6)
        
        plt.tight_layout()
        plt.show()

# Добавляем вызов анализатора в основную программу
def main_with_potential_well_analysis():
    """Основная функция с анализом потенциальной ямы"""
    print("=" * 60)
    print("АНАЛИЗ ПОТЕНЦИАЛЬНОЙ ЯМЫ В НАНОПРОВОДЕ")
    print("=" * 60)
    
    # Создаем систему
    nano_system = CompleteNanoSystem()
    nano_system.create_complete_system()
    
    # Создаем solver
    field_solver = ElectricFieldSolver(nano_system, grid_resolution=60)
    
    # Проверяем наличие сохраненных результатов
    saved_file = "electric_field_results.pkl"
    if os.path.exists(saved_file):
        print(f"Найден сохраненный файл: {saved_file}")
        choice = input("Загрузить результаты из файла? (y/n): ")
        if choice.lower() == 'y':
            if field_solver.load_results(saved_file):
                # Создаем анализатор потенциальной ямы
                well_analyzer = NanowirePotentialAnalyzer(field_solver, nano_system)
                
                print("\n1. Анализ профиля потенциала:")
                well_analyzer.plot_potential_profile_along_nanowire()
                
                print("\n2. 3D визуализация потенциальной ямы:")
                well_analyzer.visualize_3d_potential_well()
                
                print("\n3. Контурные карты:")
                well_analyzer.plot_potential_contour_map()
                
                return
    
    # Выполняем новый расчет
    print("Выполнение нового расчета...")
    field_solver.solve_laplace_sor(gate_potential=10.0)
    
    # Анализ потенциальной ямы
    well_analyzer = NanowirePotentialAnalyzer(field_solver, nano_system)
    
    print("\n1. Детальный анализ потенциальной ямы:")
    well_analyzer.plot_potential_profile_along_nanowire()
    
    print("\n2. 3D визуализация:")
    well_analyzer.visualize_3d_potential_well()
    
    print("\n3. Контурные карты:")
    well_analyzer.plot_potential_contour_map()

# Быстрая функция для анализа из файла
def quick_potential_well_analysis(filename="electric_field_results.pkl"):
    """Быстрый анализ потенциальной ямы из файла"""
    print("=" * 60)
    print("БЫСТРЫЙ АНАЛИЗ ПОТЕНЦИАЛЬНОЙ ЯМЫ ИЗ ФАЙЛА")
    print("=" * 60)
    
    # Создаем систему (только геометрия)
    nano_system = CompleteNanoSystem()
    nano_system.create_complete_system()
    
    # Создаем solver и загружаем результаты
    field_solver = ElectricFieldSolver(nano_system)
    
    if field_solver.load_results(filename):
        # Создаем анализатор
        well_analyzer = NanowirePotentialAnalyzer(field_solver, nano_system)
        
        print("1. Профиль потенциала вдоль нанопровода:")
        well_analyzer.plot_potential_profile_along_nanowire()
        
        print("2. 3D визуализация потенциальной ямы:")
        well_analyzer.visualize_3d_potential_well()
        
        print("3. Контурные карты:")
        well_analyzer.plot_potential_contour_map()
    else:
        print("Не удалось загрузить результаты расчета!")

def test_substrate_is_dielectric(self):
    """Тест, что подложка ведет себя как диэлектрик"""
    print("\n" + "="*50)
    print("ТЕСТ: ПОДЛОЖКА - ДИЭЛЕКТРИК")
    print("="*50)
    
    X, Y, Z = self.solver.grid
    potential = self.solver.potential
    
    # Собираем потенциалы в подложке
    substrate_potentials = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                point = (X[i,j,k], Y[i,j,k], Z[i,j,k])
                if self.solver.is_point_in_material(point, 'substrate'):
                    substrate_potentials.append(potential[i,j,k])
    
    if substrate_potentials:
        substrate_mean = np.mean(substrate_potentials)
        substrate_std = np.std(substrate_potentials)
        
        print(f"Потенциал в подложке:")
        print(f"  Среднее: {substrate_mean:.3f} V")
        print(f"  Стандартное отклонение: {substrate_std:.3f} V")
        print(f"  Диапазон: [{np.min(substrate_potentials):.3f}, {np.max(substrate_potentials):.3f}] V")
        
        # Критерии диэлектрика:
        # 1. Потенциал НЕ должен быть фиксированным 0V
        # 2. Должны быть вариации потенциала
        is_dielectric = True
        
        if abs(substrate_mean) < 0.1 and substrate_std < 0.1:
            print("✗ ПОДЛОЖКА ВЕДЕТ СЕБЯ КАК ПРОВОДНИК (почти постоянный 0V)")
            is_dielectric = False
        else:
            print("✓ Подложка ведет себя как диэлектрик (потенциал изменяется)")
            
        # Проверяем, что есть точки с ненулевым потенциалом
        nonzero_points = sum(1 for p in substrate_potentials if abs(p) > 0.1)
        if nonzero_points > 0:
            print(f"✓ Найдено точек с ненулевым потенциалом: {nonzero_points}")
        else:
            print("✗ Все точки подложки имеют почти нулевой потенциал")
            is_dielectric = False
            
        return is_dielectric
    else:
        print("✗ Не найдено точек в подложке")
        return False
    
if __name__ == "__main__":
    # Основной запуск
    main()
    # Запуск анализа потенциальной ямы
    # main_with_potential_well_analysis()
    
    # Или быстрый анализ из файла:
    # quick_potential_well_analysis()