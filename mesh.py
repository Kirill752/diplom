import numpy as np
import pyvista as pv
from typing import List, Tuple, Dict
import time

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

import numpy as np
import pyvista as pv
from typing import List, Tuple, Dict
import time
import pickle
import os

# [Весь предыдущий код классов Point, Box, Envelope, NanoBridge, Substrate, 
# AirEnvironment, GateElectrode, CompleteNanoSystem остается без изменений]

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

# Основная демонстрация с улучшенным управлением
def main():
    print("Создание полной системы наномостика...")
    
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
    
    # Проверяем, есть ли сохраненные результаты
    saved_file = "electric_field_results.pkl"
    if os.path.exists(saved_file):
        print(f"Найден сохраненный файл {saved_file}")
        choice = input("Загрузить результаты из файла? (y/n): ")
        if choice.lower() == 'y':
            field_solver.quick_visualize_from_file(saved_file)
            return
    
    print("\n3. Выполнение нового расчета...")
    
    # Демонстрация изменения диэлектрических проницаемостей
    print("\n4. Демонстрация влияния диэлектрических проницаемостей:")
    
    # Первый расчет с базовыми параметрами
    print("Расчет с базовыми диэлектрическими проницаемостями:")
    field_solver.solve_laplace_sor(gate_potential=10.0)
    
    # Второй расчет с измененными параметрами оксида
    print("\nРасчет с увеличенной диэлектрической проницаемостью оксида (ε=25):")
    field_solver.set_dielectric_constants({'oxide': 25.0})
    field_solver.solve_laplace_sor(gate_potential=10.0)
    field_solver.save_results("high_epsilon_oxide.pkl")
    
    print("\n5. Сравнение результатов:")
    field_solver.visualize_nanobridge_cross_sections()

if __name__ == "__main__":
    main()
