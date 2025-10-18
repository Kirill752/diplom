import gmsh
from typing import List, Tuple
from lib.entity import Point, Line, Surface, Volume


class Box:
    """Класс для создания параллелепипеда"""
    
    def __init__(self, x: float, y: float, z: float, dx: float, dy: float, dz: float, mesh_size: float = None):
        """
        Создает параллелепипед
        
        Args:
            origin: начальная точка (минимальные координаты)
            dx: размер по X
            dy: размер по Y  
            dz: размер по Z
            mesh_size: опциональный размер сетки
        """
        self.x = x
        self.y = y
        self.z = z
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.mesh_size = mesh_size
        self.volume = self._create_box()
    
    def _create_box(self) -> Volume:
        """Создает параллелепипед и возвращает Volume"""
        x, y, z = self.x, self.y, self.z
        dx, dy, dz = self.dx, self.dy, self.dz
        
        # Создание точек для параллелепипеда
        p1 = Point(x, y, z)
        p2 = Point(x + dx, y, z)
        p3 = Point(x + dx, y + dy, z)
        p4 = Point(x, y + dy, z)
        p5 = Point(x, y, z + dz)
        p6 = Point(x + dx, y, z + dz)
        p7 = Point(x + dx, y + dy, z + dz)
        p8 = Point(x, y + dy, z + dz)
        
        # Создание линий
        # Нижняя грань
        l1 = Line(p1, p2)
        l2 = Line(p2, p3)
        l3 = Line(p3, p4)
        l4 = Line(p4, p1)
        
        # # Верхняя грань
        l5 = Line(p5, p6)
        l6 = Line(p6, p7)
        l7 = Line(p7, p8)
        l8 = Line(p8, p5)
        
        # Вертикальные линии
        l9 = Line(p1, p5)
        l10 = Line(p2, p6)
        l11 = Line(p3, p7)
        l12 = Line(p8, p4)
        
        # Создание поверхностей
        bottom = Surface([l1, l2, l3, l4], self.mesh_size)  # нижняя
        top = Surface([l5, l6, l7, l8], self.mesh_size)     # верхняя
        front = Surface([l1, l10, l5, l9], self.mesh_size)  # передняя
        back = Surface([l11, l7, l12, l3], self.mesh_size)  # задняя
        left = Surface([l4, l9, l8, l12], self.mesh_size)   # левая
        right = Surface([l2, l11, l6, l10], self.mesh_size) # правая
        
        # Создание объема
        return Volume([bottom, top, front, back, left, right], self.mesh_size)
    
    def GetVolume(self) -> Volume:
        """Возвращает созданный объем"""
        return self.volume
    
    def GetId(self) -> int:
        """Возвращает ID объема"""
        return self.volume.GetId()
    
    def GetDimensions(self) -> Tuple[float, float, float]:
        """Возвращает размеры параллелепипеда"""
        return (self.dx, self.dy, self.dz)
    
    def GetOrigin(self) -> Point:
        """Возвращает начальную точку"""
        return self.origin
    
    def __repr__(self):
        return f"Box(origin={self.origin}, dx={self.dx}, dy={self.dy}, dz={self.dz}, id={self.GetId()})"
        

class NanoBridge:
    """Класс для создания наномостика в форме собачей кости на основе 3 объектов Box"""
    
    def __init__(self, total_length: float = 12, grip_width: float = 2, 
                 end_size: float = 3, thickness: float = 2, mesh_size: float = 0.5):
        self.total_length = total_length
        self.grip_width = grip_width
        self.end_size = end_size
        self.thickness = thickness
        self.mesh_size = mesh_size
        
        self.boxes = []
        self.volumes = []
    
    def create_nano_bridge_geometry(self):
        """Создает геометрию наномостика в форме собачей кости"""
        L, G, E, T = self.total_length, self.grip_width, self.end_size, self.thickness
        
        # Расчет позиций для трех Box объектов
        
        # Левый концевой бокс (утолщение)
        left_box = Box(
            -L/2, -E/2, -T/2,  # origin
            E, E, T,                   # dimensions (куб)
            self.mesh_size
        )
        
        # Центральный бокс (узкая часть - ручка)
        center_box = Box(
            -L/2 + E, -E/4, -T/2,  # origin
            L - 2*E, E/2, T,                   # dimensions (узкий параллелепипед)
            self.mesh_size
        )
        
        # Правый концевой бокс (утолщение)
        right_box = Box(
            L/2 - E, -E/2, -T/2,  # origin
            E, E, T,                      # dimensions (куб)
            self.mesh_size
        )
        
        self.boxes = [left_box, center_box, right_box]
        self.volumes = [box.GetVolume() for box in self.boxes]
    
    def get_volumes(self) -> List[Volume]:
        """Возвращает список объемов наномостика"""
        return self.volumes
    
    def get_boxes(self) -> List[Box]:
        """Возвращает список Box объектов"""
        return self.boxes
    
    def get_volume_ids(self) -> List[int]:
        """Возвращает список ID объемов"""
        return [volume.GetId() for volume in self.volumes]
    
    def generate_and_show(self):
        """Генерирует сетку и показывает модель"""
        try:
            # self.model.synchronize()
            
            # Настройка физических групп для удобства постобработки
            volume_ids = self.get_volume_ids()
            gmsh.model.addPhysicalGroup(3, volume_ids, name="NanoBridge")
            
            # Добавляем физические группы для отдельных частей
            if len(self.boxes) >= 3:
                gmsh.model.addPhysicalGroup(3, [self.boxes[0].GetId()], name="LeftEnd")
                gmsh.model.addPhysicalGroup(3, [self.boxes[1].GetId()], name="Grip")
                gmsh.model.addPhysicalGroup(3, [self.boxes[2].GetId()], name="RightEnd")
            
            # Генерация сетки
            # self.model.generate_mesh(3)
            
            # Сохранение результатов в разных форматах
            # self.model.save_mesh("nano_bridge.msh")
            # self.model.save_mesh("nano_bridge.stl")  # Для 3D печати
            # self.model.save_mesh("nano_bridge.vtk")  # Для анализа в ParaView
            
            print("NanoBridge model created successfully!")
            print(f"Total length: {self.total_length}")
            print(f"Grip width: {self.grip_width}")
            print(f"End size: {self.end_size}")
            print(f"Thickness: {self.thickness}")
            print(f"Created {len(self.boxes)} Box objects")
            print(f"Volume IDs: {self.get_volume_ids()}")
            
            # Показ модели
        #     self.model.show()
            
        except Exception as e:
            print(f"Error generating mesh: {e}")
        # finally:
        #     self.model.finalize()


class AdvancedNanoBridge(NanoBridge):
    """Расширенный класс наномостика с дополнительными функциями"""
    
    def __init__(self, total_length: float = 12, grip_width: float = 2, 
                 end_size: float = 3, thickness: float = 2, mesh_size: float = 0.5,
                 material_properties: dict = None):
        super().__init__(total_length, grip_width, end_size, thickness, mesh_size)
        self.material_properties = material_properties or {}
    
    def create_nano_bridge_with_base(self, base_thickness: float = 0.5):
        """Создает наномостик с подложкой"""
        # Сначала создаем основную структуру
        self.create_nano_bridge_geometry()
        
        # Добавляем подложку
        base_box = Box(
            Point(-self.total_length/2 - 1, -self.end_size/2 - 1, -self.thickness/2 - base_thickness),
            self.total_length + 2,
            self.end_size + 2,
            base_thickness,
            self.mesh_size
        )
        
        self.boxes.append(base_box)
        self.volumes.append(base_box.GetVolume())
    
    def set_material_properties(self, youngs_modulus: float, poissons_ratio: float, density: float):
        """Устанавливает механические свойства материала"""
        self.material_properties = {
            'youngs_modulus': youngs_modulus,
            'poissons_ratio': poissons_ratio,
            'density': density
        }
    
    def calculate_volume(self) -> float:
        """Вычисляет общий объем наномостика"""
        total_volume = 0.0
        for box in self.boxes:
            dx, dy, dz = box.GetDimensions()
            total_volume += dx * dy * dz
        return total_volume
    
    def calculate_mass(self) -> float:
        """Вычисляет массу наномостика (требует установки density)"""
        if 'density' not in self.material_properties:
            print("Warning: Density not set. Please set material properties first.")
            return 0.0
        return self.calculate_volume() * self.material_properties['density']
    


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
        
        self.electrode_volume = None
        
    
    def create_c_shaped(self):
        """Создает C-образный электрод, огибающий наномостик с трех сторон (с разрезом)"""
        # Размеры внутренней полости (где будет наномостик)
        width = self.grip_width + 2 * self.envelope_gap + 2 * self.envelope_thickness
        height = self.grip_height + self.envelope_gap
        
        # Создаем электрод как набор из трех частей, образующих букву C
        parts = []
        
        # Левая часть (охватывает слева)
        parts.append(Box(
            self.bridge_center_x - self.envelope_length/2,
            self.bridge_center_y - width/2,
            self.bridge_center_z - height/2,
            self.envelope_length,
            self.envelope_thickness,
            height + self.envelope_thickness
        ))
        
        # Верхняя часть (охватывает сверху)
        parts.append(Box(
            self.bridge_center_x - self.envelope_length/2,
            self.bridge_center_y - self.grip_width/2,
            self.bridge_center_z + height/2,
            self.envelope_length,
            self.grip_width,
            self.envelope_thickness
        ))
        
        # Нижняя часть (охватывает снизу)
        parts.append(Box(
            self.bridge_center_x - self.envelope_length/2,
            self.bridge_center_y + 0.75,
            self.bridge_center_z - height/2,
            self.envelope_length,
            self.envelope_thickness,
            height + self.envelope_thickness
        ))