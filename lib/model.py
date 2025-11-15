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
    
    def GetCenter(self) -> Point:
        x, y, z = self.x, self.y, self.z
        dx, dy, dz = self.dx, self.dy, self.dz
        return Point(x + dx/2, y + dy/2, z + dz/2)

    def _create_box(self) -> Volume:
        """Создает параллелепипед и возвращает Volume"""
        x, y, z = self.x, self.y, self.z
        dx, dy, dz = self.dx, self.dy, self.dz
        
        # Создание объема
        return Volume((x, y, z, dx, dy, dz))
    
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
        return Point(self.x, self.y, self.z)
    
    def __repr__(self):
        return f"Box(origin={self.origin}, dx={self.dx}, dy={self.dy}, dz={self.dz}, id={self.GetId()})"
        

class NanoBridge:
    """Класс для создания наномостика в форме собачей кости на основе 3 объектов Box"""
    
    def __init__(self, grip_length = 10, grip_width = 5, grip_height = 5,
                 end_length = 5, end_width = 10, mesh_size = 0.2):
        self.grip_length = grip_length
        self.grip_width = grip_width
        self.grip_height = grip_height
        self.end_length = end_length
        self.end_width = end_width
        self.mesh_size = mesh_size
        
        self.boxes = []
        self.volumes = []
        self.oxide_volumes = []
    
    def create_nano_bridge_geometry(self):
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
        self.volumes = [box.GetVolume() for box in self.boxes]
    
    def create_oxide(self):
        (left, center, right) = self.get_boxes()
        leftCenter = left.GetCenter().GetCoords()
        centerCenter = center.GetCenter().GetCoords()
        rightCenter = right.GetCenter().GetCoords()
        volumes = []
        ox1 = Envelope(
            bridge_center_x=centerCenter[0], bridge_center_y=centerCenter[1], bridge_center_z=centerCenter[2],
            grip_width=self.grip_width, grip_height=self.grip_height,
            envelope_gap=0, envelope_thickness=0.3, envelope_length=self.grip_length,
            mesh_size=0.4
        )
        ox1.create_c_shaped()  
        volumes.append(ox1.GetVolumes())
        ox2 = Envelope(
            bridge_center_x=leftCenter[0], bridge_center_y=leftCenter[1], bridge_center_z=leftCenter[2],
            grip_width=self.end_width, grip_height=self.grip_height,
            envelope_gap=0, envelope_thickness=0.3, envelope_length=self.end_length,
            mesh_size=0.4
        )
        ox2.create_c_shaped()  
        volumes.append(ox2.GetVolumes())
        ox3 = Envelope(
            bridge_center_x=rightCenter[0], bridge_center_y=rightCenter[1], bridge_center_z=rightCenter[2],
            grip_width=self.end_width, grip_height=self.grip_height,
            envelope_gap=0, envelope_thickness=0.3, envelope_length=self.end_length,
            mesh_size=0.4
        )
        ox3.create_c_shaped()
        volumes.append(ox3.GetVolumes())

        for volume in volumes:
            for box in volume:
                self.oxide_volumes.append(box.GetId())
        

    def get_volumes(self) -> List[Volume]:
        """Возвращает список объемов наномостика"""
        return self.volumes
    
    def get_boxes(self) -> List[Box]:
        """Возвращает список Box объектов"""
        return self.boxes
    
    def get_volume_ids(self) -> List[int]:
        """Возвращает список ID объемов"""
        return [volume.GetId() for volume in self.volumes]
    


class AdvancedNanoBridge(NanoBridge):
    """Расширенный класс наномостика с дополнительными функциями"""
    
    def __init__(self, total_length: float = 12, grip_width: float = 2, 
                 end_size: float = 3, thickness: float = 2, mesh_size: float = 0.5,
                 material_properties: dict = None):
        super().__init__(total_length, grip_width, end_size, thickness, mesh_size)
        self.material_properties = material_properties or {}
    
    def create_nano_bridge_with_base(self, base_thickness: float = 0.5):
        self.create_nano_bridge_geometry()

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
        
        self.volumes = []
        
    def GetVolumes(self):
        return self.volumes

    def create_c_shaped(self):
        width = self.grip_width + 2 * self.envelope_gap + 2 * self.envelope_thickness
        height = self.grip_height + self.envelope_gap

        self.volumes.append(Box(
            self.bridge_center_x - self.envelope_length/2,
            self.bridge_center_y - width/2,
            self.bridge_center_z - height/2,
            self.envelope_length,
            self.envelope_thickness,
            height + self.envelope_thickness,
            self.mesh_size
        ))

        self.volumes.append(Box(
            self.bridge_center_x - self.envelope_length/2,
            self.bridge_center_y - self.grip_width/2,
            self.bridge_center_z + height/2,
            self.envelope_length,
            self.grip_width,
            self.envelope_thickness,
            self.mesh_size
        ))

        self.volumes.append(Box(
            self.bridge_center_x - self.envelope_length/2,
            self.bridge_center_y + width/2,
            self.bridge_center_z - height/2,
            self.envelope_length,
            -self.envelope_thickness,
            height + self.envelope_thickness,
            self.mesh_size
        ))