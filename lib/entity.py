import gmsh
from typing import List, Tuple

class Point:
    def __init__(self, x: float, y: float, z: float = 0):
        self.x = x
        self.y = y
        self.z = z
        self.id = -1
        self.created = False
    
    def Create(self):
        if not self.created:
            self.id = gmsh.model.occ.addPoint(self.x, self.y, self.z)
        else:
            print(f"Point {self.id} with coords {self.GetCoords()} has already created")
        return self

    def GetCoords(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)
    
    def GetId(self) -> int:
        return self.id
    
    def __repr__(self):
        return f"Point({self.x}, {self.y}, {self.z}, id={self.id})"

class Line:
    def __init__(self, p1: Point, p2: Point):
        self.p1 = p1
        self.p2 = p2
        self.id = gmsh.model.occ.addLine(p1.GetId(), p2.GetId())

    def reverse(self) -> 'Line':
        """Создает линию в обратном направлении"""
        return Line(self.p2, self.p1)
    
    def is_connected_to(self, other: 'Line') -> bool:
        """Проверяет, соединена ли эта линия с другой"""
        return (self.p2.GetId() == other.p1.GetId() or 
                self.p1.GetId() == other.p2.GetId() or
                self.p2.GetId() == other.p2.GetId() or
                self.p1.GetId() == other.p1.GetId())
    
    def get_connection_type(self, other: 'Line') -> str:
        """Возвращает тип соединения с другой линией"""
        if self.p2.GetId() == other.p1.GetId():
            return "end_to_start"
        elif self.p1.GetId() == other.p2.GetId():
            return "start_to_end" 
        elif self.p2.GetId() == other.p2.GetId():
            return "end_to_end"
        elif self.p1.GetId() == other.p1.GetId():
            return "start_to_start"
        return "not_connected"
    
    def GetPoints(self) -> Tuple[Point, Point]:
        return (self.p1, self.p2)
    
    def GetId(self) -> int:
        return self.id
    
    def __repr__(self):
        return f"Line({self.p1.id}->{self.p2.id}, id={self.id})"


class Surface:
    def __init__(self, lines: List[Line], mesh_size: float = None):
        self.lines = lines
        
        # Автоматически стыкуем линии перед созданием curve loop
        connected_lines = self._connect_lines(lines)
        curve_loop_id = gmsh.model.occ.addCurveLoop([l.GetId() for l in connected_lines])
        self.id = gmsh.model.occ.addPlaneSurface([curve_loop_id])
        
        # Автоматическая настройка сетки для четырехугольников
        if len(connected_lines) == 4:
            try:
                gmsh.model.occ.mesh.setTransfiniteSurface(self.id)
                gmsh.model.occ.mesh.setRecombine(2, self.id)
            except Exception as e:
                print(f"Warning: Could not set transfinite surface {self.id}: {e}")
        
        # Установка размера сетки если указан
        if mesh_size is not None:
            gmsh.model.occ.mesh.setSize([(0, p.GetId()) for line in connected_lines for p in line.GetPoints()], mesh_size)
    
    def _connect_lines(self, lines: List[Line]) -> List[Line]:
        """Автоматически стыкует линии в правильном порядке, переворачивая при необходимости"""
        if len(lines) < 3:
            return lines
        
        # Начинаем с первой линии
        connected = [lines[0]]
        used_indices = {0}
        current_end_point = lines[0].p2
        
        while len(connected) < len(lines):
            found_match = False
            
            for i, line in enumerate(lines):
                if i in used_indices:
                    continue
                
                # Проверяем все возможные соединения
                if line.p1.GetId() == current_end_point.GetId():
                    # Прямое соединение: конец текущей -> начало следующей
                    connected.append(line)
                    used_indices.add(i)
                    current_end_point = line.p2
                    found_match = True
                    break
                elif line.p2.GetId() == current_end_point.GetId():
                    # Обратное соединение: конец текущей -> конец следующей, нужно перевернуть
                    reversed_line = Line(line.p2, line.p1)
                    connected.append(reversed_line)
                    used_indices.add(i)
                    current_end_point = reversed_line.p2
                    found_match = True
                    break
                elif line.p2.GetId() == connected[0].p1.GetId():
                    # Соединение с начальной точкой (замыкание контура)
                    connected.append(line)
                    used_indices.add(i)
                    current_end_point = line.p1
                    found_match = True
                    break
                elif line.p1.GetId() == connected[0].p1.GetId():
                    # Обратное соединение с начальной точкой
                    reversed_line = Line(line.p2, line.p1)
                    connected.append(reversed_line)
                    used_indices.add(i)
                    current_end_point = reversed_line.p1
                    found_match = True
                    break
                else:
                    print(f"Warning: Could not properly connect line. lineID {line.GetId()}, curentEndPoint {current_end_point.GetId()}")
                    
            
            if not found_match:
                # Если не нашли соединение, пробуем найти любую неиспользованную линию
                for i, line in enumerate(lines):
                    if i not in used_indices:
                        connected.append(line)
                        used_indices.add(i)
                        current_end_point = line.p2
                        found_match = True
                        break
            
            if not found_match:
                # Если совсем не можем найти соединение, возвращаем что есть
                print(f"Warning: Could not properly connect all lines. Connected {len(connected)} of {len(lines)}")
                break
        
        # Проверяем, замкнут ли контур
        if len(connected) > 2 and connected[-1].p2.GetId() != connected[0].p1.GetId():
            print(f"Warning: Surface contour is not closed. EndPoints: 1. {connected[-1].p2.GetId()}, 2: {connected[0].p1.GetId()}")
        
        return connected
    
    def _create_curve_loop_safe(self, lines: List[Line]) -> int:
        """Безопасное создание curve loop с обработкой ошибок"""
        try:
            return gmsh.model.occ.addCurveLoop([l.GetId() for l in lines])
        except Exception as e:
            print(f"Error creating curve loop: {e}")
            print("Lines order:", [l.GetId() for l in lines])
            print("Trying alternative approach...")
            
            # Пробуем создать surface через альтернативный метод
            return self._create_surface_alternative(lines)
    
    def _create_surface_alternative(self, lines: List[Line]) -> int:
        """Альтернативный метод создания поверхности"""
        try:
            # Создаем wire из линий
            wire = gmsh.model.occ.addWire([l.GetId() for l in lines])
            # Создаем surface из wire
            surface_id = gmsh.model.occ.addPlaneSurface([wire])
            gmsh.model.occ.synchronize()
            return surface_id
        except Exception as e:
            print(f"Alternative method also failed: {e}")
            # Последняя попытка - создаем простой прямоугольник
            return self._create_fallback_surface()
    
    def _create_fallback_surface(self) -> int:
        """Создает простую поверхность как запасной вариант"""
        print("Creating fallback surface...")
        # Создаем простой квадрат 1x1
        p1 = gmsh.model.occ.addPoint(0, 0, 0)
        p2 = gmsh.model.occ.addPoint(1, 0, 0)
        p3 = gmsh.model.occ.addPoint(1, 1, 0)
        p4 = gmsh.model.occ.addPoint(0, 1, 0)
        
        l1 = gmsh.model.occ.addLine(p1, p2)
        l2 = gmsh.model.occ.addLine(p2, p3)
        l3 = gmsh.model.occ.addLine(p3, p4)
        l4 = gmsh.model.occ.addLine(p4, p1)
        
        curve_loop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
        return gmsh.model.occ.addPlaneSurface([curve_loop])
    
    def GetLines(self) -> List[Line]:
        return self.lines
    
    def GetId(self) -> int:
        return self.id
    
    def __repr__(self):
        return f"Surface(lines={len(self.lines)}, id={self.id})"

class Volume:
    def __init__(self, id = None):
        self.mesh_size = None
        self.id = id

    @classmethod
    def from_surfaces(self, surfaces: List[Surface], mesh_size: float = None):
        instance = self()
        surface_loop_id = gmsh.model.occ.addSurfaceLoop([s.GetId() for s in surfaces])
        self.id = gmsh.model.occ.addVolume([surface_loop_id])
        
        # try:
        #     if len(surfaces) == 6:
        #         gmsh.model.occ.mesh.setTransfiniteVolume(self.id)
        #         gmsh.model.occ.mesh.setRecombine(3, self.id)
        # except Exception as e:
        #     print(f"Warning: Could not set transfinite volume {self.id}: {e}")

        if mesh_size is not None:
            all_points = []
            for surface in surfaces:
                for line in surface.GetLines():
                    all_points.extend(line.GetPoints())
            unique_points = list(set(all_points))
            gmsh.model.occ.mesh.setSize([(0, p.GetId()) for p in unique_points], mesh_size)

        return instance
    
    def GetId(self) -> int:
        return self.id
    
    def __repr__(self):
        return f"Volume(id={self.id})"