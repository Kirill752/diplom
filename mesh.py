import gmsh
import sys


class Nanobridge:
    """
    Класс для создания и генерации сетки для модели наномостика,
    состоящей из подложки и опционального оксидного слоя.
    Использует трансфинитное разбиение для создания идеальной кубической сетки.
    """
    def __init__(self,
                 mesh_size: float = 2.0,
                 grip_length: float = 30.0,
                 gauge_length: float = 40.0,
                 grip_width: float = 25.0,
                 gauge_width: float = 15.0,
                 substrate_thickness: float = 10.0,
                 oxide_thickness: float = 4.0,
                 include_oxide: bool = True):
        """
        Инициализирует параметры модели наномостика.
        """
        self.mesh_size = mesh_size
        self.grip_length = grip_length
        self.gauge_length = gauge_length
        self.grip_width = grip_width
        self.gauge_width = gauge_width
        self.substrate_thickness = substrate_thickness
        self.oxide_thickness = oxide_thickness
        self.include_oxide = include_oxide
    def _create_geometry(self):
        """Создает 2D-геометрию (фундамент) из трех состыкованных прямоугольников."""
        print("Шаг 1: Создание 2D-фундамента...")
        total_length = 2 * self.grip_length + self.gauge_length
        x_grip1_end = -self.gauge_length / 2
        x_grip2_start = self.gauge_length / 2

        # Точки
        p = {
            1: gmsh.model.geo.addPoint(-total_length / 2, -self.grip_width / 2, 0),
            2: gmsh.model.geo.addPoint(x_grip1_end, -self.grip_width / 2, 0),
            3: gmsh.model.geo.addPoint(x_grip1_end, -self.gauge_width / 2, 0),
            4: gmsh.model.geo.addPoint(x_grip2_start, -self.gauge_width / 2, 0),
            5: gmsh.model.geo.addPoint(x_grip2_start, -self.grip_width / 2, 0),
            6: gmsh.model.geo.addPoint(total_length / 2, -self.grip_width / 2, 0),
            7: gmsh.model.geo.addPoint(total_length / 2, self.grip_width / 2, 0),
            8: gmsh.model.geo.addPoint(x_grip2_start, self.grip_width / 2, 0),
            9: gmsh.model.geo.addPoint(x_grip2_start, self.gauge_width / 2, 0),
            10: gmsh.model.geo.addPoint(x_grip1_end, self.gauge_width / 2, 0),
            11: gmsh.model.geo.addPoint(x_grip1_end, self.grip_width / 2, 0),
            12: gmsh.model.geo.addPoint(-total_length / 2, self.grip_width / 2, 0),
        }
        # Линии
        self.lines = {
            'l1': gmsh.model.geo.addLine(p[1], p[2]),
            'l2': gmsh.model.geo.addLine(p[2], p[3]),
            'l3': gmsh.model.geo.addLine(p[3], p[4]),
            'l4': gmsh.model.geo.addLine(p[4], p[5]),
            'l5': gmsh.model.geo.addLine(p[5], p[6]),
            'l6': gmsh.model.geo.addLine(p[6], p[7]),
            'l7': gmsh.model.geo.addLine(p[7], p[8]),
            'l8': gmsh.model.geo.addLine(p[8], p[9]),
            'l9': gmsh.model.geo.addLine(p[9], p[10]),
            'l10': gmsh.model.geo.addLine(p[10], p[11]),
            'l11': gmsh.model.geo.addLine(p[11], p[12]),
            'l12': gmsh.model.geo.addLine(p[12], p[1]),
        }
        # Создаем поверхности из правильных, последовательных контуров
        cl1 = gmsh.model.geo.addCurveLoop([self.lines['l1'], self.lines['l2'], self.lines['l3'], self.lines['l4'], self.lines['l5'], self.lines['l6'],
                                           self.lines['l7'], self.lines['l8'], self.lines['l9'], self.lines['l10'], self.lines['l11'], self.lines['l12']])
        self.s1 = gmsh.model.geo.addPlaneSurface([cl1])
# --- Инициализация Gmsh ---
gmsh.initialize()
gmsh.model.add("structured_block_specimen")

# ==============================================================================
# === 1. ГЛАВНЫЕ ПАРАМЕТРЫ, КОТОРЫЕ МОЖНО МЕНЯТЬ ===
# ==============================================================================

# Размер одного элемента сетки (кубика)
mesh_size = 2.0

# --- Размеры образца ---
grip_length = 30.0    # Длина одного "захвата" (крайний блок)
gauge_length = 40.0   # Длина узкого "мостика" (центральный блок)
grip_width = 25.0     # Ширина захвата
gauge_width = 15.0    # Ширина мостика
substrate_thickness = 10.0 # Толщина основной подложки
oxide_thickness = 4.0      # Толщина оксидного слоя сверху

# --- Расчетные переменные ---
total_length = 2 * grip_length + gauge_length
x_grip1_end = -gauge_length / 2
x_grip2_start = gauge_length / 2

bridge = Nanobridge()
bridge._create_geometry()
# # ==============================================================================
# # === 2. СОЗДАНИЕ 2D ГЕОМЕТРИИ (ФУНДАМЕНТА) ===
# # ==============================================================================
# print("Шаг 1: Создание 2D-фундамента из трех прямоугольников...")

# p1 = gmsh.model.geo.addPoint(-total_length / 2, -grip_width / 2, 0)
# p2 = gmsh.model.geo.addPoint(x_grip1_end,      -grip_width / 2, 0)
# p3 = gmsh.model.geo.addPoint(x_grip1_end,      grip_width / 2, 0)
# p4 = gmsh.model.geo.addPoint(-total_length / 2, grip_width / 2, 0)
# p5 = gmsh.model.geo.addPoint(x_grip1_end,      -gauge_width / 2, 0)
# p6 = gmsh.model.geo.addPoint(x_grip2_start,    -gauge_width / 2, 0)
# p7 = gmsh.model.geo.addPoint(x_grip2_start,    gauge_width / 2, 0)
# p8 = gmsh.model.geo.addPoint(x_grip1_end,      gauge_width / 2, 0)
# p9 = gmsh.model.geo.addPoint(x_grip2_start,    -grip_width / 2, 0)
# p10 = gmsh.model.geo.addPoint(total_length / 2, -grip_width / 2, 0)
# p11 = gmsh.model.geo.addPoint(total_length / 2, grip_width / 2, 0)
# p12 = gmsh.model.geo.addPoint(x_grip2_start,    grip_width / 2, 0)

# # Линии левого захвата
# l1 = gmsh.model.geo.addLine(p1, p2); l2 = gmsh.model.geo.addLine(p2, p3)
# l3 = gmsh.model.geo.addLine(p3, p4); l4 = gmsh.model.geo.addLine(p4, p1)
# # Линии мостика
# l5 = gmsh.model.geo.addLine(p5, p6); l6 = gmsh.model.geo.addLine(p6, p7)
# l7 = gmsh.model.geo.addLine(p7, p8); l8 = gmsh.model.geo.addLine(p8, p5)
# # Линии правого захвата
# l9 = gmsh.model.geo.addLine(p9, p10); l10 = gmsh.model.geo.addLine(p10, p11)
# l11 = gmsh.model.geo.addLine(p11, p12); l12 = gmsh.model.geo.addLine(p12, p9)

# # Поверхности
# s1 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])])
# s2 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l5, l6, l7, l8])])
# s3 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l9, l10, l11, l12])])

# gmsh.model.geo.synchronize()

# # ==============================================================================
# # === 3. ЗАДАНИЕ ПРАВИЛ ДЛЯ СТРУКТУРИРОВАННОЙ СЕТКИ (TRANSFINITE) ===
# # ==============================================================================
# print("Шаг 2: Задание правил трансфинитного разбиения...")

# # Рассчитываем количество ЭЛЕМЕНТОВ вдоль каждой стороны
# N_grip_len = round(grip_length / mesh_size)
# N_gauge_len = round(gauge_length / mesh_size)
# N_grip_w = round(grip_width / mesh_size)
# N_gauge_w = round(gauge_width / mesh_size)

# # Устанавливаем количество УЗЛОВ (элементов + 1) на каждой линии
# gmsh.model.geo.mesh.setTransfiniteCurve(l1, N_grip_len + 1); gmsh.model.geo.mesh.setTransfiniteCurve(l3, N_grip_len + 1, "Progression", -1)
# gmsh.model.geo.mesh.setTransfiniteCurve(l2, N_grip_w + 1); gmsh.model.geo.mesh.setTransfiniteCurve(l4, N_grip_w + 1, "Progression", -1)

# gmsh.model.geo.mesh.setTransfiniteCurve(l5, N_gauge_len + 1); gmsh.model.geo.mesh.setTransfiniteCurve(l7, N_gauge_len + 1, "Progression", -1)
# gmsh.model.geo.mesh.setTransfiniteCurve(l6, N_gauge_w + 1); gmsh.model.geo.mesh.setTransfiniteCurve(l8, N_gauge_w + 1, "Progression", -1)
# gmsh.model.geo.mesh.setTransfiniteCurve(l9, N_grip_len + 1); gmsh.model.geo.mesh.setTransfiniteCurve(l11, N_grip_len + 1, "Progression", -1)
# gmsh.model.geo.mesh.setTransfiniteCurve(l10, N_grip_w + 1); gmsh.model.geo.mesh.setTransfiniteCurve(l12, N_grip_w + 1, "Progression", -1)

# # Устанавливаем, что поверхности являются трансфинитными
# gmsh.model.geo.mesh.setTransfiniteSurface(s1)
# gmsh.model.geo.mesh.setTransfiniteSurface(s2)
# gmsh.model.geo.mesh.setTransfiniteSurface(s3)

# # Указываем, что их нужно разбивать на четырехугольники
# gmsh.model.geo.mesh.setRecombine(2, s1)
# gmsh.model.geo.mesh.setRecombine(2, s2)
# gmsh.model.geo.mesh.setRecombine(2, s3)

# gmsh.model.geo.synchronize()

# # ==============================================================================
# # === 4. ВЫДАВЛИВАНИЕ В 3D И СОЗДАНИЕ ФИЗИЧЕСКИХ ГРУПП ===
# # ==============================================================================
# print("Шаг 3: Выдавливание геометрии для создания 3D-блоков и физ. групп...")

# # Рассчитываем количество элементов по толщине
# N_sub_thick = round(substrate_thickness / mesh_size)
# N_ox_thick = round(oxide_thickness / mesh_size)

# # --- Выдавливаем ПОДЛОЖКУ (Substrate) ---
# ext1 = gmsh.model.geo.extrude([(2, s1), (2, s2), (2, s3)], 0, 0, substrate_thickness, 
#                               numElements=[N_sub_thick], recombine=True)
# substrate_vols = [ext1[1][1], ext1[5][1], ext1[9][1]]


gmsh.model.geo.synchronize()

# # --- Создаем Физические Группы ---
# gmsh.model.addPhysicalGroup(3, substrate_vols, name="Substrate")

# ==============================================================================
# === 5. ГЕНЕРАЦИЯ СЕТКИ И СОХРАНЕНИЕ ===
# ==============================================================================
print("\nШаг 4: Генерация структурированной кубической сетки...")
gmsh.model.mesh.generate(2)

# --- Сохранение и просмотр ---
output_filename = "structured_blocks.msh"
gmsh.write(output_filename)
print(f"Модель сохранена в файл '{output_filename}'")

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()