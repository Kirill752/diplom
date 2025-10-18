import gmsh
import sys
from lib.model import  NanoBridge, Envelope

gmsh.initialize()
gmsh.model.add("structured_block_specimen")

b1 = NanoBridge()
b1.create_nano_bridge_geometry()



electrode = Envelope(
    bridge_center_x=0, bridge_center_y=0, bridge_center_z=0,
    grip_width=1.5, grip_height=2,
    envelope_gap=0, envelope_thickness=0.3, envelope_length=6,
    mesh_size=0.2
)

electrode.create_c_shaped()      # Полный охват

# gmsh.model.geo.rotate([[3, 1]], 0, 0, 0, 1, 1, 1, 3.14)

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(1)

output_filename = "structured_blocks.msh"
gmsh.write(output_filename)
print(f"Модель сохранена в файл '{output_filename}'")

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()