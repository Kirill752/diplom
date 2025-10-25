import gmsh
import sys
from lib.model import  NanoBridge, Envelope, Box


gmsh.initialize()
gmsh.model.add("structured_block_specimen")

b1 = NanoBridge(mesh_size=1)
b1.create_nano_bridge_geometry()
b1.create_oxide()

(_, c, _) = b1.get_boxes()

center = c.GetCenter()

gate = Envelope(
    bridge_center_x=center.x, bridge_center_y=center.y, bridge_center_z=center.z,
    grip_width=5+1, grip_height=5+0.3,
    envelope_gap=0, envelope_thickness=2, envelope_length=2,
    mesh_size=0.2
)
gate.create_c_shaped() 

gmsh.model.geo.synchronize()

b1.generate_and_show()
gmsh.model.mesh.generate(3)

output_filename = "structured_blocks.msh"
gmsh.write(output_filename)
print(f"Модель сохранена в файл '{output_filename}'")

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()