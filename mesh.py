import gmsh
import sys
from lib.model import  NanoBridge, Envelope, Box

gmsh.initialize()
gmsh.model.add("structured_block_specimen")

b1 = NanoBridge(mesh_size=1)
b1.create_nano_bridge_geometry()
b1.create_oxide()


gmsh.model.geo.synchronize()

b1.generate_and_show()
gmsh.model.mesh.generate(3)

output_filename = "structured_blocks.msh"
gmsh.write(output_filename)
print(f"Модель сохранена в файл '{output_filename}'")

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()