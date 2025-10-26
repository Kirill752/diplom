import gmsh
import sys
from lib.model import  NanoBridge, Envelope, Box
from typing import List


gmsh.initialize()
gmsh.model.add("structured_block_specimen")

b1 = NanoBridge(mesh_size=1)
b1.create_nano_bridge_geometry()
b1.create_oxide()

(_, c, _) = b1.get_boxes()

center = c.GetCenter()

gate = Envelope(
    bridge_center_x=center.x, bridge_center_y=center.y, bridge_center_z=center.z,
    grip_width=5 + 2 * 0.3, grip_height=5+0.3,
    envelope_gap=0, envelope_thickness=2, envelope_length=2,
    mesh_size=0.6
)
gate.create_c_shaped()

mainPart = List[Box]
mainPart = gate.GetVolumes()
leftHand = Box(mainPart[2].GetOrigin().x, mainPart[2].GetOrigin().y, mainPart[2].GetOrigin().z, mainPart[2].GetDimensions()[0] , -2 * mainPart[2].GetDimensions()[1], 1, 0.6)
rightHand = Box(mainPart[0].GetOrigin().x, mainPart[0].GetOrigin().y, mainPart[0].GetOrigin().z, mainPart[2].GetDimensions()[0] , -2 * mainPart[0].GetDimensions()[1], 1, 0.6)

gmsh.model.geo.synchronize()

b1.generate_and_show()
gate_volumes = []
gate_volumes.append(leftHand.GetVolume().GetId())
gmsh.model.setColor([(3, leftHand.GetVolume().GetId())], 173, 216, 230, 255)
gate_volumes.append(rightHand.GetVolume().GetId())
gmsh.model.setColor([(3, rightHand.GetVolume().GetId())], 173, 216, 230, 255)
for v in gate.GetVolumes():
    gmsh.model.setColor([(3, v.GetId())], 173, 216, 230, 255)
    gate_volumes.append(v.GetId())

gmsh.model.add_physical_group(3, gate_volumes, -1, "Gate")
gmsh.model.mesh.generate(3)

output_filename = "structured_blocks.msh"
gmsh.write(output_filename)
print(f"Модель сохранена в файл '{output_filename}'")

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()