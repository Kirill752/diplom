import gmsh
import numpy as np
from typing import Dict, List, Tuple

def check_mesh_file(mesh_file: str):
    """
    Проверяет содержимое .msh файла
    """
    
    gmsh.initialize()
    gmsh.open(mesh_file)
    
    print("=== Mesh File Analysis ===")
    
    # Физические группы
    physical_groups = gmsh.model.getPhysicalGroups()
    print(f"Physical groups: {len(physical_groups)}")
    for dim, tag in physical_groups:
        name = gmsh.model.getPhysicalName(dim, tag)
        entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
        print(f"  Dim {dim}, Tag {tag}: {name} (entities: {entities})")
    
    # Сущности
    for dim in [0, 1, 2, 3]:
        entities = gmsh.model.getEntities(dim)
        print(f"Dimension {dim} entities: {len(entities)}")
    
    # Узлы
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    print(f"Total nodes: {len(node_tags)}")
    
    # Элементы
    for dim in [0, 1, 2, 3]:
        element_types, element_tags, _ = gmsh.model.mesh.getElements(dim)
        total_elements = sum(len(tags) for tags in element_tags)
        print(f"Dimension {dim} elements: {total_elements}")
    
    gmsh.finalize()

def get_node_physical_groups(mesh_file: str) -> Dict[int, str]:
    """
    Возвращает словарь: node_tag -> physical_group_name
    """
    
    gmsh.initialize()
    gmsh.open(mesh_file)
    
    node_physical_map = {}
    
    # 1. Получаем информацию о физических группах
    physical_groups = {}
    physical_names = gmsh.model.getPhysicalGroups()
    for dim, tag in physical_names:
        name = gmsh.model.getPhysicalName(dim, tag)
        physical_groups[(dim, tag)] = name
    
    print("Found physical groups:")
    for (dim, tag), name in physical_groups.items():
        print(f"  Dimension {dim}, Tag {tag}: {name}")
    
    # 2. Создаем mapping: entity -> physical_group
    entity_physical_map = {}
    for (dim, phys_tag), phys_name in physical_groups.items():
        entities = gmsh.model.getEntitiesForPhysicalGroup(dim, phys_tag)
        for entity_tag in entities:
            entity_physical_map[(dim, entity_tag)] = phys_name
    
    # 3. Обрабатываем объемы (dimension 3) - самый надежный способ
    volumes = gmsh.model.getEntities(3)
    for dim, volume_tag in volumes:
        # Получаем физическую группу для этого объема
        phys_name = None
        for (ent_dim, ent_tag), name in entity_physical_map.items():
            if ent_dim == 3 and ent_tag == volume_tag:
                phys_name = name
                break
        
        if phys_name:
            # Получаем все узлы этого объема
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes(dim, volume_tag)
            for node_tag in node_tags:
                node_physical_map[node_tag] = phys_name
    
    # 4. Обрабатываем поверхности (dimension 2) для граничных узлов
    surfaces = gmsh.model.getEntities(2)
    for dim, surface_tag in surfaces:
        phys_name = None
        for (ent_dim, ent_tag), name in entity_physical_map.items():
            if ent_dim == 2 and ent_tag == surface_tag:
                phys_name = name
                break
        
        if phys_name:
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes(dim, surface_tag)
            for node_tag in node_tags:
                # Добавляем только если узел еще не был добавлен из объемов
                if node_tag not in node_physical_map:
                    node_physical_map[node_tag] = phys_name
    
    gmsh.finalize()
    return node_physical_map

def export_node_physical_groups(mesh_file: str, output_file: str):
    """
    Экспортирует информацию о принадлежности узлов к физическим группам
    """
    
    # Получаем mapping узлов к физическим группам
    node_physical_map = get_node_physical_groups(mesh_file)
    
    # Получаем координаты узлов
    gmsh.initialize()
    gmsh.open(mesh_file)
    
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    gmsh.finalize()
    
    # Создаем массив координат для быстрого доступа
    coords_dict = {}
    for i, tag in enumerate(node_tags):
        coords_dict[tag] = (node_coords[i*3], node_coords[i*3+1], node_coords[i*3+2])
    
    # Записываем результаты в файл
    with open(output_file, 'w') as f:
        f.write("Node_ID,Physical_Group,X,Y,Z\n")
        
        for node_tag in sorted(node_physical_map.keys()):
            phys_group = node_physical_map[node_tag]
            x, y, z = coords_dict.get(node_tag, (0, 0, 0))
            f.write(f"{node_tag},{phys_group},{x:.6f},{y:.6f},{z:.6f}\n")
    
    # Статистика
    stats = {}
    for phys_group in node_physical_map.values():
        stats[phys_group] = stats.get(phys_group, 0) + 1
    
    print(f"\nExported {len(node_physical_map)} nodes to {output_file}")
    print("Physical group distribution:")
    for phys_group, count in stats.items():
        print(f"  {phys_group}: {count} nodes")

# check_mesh_file("structured_blocks.msh")
export_node_physical_groups("structured_blocks.msh", "node_physical_groups.csv")