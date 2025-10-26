import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import gmsh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from typing import Dict, List, Tuple

class MeshVisualizer:
    """Класс для визуализации сетки с цветами по физическим группам"""
    
    # Цвета для физических групп
    GROUP_COLORS = {
        'NanoBridge': 'red',
        'Oxide': 'blue',
        'Electrode': 'green',
        'Substrate': 'orange',
        'Grip': 'purple',
        'LeftEnd': 'yellow',
        'RightEnd': 'cyan'
    }
    
    def __init__(self, mesh_file: str):
        self.mesh_file = mesh_file
        self.node_data = None
        self.physical_groups = {}
        
    def load_mesh_data(self):
        """Загружает данные из .msh файла"""
        gmsh.initialize()
        gmsh.open(self.mesh_file)
        
        # Получаем информацию о физических группах
        physical_groups_info = gmsh.model.getPhysicalGroups()
        for dim, tag in physical_groups_info:
            name = gmsh.model.getPhysicalName(dim, tag)
            self.physical_groups[tag] = name
            print(f"Physical group: {name} (tag: {tag}, dim: {dim})")
        
        # Собираем данные о узлах и их принадлежности
        node_data = []
        
        # Обрабатываем каждую физическую группу
        for phys_tag, phys_name in self.physical_groups.items():
            entities = gmsh.model.getEntitiesForPhysicalGroup(3, phys_tag)  # только объемы
            
            for entity in entities:
                node_tags, node_coords, _ = gmsh.model.mesh.getNodes(3, entity)
                
                for i, node_tag in enumerate(node_tags):
                    x = node_coords[i*3]
                    y = node_coords[i*3 + 1]
                    z = node_coords[i*3 + 2]
                    
                    node_data.append({
                        'node_id': node_tag,
                        'x': x,
                        'y': y,
                        'z': z,
                        'physical_group': phys_name,
                        'physical_tag': phys_tag
                    })
        
        gmsh.finalize()
        
        self.node_data = pd.DataFrame(node_data)
        print(f"Loaded {len(self.node_data)} nodes")
        
        # Статистика по группам
        group_stats = self.node_data['physical_group'].value_counts()
        print("\nNode distribution by physical group:")
        for group, count in group_stats.items():
            print(f"  {group}: {count} nodes")
    
    def plot_3d_points(self, show_plot: bool = True, save_path: str = None):
        """Визуализирует точки в 3D с цветами по физическим группам"""
        if self.node_data is None:
            self.load_mesh_data()
        
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Рисуем точки для каждой физической группы
        unique_groups = self.node_data['physical_group'].unique()
        
        for group in unique_groups:
            group_data = self.node_data[self.node_data['physical_group'] == group]
            color = self.GROUP_COLORS.get(group, 'gray')
            
            ax.scatter(group_data['x'], group_data['y'], group_data['z'],
                      c=color, label=group, s=20, alpha=0.7)
        
        # Настройки графика
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Node Distribution by Physical Groups')
        ax.legend()
        
        # Настраиваем вид
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig, ax
    
    def plot_2d_projection(self, projection: str = 'xy', show_plot: bool = True, 
                          save_path: str = None):
        """Визуализирует 2D проекцию точек"""
        if self.node_data is None:
            self.load_mesh_data()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        unique_groups = self.node_data['physical_group'].unique()
        
        for group in unique_groups:
            group_data = self.node_data[self.node_data['physical_group'] == group]
            color = self.GROUP_COLORS.get(group, 'gray')
            
            if projection == 'xy':
                ax.scatter(group_data['x'], group_data['y'], c=color, 
                          label=group, s=15, alpha=0.7)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
            elif projection == 'xz':
                ax.scatter(group_data['x'], group_data['z'], c=color,
                          label=group, s=15, alpha=0.7)
                ax.set_xlabel('X')
                ax.set_ylabel('Z')
            elif projection == 'yz':
                ax.scatter(group_data['y'], group_data['z'], c=color,
                          label=group, s=15, alpha=0.7)
                ax.set_xlabel('Y')
                ax.set_ylabel('Z')
        
        ax.set_title(f'Node Distribution - {projection.upper()} Projection')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"2D plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig, ax

class InteractiveMeshVisualizer:
    """Интерактивная визуализация с использованием Plotly"""
    
    def __init__(self, mesh_file: str):
        self.mesh_file = mesh_file
        self.node_data = None
        
    def load_data(self):
        """Загружает данные (используем тот же метод)"""
        visualizer = MeshVisualizer(self.mesh_file)
        visualizer.load_mesh_data()
        self.node_data = visualizer.node_data
        return self.node_data
    
    def create_interactive_plot(self, save_path: str = None):
        """Создает интерактивный 3D график"""
        if self.node_data is None:
            self.load_data()
        
        fig = go.Figure()
        
        unique_groups = self.node_data['physical_group'].unique()
        
        for group in unique_groups:
            group_data = self.node_data[self.node_data['physical_group'] == group]
            
            # Получаем цвет из палитры
            color = MeshVisualizer.GROUP_COLORS.get(group, 'gray')
            
            fig.add_trace(go.Scatter3d(
                x=group_data['x'],
                y=group_data['y'], 
                z=group_data['z'],
                mode='markers',
                marker=dict(
                    size=3,
                    color=color,
                    opacity=0.7
                ),
                name=group,
                text=[f"Node: {row['node_id']}<br>Group: {group}" 
                      for _, row in group_data.iterrows()],
                hovertemplate='<b>Node %{text}</b><br>' +
                             'X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
            ))
        
        fig.update_layout(
            title='3D Node Distribution by Physical Groups',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1000,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Interactive plot saved to {save_path}")
        
        fig.show()
        return fig
    
    def create_2d_interactive_subplots(self, save_path: str = None):
        """Создает интерактивные 2D проекции"""
        if self.node_data is None:
            self.load_data()
        
        # Создаем subplots для трех проекций
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('XY Projection', 'XZ Projection', 'YZ Projection', '3D View'),
            specs=[[{'type': 'xy'}, {'type': 'xy'}],
                   [{'type': 'xy'}, {'type': 'scene'}]]
        )
        
        unique_groups = self.node_data['physical_group'].unique()
        
        for group in unique_groups:
            group_data = self.node_data[self.node_data['physical_group'] == group]
            color = MeshVisualizer.GROUP_COLORS.get(group, 'gray')
            
            # XY проекция
            fig.add_trace(go.Scatter(
                x=group_data['x'], y=group_data['y'],
                mode='markers', name=group,
                marker=dict(color=color, size=4, opacity=0.7),
                legendgroup=group, showlegend=True
            ), row=1, col=1)
            
            # XZ проекция  
            fig.add_trace(go.Scatter(
                x=group_data['x'], y=group_data['z'],
                mode='markers', name=group,
                marker=dict(color=color, size=4, opacity=0.7),
                legendgroup=group, showlegend=False
            ), row=1, col=2)
            
            # YZ проекция
            fig.add_trace(go.Scatter(
                x=group_data['y'], y=group_data['z'],
                mode='markers', name=group,
                marker=dict(color=color, size=4, opacity=0.7),
                legendgroup=group, showlegend=False
            ), row=2, col=1)
            
            # 3D вид
            fig.add_trace(go.Scatter3d(
                x=group_data['x'], y=group_data['y'], z=group_data['z'],
                mode='markers', name=group,
                marker=dict(color=color, size=3, opacity=0.7),
                legendgroup=group, showlegend=False
            ), row=2, col=2)
        
        fig.update_layout(
            title_text="Mesh Node Analysis - Multiple Projections",
            height=800,
            width=1200
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Subplots saved to {save_path}")
        
        fig.show()
        return fig
    
def export_visualization_data(mesh_file: str, output_csv: str = "node_visualization_data.csv"):
    """Экспортирует данные для визуализации в CSV"""
    visualizer = MeshVisualizer(mesh_file)
    visualizer.load_mesh_data()
    
    if visualizer.node_data is not None:
        visualizer.node_data.to_csv(output_csv, index=False)
        print(f"Visualization data exported to {output_csv}")
        
        # Создаем файл с описанием цветов
        color_info = []
        for group, color in MeshVisualizer.GROUP_COLORS.items():
            color_info.append(f"{group},{color}")
        
        with open("color_mapping.txt", "w") as f:
            f.write("PhysicalGroup,Color\n")
            f.write("\n".join(color_info))
        
        print("Color mapping saved to color_mapping.txt")
        
    return visualizer.node_data
    
def main():
    # Укажите путь к вашему .msh файлу
    mesh_file = "structured_blocks.msh"  # замените на ваш файл
    
    print("=== Mesh Node Visualization ===")
    
    try:
        # 1. Создаем визуализатор
        visualizer = MeshVisualizer(mesh_file)
        
        # 2. Загружаем данные
        print("Loading mesh data...")
        visualizer.load_mesh_data()
        
        # 3. Создаем 3D визуализацию
        print("Creating 3D visualization...")
        visualizer.plot_3d_points(show_plot=True, save_path="3d_nodes.png")
        
        # 4. Создаем 2D проекции
        print("Creating 2D projections...")
        visualizer.plot_2d_projection('xy', show_plot=True, save_path="2d_xy.png")
        visualizer.plot_2d_projection('xz', show_plot=True, save_path="2d_xz.png")
        visualizer.plot_2d_projection('yz', show_plot=True, save_path="2d_yz.png")
        
        # 5. Экспортируем данные
        print("Exporting data...")
        export_visualization_data(mesh_file)
        
        # 6. Интерактивная визуализация (опционально)
        try:
            print("Creating interactive visualization...")
            interactive_viz = InteractiveMeshVisualizer(mesh_file)
            interactive_viz.create_interactive_plot(save_path="interactive_plot.html")
            interactive_viz.create_2d_interactive_subplots(save_path="interactive_subplots.html")
        except ImportError:
            print("Plotly not installed. Skipping interactive plots.")
            print("Install with: pip install plotly")
        
        print("\n=== Visualization Complete ===")
        print("Generated files:")
        print("  - 3d_nodes.png (3D scatter plot)")
        print("  - 2d_xy.png, 2d_xz.png, 2d_yz.png (2D projections)")
        print("  - node_visualization_data.csv (raw data)")
        print("  - color_mapping.txt (color information)")
        print("  - interactive_plot.html (interactive 3D)")
        print("  - interactive_subplots.html (interactive subplots)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def quick_visualization(mesh_file: str):
    """Быстрая визуализация без сохранения файлов"""
    visualizer = MeshVisualizer(mesh_file)
    visualizer.load_mesh_data()
    visualizer.plot_3d_points(show_plot=True)
    # visualizer.plot_2d_projection('xy', show_plot=True)

if __name__ == "__main__":
    # Для быстрой визуализации:
    # quick_visualization("structured_blocks.msh")
    
    # Для полной визуализации с сохранением:
    main()