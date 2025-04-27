import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon
import os
import heapq
import csv
from shapely.geometry import LineString, Polygon as ShapelyPolygon, Point
from pyproj import Transformer
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import scrolledtext
import threading
import geopandas as gpd
import time
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义全局变量
all_paths = []
transformer = None
takeoff_points = []
landing_points = []
obstacle_nodes = []
grid = None
min_x = None
min_y = None
total_length = 0
current_params = []

# 新的强化学习思路的路径简化方法
def rl_simplify_path(path, obstacle_nodes):
    if len(path) < 3:
        return path
    simplified_path = path.copy()
    i = 1
    while i < len(simplified_path) - 1:
        prev_point = simplified_path[i - 1]
        current_point = simplified_path[i]
        next_point = simplified_path[i + 1]
        line = LineString([prev_point, next_point])
        intersects = False
        for obstacle in obstacle_nodes:
            obs_polygon = ShapelyPolygon(obstacle)
            if line.intersects(obs_polygon):
                intersects = True
                break
        if not intersects:
            simplified_path.pop(i)
        else:
            i += 1
    return simplified_path

def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def is_valid_move(point, obstacle_nodes):
    point_geom = Point(point)
    for obstacle in obstacle_nodes:
        obs_polygon = ShapelyPolygon(obstacle)
        if obs_polygon.contains(point_geom):
            return False
    return True

def get_neighbors(point, step_size, obstacle_nodes, visited):
    x, y = point
    neighbors = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
    for dx, dy in directions:
        new_point = (x + dx * step_size, y + dy * step_size)
        if is_valid_move(new_point, obstacle_nodes) and new_point not in visited:
            neighbors.append(new_point)
    return neighbors

# 全局路径规划函数
def global_path_planning():
    global all_paths, transformer, takeoff_points, landing_points, obstacle_nodes, grid, min_x, min_y, total_length
    takeoff_points_file = takeoff_points_entry.get()
    landing_points_file = landing_points_entry.get()
    obstacle_nodes_file = obstacle_nodes_entry.get()
    grid_resolution = float(grid_resolution_entry.get())
    source_proj = source_proj_entry.get()
    target_proj = target_proj_entry.get()

    try:
        height_threshold = float(height_entry.get())
    except ValueError:
        height_threshold = None
    PARAMETERS = {
        "takeoff_points_file": takeoff_points_file,
        "landing_points_file": landing_points_file,
        "obstacle_nodes_file": obstacle_nodes_file,
        "fig_size": (12, 12),
        "grid_resolution": grid_resolution,
        "source_proj": source_proj,
        "target_proj": target_proj
    }
    start_time = time.time()  # 记录开始时间

    def path_planning_task():
        global all_paths, transformer, takeoff_points, landing_points, obstacle_nodes, grid, min_x, min_y, total_length
        # 创建投影转换器
        transformer = Transformer.from_crs(source_proj, target_proj, always_xy=True)
        try:
            obstacle_nodes = read_obstacle_shp(obstacle_nodes_file, height_threshold=height_threshold)
        except ValueError as e:
            root.after(0, lambda: output_text.insert(tk.END, f"错误：{str(e)}\n"))
            root.after(0, lambda: progress_bar.stop())
            root.after(0, lambda: progress_bar.grid_remove())
            return
        if not all([os.path.exists(obstacle_nodes_file), os.path.exists(takeoff_points_file), os.path.exists(landing_points_file)]):
            root.after(0, lambda: result_label.config(text="部分文件未找到，请检查文件路径。"))
            root.after(0, lambda: output_text.insert(tk.END, "部分文件未找到，请检查文件路径。\n"))
            root.after(0, lambda: progress_bar.stop())
            root.after(0, lambda: progress_bar.grid_remove())
            return

        takeoff_points = read_points_shp(takeoff_points_file)
        landing_points = read_points_shp(landing_points_file)
        grid, min_x, min_y = create_grid(obstacle_nodes, takeoff_points, landing_points, grid_resolution)
        all_paths = []
        total_length = 0
        output_messages = []
        for takeoff in takeoff_points:
            for landing in landing_points:
                start_grid = (
                    int((takeoff[0] - min_x) / grid_resolution),
                    int((takeoff[1] - min_y) / grid_resolution))
                end_grid = (
                    int((landing[0] - min_x) / grid_resolution),
                    int((landing[1] - min_y) / grid_resolution))

                path_grid = a_star_grid(grid, start_grid, Node(end_grid[0], end_grid[1]), obstacle_nodes)
                if path_grid:
                    path = [(min_x + x * grid_resolution, min_y + y * grid_resolution) for x, y in path_grid]
                    simplified_points = rl_simplify_path(path, obstacle_nodes)
                    all_paths.append((takeoff, landing, simplified_points))
                    length = sum(
                        np.linalg.norm(np.array(simplified_points[i]) - np.array(simplified_points[i + 1])) for i in
                        range(len(simplified_points) - 1))
                    length = int(length)
                    total_length += length
                    output_text.insert(tk.END, f"{takeoff[2]} 到 {landing[2]} 路径已找到，长度: {length} 米\n")
                else:
                    output_text.insert(tk.END, f"未找到从 {takeoff[2]} 到 {landing[2]} 的有效路径。\n")

        end_time = time.time()
        elapsed_time = end_time - start_time
        output_text.insert(tk.END, f"路径计算消耗时间: {elapsed_time} 秒\n")  # 输出消耗时间

        root.after(0, lambda: plot_on_canvas())

        # 在最终完成时停止并隐藏进度条
        root.after(0, lambda: progress_bar.stop())
        root.after(0, lambda: progress_bar.grid_remove())

        root.after(0, lambda: result_label.config(text="路径规划完成，请确认是否保存。"))
        root.after(0, lambda: save_button.config(state=tk.NORMAL))

    # 显示并启动进度条
    root.after(0, lambda: progress_bar.grid())
    root.after(0, lambda: progress_bar.start())

    # 在新线程中运行路径规划任务
    thread = threading.Thread(target=path_planning_task)
    thread.start()


# 分层路径规划函数
def hierarchical_path_planning():
    global all_paths, transformer, takeoff_points, landing_points, obstacle_nodes, grid, min_x, min_y, total_length
    takeoff_points_file = takeoff_points_entry.get()
    landing_points_file = landing_points_entry.get()
    obstacle_nodes_file = obstacle_nodes_entry.get()
    grid_resolution_global = float(grid_resolution_entry.get())
    try:
        local_resolution_percentage = float(local_resolution_percentage_entry.get()) / 100
    except ValueError:
        root.after(0, lambda: output_text.insert(tk.END, "局部精细分辨率百分比输入无效，使用默认值20%\n"))
        local_resolution_percentage = 0.2
    grid_resolution_local = grid_resolution_global * local_resolution_percentage
    source_proj = source_proj_entry.get()
    target_proj = target_proj_entry.get()

    def path_planning_task1():
        global all_paths, transformer, takeoff_points, landing_points, obstacle_nodes, grid, min_x, min_y, total_length
        transformer = Transformer.from_crs(source_proj, target_proj, always_xy=True)
        height_threshold = height_entry.get()
        if height_threshold == "" or float(height_threshold) == 0:
            height_threshold = None
        else:
            try:
                height_threshold = float(height_threshold)
            except ValueError:
                root.after(0, lambda: output_text.insert(tk.END, "高度阈值输入无效，使用默认值0米\n"))
                height_threshold = 0

        try:
            obstacle_nodes = read_obstacle_shp(obstacle_nodes_file,
                                               height_threshold=height_threshold)
        except ValueError as e:
            root.after(0, lambda: output_text.insert(tk.END, f"错误：{str(e)}\n"))
            root.after(0, lambda: progress_bar.stop())
            root.after(0, lambda: progress_bar.grid_remove())
            return
        if not all([os.path.exists(obstacle_nodes_file), os.path.exists(takeoff_points_file), os.path.exists(landing_points_file)]):
            root.after(0, lambda: result_label.config(text="部分文件未找到，请检查文件路径。"))
            root.after(0, lambda: output_text.insert(tk.END, "部分文件未找到，请检查文件路径。\n"))
            root.after(0, lambda: progress_bar.stop())
            root.after(0, lambda: progress_bar.grid_remove())
            return

        takeoff_points = read_points_shp(takeoff_points_file)
        landing_points = read_points_shp(landing_points_file)

        grid_global, min_x, min_y = create_grid(obstacle_nodes, takeoff_points, landing_points,
                                                grid_resolution_global)

        start_time = time.time()  # 记录开始时间

        all_paths = []
        total_length = 0
        for takeoff in takeoff_points:
            for landing in landing_points:
                start_grid = (
                    int((takeoff[0] - min_x) / grid_resolution_global),
                    int((takeoff[1] - min_y) / grid_resolution_global))
                end_grid = (
                    int((landing[0] - min_x) / grid_resolution_global),
                    int((landing[1] - min_y) / grid_resolution_global))

                path_grid_global = a_star_grid(grid_global, start_grid, Node(end_grid[0], end_grid[1]), obstacle_nodes)
                if path_grid_global:
                    path_global = [(min_x + x * grid_resolution_global,
                                    min_y + y * grid_resolution_global) for
                                   x, y in
                                   path_grid_global]
                    path_local = path_global.copy()
                    for i in range(len(path_local) - 1):
                        start_point = path_local[i]
                        end_point = path_local[i + 1]
                        local_min_x = min(start_point[0], end_point[0])
                        local_max_x = max(start_point[0], end_point[0])
                        local_min_y = min(start_point[1], end_point[1])
                        local_max_y = max(start_point[1], end_point[1])

                        local_obstacles = [obs for obs in obstacle_nodes if
                                           any(point_in_polygon(point, obs) for point in [start_point, end_point])]
                        local_takeoff = (start_point[0], start_point[1], "", "")
                        local_landing = (end_point[0], end_point[1], "", "")

                        local_grid, _, _ = create_grid(local_obstacles, [local_takeoff], [local_landing],
                                                       grid_resolution_local)
                        local_start_grid = (
                            int((local_takeoff[0] - local_min_x) / grid_resolution_local),
                            int((local_takeoff[1] - local_min_y) / grid_resolution_local))
                        local_end_grid = (
                            int((local_landing[0] - local_min_x) / grid_resolution_local),
                            int((local_landing[1] - local_min_y) / grid_resolution_local))

                        path_grid_local = a_star_grid(local_grid, local_start_grid,
                                                      Node(local_end_grid[0], local_end_grid[1]), local_obstacles)
                        if path_grid_local:
                            path_local[i:i + 1] = [(local_min_x + x * grid_resolution_local,
                                                    local_min_y + y * grid_resolution_local) for
                                                   x, y in
                                                   path_grid_local]

                    simplified_points = rl_simplify_path(path_local, obstacle_nodes)
                    all_paths.append((takeoff, landing, simplified_points))
                    length = sum(
                        np.linalg.norm(np.array(simplified_points[i]) - np.array(simplified_points[i + 1])) for i in
                        range(len(simplified_points) - 1))
                    length = int(length)
                    total_length += length
                    root.after(0, lambda t=takeoff, l=landing, len=length: output_text.insert(tk.END, f"{t[2]} 到 {l[2]} 路径已找到，长度: {len} 米\n"))
                else:
                    root.after(0, lambda t=takeoff, l=landing: output_text.insert(tk.END, f"未找到从 {t[2]} 到 {l[2]} 的有效路径。\n"))

        end_time = time.time()
        elapsed_time = end_time - start_time
        root.after(0, lambda: output_text.insert(tk.END, f"路径计算消耗时间: {elapsed_time} 秒\n"))
        root.after(0, lambda: plot_on_canvas())
        root.after(0, lambda: progress_bar.stop())
        root.after(0, lambda: progress_bar.grid_remove())
        root.after(0, lambda: result_label.config(text="路径规划完成，请确认是否保存。"))
        root.after(0, lambda: save_button.config(state=tk.NORMAL))

    # 显示并启动进度条
    root.after(0, lambda: progress_bar.grid())
    root.after(0, lambda: progress_bar.start())
    # 在新线程中运行路径规划任务
    thread = threading.Thread(target=path_planning_task1)
    thread.start()

# 读取起飞点和降落点的 shp 文件
def read_points_shp(file_path):
    points = []
    gdf = gpd.read_file(file_path)
    for index, row in gdf.iterrows():
        name = row.get('name', '')
        point = row['geometry']
        if point:
            if point.geom_type == 'Point':
                lon, lat = point.x, point.y
                points.append((lon, lat, name, ""))
            elif point.geom_type == 'MultiPoint':
                for p in point.geoms:
                    lon, lat = p.x, p.y
                    points.append((lon, lat, name, ""))
    return points

# 读取障碍物的 shp 文件
def read_obstacle_shp(file_path, height_threshold=None):
    obstacle_nodes = []
    gdf = gpd.read_file(file_path)

    if 'height' not in gdf.columns and height_threshold is not None:
        raise ValueError("障碍物文件缺少'height'字段")

    for index, row in gdf.iterrows():
        if height_threshold is None:
            polygon = row['geometry']
        else:
            if row['height'] < height_threshold:
                continue
            polygon = row['geometry']

        if polygon:
            if polygon.geom_type == 'Polygon':
                coords = list(polygon.exterior.coords)
                two_d_coords = [(x, y) for x, y, *extra in coords]
                obstacle_nodes.append(two_d_coords)
            elif polygon.geom_type == 'MultiPolygon':
                for sub_polygon in polygon.geoms:
                    coords = list(sub_polygon.exterior.coords)
                    two_d_coords = [(x, y) for x, y, *extra in coords]
                    obstacle_nodes.append(two_d_coords)
    return obstacle_nodes

# 检查点是否在多边形内
def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

# 创建地图网格并标记障碍物
def create_grid(obstacle_nodes, takeoff_points, landing_points, resolution):
    all_coords = [point[:2] for point in takeoff_points + landing_points] + [point for obs in obstacle_nodes for point in obs]
    min_x = min([coord[0] for coord in all_coords])
    max_x = max([coord[0] for coord in all_coords])
    min_y = min([coord[1] for coord in all_coords])
    max_y = max([coord[1] for coord in all_coords])

    x_size = int((max_x - min_x) / resolution) + 1
    y_size = int((max_y - min_y) / resolution) + 1
    grid = np.zeros((x_size, y_size))
    for i in range(x_size):
        for j in range(y_size):
            point = (min_x + i * resolution, min_y + j * resolution)
            for obstacle in obstacle_nodes:
                if point_in_polygon(point, obstacle):
                    grid[i, j] = 1
                    break

    return grid, min_x, min_y

# 定义节点类
class Node:
    def __init__(self, x, y, parent=None, g=0, h=0):
        self.x = x
        self.y = y
        self.parent = parent
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f

# 最基础的启发式函数，使用欧几里得距离

def heuristic(a, b):
    if heuristic_var.get() == "欧式距离":
        return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
    elif heuristic_var.get() == "曼哈顿距离":
        return abs(b[0] - a[0]) + abs(b[1] - a[1])

def heuristic_grid(node, goal):
    if heuristic_var.get() == "欧式距离":
        return np.sqrt((node.x - goal.x) ** 2 + (node.y - goal.y) ** 2)
    elif heuristic_var.get() == "曼哈顿距离":
        return abs(node.x - goal.x) + abs(node.y - goal.y)

# A*算法（网格版）
def a_star_grid(grid, start, goal, obstacle_nodes):
    open_list = []
    closed_list = set()
    start_node = Node(start[0], start[1], None, 0, heuristic_grid(Node(start[0], start[1]), goal))
    heapq.heappush(open_list, (start_node.f, start_node))
    while open_list:
        _, current_node = heapq.heappop(open_list)
        if (current_node.x, current_node.y) in closed_list:
            continue
        closed_list.add((current_node.x, current_node.y))
        if current_node.x == goal.x and current_node.y == goal.y:
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1]
        neighbors = [(current_node.x + dx, current_node.y + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if
                     (dx != 0 or dy != 0)]
        for neighbor in neighbors:
            if neighbor[0] < 0 or neighbor[0] >= grid.shape[0] or neighbor[1] < 0 or neighbor[1] >= grid.shape[1]:
                continue
            if grid[neighbor[0], neighbor[1]] == 1:
                continue
            neighbor_node = Node(neighbor[0], neighbor[1], current_node, current_node.g + 1,
                                 heuristic_grid(Node(neighbor[0], neighbor[1]), goal))
            if (neighbor_node.x, neighbor_node.y) in closed_list:
                continue
            heapq.heappush(open_list, (neighbor_node.f, neighbor_node))
    return None

def calculate_grid_resolution(grid_width, grid_height):
    min_length = min(grid_width, grid_height)
    resolution = min_length / 800
    return resolution

def calculate_vector_offset(grid_width, grid_height):
    min_length = min(grid_width, grid_height)
    offset = min_length / 80
    return offset

# 新增矢量直接计算法相关函数
def vector_path_planning():
    global all_paths, transformer, total_length

    # 显示并启动进度条
    root.after(0, lambda: progress_bar.grid())
    root.after(0, lambda: progress_bar.start())

    step_size = float(step_size_entry.get())


    # 读取数据
    takeoff_points = read_points_shp(takeoff_points_entry.get())
    landing_points = read_points_shp(landing_points_entry.get())
    obstacle_nodes = read_obstacle_shp(obstacle_nodes_entry.get())

    all_paths = []
    total_length = 0
    # 记录开始时间
    start_time = time.time()
    # 矢量算法核心逻辑
    for takeoff in takeoff_points:
        for landing in landing_points:
            start_point = (takeoff[0], takeoff[1])
            end_point = (landing[0], landing[1])

            path = vector_a_star(start_point, end_point, obstacle_nodes, step_size)

            if path:
                simplified_path = vector_rl_simplify(path, obstacle_nodes)
                all_paths.append((takeoff, landing, simplified_path))

                # 计算路径长度
                length = sum(np.linalg.norm(np.array(simplified_path[i]) -
                                            np.array(simplified_path[i + 1])) for i in range(len(simplified_path) - 1))
                total_length += length
                # 在主线程中更新文本框
                root.after(0, lambda t=takeoff, l=landing, len=length: output_text.insert(tk.END, f"{t[2]} 到 {l[2]} 矢量路径找到，长度: {int(len)}米\n"))

    # 在主线程中停止并隐藏进度条
    root.after(0, lambda: progress_bar.stop())
    root.after(0, lambda: progress_bar.pack_forget())

    # 记录结束时间
    end_time = time.time()
    # 计算总运行时间
    total_time = end_time - start_time
    root.after(0, output_text.insert(tk.END, f"路径计算消耗时间: {total_time} 秒"))

def vector_a_star(start, end, obstacles, step_size):
    open_set = {start}
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}
    visited = set()
    while open_set:
        current = min(open_set, key=lambda x: f_score[x])
        visited.add(current)
        distance_to_end = heuristic(current, end)
        if distance_to_end <= step_size:
            path = []
            current_node = current
            while current_node in came_from:
                path.append(current_node)
                current_node = came_from[current_node]
            path.append(start)
            path.reverse()
            path.append(end)
            return path
        open_set.remove(current)
        neighbors = get_neighbors(current, step_size, obstacles, visited)
        for neighbor in neighbors:
            tentative_g_score = g_score[current] + heuristic(current, neighbor)
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                open_set.add(neighbor)
    return None

def vector_rl_simplify(path, obstacles):
    # 实现矢量版路径简化
    return rl_simplify_path(path, obstacles)
# 运行路径规划函数
def run_path_planning():

    def long_running_task():
        try:
            if method_var.get() == "网格方法":
                if use_hierarchical_var.get():
                    hierarchical_path_planning()
                else:
                    global_path_planning()
            else:
                vector_path_planning()
            # 在主线程中更新画布和按钮状态
            root.after(0, plot_on_canvas)
            root.after(0, lambda: save_button.config(state=tk.NORMAL))
        except Exception as e:
            # 在主线程中输出错误信息
            root.after(0, lambda: print(f"路径规划出错: {e}"))



    # 创建并启动线程
    thread = threading.Thread(target=long_running_task)
    thread.start()


# 根据选择的规划方法动态显示参数输入框
def update_parameter_fields(input_widgets_frame):
    global current_params, grid_resolution_label, local_res_label, step_size_label, step_size_entry, checkbutton, button_frame, result_label, progress_bar, output_text, rows_start
    # 清除旧参数
    for param in current_params:
        param.grid_remove()
    current_params.clear()

    row = rows_start
    if method_var.get() == "网格方法":
        grid_resolution_label.grid(row=row, column=0, in_=input_widgets_frame, padx=2, pady=2)
        grid_resolution_entry.grid(row=row, column=1, in_=input_widgets_frame, padx=2, pady=2)
        current_params.extend([grid_resolution_label, grid_resolution_entry])
        row += 1

        local_res_label.grid(row=row, column=0, in_=input_widgets_frame, padx=2, pady=2)
        local_resolution_percentage_entry.grid(row=row, column=1, in_=input_widgets_frame, padx=2, pady=2)
        current_params.extend([local_res_label, local_resolution_percentage_entry])
        row += 1

        checkbutton.grid(row=row, column=0, columnspan=3, in_=input_widgets_frame, padx=2, pady=2)
        current_params.append(checkbutton)
        update_local_resolution_entry_state()
    else:  # 矢量直接计算法
        step_size_label.grid(row=row, column=0, in_=input_widgets_frame, padx=2, pady=2)
        step_size_entry.grid(row=row, column=1, in_=input_widgets_frame, padx=2, pady=2)
        current_params.extend([step_size_label, step_size_entry])
        checkbutton.grid_remove()

    # 初始时不显示进度条
    progress_bar.pack_forget()
def create_gui():
    global root, input_frame, takeoff_points_entry, landing_points_entry, obstacle_nodes_entry, height_entry, \
        grid_resolution_entry, local_resolution_percentage_entry, source_proj_entry, target_proj_entry, \
        use_hierarchical_var, method_var, result_label, output_text, canvas_frame, save_button, progress_bar, \
        step_size_label, step_size_entry, rows_start, grid_resolution_label, local_res_label, current_params, checkbutton, button_frame, heuristic_var
    current_params = []
    # 创建 GUI 窗口
    root = tk.Tk()
    root.title("路径规划工具")
    root.geometry("1100x600")
    # 创建主框架，使用 grid 布局管理左右两部分
    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)
    # 设置主框架的行和列权重，使其能自适应窗口大小
    main_frame.grid_rowconfigure(0, weight=1)
    main_frame.grid_columnconfigure(0, weight=1)
    main_frame.grid_columnconfigure(1, weight=0)
    # 创建可视化区域（修改背景颜色为灰色）
    canvas_frame = tk.Frame(main_frame, bg="#E6E6E6")
    canvas_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
    # 设置可视化区域的行和列权重，使其能自适应窗口大小
    canvas_frame.grid_rowconfigure(0, weight=1)
    canvas_frame.grid_columnconfigure(0, weight=1)
    # 创建输入区域，固定宽度为 400 像素
    input_frame = tk.Frame(main_frame, width=400)
    input_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=2)
    input_frame.pack_propagate(0)  # 防止框架根据内容调整大小
    # 设置输入区域的行权重，使其能自适应窗口高度
    input_frame.grid_rowconfigure(0, weight=1)
    # 第一层：输入区域
    input_widgets_frame = tk.Frame(input_frame)
    input_widgets_frame.pack(fill=tk.X, expand=True, padx=10, pady=10, anchor=tk.CENTER)
    rows = 0
    # 起飞点
    tk.Label(input_widgets_frame, text="起飞点 shp :").grid(row=rows, column=0, padx=2, pady=2)
    takeoff_points_entry = tk.Entry(input_widgets_frame)
    takeoff_points_entry.grid(row=rows, column=1, padx=2, pady=2)
    tk.Button(input_widgets_frame, text="选择文件",
              command=lambda: [
                  takeoff_points_entry.insert(0, filedialog.askopenfilename(filetypes=[('Shapefiles', '*.shp')])),
                  update_visualization()]).grid(
        row=rows, column=2, padx=2, pady=2)
    rows += 1
    tk.Label(input_widgets_frame, text="降落点 shp :").grid(row=rows, column=0, padx=2, pady=2)
    landing_points_entry = tk.Entry(input_widgets_frame)
    landing_points_entry.grid(row=rows, column=1, padx=2, pady=2)
    tk.Button(input_widgets_frame, text="选择文件",
              command=lambda: [
                  landing_points_entry.insert(0, filedialog.askopenfilename(filetypes=[('Shapefiles', '*.shp')])),
                  update_visualization()]).grid(
        row=rows, column=2, padx=2, pady=2)
    rows += 1
    tk.Label(input_widgets_frame, text="障碍物 shp :").grid(row=rows, column=0, padx=2, pady=2)
    obstacle_nodes_entry = tk.Entry(input_widgets_frame)
    obstacle_nodes_entry.grid(row=rows, column=1, padx=2, pady=2)
    tk.Button(input_widgets_frame, text="选择文件",
              command=lambda: [
                  obstacle_nodes_entry.insert(0, filedialog.askopenfilename(filetypes=[('Shapefiles', '*.shp')])),
                  update_visualization()]).grid(
        row=rows, column=2, padx=2, pady=2)
    rows += 1
    tk.Label(input_widgets_frame, text="障碍物高度阈值(m):").grid(row=rows, column=0, padx=2, pady=2)
    height_entry = tk.Entry(input_widgets_frame)
    height_entry.grid(row=rows, column=1, padx=2, pady=2)
    height_entry.insert(0, "")
    # 绑定高度输入框的按键释放事件
    height_entry.bind("<KeyRelease>", lambda event: update_visualization())
    rows += 1
    # 投影输入框放在高度阈值下面
    tk.Label(input_widgets_frame, text="源投影系统 (EPSG):").grid(row=rows, column=0, padx=2, pady=2)
    source_proj_entry = tk.Entry(input_widgets_frame)
    source_proj_entry.grid(row=rows, column=1, padx=2, pady=2)
    source_proj_entry.insert(0, "EPSG:32650")
    rows += 1
    tk.Label(input_widgets_frame, text="目标投影系统 (EPSG):").grid(row=rows, column=0, padx=2, pady=2)
    target_proj_entry = tk.Entry(input_widgets_frame)
    target_proj_entry.grid(row=rows, column=1, padx=2, pady=2)
    target_proj_entry.insert(0, "EPSG:4326")
    rows += 1
    # 新增启发式函数选择下拉菜单
    tk.Label(input_widgets_frame, text="启发式函数:").grid(row=rows, column=0, padx=5, pady=5)
    heuristic_var = tk.StringVar(value="欧式距离")
    heuristic_menu = ttk.Combobox(input_widgets_frame, textvariable=heuristic_var,
                                  values=["欧式距离", "曼哈顿距离"])
    heuristic_menu.grid(row=rows, column=1, padx=5, pady=5)
    rows += 1
    tk.Label(input_widgets_frame, text="规划方法:").grid(row=rows, column=0, padx=5, pady=5)
    method_var = tk.StringVar(value="网格方法")
    method_menu = ttk.Combobox(input_widgets_frame, textvariable=method_var,
                               values=["网格方法", "矢量直接计算法"])
    method_menu.grid(row=rows, column=1, padx=5, pady=5)
    method_menu.bind("<<ComboboxSelected>>", lambda e: update_parameter_fields(input_widgets_frame))
    rows_start = rows + 1
    grid_resolution_label = tk.Label(input_widgets_frame, text="网格分辨率:")
    grid_resolution_entry = tk.Entry(input_widgets_frame)
    grid_resolution_entry.insert(0, "5")
    local_res_label = tk.Label(input_widgets_frame, text="局部精细分辨率(%):")
    local_resolution_percentage_entry = tk.Entry(input_widgets_frame)
    local_resolution_percentage_entry.insert(0, "20")
    step_size_label = tk.Label(input_widgets_frame, text="步长(米):")
    step_size_entry = tk.Entry(input_widgets_frame)
    step_size_entry.insert(0, "100")
    # 先定义 use_hierarchical_var
    use_hierarchical_var = tk.IntVar()
    checkbutton = tk.Checkbutton(input_widgets_frame, text="使用分层规划", variable=use_hierarchical_var,
                                 command=update_local_resolution_entry_state)
    # 第二层：按钮区域
    button_frame = tk.Frame(input_frame)
    button_frame.pack(fill=tk.X, expand=True, padx=10, pady=10, anchor=tk.CENTER)
    button_frame.grid_columnconfigure(0, weight=1)
    button_frame.grid_columnconfigure(1, weight=1)
    button_frame.grid_columnconfigure(2, weight=1)
    tk.Button(button_frame, text="运行路径规划", command=lambda: threading.Thread(target=run_path_planning).start()).grid(row=0, column=0, padx=2, pady=2, sticky='ew')
    save_button = tk.Button(button_frame, text="保存路径", command=save_paths, state=tk.DISABLED)
    save_button.grid(row=0, column=1, padx=2, pady=2, sticky='ew')
    reset_button = tk.Button(button_frame, text="一键取消", command=reset_all)
    reset_button.grid(row=0, column=2, padx=2, pady=2, sticky='ew')
    result_label = tk.Label(input_frame, text="")
    # 进度条区域
    progress_frame = tk.Frame(input_frame)
    progress_frame.pack(fill=tk.X, expand=True, padx=10, pady=10, anchor=tk.CENTER)
    # 进度条（初始隐藏）
    progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", length=375, mode="indeterminate")
    progress_bar.pack_forget()  # 初始时隐藏进度条
    # 第三层：打印文本框区域
    output_frame = tk.Frame(input_frame)
    output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10, anchor=tk.CENTER)
    output_text = scrolledtext.ScrolledText(output_frame, height=8, width=50)
    output_text.pack(fill=tk.BOTH, expand=True)
    # 初始化局部精细分辨率百分比输入框的状态
    update_local_resolution_entry_state()
    update_parameter_fields(input_widgets_frame)
    # 确保输入区域高度固定为网格方法时的高度
    input_widgets_frame.update_idletasks()
    input_widgets_frame_height = input_widgets_frame.winfo_height()
    input_widgets_frame.config(height=input_widgets_frame_height)
    return root

# 根据勾选框状态更新局部精细分辨率百分比输入框的状态
def update_local_resolution_entry_state():
    global use_hierarchical_var, local_resolution_percentage_entry
    if use_hierarchical_var.get():
        local_resolution_percentage_entry.config(state=tk.NORMAL, bg='white')
    else:
        local_resolution_percentage_entry.config(state=tk.DISABLED, bg='gray')

def plot_on_canvas():
    global all_paths, takeoff_points, landing_points, obstacle_nodes, canvas_frame, transformer

    # 清空之前的绘图
    for widget in canvas_frame.winfo_children():
        widget.destroy()

    # 获取 canvas_frame 的宽度和高度
    frame_width = canvas_frame.winfo_width()
    frame_height = canvas_frame.winfo_height()

    # 根据 canvas_frame 的尺寸调整图形大小
    fig_width = frame_width / 100  # 转换为英寸
    fig_height = frame_height / 100  # 转换为英寸
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # 绘制障碍物
    for obstacle in obstacle_nodes:
        polygon = Polygon(obstacle, edgecolor='red', facecolor='red', alpha=0.5)
        ax.add_patch(polygon)

    # 绘制起飞点和降落点
    takeoff_x = [point[0] for point in takeoff_points]
    takeoff_y = [point[1] for point in takeoff_points]
    landing_x = [point[0] for point in landing_points]
    landing_y = [point[1] for point in landing_points]
    ax.scatter(takeoff_x, takeoff_y, color='green', label='起飞点')
    ax.scatter(landing_x, landing_y, color='blue', label='降落点')

    # 绘制路径
    for takeoff, landing, path in all_paths:
        path_x = [point[0] for point in path]
        path_y = [point[1] for point in path]
        ax.plot(path_x, path_y, color='orange', linewidth=2)

    ax.set_xlabel('X 坐标')
    ax.set_ylabel('Y 坐标')
    ax.set_title('路径规划可视化')
    ax.legend()

    # 调整图形布局以适应 canvas_frame
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def save_paths():
    output_file = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
    if not output_file:
        return

    if method_var.get() == "网格方法":
        columns = ['起始点名称', '降落点名称']
        max_nodes = max([len(path[2]) for path in all_paths])
        for i in range(max_nodes):
            columns.extend([f'节点{i + 1}_经度', f'节点{i + 1}_纬度'])

        # 构建数据行
        data = []
        for start, end, path in all_paths:
            row = [start[2], end[2]]
            for point in path:
                lon, lat = transformer.transform(*point)
                row.extend([lon, lat])
            # 补齐空白
            if len(path) < max_nodes:
                row.extend([''] * (2 * (max_nodes - len(path))))
            data.append(row)

        # 保存为 CSV
        pd.DataFrame(data, columns=columns).to_csv(output_file, index=False)
    else:
        # 矢量方法保存
        columns = ['起始点名称', '降落点名称']
        max_nodes = max([len(path[2]) for path in all_paths])
        for i in range(max_nodes):
            columns.extend([f'节点{i + 1}_经度', f'节点{i + 1}_纬度'])

        # 构建数据行
        data = []
        for start, end, path in all_paths:
            row = [start[2], end[2]]
            for point in path:
                lon, lat = transformer.transform(*point)
                row.extend([lon, lat])
            # 补齐空白
            if len(path) < max_nodes:
                row.extend([''] * (2 * (max_nodes - len(path))))
            data.append(row)

        # 保存为 CSV
        pd.DataFrame(data, columns=columns).to_csv(output_file, index=False)

    result_label.config(text="路径规划结果已保存。")
    output_text.insert(tk.END, "路径规划结果已保存。\n")


# 更新可视化函数
def update_visualization():
    global takeoff_points, landing_points, obstacle_nodes
    takeoff_points_file = takeoff_points_entry.get()
    landing_points_file = landing_points_entry.get()
    obstacle_nodes_file = obstacle_nodes_entry.get()

    try:
        height_threshold = float(height_entry.get())
    except ValueError:
        height_threshold = None

    takeoff_points = read_points_shp(takeoff_points_file) if os.path.exists(takeoff_points_file) else []
    landing_points = read_points_shp(landing_points_file) if os.path.exists(landing_points_file) else []

    try:
        obstacle_nodes = read_obstacle_shp(obstacle_nodes_file, height_threshold) \
            if os.path.exists(obstacle_nodes_file) else []
    except Exception as e:
        output_text.insert(tk.END, f"更新可视化失败：{str(e)}\n")
        obstacle_nodes = []

    all_paths = []

    # 收集所有点的坐标
    all_coords = [point[:2] for point in takeoff_points + landing_points] + [
        point for obs in obstacle_nodes for point in obs
    ]
    min_x = min([coord[0] for coord in all_coords])
    max_x = max([coord[0] for coord in all_coords])
    min_y = min([coord[1] for coord in all_coords])
    max_y = max([coord[1] for coord in all_coords])

    grid_width = max_x - min_x
    grid_height = max_y - min_y

    # 动态计算网格分辨率
    resolution = calculate_grid_resolution(grid_width, grid_height)
    grid_resolution_entry.delete(0, tk.END)
    grid_resolution_entry.insert(0, str(resolution))
    # 动态计算步长
    offset =calculate_vector_offset(grid_width, grid_height)
    step_size_entry.delete(0, tk.END)
    step_size_entry.insert(0, str(offset))


    # 更新可视化
    plot_on_canvas()

    # 检查三个文件是否都已输入
    if takeoff_points_file and landing_points_file and obstacle_nodes_file:
        width = max_x - min_x
        height = max_y - min_y

        output_text.insert(tk.END, f"显示范围宽度: {width} 米\n")
        output_text.insert(tk.END, f"显示范围高度: {height} 米\n")

# 一键取消功能
def reset_all():
    global all_paths, takeoff_points, landing_points, obstacle_nodes
    all_paths = []
    takeoff_points = []
    landing_points = []
    obstacle_nodes = []

    takeoff_points_entry.delete(0, tk.END)
    landing_points_entry.delete(0, tk.END)
    obstacle_nodes_entry.delete(0, tk.END)
    height_entry.delete(0, tk.END)
    height_entry.insert(0, "")
    grid_resolution_entry.delete(0, tk.END)
    local_resolution_percentage_entry.delete(0, tk.END)
    local_resolution_percentage_entry.insert(0, "20")
    source_proj_entry.delete(0, tk.END)
    source_proj_entry.insert(0, "EPSG:32650")
    target_proj_entry.delete(0, tk.END)
    target_proj_entry.insert(0, "EPSG:4326")
    use_hierarchical_var.set(0)  # 取消勾选
    update_local_resolution_entry_state()

    result_label.config(text="")
    output_text.delete(1.0, tk.END)

    for widget in canvas_frame.winfo_children():
        widget.destroy()

    # 收集所有点的坐标（假设在更新可视化时已有相同的逻辑）
    all_coords = [point[:2] for point in takeoff_points + landing_points] + [
        point for obs in obstacle_nodes for point in obs
    ]
    min_x = min([coord[0] for coord in all_coords])
    max_x = max([coord[0] for coord in all_coords])
    min_y = min([coord[1] for coord in all_coords])
    max_y = max([coord[1] for coord in all_coords])

    grid_width = max_x - min_x
    grid_height = max_y - min_y

    # 动态计算网格分辨率
    resolution = calculate_grid_resolution(grid_width, grid_height)
    grid_resolution_entry.insert(0, str(resolution))

    offset =calculate_vector_offset(grid_width, grid_height)
    step_size_entry.insert(0, str(offset))
if __name__ == "__main__":
    root = create_gui()
    root.mainloop()
