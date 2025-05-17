import os
import atexit
from flask import Flask, request, jsonify
import geopandas as gpd
from pyproj import Transformer, CRS
import numpy as np
from shapely.geometry import LineString, Polygon as ShapelyPolygon, Point
import heapq
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins="*")

# 定义上传临时文件夹路径
UPLOAD_TEMP_FOLDER = 'upload_temp'
if not os.path.exists(UPLOAD_TEMP_FOLDER):
    os.makedirs(UPLOAD_TEMP_FOLDER)


def delete_temp_files():
    if not os.path.exists(UPLOAD_TEMP_FOLDER):
        return
    for root, dirs, files in os.walk(UPLOAD_TEMP_FOLDER, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"删除临时文件 {file_path} 出错: {e}")
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            try:
                os.rmdir(dir_path)
            except Exception as e:
                print(f"删除临时目录 {dir_path} 出错: {e}")

atexit.register(delete_temp_files)



def geojson_to_shapefile(geojson_path, output_dir=None, crs=None):
    """
    将GeoJSON文件转换为Shapefile文件集合

    参数:
    geojson_path (str): GeoJSON文件路径
    output_dir (str, optional): 输出目录，默认为GeoJSON所在目录
    crs (str or pyproj.CRS, optional): 输出Shapefile的坐标参考系统，默认为EPSG:4326

    返回:
    dict: 包含生成的Shapefile文件路径的字典
    """
    try:
        # 读取GeoJSON文件
        gdf = gpd.read_file(geojson_path)

        # 如果未指定输出目录，使用GeoJSON所在目录
        if output_dir is None:
            output_dir = os.path.dirname(geojson_path)

        # 如果未指定CRS，默认为EPSG:4326
        if crs is None:
            crs = CRS.from_epsg(4326)
        elif isinstance(crs, str):
            crs = CRS.from_string(crs)

        # 设置输出文件名（去掉扩展名）
        base_name = os.path.splitext(os.path.basename(geojson_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.shp")

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 设置Shapefile的CRS
        gdf.crs = crs

        # 保存为Shapefile
        gdf.to_file(output_path, driver='ESRI Shapefile')

        # 生成.prj文件（如果没有自动生成）
        prj_path = os.path.join(output_dir, f"{base_name}.prj")
        if not os.path.exists(prj_path):
            with open(prj_path, 'w') as prj_file:
                prj_file.write(crs.to_wkt())

        # 返回生成的文件路径
        return {
            'shp': output_path,
            'shx': os.path.join(output_dir, f"{base_name}.shx"),
            'dbf': os.path.join(output_dir, f"{base_name}.dbf"),
            'prj': prj_path
        }

    except Exception as e:
        print(f"GeoJSON转Shapefile出错: {e}")
        return None


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


def create_grid(obstacle_nodes, takeoff_points, landing_points, resolution):
    all_coords = [point[:2] for point in takeoff_points + landing_points] + [point for obs in obstacle_nodes for point
                                                                             in obs]
    min_x = min([coord[0] for coord in all_coords])
    max_x = max([coord[0] for coord in all_coords])
    min_y = min([coord[1] for coord in all_coords])
    max_y = max([coord[1] for coord in all_coords])
    x_size = int((max_x - min_x) / resolution) + 1
    y_size = int((max_y - min_y) / resolution) + 1
    grid = np.zeros((x_size, y_size))

    # 安全缓冲区（相对于网格大小的比例）
    buffer_ratio = 0.4  # 增加40%的缓冲区
    cell_buffer = resolution * buffer_ratio

    for i in range(x_size):
        for j in range(y_size):
            x1 = min_x + i * resolution
            y1 = min_y + j * resolution
            x2 = x1 + resolution
            y2 = y1 + resolution

            # 创建带缓冲区的网格单元
            grid_cell = ShapelyPolygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)])
            buffered_cell = grid_cell.buffer(cell_buffer)  # 扩大网格单元

            for obstacle in obstacle_nodes:
                obs_polygon = ShapelyPolygon(obstacle)
                if buffered_cell.intersects(obs_polygon):  # 使用带缓冲区的单元检测
                    grid[i, j] = 1
                    break
    return grid, min_x, min_y



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


def heuristic(a, b, method="欧式距离"):
    if method == "欧式距离":
        return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
    elif method == "曼哈顿距离":
        return abs(b[0] - a[0]) + abs(b[1] - a[1])


def heuristic_grid(node, goal, method="欧式距离"):
    if method == "欧式距离":
        return np.sqrt((node.x - goal.x) ** 2 + (node.y - goal.y) ** 2)
    elif method == "曼哈顿距离":
        return abs(node.x - goal.x) + abs(node.y - goal.y)


def a_star_grid(grid, start, goal, obstacle_nodes, heuristic_method="欧式距离"):
    open_list = []
    closed_list = set()
    start_node = Node(start[0], start[1], None, 0, heuristic_grid(Node(start[0], start[1]), goal, heuristic_method))
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
                                 heuristic_grid(Node(neighbor[0], neighbor[1]), goal, heuristic_method))
            if (neighbor_node.x, neighbor_node.y) in closed_list:
                continue
            heapq.heappush(open_list, (neighbor_node.f, neighbor_node))
    return None


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


def vector_a_star(start, end, obstacles, step_size, heuristic_method="欧式距离"):
    open_set = {start}
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end, heuristic_method)}
    visited = set()
    while open_set:
        current = min(open_set, key=lambda x: f_score[x])
        visited.add(current)
        distance_to_end = heuristic(current, end, heuristic_method)
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
            tentative_g_score = g_score[current] + heuristic(current, neighbor, heuristic_method)
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, end, heuristic_method)
                open_set.add(neighbor)
    return None


def vector_rl_simplify_path(path, obstacles):
    return rl_simplify_path(path, obstacles)


def get_neighbors(point, step_size, obstacle_nodes, visited):
    x, y = point
    neighbors = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
    for dx, dy in directions:
        new_point = (x + dx * step_size, y + dy * step_size)

        # 关键改进：新增路径碰撞检测
        ##############################
        # 1. 检查新点本身是否有效（原逻辑保留）
        if not is_valid_move(new_point, obstacle_nodes):
            continue  # 新点在障碍物内，跳过

        # 2. 检查当前点到新点的线段是否与障碍物相交
        # 创建当前点到新点的线段（Shapely的LineString对象）
        move_line = LineString([point, new_point])
        # 遍历所有障碍物，检测线段是否与障碍物相交
        path_collision = False
        for obstacle in obstacle_nodes:
            obs_polygon = ShapelyPolygon(obstacle)  # 假设obstacle是多边形坐标序列
            if move_line.intersects(obs_polygon):  # 线段与障碍物相交
                path_collision = True
                break
        if path_collision:
            continue  # 路径穿障，跳过该方向
        ##############################

        # 3. 检查是否已访问（原逻辑保留）
        if new_point not in visited:
            neighbors.append(new_point)
    return neighbors


def is_valid_move(point, obstacle_nodes):
    point_geom = Point(point)
    for obstacle in obstacle_nodes:
        obs_polygon = ShapelyPolygon(obstacle)
        if obs_polygon.contains(point_geom):
            return False
    return True


def path_planning(start_file, end_file, obstacle_file, height_threshold=0,
                  source_proj=None, target_proj="4326",
                  heuristic_method="欧式距离", path_method="网格方法",
                  grid_resolution=5, local_resolution_percentage=0, step_size=100):
    # 读取数据
    start_points = read_points_shp(start_file)
    end_points = read_points_shp(end_file)
    obstacle_nodes = read_obstacle_shp(obstacle_file, height_threshold)
    # 检查投影系统
    start_gdf = gpd.read_file(start_file)
    end_gdf = gpd.read_file(end_file)
    obstacle_gdf = gpd.read_file(obstacle_file)
    start_proj = start_gdf.crs.to_epsg()
    end_proj = end_gdf.crs.to_epsg()
    obstacle_proj = obstacle_gdf.crs.to_epsg()
    if start_proj != end_proj or start_proj != obstacle_proj or end_proj != obstacle_proj:
        raise ValueError("输入的SHP文件投影坐标不一致")

    # 创建投影转换器
    transformer = Transformer.from_crs(f"EPSG:{source_proj}", f"EPSG:{target_proj}", always_xy=True)
    all_paths = []
    total_length = 0
    if path_method == "网格方法":
        grid, min_x, min_y = create_grid(obstacle_nodes, start_points, end_points, grid_resolution)
        for start in start_points:
            for end in end_points:
                start_grid = (
                    int((start[0] - min_x) / grid_resolution),
                    int((start[1] - min_y) / grid_resolution))
                end_grid = (
                    int((end[0] - min_x) / grid_resolution),
                    int((end[1] - min_y) / grid_resolution))
                path_grid = a_star_grid(grid, start_grid, Node(end_grid[0], end_grid[1]), obstacle_nodes,
                                        heuristic_method)
                if path_grid:
                    path = [(start[0], start[1]),  *[(min_x + x * grid_resolution, min_y + y * grid_resolution) for x, y in path_grid],(end[0], end[1]) ]      # 固定终点为原始UTM坐标
                    if local_resolution_percentage > 0:
                        # 分层路径规划
                        grid_resolution_local = grid_resolution * local_resolution_percentage / 100
                        path = hierarchical_path_refinement(path, obstacle_nodes, grid_resolution_local)
                    simplified_points = rl_simplify_path(path, obstacle_nodes)
                    length = sum(
                        np.linalg.norm(np.array(simplified_points[i]) - np.array(simplified_points[i + 1])) for i in
                        range(len(simplified_points) - 1))
                    all_paths.append((start, end, simplified_points, length))
                else:
                    all_paths.append((start, end, [], 0))
    elif path_method == "矢量直接计算法":
        for start in start_points:
            for end in end_points:
                start_point = (start[0], start[1])
                end_point = (end[0], end[1])
                path = vector_a_star(start_point, end_point, obstacle_nodes, step_size, heuristic_method)
                if path:
                    simplified_path = rl_simplify_path(path, obstacle_nodes)
                    length = sum(
                        np.linalg.norm(np.array(simplified_path[i]) - np.array(simplified_path[i + 1])) for i in
                        range(len(simplified_path) - 1))
                    all_paths.append((start, end, simplified_path, length))
                else:
                    all_paths.append((start, end, [], 0))
    # 构建结果数据
    result = []
    for start, end, path, length in all_paths:
        path_data = []
        for point in path:
            lon, lat = transformer.transform(*point)
            path_data.append([lon, lat])
        result.append({
            "start_name": start[2],
            "end_name": end[2],
            "path": path_data,
            "length": length
        })
    return result


def hierarchical_path_refinement(path, obstacle_nodes, local_resolution):
    refined_path = path.copy()
    for i in range(len(refined_path) - 1):
        start_point = refined_path[i]
        end_point = refined_path[i + 1]
        local_min_x = min(start_point[0], end_point[0])
        local_max_x = max(start_point[0], end_point[0])
        local_min_y = min(start_point[1], end_point[1])
        local_max_y = max(start_point[1], end_point[1])
        local_obstacles = [obs for obs in obstacle_nodes if
                           any(point_in_polygon(point, obs) for point in [start_point, end_point])]
        local_start = (start_point[0], start_point[1], "", "")
        local_end = (end_point[0], end_point[1], "", "")
        local_grid, _, _ = create_grid(local_obstacles, [local_start], [local_end], local_resolution)
        local_start_grid = (
            int((local_start[0] - local_min_x) / local_resolution),
            int((local_start[1] - local_min_y) / local_resolution))
        local_end_grid = (
            int((local_end[0] - local_min_x) / local_resolution),
            int((local_end[1] - local_min_y) / local_resolution))
        path_grid_local = a_star_grid(local_grid, local_start_grid, Node(local_end_grid[0], local_end_grid[1]),
                                      local_obstacles)
        if path_grid_local:
            refined_path[i:i + 1] = [(local_min_x + x * local_resolution, local_min_y + y * local_resolution) for x, y
                                     in
                                     path_grid_local]
    return refined_path


atexit.register(delete_temp_files)
@app.route('/cesium_path_planning', methods=['POST'])
def api_path_planning():
    try:
        # 处理上传的文件
        start_file = request.files['start_file']
        end_file = request.files['end_file']
        obstacle_file = request.files['obstacle_file']

        # 保存GeoJSON文件
        start_path = os.path.join(UPLOAD_TEMP_FOLDER, start_file.filename)
        end_path = os.path.join(UPLOAD_TEMP_FOLDER, end_file.filename)
        obstacle_path = os.path.join(UPLOAD_TEMP_FOLDER, obstacle_file.filename)

        start_file.save(start_path)
        end_file.save(end_path)
        obstacle_file.save(obstacle_path)


        # 构建Shapefile文件名（去掉原文件扩展名，添加.shp）
        start_base_name = os.path.splitext(os.path.basename(start_file.filename))[0]
        end_base_name = os.path.splitext(os.path.basename(end_file.filename))[0]
        obstacle_base_name = os.path.splitext(os.path.basename(obstacle_file.filename))[0]

        # 构建完整的Shapefile路径
        start_shp_path = os.path.join(UPLOAD_TEMP_FOLDER, f"{start_base_name}.shp")
        end_shp_path = os.path.join(UPLOAD_TEMP_FOLDER, f"{end_base_name}.shp")
        obstacle_shp_path = os.path.join(UPLOAD_TEMP_FOLDER, f"{obstacle_base_name}.shp")

        # 执行转换
        geojson_to_shapefile(start_path, UPLOAD_TEMP_FOLDER)
        geojson_to_shapefile(end_path, UPLOAD_TEMP_FOLDER)
        geojson_to_shapefile(obstacle_path, UPLOAD_TEMP_FOLDER)


        # 处理其他参数
        height_threshold = float(request.form.get('height_threshold', 0))

        source_proj = request.form.get('source_proj')
        if not source_proj:
            source_proj = '4326'

        target_proj = request.form.get('target_proj', '4326')
        heuristic_method = request.form.get('heuristic_method', '欧式距离')
        path_method = request.form.get('path_method', '网格方法')

        grid_resolution = request.form.get('grid_resolution')
        if grid_resolution is None:
            raise ValueError("缺少必需参数: grid_resolution")
        grid_resolution = float(grid_resolution)

        local_resolution_percentage = float(request.form.get('local_resolution_percentage', 0))
        step_size = float(request.form.get('step_size', 100))


        # 执行路径规划（使用构建的Shapefile路径）
        result = path_planning(
            start_shp_path,
            end_shp_path,
            obstacle_shp_path,
            height_threshold,
            source_proj,
            target_proj,
            heuristic_method,
            path_method,
            grid_resolution,
            local_resolution_percentage,
            step_size
        )


        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True,port=5001)