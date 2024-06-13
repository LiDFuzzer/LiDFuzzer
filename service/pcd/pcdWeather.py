import math
import numpy as np
import random
import open3d as o3d
import time

def augment_rain(pcdArr, intensities, indices, ran):
    random.seed(time.time())
    """Simulate 'rain' by randomly reducing points and adjusting intensities."""
    # Define rain parameters
    # quantity = np.linspace(1, 10, 10)
    # changerate = np.linspace(0.475, 0.25, 10)
    quantity = np.linspace(2, 6, 3)
    changerate = np.linspace(0.95, 0.85, 3)
    if ran == None:
        ran = random.randint(0, 2)

    # Reduce the number of points
    if indices is None:
        new_indices = np.random.choice(len(pcdArr), int(len(pcdArr) * changerate[ran]), replace=False)
    else:
        new_indices = indices[indices < len(intensities)]

    # 使用过滤后的索引访问数组
    reduced_cloud = pcdArr[new_indices]
    reduced_intensities = intensities[new_indices]

    # reduced_semantics = semantics[indices]
    # reduced_instancs = instances[indices]

    # Adjust intensities based on 'rain' effect
    for i, point in enumerate(reduced_cloud):
        d = np.linalg.norm(point)
        result = math.exp(-0.02 * quantity[ran] ** 0.6 * d)
        reduced_intensities[i] *= result

    return reduced_cloud, reduced_intensities, new_indices, ran

def augment_snow(points, intensity, new_points, ran):
    random.seed(time.time())
    # quantity = np.linspace(1, 2, 10)
    # changerate = np.linspace(0.025, 0.25, 10)
    quantity = np.linspace(2,6,3)
    changerate = np.linspace(0.05, 0.15, 3)

    if ran == None:
        ran = random.randint(0, 2)

    cloud_filtered_size = int(points.shape[0] * changerate[ran])

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)

    min_bound = cloud.get_min_bound()
    max_bound = cloud.get_max_bound()

    density = 8
    para_nd = [(max(abs(min_bound[i]), abs(max_bound[i])) ** 2) / 9 / density / density for i in range(3)]

    if new_points is None:
        new_points = np.empty((0, 3))

        while new_points.shape[0] < cloud_filtered_size:
            remaining_points = cloud_filtered_size - new_points.shape[0]
            
            generated_points = np.random.normal(0, para_nd, size=(remaining_points, 3))
            
            # 过滤超出原始点云边界的点
            valid_indices = np.all([generated_points[:, i] > min_bound[i] for i in range(3)] +
                                [generated_points[:, i] < max_bound[i] for i in range(3)], axis=0)
            valid_points = generated_points[valid_indices]
            
            # 将有效的点添加到new_points数组中
            new_points = np.vstack((new_points, valid_points))

            # 如果生成的有效点过多，则截断new_points以确保其大小正好是cloud_filtered_size
            if new_points.shape[0] > cloud_filtered_size:
                new_points = new_points[:cloud_filtered_size, :]

    # 将新增点添加到点云中
    all_points = np.vstack([points, new_points])
    cloud.points = o3d.utility.Vector3dVector(points)


    add_intensity = np.zeros(len(new_points))

    new_intensity = np.hstack([intensity, add_intensity])

    pcd_tree = o3d.geometry.KDTreeFlann(cloud)
    for i in range(len(points), len(all_points)):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(all_points[i], 5)
        if k > 0:
            new_intensity[i] = np.mean([intensity[index] for index in idx[1:] if index < len(intensity)])

    resparameter = quantity[ran] ** 0.5
    for i, point in enumerate(all_points):
        d = np.linalg.norm(point)
        result = math.exp(-0.01 * resparameter * d)
        new_intensity[i] *= result

    return all_points, new_intensity, new_points, ran


def augment_fog(points, intensity, additional_points, fog_intensities, ran):
    random.seed(time.time())
    # quantity = np.linspace(1, 10, 10)
    # changerate = np.linspace(2.5, 25, 10)
    quantity = np.linspace(2, 6, 3)
    changerate = np.linspace(10, 30, 3)

    if ran is None:
        ran = random.randint(0, 2)

    R = 4
    H = 3
    K = 5

    density = changerate[ran]
    cloudfilteredsize = int(360 * density)

    if additional_points is None and fog_intensities is None:
        additional_points = []
        fog_intensities = []

        pcd_snow = o3d.geometry.PointCloud()
        pcd_snow.points = o3d.utility.Vector3dVector(points)

        kdtree = o3d.geometry.KDTreeFlann(pcd_snow)

        for _ in range(cloudfilteredsize):
            angle = random.uniform(0, 2 * math.pi)
            x = R * math.cos(angle)
            y = R * math.sin(angle)
            z = 0.0
            while z < H:
                z += (H / density)
                additional_points.append([x, y, z])
                [k, idx, _] = kdtree.search_knn_vector_3d([x, y, z], K)
                if k > 0:
                    sum_intensity = np.mean([intensity[index] for index in idx[1:] if index < len(intensity)])
                    fog_intensities.append(sum_intensity)

    additional_points = np.array(additional_points)
    all_points = np.vstack((points, additional_points))
    all_intensities = np.concatenate((intensity, fog_intensities))

    resparameter = quantity[ran] ** 0.7
    distances = np.linalg.norm(all_points, axis=1)
    results = np.exp(-0.03 * resparameter * distances)
    all_intensities *= results

    return all_points, all_intensities, additional_points, fog_intensities, ran