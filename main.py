import numpy as np

import taichi as ti

ti.init(ti.cuda)

# dim, n_grid, steps, dt = 2, 128, 20, 2e-4
# dim, n_grid, steps, dt = 2, 256, 32, 1e-4
# dim, n_grid, steps, dt = 3, 32, 25, 4e-4
dim, n_grid, steps, dt = 3, 64, 25, 2e-4
# dim, n_grid, steps, dt = 3, 128, 5, 1e-4

n_particles = n_grid ** dim // 2 ** (dim - 1)

print(n_particles)

dx = 1 / n_grid

p_rho = 1
p_vol = (dx * 0.5) ** 2
p_mass = p_vol * p_rho
g_x = 0
g_y = -9.8
g_z = 0
bound = 3
E = 1000  # Young's modulus
nu = 0.2  # Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / (
        (1 + nu) * (1 - 2 * nu))  # Lame parameters

x = ti.Vector.field(dim, float, n_particles)
v = ti.Vector.field(dim, float, n_particles)
C = ti.Matrix.field(dim, dim, float, n_particles)
F = ti.Matrix.field(3, 3, dtype=float,
                    shape=n_particles)  # deformation gradient
Jp = ti.field(float, n_particles)

colors = ti.Vector.field(3, float, n_particles)
colors_random = ti.Vector.field(3, float, n_particles)
colors_depth = ti.Vector.field(3, float, n_particles * 20)
colors_light = ti.Vector.field(3, float, n_particles * 20)
materials = ti.field(int, n_particles)
grid_v = ti.Vector.field(dim, float, (n_grid,) * dim)
grid_m = ti.field(float, (n_grid,) * dim)
used = ti.field(int, n_particles)
light_position = ti.Vector.field(3, float, 1)
view_position = ti.Vector.field(3, float, 1)

min_max = ti.Vector.field(2, float, 3)
wall = ti.Vector.field(3, float, 18)
wall_image = ti.Vector.field(2, float, 18)
wall_image_left = ti.Vector.field(2, float, 6)
wall_image_down = ti.Vector.field(2, float, 6)
wall_image_back = ti.Vector.field(2, float, 6)
wall_image_white = ti.Vector.field(2, float, 144)
wall_image_black = ti.Vector.field(2, float, 144)
wall_white = ti.Vector.field(3, float, 144)
wall_black = ti.Vector.field(3, float, 144)

neighbour = (3,) * dim

WATER = 0
JELLY = 1
SNOW = 2

width = 1080
height = 980
res = (width, height)
aspect = width / height
circles = ti.Vector.field(2, float, n_particles * 20)
circles_colors = ti.Vector.field(3, float, n_particles * 20)
height_image = ti.Vector.field(1, float, shape=(width, height))
thickness_image = ti.Vector.field(1, float, shape=(width, height))
gause_height_image = ti.Vector.field(1, float, shape=(width, height))
gause_height_image_new = ti.Vector.field(1, float, shape=(width, height))
height_image_color = ti.Vector.field(3, float, shape=(width, height))
back_ground_color = ti.Vector.field(3, float, shape=(width, height))
mixed_color = ti.Vector.field(3, float, shape=(width, height))
thickness_image_color = ti.Vector.field(3, float, shape=(width, height))
gause_thickness_image = ti.Vector.field(1, float, shape=(width, height))
gause_thickness_image_new = ti.Vector.field(1, float, shape=(width, height))
normal_map = ti.Vector.field(3, float, shape=(width, height))  # 法线向量
normal_point = ti.Vector.field(2, float, n_particles)  # 法线延伸出的点
project_x = ti.Vector.field(dim, float, n_particles)  # 投影视角下的点的全坐标
project_index = ti.Vector.field(1, float, shape=(width, height))  # 当前像素下点的索引
depth_x = ti.Vector.field(2, float, n_particles * 20)
depth = ti.Vector.field(1, float, n_particles)
max_z = ti.field(float, 1)
min_z = ti.field(float, 1)
line_indeces = ti.field(int, n_particles * 2)
lines = ti.Vector.field(2, float, n_particles * 2)
line_indeces[0] = 0
line_indeces[1] = 1
tri_meshes = ti.Vector.field(2, float, n_particles * 4)


def lookat(pos: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    vecz = pos - target
    vecz = vecz / np.sqrt(vecz.dot(vecz))

    vecx = np.cross(up, vecz)
    vecx = vecx / np.sqrt(vecx.dot(vecx))

    vecy = np.cross(vecz, vecx)

    rot_mat = np.r_[vecx.reshape((1, 3)), vecy.reshape((1, 3)), vecz.reshape((1, 3))]
    d_v = -rot_mat.dot(pos)

    rlt_mat = np.array([
        [rot_mat[0, 0], rot_mat[0, 1], rot_mat[0, 2], d_v[0]],
        [rot_mat[1, 0], rot_mat[1, 1], rot_mat[1, 2], d_v[1]],
        [rot_mat[2, 0], rot_mat[2, 1], rot_mat[2, 2], d_v[2]],
        [0, 0, 0, 1]
    ])
    print(rlt_mat)
    return rlt_mat


@ti.func
def point_scale(pos0: ti.f32, pos1: ti.f32, pos2: ti.f32, radius: ti.f32):
    base = pos0 * proj_mat[0][0] + pos1 * proj_mat[0][1] + pos2 * proj_mat[0][2] + 1 * proj_mat[0][3]
    dx = (pos0 + radius) * proj_mat[0][0] + pos1 * proj_mat[0][1] + pos2 * proj_mat[0][2] + 1 * proj_mat[0][3]
    dy = pos0 * proj_mat[0][0] + (pos1 + radius) * proj_mat[0][1] + pos2 * proj_mat[0][2] + 1 * proj_mat[0][3]
    dx = dx - base
    dy = dy - base
    return dx

@ti.kernel
def adjust_to_window():
    for p in depth_x:
        # depth_x[p][0] = (depth_x[p][0] + 1.0) * 500 / width
        # depth_x[p][1] = (depth_x[p][1] + 1.0) * 500 / height
        # project_x[p][0] = depth_x[p][0]
        # project_x[p][1] = depth_x[p][1]
        depth_x[p][0] = (depth_x[p][0] + 1.0) / 2
        depth_x[p][1] = (depth_x[p][1] + 1.0) / 2
        project_x[p][0] = depth_x[p][0]
        project_x[p][1] = depth_x[p][1]

        # normal_point[p][0] = (normal_point[p][0] + 0.3) * 500 / width
        # normal_point[p][1] = normal_point[p][1] * 500 / height
        # lines[p][0] = (lines[p][0] + 0.3) * 500 / width
        # lines[p][1] = (lines[p][1]) * 500 / height


@ti.kernel
def make_height_map():
    delt = max_z[0] - min_z[0]
    for i, j in height_image:
        height_image[i, j] = ti.Vector([1.0])
        height_image_color[i, j] = ti.Vector([1.0, 1.0, 1.0])
        # project_index[i, j] = ti.Vector([-1])
    for z in range(n_particles):
        depth[z][0] = (depth[z][0] - min_z[0]) / delt
        colors_depth[z] = ti.Vector([0.1 * depth[z][0], 0.6 * depth[z][0], 0.9 * depth[z][0]])
        # colors_depth[z] = ti.Vector([0.1, 0.6, 0.9])
        xx = int(depth_x[z][0] * width)
        yy = int(depth_x[z][1] * height)

        delt_r = point_scale(x[z][0], x[z][1], x[z][2], particles_radius)
        delt_r = int(delt_r * width)
        base_x = xx - delt_r
        base_y = yy - delt_r
        for i in range(2 * delt_r):
            for j in range(2 * delt_r):
                delt_o = ti.sqrt((delt_r - i) * (delt_r - i) + (delt_r - j) * (delt_r - j))
                if delt_o > delt_r:
                    continue
                dz = depth[z][0] + (delt_r - delt_o) / width
                dz = 1 - dz
                if height_image[base_x + i, base_y + j][0] > dz:
                    height_image[base_x + i, base_y + j][0] = dz
                    height_image_color[base_x + i, base_y + j] = ti.Vector([1.0 * dz, 1.0 * dz, 1.0 * dz])

                    # height_image_color[base_x + i, base_y + j] = ti.Vector([1.0, 0, 0])

                    # height_image_color[xx, yy] = ti.Vector([0.1, 0.6, 0.9])
                    # project_index[base_x + i, base_y + j][0] = z

                # height_image_color[base_x + i, base_y + j] = ti.Vector([1.0 * depth[z][0], 1.0 * depth[z][0], 1.0 * depth[z][0]])


        # if height_image[xx, yy][0] < depth[z][0]:
        #     height_image[xx, yy][0] = depth[z][0]
        #     height_image_color[xx, yy] = ti.Vector([1.0 - 1.0 * depth[z][0],1.0 - 1.0 * depth[z][0],1.0 - 1.0 * depth[z][0]])
        #     # project_index[xx, yy][0] = z
        #
        #     delt_r = point_scale(x[z][0], x[z][1], x[z][2], particles_radius)
        #     delt_r = int(delt_r * width)
        #     base_x = xx - delt_r
        #     base_y = yy - delt_r
        #     for i in range(2 * delt_r):
        #         for j in range(2 * delt_r):
        #             delt_o = ti.sqrt((delt_r - i) * (delt_r - i) + (delt_r - j) * (delt_r - j))
        #             if delt_o > delt_r:
        #                 continue
        #             height_image[base_x + i, base_y + j][0] = depth[z][0]
        #             height_image_color[base_x + i, base_y + j] = ti.Vector(
        #                 [1.0 * depth[z][0], 1.0 * depth[z][0], 1.0 * depth[z][0]])


@ti.kernel
def make_thickness_map():
    for i, j in thickness_image:
        thickness_image[i, j][0] = 0
        thickness_image_color[i, j] = ti.Vector([0.0, 0.0, 0.0])
    for z in range(n_particles):
        xx = int(depth_x[z][0] * width)
        yy = int(depth_x[z][1] * height)
        delt_r = point_scale(x[z][0], x[z][1], x[z][2], particles_radius)
        delt_r = int(delt_r * width)
        base_x = xx - delt_r
        base_y = yy - delt_r
        for i in range(2 * delt_r):
            for j in range(2 * delt_r):
                delt_o = ti.sqrt((delt_r - i) * (delt_r - i) + (delt_r - j) * (delt_r - j))
                if delt_o > delt_r:
                    continue
                dz = (delt_r - delt_o) / width
                thickness = (thickness_image[base_x + i, base_y + j][0] + dz)
                thickness = thickness * 1.1
                thickness_image[base_x + i, base_y + j][0] = thickness

                # thickness = 1 - thickness * 5.0
                # print(thickness)
                thickness_image_color[base_x + i, base_y + j] = ti.Vector([1.0 * thickness, 1.0 * thickness, 1.0 * thickness])



weight = [
    [0.003765, 0.015019, 0.023792, 0.015019, 0.003765],
    [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
    [0.023792, 0.094907, 0.1503462, 0.094907, 0.023792],
    [0.003765, 0.015019, 0.023792, 0.015019, 0.003765],
    [0.015019, 0.059912, 0.094907, 0.059912, 0.015019]
]
weight1 = [0.197448, 0.174697, 0.120999, 0.065602, 0.02784, 0.009246, 0.002403, 0.000489]


@ti.kernel
def gause():
    for i, j in gause_height_image:
        if i < 3 or i > width - 4 or j < 3 or j > height - 4:
            gause_height_image[i, j][0] = 1
        else:
            gause_height_image[i, j][0] = 0
    for i, j in height_image:
        if i < 3 or i > width - 4 or j < 3 or j > height - 4:
            # gause_height_image[i,j][0] = 1
            # if i == 1 and j == 1:
            #     print(height_image[i,j])
            continue

        base_x = i - 2
        base_y = j - 2
        for a in ti.static(range(5)):
            for b in ti.static(range(5)):
                # if height_image[base_x + a, base_y + b][0] == 1:
                #     gause_height_image[i, j][0] = gause_height_image[i, j][0]
                # else:
                gause_height_image[i, j][0] = gause_height_image[i, j][0] + height_image[base_x + a, base_y + b][0] * weight[a][b]

    for i, j in height_image:
        height_image[i, j] = gause_height_image[i, j]
        # if height_image[i, j][0] > 0:
            # height_image_color[i, j] = ti.Vector([0.1, 0.6, 0.9])
        height_image_color[i, j] = ti.Vector([1.0 * height_image[i, j][0], 1.0 * height_image[i, j][0], 1.0 * height_image[i, j][0]])
        # if height_image[i, j][0] >= 1:
            # height_image_color[i, j] = ti.Vector([1,0,0])


@ti.kernel
def gause_thickness():
    for i, j in gause_thickness_image:
        gause_thickness_image[i, j][0] = 0
    for i, j in gause_thickness_image:
        if i < 2 or i > width - 3 or j < 2 or j > height - 3:
            continue
        base_x = i - 2
        base_y = j - 2
        for a in ti.static(range(5)):
            for b in ti.static(range(5)):
                gause_thickness_image[i, j][0] = gause_thickness_image[i, j][0] + thickness_image[base_x + a, base_y + b][0] * weight[a][b]

    for i, j in thickness_image:
        thickness_image[i, j] = gause_thickness_image[i, j]
        if thickness_image[i, j][0] > 0:
            # height_image_color[i, j] = ti.Vector([0.1, 0.6, 0.9])
            thickness_image_color[i, j] = ti.Vector([1.0 * thickness_image[i, j][0], 1.0 * thickness_image[i, j][0], 1.0 * thickness_image[i, j][0]])


@ti.kernel
def rebuild_normal():
    # for i in normal_point:
    #     normal_point[i] = ti.Vector([0.9, 0.9])
    #     lines[i * 2 + 0] = ti.Vector([0, 0])
    #     lines[i * 2 + 1] = ti.Vector([0, 0])
    for i, j in normal_map:
        normal_map[i, j] = ti.Vector([0, 0, 0])
    for i, j in height_image:
        if i <= 3 or i >= width - 4 or j <= 3 or j >= height - 4:
            continue

        lleft = height_image[i - 1, j][0]  # 中心差分法
        rright = height_image[i + 1, j][0]
        uup = height_image[i, j + 1][0]
        ddown = height_image[i, j - 1][0]
        h1 = ti.Vector([2 / width, 0, rright - lleft])
        h2 = ti.Vector([0, 2 / height, uup - ddown])

        # base = height_image[i, j][0]
        # lx = height_image[i - 1, j][0]
        # rx = height_image[i + 1, j][0]
        # uy = height_image[i, j + 1][0]
        # dy = height_image[i, j - 1][0]
        # delt_left = base - lx
        # delt_right = base - rx
        # delt_up = base - uy
        # delt_down = base - dy
        # h1 = ti.Vector([1 / width, 0, delt_left])
        # if ti.abs(delt_left) > ti.abs(delt_right):
        #     h1 = ti.Vector([1 / width, 0, delt_right])
        # h2 = ti.Vector([0, 1 / height, delt_up])
        # if ti.abs(delt_up) > ti.abs(delt_down):
        #     h2 = ti.Vector([0, 1 / height, delt_down])
        norm = h2.cross(h1)
        normal_map[i, j] = norm.normalized()
        #在未经高斯模糊的情况下，法线贴图中每个粒子出现黑色的十字，是因为h1h2相互垂直
        if ti.abs(normal_map[i, j][0]) > 0.0001 and ti.abs(normal_map[i, j][1]) > 0.0001 and ti.abs(normal_map[i, j][2] + 1) > 0.0001:
            height_image_color[i, j] = normal_map[i, j] * 0.5 + ti.Vector([0.5, 0.5, 0.5])
            # height_image_color[i, j] = ti.Vector([1, 0, 0])
        # elif height_image[i, j][0] != 1:
        #     height_image_color[i, j] = ti.Vector([0, 1, 0])

@ti.kernel
def light_reflect():
    delt = max_z[0] - min_z[0]
    for i in depth_x:
        depth_x[i] = ti.Vector([0, 0])
        colors_light[i] = ti.Vector([0, 0, 0])
    for i, j in height_image:
        if height_image[i, j][0] >= 0.9999:
            continue
        pos = ti.Vector([(i + 0.5) / width, (j + 0.5) / height, height_image[i, j][0] * delt + min_z[0]])
        light_dir = light_position[0] - pos
        light_dir = -ti.normalized(light_dir)
        view_dir = view_position[0] - pos
        view_dir = -ti.normalized(view_dir)
        half_dir = view_dir + light_dir
        half_dir = ti.normalized(half_dir)
        normal_dir = normal_map[i, j]

        specular = ti.Vector([1.0, 1.0, 1.0]) * ti.Vector((0 / 256, 191 / 256, 255 / 256)) * ti.pow(max(ti.dot(normal_dir, half_dir), 0), 10)
        diffuse = ti.Vector([1.0, 1.0, 1.0]) * ti.Vector((135 / 256, 206 / 256, 255 / 256)) * max(ti.dot(normal_dir, light_dir), 0)
        height_image_color[i, j] = specular * 15 + diffuse * 0.8 #+ height_image_color[i, j] * 0.2
        height_image_color[i, j] = height_image_color[i, j] * 0.7
        # depth_x[i * width + j] = ti.Vector([i / width, j / height])
        # colors_light[i * width + j] = height_image_color[i, j]
        # height_image_color[i, j] = specular
        # print(height_image_color[i, j])
        # print(ti.dot(normal_dir, half_dir), ti.dot(normal_dir, light_dir))

@ti.kernel
def paint_background() :
    for i, j in back_ground_color:
        if i < 300 or i >= 812 or j < 200 or j >= 712:
            back_ground_color[i, j] = ti.Vector([1, 1, 1])
    for i, j in ti.ndrange(512 * 4, 512 * 4):
        ret = ti.taichi_logo(ti.Vector([i / (512 * 4), j / (512 * 4)]))
        back_ground_color[300 + i // 4, 200 + j // 4] += ti.Vector([ret / 16, ret / 16, ret / 16])


@ti.kernel
def light_refract():
    # for i, j in height_image_color:
    #     height_image_color[i, j] = ti.Vector([0, 0, 0])
    for i, j in mixed_color:
        mixed_color[i, j] = ti.Vector([0,0,0])
    for i, j in height_image_color:
        if height_image[i, j][0] >= 0.999:
            mixed_color[i, j] = back_ground_color[i, j]
        else:
            thickness = thickness_image[i, j][0] * 2.0
            transmission = ti.exp(-(ti.Vector([1.0, 1.0, 1.0]) - ti.Vector([135 / 256, 206 / 256, 255 / 256])) * thickness)
            new_i = i + normal_map[i, j][0] * 10
            new_j = j + normal_map[i, j][1] * 10
            # height_image_color[i, j] += ti.Vector([0 / 256, 191 / 256, 255 / 256]) * transmission * 0.5
            # height_image_color[i, j] += back_ground_color[new_i, new_j] * transmission * 0.5
            mixed_color[i, j] = height_image_color[i, j] * 0.5 + back_ground_color[new_i, new_j] * transmission * 0.5

@ti.kernel
def make_wall_image():
    pass


@ti.kernel
def make_circles():
    # cnt = 0
    for z in circles:
        circles[z] = ti.Vector([0, 0])
        circles_colors[z] = ti.Vector([0, 0, 0])
    for i, j in height_image_color:
        if height_image[i, j][0] >= 1:
            continue
        xx = i / width
        yy = j / height

        # circles[i * width + j] = ti.Vector([xx, yy])
        # circles_colors[i * width + j] = height_image_color[i, j]
        # print(i * width + j, i + j * height)
        circles[i + j * height - 200000] = ti.Vector([xx, yy])
        circles_colors[i + j * height - 200000] = height_image_color[i, j]
        # cnt += 1
        # print(cnt)


@ti.kernel
def substep(g_x: float, g_y: float, g_z: float):
    for I in ti.grouped(grid_m):
        grid_v[I] = ti.zero(grid_v[I])
        grid_m[I] = 0
    ti.block_dim(n_grid)
    for p in x:
        if used[p] == 0:
            continue
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

        F[p] = (ti.Matrix.identity(float, 3) +
                dt * C[p]) @ F[p]  # deformation gradient update

        h = ti.exp(
            10 *
            (1.0 -
             Jp[p]))  # Hardening coefficient: snow gets harder when compressed
        if materials[p] == JELLY:  # jelly, make it softer
            h = 0.3
        mu, la = mu_0 * h, lambda_0 * h
        if materials[p] == WATER:  # liquid
            mu = 0.0

        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(3)):
            new_sig = sig[d, d]
            if materials[p] == SNOW:  # Snow
                new_sig = min(max(sig[d, d], 1 - 2.5e-2),
                              1 + 4.5e-3)  # Plasticity
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        if materials[
            p] == WATER:  # Reset deformation gradient to avoid numerical instability
            new_F = ti.Matrix.identity(float, 3)
            new_F[0, 0] = J
            F[p] = new_F
        elif materials[p] == SNOW:
            F[p] = U @ sig @ V.transpose(
            )  # Reconstruct elastic deformation gradient after plasticity
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose(
        ) + ti.Matrix.identity(float, 3) * la * J * (J - 1)
        stress = (-dt * p_vol * 4) * stress / dx ** 2
        affine = stress + p_mass * C[p]

        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
    for I in ti.grouped(grid_m):
        if grid_m[I] > 0:
            grid_v[I] /= grid_m[I]
        grid_v[I] += dt * ti.Vector([g_x, g_y, g_z])
        cond = I < bound and grid_v[I] < 0 or I > n_grid - bound and grid_v[
            I] > 0
        grid_v[I] = 0 if cond else grid_v[I]
    ti.block_dim(n_grid)
    for p in x:
        if used[p] == 0:
            continue
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.zero(v[p])
        new_C = ti.zero(C[p])
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            g_v = grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx ** 2
        v[p] = new_v
        x[p] += dt * v[p]
        depth_x[p][0] = (
                x[p][0] * proj_mat[0][0] + x[p][1] * proj_mat[0][1] + x[p][2] * proj_mat[0][2] + 1 * proj_mat[0][3])
        # depth_x[p][0] = depth_x[p][0] * 500 / width
        # project_x[p][0] = depth_x[p][0]
        depth_x[p][1] = (
                x[p][0] * proj_mat[1][0] + x[p][1] * proj_mat[1][1] + x[p][2] * proj_mat[1][2] + 1 * proj_mat[1][3])
        # depth_x[p][1] = depth_x[p][1] * 500 / height
        # project_x[p][1] = depth_x[p][1]
        # depth[p][0] = -(x[p][0] * proj_mat[2][0] + x[p][1] * proj_mat[2][1] + x[p][2] * proj_mat[2][2] + 1 * proj_mat[2][3]) / 5
        # if depth[p][0] > 1:
        #     depth[0][0] = 1
        # else:
        #     depth[p][0] = 0.5
        # colors_depth[p] = ti.Vector([0.1 * depth[p][0], 0.6 * depth[p][0], 0.9 * depth[p][0]])
        depth[p][0] = x[p][0] * proj_mat[2][0] + x[p][1] * proj_mat[2][1] + x[p][2] * proj_mat[2][2] + 1 * proj_mat[2][
            3]
        project_x[p][2] = depth[p][0]
        if depth[p][0] < min_z[0]:
            min_z[0] = depth[p][0]
        if depth[p][0] > max_z[0]:
            max_z[0] = depth[p][0]
        C[p] = new_C

        # for i in ti.static(range(3)):
        #     if x[p][i] > min_max[i][1]:
        #         min_max[i][1] = x[p][i]
        #     if x[p][i] < min_max[i][0]:
        #         min_max[i][0] = x[p][i]


class CubeVolume:
    def __init__(self, minimum, size, material):
        self.minimum = minimum
        self.size = size
        self.volume = self.size.x * self.size.y * self.size.z
        self.material = material


@ti.kernel
def init_cube_vol(first_par: int, last_par: int, x_begin: float,
                  y_begin: float, z_begin: float, x_size: float, y_size: float,
                  z_size: float, material: int):
    for i in range(first_par, last_par):
        x[i] = ti.Vector([ti.random() for i in range(dim)]) * ti.Vector(
            [x_size, y_size, z_size]) + ti.Vector([x_begin, y_begin, z_begin])
        Jp[i] = 1
        F[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        v[i] = ti.Vector([0.0, 0.0, 0.0])
        materials[i] = material
        colors_random[i] = ti.Vector([ti.random(), ti.random(), ti.random()])
        used[i] = 1


@ti.kernel
def set_all_unused():
    for p in used:
        used[p] = 0
        # basically throw them away so they aren't rendered
        x[p] = ti.Vector([533799.0, 533799.0, 533799.0])
        Jp[p] = 1
        F[p] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        C[p] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        v[p] = ti.Vector([0.0, 0.0, 0.0])


def init_vols(vols):
    set_all_unused()
    total_vol = 0
    for v in vols:
        total_vol += v.volume

    next_p = 0
    for i in range(len(vols)):
        v = vols[i]
        if isinstance(v, CubeVolume):
            par_count = int(v.volume / total_vol * n_particles)
            if i == len(
                    vols
            ) - 1:  # this is the last volume, so use all remaining particles
                par_count = n_particles - next_p
            init_cube_vol(next_p, next_p + par_count, *v.minimum, *v.size,
                          v.material)
            next_p += par_count
        else:
            raise Exception("???")


@ti.kernel
def set_color_by_material(material_colors: ti.ext_arr()):
    for i in range(n_particles):
        mat = materials[i]
        colors[i] = ti.Vector([
            material_colors[mat, 0], material_colors[mat, 1],
            material_colors[mat, 2]
        ])


print("Loading presets...this might take a minute")

presets = [
    [
        CubeVolume(ti.Vector([0.55, 0.05, 0.55]), ti.Vector([0.4, 0.4, 0.4]),
                   WATER),
    ],
    [
        CubeVolume(ti.Vector([0.05, 0.05, 0.05]),
                   ti.Vector([0.3, 0.4, 0.3]), WATER),
        CubeVolume(ti.Vector([0.65, 0.05, 0.65]),
                   ti.Vector([0.3, 0.4, 0.3]), WATER),
    ],
    [
        CubeVolume(ti.Vector([0.6, 0.05, 0.6]),
                   ti.Vector([0.25, 0.25, 0.25]), WATER),
        CubeVolume(ti.Vector([0.35, 0.35, 0.35]),
                   ti.Vector([0.25, 0.25, 0.25]), SNOW),
        CubeVolume(ti.Vector([0.05, 0.6, 0.05]),
                   ti.Vector([0.25, 0.25, 0.25]), JELLY),
    ]]
preset_names = [
    "Single Dam Break",
    "Double Dam Break",
    "Water Snow Jelly",
]

curr_preset_id = 0

paused = False

show_depth_image = False
use_random_colors = False
particles_radius = 0.005

material_colors = [(0.1, 0.6, 0.9), (0.93, 0.33, 0.23), (1.0, 1.0, 1.0)]


def init():
    global paused
    init_vols(presets[curr_preset_id])


init()

window = ti.ui.Window("Real MPM 3D", res, vsync=True)

frame_id = 0
canvas = window.get_canvas()


proj_mat = lookat(np.array([1.1, 3.0, 3.45]), np.array([0.3, 0.3, 0.5]), np.array([0, 1, 0]))
# light_position[0] = ti.Vector([0.9, 1.5, -0.5])
light_position[0] = ti.Vector([-0.9, -3.0, 0.5])
view_position[0] = ti.Vector([1.1, 3.0, 3.45])
x1, y1, z1, w1 = light_position[0][0], light_position[0][1], light_position[0][2], 1
light_position[0][0] = x1 * proj_mat[0][0] + y1 * proj_mat[0][1] + z1 * proj_mat[0][2] + w1 * proj_mat[0][3]
light_position[0][0] = x1 * proj_mat[1][0] + y1 * proj_mat[1][1] + z1 * proj_mat[1][2] + w1 * proj_mat[0][3]
light_position[0][0] = x1 * proj_mat[2][0] + y1 * proj_mat[2][1] + z1 * proj_mat[2][2] + w1 * proj_mat[0][3]
x1, y1, z1, w1 = view_position[0][0], view_position[0][1], view_position[0][2], 1
view_position[0][0] = x1 * proj_mat[0][0] + y1 * proj_mat[0][1] + z1 * proj_mat[0][2] + w1 * proj_mat[0][3]
view_position[0][0] = x1 * proj_mat[1][0] + y1 * proj_mat[1][1] + z1 * proj_mat[1][2] + w1 * proj_mat[0][3]
view_position[0][0] = x1 * proj_mat[2][0] + y1 * proj_mat[2][1] + z1 * proj_mat[2][2] + w1 * proj_mat[0][3]
min_max[0][0], min_max[1][0], min_max[2][0] = 0.03126428, 0.03091107, 0.03128461
min_max[0][1], min_max[1][1], min_max[2][1] = 0.96867436, 0.96853966, 0.9687209


def render():
    # canvas.set_background_color((1.0, 1.0, 1.0))
    # for i in range(3):
    #     tri_meshes[i] = x[i]
    #
    # camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    # scene.set_camera(camera)
    #
    # scene.ambient_light((0, 0, 0))
    #
    # colors_used = colors_random if use_random_colors else colors
    # scene.particles(x, per_vertex_color=colors_used, radius=particles_radius)
    # scene.mesh(tri_meshes, color=(1.0, 0, 0))
    # #print(depth_x)
    # #print(x)
    # scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
    # scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.5, 0.5, 0.5))
    #
    # canvas.scene(scene)

    # print(depth)

    # canvas.triangles(wall_image, color=(1.0, 0.0, 0.0))
    # canvas.triangles(wall_image_left, color=(0.5, 0.0, 0.0))
    # canvas.triangles(wall_image_back, color=(0.0, 0.5, 0.0))
    # canvas.triangles(wall_image_down, color=(0.0, 0.0, 0.5))
    # canvas.triangles(wall_image_white, color=(0.41, 0.41, 0.41))
    # canvas.triangles(wall_image_black, color=(0.55, 0.73, 0.55))
    # canvas.circles(depth_x, radius=0.00205, color=(0.1, 0.6, 0.9))
    # canvas.circles(depth_x, radius=0.00405, per_vertex_color=colors_light)
    # canvas.set_image(height_image_color)
    canvas.set_image(thickness_image_color)
    # canvas.circles(circles, radius=0.00105, per_vertex_color=circles_colors)
    # canvas.circles(depth_x, radius=0.00205, per_vertex_color=colors_depth)
    # canvas.circles(normal_point, radius=0.00105, color=(1.0, 0, 0))
    # canvas.lines(lines, width=0.000015)
def render_image():
    # canvas.set_image(back_ground_color)
    # canvas.set_image(height_image_color)
    canvas.set_image(mixed_color)
    # canvas.set_image(thickness_image_color)
def render_circles():
    canvas.triangles(wall_image_white, color=(0.41, 0.41, 0.41))
    canvas.triangles(wall_image_black, color=(0.55, 0.73, 0.55))
    canvas.circles(circles, radius=0.00105, per_vertex_color=circles_colors)
    # canvas.circles(circles, radius=0.00105, color=(0.1, 0.6, 0.9, 0.1))

paint_background()
while window.running:
    # print("heyyy ",frame_id)
    frame_id += 1
    frame_id = frame_id % 256
    min_z[0] = 999
    max_z[0] = -999
    if not paused:
        for s in range(steps):
            substep(g_x, g_y, g_z)
    adjust_to_window()
    make_height_map()
    make_thickness_map()
    for _ in range(10):
        gause_thickness()
    for _ in range(20):
        gause()
    rebuild_normal()
    light_reflect()
    light_refract()
    # make_circles()
    # render_circles()
    render_image()

    # print(min_max)
    # show_options()

    window.show()
