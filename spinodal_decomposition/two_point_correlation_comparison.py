import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import correlate
from sklearn.metrics import mean_squared_error


def autocor(data, periodicity):
    m, n = data.shape
    per = periodicity == 'nonperiodic'

    if per:  # 'nonperiodic'
        cutoff_1 = int(m * 0.5)
        cutoff_2 = int(n * 0.5)
        image_pad = np.zeros([m + cutoff_1, n + cutoff_2])
        image_pad[0:m, 0:n] = data
        H1 = np.fft.fftn(image_pad)
        H1 = H1 * (np.conj(H1))
        H1 = np.fft.ifftn(H1)
        H1 = np.fft.fftshift(H1.real)

        mask = np.zeros([m + cutoff_1, n + cutoff_2])
        mask[0:m, 0:n] = 1
        H2 = np.fft.fftn(mask)
        H2 = H2 * (np.conj(H2))
        H2 = np.fft.ifftn(H2)
        H2 = np.fft.fftshift(H2.real)

        image_2pt = H1 / H2
    else:  # 'periodic'
        H = np.fft.fftn(data)
        H = H * (np.conj(H))
        H = np.fft.ifftn(H)
        H = np.fft.fftshift(H.real)
        image_2pt = H / (m * n)
    return image_2pt


def two_point_correlation(image, max_radius):
    # 获取图像尺寸
    h, w = image.shape
    # 创建网格坐标
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2

    # 计算到中心点的距离
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # 初始化结果数组
    radii = np.arange(0, max_radius)
    correlation = np.zeros_like(radii, dtype=float)

    # 计算自相关函数
    for i, radius in enumerate(radii):
        mask = (r >= radius - 0.5) & (r < radius + 0.5)
        if mask.sum() > 0:
            correlation[i] = np.mean(image[mask])

    return radii, correlation

def parse_zone_header(header_line):
    """Parse grid dimensions I and J from ZONE header line."""
    # 清理字符串，移除多余的字符
    header_line = header_line.replace(",", "").replace('"', "").strip()

    # 分割字符串后按顺序解析 I 和 J
    tokens = header_line.split()
    I = int([token.split("=")[1] for token in tokens if token.startswith("I=")][0])
    J = int([token.split("=")[1] for token in tokens if token.startswith("J=")][0])

    return I, J


def read_phi_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 调用新函数解析 I 和 J
    I, J = parse_zone_header(lines[1])

    data = np.array([float(value.strip()) for value in lines[2:]])
    grid_data = data.reshape((I, J))
    return grid_data

# Plot configuration
# FONT_FAMILY = 'Arial'  # 字体类型：'Arial', 'Times New Roman', 'DejaVu Sans', etc.
FONT_SIZE_LABELS = 20           # 坐标轴标签字体大小
FONT_SIZE_LEGEND = 18           # 图例字体大小
FONT_SIZE_TITLE = 16            # 标题字体大小
LINE_WIDTH = 2.5                # 线条粗细
# LINE_WIDTH = 1.5                # 线条粗细
DPI = 300                       # 图片清晰度 (dots per inch)
FIGURE_SIZE = (6, 5)           # 图片尺寸 (宽, 高) 英寸
# FIGURE_SIZE = (8, 5)           # 图片尺寸 (宽, 高) 英寸

# 设置matplotlib全局字体
# plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['font.size'] = FONT_SIZE_LABELS

import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_path=os.path.join(repo_root,'spinodal_decomposition','spi_noise_rollout_3stage')+os.sep

# 读取数据
# image1=read_phi_file(data_path+'final_true_phi.dat')
# image2=read_phi_file(data_path+'final_predicted_phi.dat')

# 读取数据
# image1=read_phi_file(data_path+'5w_true_phi.dat')
# image2=read_phi_file(data_path+'5w_predicted_phi.dat')

image1=read_phi_file(data_path+'2k_true_phi.dat')
image2=read_phi_file(data_path+'2k_predicted_phi.dat')

image1 = np.where(image1 > 0.6, 1, 0).astype(np.int8)
image2 = np.where(image2 > 0.6, 1, 0).astype(np.int8)

image_2pt_1 = autocor(image1, 'periodic')
image_2pt_2 = autocor(image2, 'periodic')

# 计算两点自相关函数
max_radius = image1.shape[0]/2
r1, S1 = two_point_correlation(image_2pt_1, max_radius)
r2, S2 = two_point_correlation(image_2pt_2, max_radius)

# 计算L2相对误差
l2_error = np.sqrt(np.sum((S1 - S2) ** 2)) / np.sqrt(np.sum(S1 ** 2))

# 提取相体积分数（两点统计函数在r=0处的值，即最大值）
phi_true = S1[0]      # True相体积分数
phi_pred = S2[0]      # Predicted相体积分数

print(phi_true)
print(phi_pred)

# 计算相体积分数的平方
phi_true_squared = phi_true ** 2
phi_pred_squared = phi_pred ** 2

print(f"True phase volume fraction: {phi_true:.4f}")
print(f"Predicted phase volume fraction: {phi_pred:.4f}")
print(f"True phase volume fraction squared: {phi_true_squared:.4f}")
print(f"Predicted phase volume fraction squared: {phi_pred_squared:.4f}")
print(f"Relative L2 Error: {l2_error:.4f}")

# 绘制自相关函数对比
plt.figure(figsize=FIGURE_SIZE, dpi=DPI)

# 绘制两点统计函数曲线
plt.plot(r1, S1, color='blue', linewidth=LINE_WIDTH,
         label='True Statistics', linestyle='-', alpha=0.7)
plt.plot(r2, S2, color='red', linewidth=LINE_WIDTH,
         label='Predicted Statistics', linestyle='-', alpha=0.7)

# 绘制相体积分数平方的水平虚线
plt.axhline(y=phi_true_squared, color='blue', linewidth=LINE_WIDTH,
            linestyle='--', alpha=0.7,
            label=f'True PVF$^2$ ({phi_true_squared:.3f})')
plt.axhline(y=phi_pred_squared, color='red', linewidth=LINE_WIDTH,
            linestyle='--', alpha=0.7,
            label=f'Predicted PVF$^2$ ({phi_pred_squared:.3f})')

# 设置坐标轴标签
plt.xlabel('Radius, r', fontsize=FONT_SIZE_LABELS)
# plt.ylabel('S$_2$(r)', fontsize=FONT_SIZE_LABELS, fontfamily=FONT_FAMILY)
# plt.ylabel(r'$\overline{\mathrm{S}_2}(r)$', fontsize=FONT_SIZE_LABELS, fontfamily=FONT_FAMILY)
plt.ylabel(r'$\overline{\mathit{S}}_2$', fontsize=FONT_SIZE_LABELS)

# 设置图例
# plt.legend(fontsize=FONT_SIZE_LEGEND, frameon=True, fancybox=True, shadow=True)
plt.legend(fontsize=FONT_SIZE_LEGEND)

# 设置网格
plt.grid(True, alpha=0.3)

# # 设置标题（可选）
# plt.title(f'Two-Point Correlation Functions (L2 Error: {l2_error:.2e})',
#           fontsize=FONT_SIZE_TITLE, fontfamily=FONT_FAMILY)

# 调整布局
plt.tight_layout()

# 保存图片
output_filename = os.path.join(data_path,'two_point_correlation_comparison_2k.png')
# output_filename = data_path+'two_point_correlation_comparison_5w.png'
# output_filename = data_path+'two_point_correlation_comparison_final.png'
plt.savefig(output_filename, dpi=DPI, bbox_inches='tight',
            facecolor='white', edgecolor='none')

# 显示图片
plt.show()

print(f"Saved: {output_filename}")