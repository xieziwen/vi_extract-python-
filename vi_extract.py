# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 11:06:34 2025

@author: Ziwen Xie
"""

import os
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from sklearn.tree import DecisionTreeClassifier
import cv2
import matplotlib.pyplot as plt

# -------------------------- 配置matplotlib中文显示 --------------------------
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


# ----------------------------- 1. 植被指数公式与颜色映射 -----------------------------
VI_FORMULAS = {
    "EXR": lambda r, g, b: 1.4 * r - g,
    "MGRVI": lambda r, g, b: (g**2 - r**2) / (1e-8 + g**2 + r**2),
    "NGRDI": lambda r, g, b: (g - r) / (1e-8 + g + r),
    "RGRI": lambda r, g, b: r / (1e-8 + g),
    "VGBDI": lambda r, g, b: (g - b) / (1e-8 + g + b),
    "EXG": lambda r, g, b: 2 * g - r - b,
    "EXGR": lambda r, g, b: 3 * g - 2.4 * r - b,
    "VEG": lambda r, g, b: g / (1e-8 + (r**0.667) * (b**0.333)),
    "CIVE": lambda r, g, b: 0.441 * r - 0.881 * g + 0.385 * b + 18.78745,
    "RGBVI": lambda r, g, b: (g**2 - b * r) / (1e-8 + g**2 + b * r),
    "GLI": lambda r, g, b: (2 * g - b - r) / (1e-8 + 2 * g + b + r),
    "VARI": lambda r, g, b: (g - r) / (1e-8 + g + r - b),
    "WI": lambda r, g, b: (g - b) / (1e-8 + r - g),
    "COM": lambda r, g, b: 0.25*(2*g - r - b) + 0.2*(3*g - 2.4*r - b) + 
                          0.33*(0.441*r - 0.881*g + 0.385*b + 18.78745) + 
                          0.12*(g / (1e-8 + (r**0.667)*(b**0.333)))
}

# 植被指数颜色映射（灰度图）
VI_CMAP = {vi: "gray" for vi in VI_FORMULAS.keys()}


# -------------------------- 2. 新增：归一化处理函数 -----------------------------
def normalize_vi(vi_array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    对植被指数进行min-max归一化（仅针对前景像素）
    将值缩放到[0, 1]范围
    """
    # 提取前景像素
    foreground = vi_array[mask == 1]
    if len(foreground) < 10:  # 前景像素不足时返回原始数组
        return vi_array.copy()
    
    # 计算前景的min和max
    vi_min = np.min(foreground)
    vi_max = np.max(foreground)
    
    # 处理极端情况（避免除零）
    if vi_max - vi_min < 1e-8:
        return np.zeros_like(vi_array)
    
    # 复制原始数组并进行归一化（仅对前景有效）
    norm_vi = vi_array.copy()
    norm_vi[mask == 1] = (foreground - vi_min) / (vi_max - vi_min)
    return norm_vi


# -------------------------- 3. 图像读取与路径处理 --------------------------
def read_image_rasterio(img_path: str) -> tuple[np.ndarray, dict]:
    """读取图像（支持32位TIFF）"""
    try:
        with rasterio.open(img_path) as src:
            metadata = src.meta.copy()
            if src.count < 3:
                raise ValueError(f"图像波段数不足3：{src.count}个波段，需RGB图像")
            
            # 读取并转换为float32处理32位样本
            bands = [src.read(i, out_dtype=np.float32) for i in range(1, 4)]
            bands = np.stack(bands, axis=0)
            rgb_array = bands.transpose(1, 2, 0)  # 转为H×W×3
            
            # 归一化到[0,1]（处理不同位深度）
            dtype_max = np.max(rgb_array) if np.max(rgb_array) > 0 else 1.0
            if dtype_max > 1.0:
                rgb_array = rgb_array / dtype_max
            
            rgb_array = np.nan_to_num(rgb_array, nan=0.0)
            return rgb_array.astype(np.float32), metadata

    except Exception as e:
        if "TIFF" not in img_path.upper():
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                raise FileNotFoundError(f"无法读取图像：{img_path}")
            rgb_array = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) / 255.0
            metadata = {
                "width": rgb_array.shape[1],
                "height": rgb_array.shape[0],
                "transform": from_origin(0, 0, 1, 1)
            }
            return rgb_array.astype(np.float32), metadata
        else:
            raise RuntimeError(f"TIFF图像读取失败：{str(e)}")


def get_output_subdir(img_path: str, input_root: str, output_root: str) -> str:
    """生成与输入子目录一致的输出目录"""
    rel_path = os.path.relpath(img_path, input_root)
    rel_dir = os.path.dirname(rel_path)
    output_subdir = os.path.join(output_root, rel_dir)
    os.makedirs(output_subdir, exist_ok=True)
    return output_subdir


# -------------------------- 4. 修改：移除图例，仅保存主图 --------------------------
def save_vi_without_legend(vi_array: np.ndarray, vi_name: str, save_dir: str, 
                          img_name: str, metadata: dict, is_normalized: bool = False):
    """仅保存植被指数图（移除图例）"""
    # 处理数据（替换NaN为0）
    vi_clean = np.nan_to_num(vi_array, nan=0.0)
    
    # 获取坐标范围
    transform = metadata["transform"]
    x_min, y_max = transform[0], transform[3]
    x_max = x_min + transform[1] * metadata["width"]
    y_min = y_max + transform[5] * metadata["height"]
    extent = [x_min, x_max, y_min, y_max]
    
    # 创建画布（仅主图）
    plt.figure(figsize=(10, 8))
    cmap = VI_CMAP.get(vi_name, "gray")
    plt.imshow(vi_clean, cmap=cmap, extent=extent, origin="upper")
    
    # 标题添加归一化标识
    norm_tag = "（归一化）" if is_normalized else ""
    plt.title(f'{img_name} - {vi_name} 分布图{norm_tag}', fontsize=14, pad=10)
    plt.xlabel('X坐标', fontsize=12)
    plt.ylabel('Y坐标', fontsize=12)
    
    # # 保存图像（添加归一化标识）
    # norm_suffix = "_norm" if is_normalized else ""
    # save_path = os.path.join(save_dir, f"{vi_name}{norm_suffix}.png")
    # plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # plt.close()
    # print(f"📊 已保存{norm_tag}{vi_name}图像：{save_path}")


# -------------------------- 5. 保存功能模块 --------------------------
def save_vi_tiff(vi_array: np.ndarray, save_path: str, metadata: dict):
    """保存VI为标准TIFF"""
    vi_clean = np.nan_to_num(vi_array, nan=0.0)
    out_meta = metadata.copy()
    out_meta.update({
        "count": 1,
        "dtype": "float32",
        "driver": "GTiff",
        "compress": "lzw"
    })
    with rasterio.open(save_path, "w", **out_meta) as dst:
        dst.write(vi_clean.astype(np.float32), 1)


# -------------------------- 6. 背景分割与特征提取 --------------------------
def convert_colour_space(img_rgb: np.ndarray) -> np.ndarray:
    """提取颜色空间特征"""
    H, W, _ = img_rgb.shape
    rgb_flat = img_rgb.reshape(-1, 3)
    img_uint8 = (img_rgb * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_hsv = img_hsv / np.array([179.0, 255.0, 255.0])
    hsv_flat = img_hsv.reshape(-1, 3)
    
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    img_lab = img_lab / 255.0
    lab_flat = img_lab.reshape(-1, 3)
    
    return np.hstack([rgb_flat, hsv_flat, lab_flat])


def train_background_model(veg_path: str, bg_path: str) -> DecisionTreeClassifier:
    """训练背景分割模型"""
    veg_rgb, _ = read_image_rasterio(veg_path)
    veg_feat = convert_colour_space(veg_rgb)
    veg_label = np.ones(veg_feat.shape[0])
    
    bg_rgb, _ = read_image_rasterio(bg_path)
    bg_feat = convert_colour_space(bg_rgb)
    bg_label = np.zeros(bg_feat.shape[0])
    
    X_train = np.vstack([veg_feat, bg_feat])
    y_train = np.hstack([veg_label, bg_label])
    model = DecisionTreeClassifier(random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    print("✅ 背景分割模型训练完成")
    return model


def segment_background(img_rgb: np.ndarray, model: DecisionTreeClassifier) -> np.ndarray:
    """背景分割"""
    H, W, _ = img_rgb.shape
    feat = convert_colour_space(img_rgb)
    mask_flat = model.predict(feat)
    mask = mask_flat.reshape(H, W).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.erode(mask, kernel, iterations=3)


# -------------------------- 7. 统计计算模块（支持归一化值） --------------------------
def calculate_vi_stats(vi_array: np.ndarray, mask: np.ndarray) -> float:
    """计算前景VI均值"""
    vi_foreground = vi_array[mask == 1]
    if len(vi_foreground) < 10:
        print("⚠️  前景像素不足，VI均值设为NaN")
        return np.nan
    vi_sorted = np.sort(vi_foreground)
    n = len(vi_sorted)
    return vi_sorted[int(n*0.1):int(n*0.9)].mean()  # 去除10%极端值的均值


# -------------------------- 8. 单图像处理与批量处理 --------------------------
def process_single_image(img_path: str, model: DecisionTreeClassifier,
                         input_root: str, output_root: str) -> tuple[pd.Series, str]:
    """处理单张图像，生成原始与归一化植被指数结果"""
    img_filename = os.path.basename(img_path)
    img_name = os.path.splitext(img_filename)[0]
    
    # 生成输出目录
    output_subdir = get_output_subdir(img_path, input_root, output_root)
    img_result_dir = os.path.join(output_subdir, img_name)
    os.makedirs(img_result_dir, exist_ok=True)
    
    # 读取图像
    try:
        img_rgb, img_metadata = read_image_rasterio(img_path)
    except Exception as e:
        print(f"❌ 跳过图像 {img_filename}：{str(e)}")
        return pd.Series(), ""
    
    # 背景分割并保存掩码
    mask = segment_background(img_rgb, model)
    mask_save_path = os.path.join(img_result_dir, f"{img_name}_mask.png")
    cv2.imwrite(mask_save_path, mask * 255)
    
    # 提取RGB通道
    r, g, b = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]
    
    # 计算VI、归一化VI并保存结果
    vi_result = pd.Series({"image_path": img_path, "image_name": img_name})
    for vi_name, vi_func in VI_FORMULAS.items():
        try:
            # 计算原始植被指数
            vi_array = vi_func(r, g, b)
            vi_array[mask == 0] = np.nan  # 背景设为NaN
            # 计算有效像素数量（非NaN值）
            valid_pixels = np.sum(~np.isnan(vi_array))
            vi_mean = calculate_vi_stats(vi_array, mask)
            vi_result[vi_name] = vi_mean
            
            # 仅当有效像素足够时才保存原始VI的TIFF和图像
            if valid_pixels >= 10:
                vi_tiff_path = os.path.join(img_result_dir, f"{vi_name}.tif")
                save_vi_tiff(vi_array, vi_tiff_path, img_metadata)
                save_vi_without_legend(vi_array, vi_name, img_result_dir, img_name, img_metadata)
            else:
                print(f"⚠️ {vi_name} 有效像素不足，不保存原始VI图像和TIFF")
            
            # 计算归一化植被指数
            norm_vi_array = normalize_vi(vi_array, mask)
            norm_valid_pixels = np.sum(~np.isnan(norm_vi_array))
            norm_vi_mean = calculate_vi_stats(norm_vi_array, mask)
            vi_result[f"{vi_name}_norm"] = norm_vi_mean
            
            # 仅当有效像素足够时才保存归一化VI的TIFF和图像
            if norm_valid_pixels >= 10:
                norm_vi_tiff_path = os.path.join(img_result_dir, f"{vi_name}_norm.tif")
                save_vi_tiff(norm_vi_array, norm_vi_tiff_path, img_metadata)
                save_vi_without_legend(norm_vi_array, vi_name, img_result_dir, img_name, 
                                     img_metadata, is_normalized=True)
            else:
                print(f"⚠️ {vi_name} 归一化后有效像素不足，不保存归一化VI图像和TIFF")
            
        except Exception as e:
            vi_result[vi_name] = np.nan
            vi_result[f"{vi_name}_norm"] = np.nan
            print(f"❌ 计算 {vi_name} 失败：{str(e)}")
    
    rel_dir = os.path.dirname(os.path.relpath(img_path, input_root))
    return vi_result, rel_dir


def main():
    # 配置参数（根据实际路径修改）
    VEGETATION_SAMPLE = "Training_set/vegetation.png"
    BACKGROUND_SAMPLE = "Training_set/background.png"
    INPUT_ROOT_DIR = "data/"
    OUTPUT_ROOT_DIR = "result/"
    IMAGE_EXTENSIONS = ["*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.png"]
    
    # 确保输出根目录存在
    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)
    
    # 训练模型
    print("📈 训练背景分割模型...")
    try:
        seg_model = train_background_model(VEGETATION_SAMPLE, BACKGROUND_SAMPLE)
    except Exception as e:
        print(f"❌ 模型训练失败：{str(e)}，程序退出")
        return
    
    # 查找图像
    all_img_paths = []
    for ext in IMAGE_EXTENSIONS:
        all_img_paths.extend(glob.glob(os.path.join(INPUT_ROOT_DIR, "**", ext), recursive=True))
    if not all_img_paths:
        print(f"❌ 在 {INPUT_ROOT_DIR} 未找到图像，程序退出")
        return
    print(f"📷 找到 {len(all_img_paths)} 张图像，开始处理...")
    
    # 批量处理并按子目录保存CSV（包含归一化值）
    results_by_subdir = {}
    for img_path in all_img_paths:
        print(f"\n🔄 处理图像：{os.path.basename(img_path)}")
        single_result, subdir = process_single_image(img_path, seg_model, INPUT_ROOT_DIR, OUTPUT_ROOT_DIR)
        if not single_result.empty and subdir is not None:
            if subdir not in results_by_subdir:
                results_by_subdir[subdir] = []
            results_by_subdir[subdir].append(single_result)
    
    # 保存包含归一化值的表格
    if results_by_subdir:
        for subdir, results in results_by_subdir.items():
            if subdir == '.':
                csv_name = "root_results_with_norm.csv"
            else:
                csv_name = f"{subdir.replace(os.sep, '_')}_results_with_norm.csv"
            
            csv_path = os.path.join(OUTPUT_ROOT_DIR, csv_name)
            result_df = pd.DataFrame(results)
            result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"\n📁 子目录 {subdir} 结果（含归一化值）已保存：{csv_path}")
    else:
        print("\n❌ 未生成有效结果")
    
    print("\n🎉 所有处理完成！")


if __name__ == "__main__":
    # 安装依赖
    # pip install rasterio numpy pandas scikit-learn opencv-python matplotlib
    main()