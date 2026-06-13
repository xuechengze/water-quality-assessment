"""
光学特征提取模块
==================
基于计算机视觉技术，从水体图像中提取7类光学物理特征：
1. RGB颜色特征
2. HSV颜色特征
3. 图像亮度
4. 图像对比度
5. 浑浊度指标
6. 悬浮颗粒特征
7. 纹理特征

适用于河道、水库、池塘等自然水体的水质光学分析。
"""

import cv2
import numpy as np
from skimage import feature, filters, morphology, measure
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
import warnings
warnings.filterwarnings('ignore')


class WaterOpticalFeatureExtractor:
    """水体光学特征提取器"""

    def __init__(self):
        self.feature_names = [
            # RGB特征 (6个)
            'R_mean', 'G_mean', 'B_mean',
            'R_std', 'G_std', 'B_std',
            # HSV特征 (6个)
            'H_mean', 'S_mean', 'V_mean',
            'H_std', 'S_std', 'V_std',
            # 亮度与对比度 (2个)
            'brightness', 'contrast',
            # 浑浊度指标 (4个)
            'turbidity_ratio', 'turbidity_entropy',
            'blue_green_ratio', 'turbidity_index',
            # 悬浮颗粒特征 (4个)
            'edge_density', 'particle_count',
            'particle_area_ratio', 'particle_mean_size',
            # 纹理特征 (4个)
            'texture_contrast', 'texture_energy',
            'texture_homogeneity', 'texture_entropy',
        ]
        self.n_features = len(self.feature_names)

    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        从水体图像中提取所有光学特征

        参数:
            image: BGR格式的OpenCV图像数组

        返回:
            包含所有特征的numpy数组 (1 x n_features)
        """
        features = []

        # ============ 1. RGB颜色特征 ============
        rgb_features = self._extract_rgb_features(image)
        features.extend(rgb_features)

        # ============ 2. HSV颜色特征 ============
        hsv_features = self._extract_hsv_features(image)
        features.extend(hsv_features)

        # ============ 3. 图像亮度特征 ============
        brightness_features = self._extract_brightness_features(image)
        features.extend(brightness_features)

        # ============ 4. 图像对比度特征 ============
        contrast_features = self._extract_contrast_features(image)
        features.extend(contrast_features)

        # ============ 5. 浑浊度指标 ============
        turbidity_features = self._extract_turbidity_features(image)
        features.extend(turbidity_features)

        # ============ 6. 悬浮颗粒特征 ============
        particle_features = self._extract_particle_features(image)
        features.extend(particle_features)

        # ============ 7. 纹理特征 ============
        texture_features = self._extract_texture_features(image)
        features.extend(texture_features)

        return np.array(features, dtype=np.float64)

    def extract_with_names(self, image: np.ndarray) -> dict:
        """提取特征并返回带名称的字典"""
        values = self.extract(image)
        return dict(zip(self.feature_names, values))

    def _extract_rgb_features(self, image: np.ndarray) -> list:
        """
        提取RGB颜色特征
        原理：不同水质的水体在RGB三通道上呈现不同的颜色特性
        - 清澈水体：B通道值较高，R通道值较低（偏蓝）
        - 富营养化：G通道值偏高（偏绿）
        - 污染水体：R通道值升高，各通道差异减小（偏黄褐）
        """
        # 分离RGB通道 (OpenCV默认BGR格式)
        b, g, r = cv2.split(image)

        rgb_features = [
            np.mean(r),          # R_mean: 红色通道均值 - 反映水中腐殖质/泥沙含量
            np.mean(g),          # G_mean: 绿色通道均值 - 反映藻类含量
            np.mean(b),          # B_mean: 蓝色通道均值 - 反映水体清澈度
            np.std(r),           # R_std: 红色通道标准差 - 反映颜色均匀性
            np.std(g),           # G_std: 绿色通道标准差
            np.std(b),           # B_std: 蓝色通道标准差
        ]
        return rgb_features

    def _extract_hsv_features(self, image: np.ndarray) -> list:
        """
        提取HSV颜色特征
        原理：HSV色彩空间更接近人眼对颜色的感知
        - H（色相）：反映水体主色调（蓝→绿→黄→褐）
        - S（饱和度）：反映水体颜色浓度
        - V（明度）：反映水体透光性
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # H通道是角度值(0-180)，需要特殊处理
        h_float = h.astype(np.float32)

        hsv_features = [
            np.mean(h_float),      # H_mean: 色相均值 - 判断水体主色调
            np.mean(s),            # S_mean: 饱和度均值 - 反映颜色浓度
            np.mean(v),            # V_mean: 明度均值 - 反映透光性
            np.std(h_float),       # H_std: 色相标准差 - 反映颜色分布均匀性
            np.std(s),             # S_std: 饱和度标准差
            np.std(v),             # V_std: 明度标准差
        ]
        return hsv_features

    def _extract_brightness_features(self, image: np.ndarray) -> list:
        """
        提取图像亮度特征
        原理：水体亮度综合反映水的透光性和悬浮物含量
        - 高亮度：水体清澈，透光性好
        - 低亮度：水体浑浊，悬浮物多
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        return [brightness]

    def _extract_contrast_features(self, image: np.ndarray) -> list:
        """
        提取图像对比度特征
        原理：对比度反映水体表面的纹理复杂度
        - 低对比度：水体均匀，可能是清澈或严重污染（均质）
        - 高对比度：水体含有悬浮物、泡沫或藻类聚集
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray)
        return [contrast]

    def _extract_turbidity_features(self, image: np.ndarray) -> list:
        """
        提取浑浊度相关指标
        原理：基于光学原理评估水体浑浊程度

        指标说明：
        - turbidity_ratio: 基于RGB通道方差，浑浊水体各通道差异减小
        - turbidity_entropy: 灰度熵，反映灰度分布混乱程度
        - blue_green_ratio: B/G通道比值，清澈水体B>G，浑浊时比值降低
        - turbidity_index: 综合浑浊指数
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        b, g, r = cv2.split(image)

        # 浑浊度比率：基于标准差归一化
        rgb_std = np.std([np.std(r), np.std(g), np.std(b)])
        rgb_mean = np.mean([np.mean(r), np.mean(g), np.mean(b)]) + 1e-6
        turbidity_ratio = rgb_std / rgb_mean

        # 灰度熵：反映灰度分布混乱度
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-6)
        hist_nonzero = hist[hist > 0]
        turbidity_entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero + 1e-10))

        # B/G比值：蓝绿通道比，评估水色
        green_mean = np.mean(g) + 1e-6
        blue_green_ratio = np.mean(b) / green_mean

        # 综合浑浊指数
        brightness_norm = np.mean(gray) / 255.0
        turbidity_index = 1.0 - brightness_norm + 0.3 * (1.0 - blue_green_ratio / 2.0)

        return [turbidity_ratio, turbidity_entropy, blue_green_ratio, turbidity_index]

    def _extract_particle_features(self, image: np.ndarray) -> list:
        """
        提取悬浮颗粒特征
        原理：通过边缘检测和形态学分析评估悬浮物含量
        - 边缘密度：反映水体中悬浮颗粒的边界信息
        - 颗粒计数：通过阈值分割统计可见颗粒数
        - 颗粒面积比：悬浮物占据图像面积比例
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. 边缘密度 - 使用Canny边缘检测
        edges = cv2.Canny(gray, 30, 100)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1] + 1e-6)

        # 2. 悬浮颗粒检测 - 自适应阈值分割
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 形态学操作去除噪声
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cleaned, connectivity=8
        )

        # 排除背景(标签0),只统计面积大于10像素的区域
        valid_regions = stats[1:, cv2.CC_STAT_AREA]
        valid_regions = valid_regions[valid_regions > 10]

        particle_count = len(valid_regions)
        total_area = image.shape[0] * image.shape[1]
        particle_area_ratio = np.sum(valid_regions) / (total_area + 1e-6)
        particle_mean_size = np.mean(valid_regions) if len(valid_regions) > 0 else 0.0

        return [edge_density, particle_count, particle_area_ratio, particle_mean_size]

    def _extract_texture_features(self, image: np.ndarray) -> list:
        """
        提取纹理特征（基于灰度共生矩阵 GLCM）
        原理：不同水质的水体表面纹理特征不同
        - 清澈水体：纹理均匀，对比度低
        - 藻类爆发：纹理不规则，对比度升高
        - 污染水体：纹理复杂，熵值增大
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_uint8 = img_as_ubyte(rgb2gray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))

        # 计算GLCM
        try:
            glcm = feature.graycomatrix(
                gray_uint8, distances=[5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                levels=256, symmetric=True, normed=True
            )

            # 提取GLCM特征（取四个方向均值）
            texture_contrast = feature.graycoprops(glcm, 'contrast').mean()
            texture_energy = feature.graycoprops(glcm, 'energy').mean()
            texture_homogeneity = feature.graycoprops(glcm, 'homogeneity').mean()
            texture_entropy = self._calc_glcm_entropy(glcm)
        except Exception:
            # 如果GLCM计算失败，返回默认值
            texture_contrast = 0
            texture_energy = 0
            texture_homogeneity = 0
            texture_entropy = 0

        return [texture_contrast, texture_energy, texture_homogeneity, texture_entropy]

    def _calc_glcm_entropy(self, glcm: np.ndarray) -> float:
        """计算GLCM熵值"""
        glcm_norm = glcm / (glcm.sum() + 1e-10)
        glcm_flat = glcm_norm.flatten()
        glcm_flat = glcm_flat[glcm_flat > 0]
        if len(glcm_flat) == 0:
            return 0.0
        entropy = -np.sum(glcm_flat * np.log2(glcm_flat + 1e-10))
        return float(entropy)

    def get_feature_summary(self, features_dict: dict) -> str:
        """生成特征分析摘要文本"""
        summary = []

        # RGB分析
        r, g, b = features_dict['R_mean'], features_dict['G_mean'], features_dict['B_mean']
        summary.append(f"RGB通道均值: R={r:.1f}, G={g:.1f}, B={b:.1f}")
        if b > r and b > g:
            summary.append("水色偏蓝，水体较清澈")
        elif g > r and g > b:
            summary.append("水色偏绿，可能存在藻类")
        elif r > b and r > g:
            summary.append("水色偏黄褐，含较多悬浮物")

        # HSV分析
        h_mean = features_dict['H_mean']
        s_mean = features_dict['S_mean']
        v_mean = features_dict['V_mean']
        summary.append(f"HSV色相={h_mean:.1f}°, 饱和度={s_mean:.2f}, 明度={v_mean:.2f}")
        if v_mean < 80:
            summary.append("水体明度较低，透光性差")
        elif v_mean > 150:
            summary.append("水体透光性良好")

        # 浑浊度分析
        turbidity = features_dict['turbidity_index']
        summary.append(f"综合浑浊指数: {turbidity:.3f}")
        if turbidity < 0.3:
            summary.append("浑浊度低，水质清澈")
        elif turbidity < 0.6:
            summary.append("浑浊度中等")
        else:
            summary.append("浑浊度高，悬浮物较多")

        # 亮度与对比度
        brightness = features_dict['brightness']
        contrast = features_dict['contrast']
        summary.append(f"图像亮度={brightness:.1f}, 对比度={contrast:.1f}")

        # 悬浮颗粒
        particle_count = features_dict['particle_count']
        edge_density = features_dict['edge_density']
        summary.append(f"悬浮颗粒检测数={int(particle_count)}, 边缘密度={edge_density:.4f}")

        # 纹理分析
        homogeneity = features_dict['texture_homogeneity']
        entropy = features_dict['texture_entropy']
        summary.append(f"纹理均匀度={homogeneity:.4f}, 纹理熵={entropy:.2f}")

        return '\n'.join(summary)


def extract_features_batch(images: list) -> np.ndarray:
    """批量提取特征"""
    extractor = WaterOpticalFeatureExtractor()
    features_list = []
    for img in images:
        feats = extractor.extract(img)
        features_list.append(feats)
    return np.array(features_list)
