"""
水质训练数据生成器
===================
基于光学物理原理，为四种水质等级生成逼真的合成特征数据。

水质等级说明：
- 优 (0): 清澈水体 — 偏蓝色调，高透光性，低浑浊度
- 良 (1): 较好水体 — 轻微偏绿或偏蓝，中等透光性
- 中 (2): 一般水体 — 偏绿色调，浑浊度上升，可见悬浮物
- 差 (3): 污染水体 — 偏黄褐，高浑浊度，悬浮物密集
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class WaterQualityDataGenerator:
    """
    水质特征数据生成器
    基于光学物理模型模拟不同水质等级的特征分布
    """

    # 特征名称列表（与feature_extractor保持一致）
    FEATURE_NAMES = [
        # RGB特征
        'R_mean', 'G_mean', 'B_mean',
        'R_std', 'G_std', 'B_std',
        # HSV特征
        'H_mean', 'S_mean', 'V_mean',
        'H_std', 'S_std', 'V_std',
        # 亮度与对比度
        'brightness', 'contrast',
        # 浑浊度指标
        'turbidity_ratio', 'turbidity_entropy',
        'blue_green_ratio', 'turbidity_index',
        # 悬浮颗粒特征
        'edge_density', 'particle_count',
        'particle_area_ratio', 'particle_mean_size',
        # 纹理特征
        'texture_contrast', 'texture_energy',
        'texture_homogeneity', 'texture_entropy',
    ]

    # ============ 各水质等级的特征参数范围 ============
    # 基于光学物理原理和实际水体观测数据设定

    # 优 (Excellent) - 清澈水体
    EXCELLENT_PARAMS = {
        'R_mean':         (30, 80),
        'G_mean':         (50, 100),
        'B_mean':         (80, 150),
        'R_std':          (5, 20),
        'G_std':          (5, 20),
        'B_std':          (8, 25),
        'H_mean':         (180, 220),
        'S_mean':         (15, 40),
        'V_mean':         (150, 200),
        'H_std':          (5, 15),
        'S_std':          (5, 15),
        'V_std':          (10, 25),
        'brightness':     (100, 180),
        'contrast':       (5, 25),
        'turbidity_ratio':   (0.05, 0.20),
        'turbidity_entropy': (3.0, 5.0),
        'blue_green_ratio':  (1.3, 2.0),
        'turbidity_index':   (0.05, 0.25),
        'edge_density':      (0.01, 0.08),
        'particle_count':    (0, 8),
        'particle_area_ratio': (0.001, 0.05),
        'particle_mean_size':  (0, 30),
        'texture_contrast':    (5, 30),
        'texture_energy':      (0.15, 0.35),
        'texture_homogeneity': (0.75, 0.95),
        'texture_entropy':     (3.0, 5.5),
    }

    # 良 (Good) - 轻度影响
    GOOD_PARAMS = {
        'R_mean':         (50, 100),
        'G_mean':         (60, 120),
        'B_mean':         (70, 130),
        'R_std':          (10, 30),
        'G_std':          (10, 30),
        'B_std':          (12, 30),
        'H_mean':         (140, 200),
        'S_mean':         (25, 55),
        'V_mean':         (120, 170),
        'H_std':          (10, 25),
        'S_std':          (8, 20),
        'V_std':          (15, 35),
        'brightness':     (80, 150),
        'contrast':       (15, 40),
        'turbidity_ratio':   (0.12, 0.35),
        'turbidity_entropy': (4.0, 6.0),
        'blue_green_ratio':  (1.0, 1.5),
        'turbidity_index':   (0.20, 0.45),
        'edge_density':      (0.05, 0.15),
        'particle_count':    (5, 25),
        'particle_area_ratio': (0.02, 0.10),
        'particle_mean_size':  (20, 60),
        'texture_contrast':    (15, 50),
        'texture_energy':      (0.10, 0.25),
        'texture_homogeneity': (0.65, 0.85),
        'texture_entropy':     (4.5, 6.5),
    }

    # 中 (Fair) - 中度污染
    FAIR_PARAMS = {
        'R_mean':         (70, 140),
        'G_mean':         (70, 130),
        'B_mean':         (50, 100),
        'R_std':          (15, 40),
        'G_std':          (15, 40),
        'B_std':          (15, 35),
        'H_mean':         (80, 160),
        'S_mean':         (35, 70),
        'V_mean':         (80, 140),
        'H_std':          (15, 35),
        'S_std':          (10, 25),
        'V_std':          (20, 45),
        'brightness':     (50, 110),
        'contrast':       (25, 60),
        'turbidity_ratio':   (0.25, 0.55),
        'turbidity_entropy': (5.5, 7.0),
        'blue_green_ratio':  (0.6, 1.1),
        'turbidity_index':   (0.40, 0.70),
        'edge_density':      (0.10, 0.25),
        'particle_count':    (20, 60),
        'particle_area_ratio': (0.08, 0.25),
        'particle_mean_size':  (40, 100),
        'texture_contrast':    (30, 80),
        'texture_energy':      (0.05, 0.18),
        'texture_homogeneity': (0.50, 0.75),
        'texture_entropy':     (5.5, 7.5),
    }

    # 差 (Poor) - 严重污染
    POOR_PARAMS = {
        'R_mean':         (100, 180),
        'G_mean':         (70, 130),
        'B_mean':         (30, 80),
        'R_std':          (25, 55),
        'G_std':          (20, 50),
        'B_std':          (15, 40),
        'H_mean':         (30, 100),
        'S_mean':         (50, 90),
        'V_mean':         (50, 110),
        'H_std':          (20, 45),
        'S_std':          (12, 30),
        'V_std':          (20, 50),
        'brightness':     (30, 80),
        'contrast':       (35, 85),
        'turbidity_ratio':   (0.40, 0.80),
        'turbidity_entropy': (6.0, 8.0),
        'blue_green_ratio':  (0.3, 0.8),
        'turbidity_index':   (0.65, 0.95),
        'edge_density':      (0.20, 0.45),
        'particle_count':    (40, 120),
        'particle_area_ratio': (0.20, 0.50),
        'particle_mean_size':  (60, 180),
        'texture_contrast':    (50, 120),
        'texture_energy':      (0.02, 0.10),
        'texture_homogeneity': (0.30, 0.60),
        'texture_entropy':     (6.5, 8.5),
    }

    # 等级映射
    LEVEL_MAP = {0: '优', 1: '良', 2: '中', 3: '差'}
    LEVEL_DESC = {
        0: '水质清澈透明，无污染迹象，适合生活饮用水源',
        1: '水质较好，轻微影响，适合渔业用水和景观用水',
        2: '水质一般，中度污染，需关注治理，仅适合工业用水',
        3: '水质较差，严重污染，需立即采取治理措施',
    }

    def __init__(self, random_seed: int = 42):
        self.rng = np.random.RandomState(random_seed)

    def _sample_from_range(self, low: float, high: float, size: int = 1,
                           distribution: str = 'uniform') -> np.ndarray:
        """从指定范围采样"""
        if distribution == 'uniform':
            return np.random.uniform(low, high, size)
        elif distribution == 'normal':
            mid = (low + high) / 2
            std = (high - low) / 4
            samples = np.random.normal(mid, std, size)
            return np.clip(samples, low, high)
        return np.random.uniform(low, high, size)

    def _generate_particle_count(self, low: int, high: int, size: int = 1) -> np.ndarray:
        """生成颗粒计数（整数）"""
        return np.random.randint(low, high + 1, size)

    def generate_samples(self, n_excellent: int = 200, n_good: int = 200,
                         n_fair: int = 200, n_poor: int = 200) -> pd.DataFrame:
        """
        生成合成训练数据集

        参数:
            n_excellent: 优等级样本数
            n_good: 良等级样本数
            n_fair: 中等级样本数
            n_poor: 差等级样本数

        返回:
            包含特征和标签的DataFrame
        """
        all_data = []
        all_labels = []
        all_levels = []

        configs = [
            (n_excellent, self.EXCELLENT_PARAMS, 0),
            (n_good, self.GOOD_PARAMS, 1),
            (n_fair, self.FAIR_PARAMS, 2),
            (n_poor, self.POOR_PARAMS, 3),
        ]

        for n_samples, params, label in configs:
            if n_samples <= 0:
                continue

            data = {}
            for feature_name in self.FEATURE_NAMES:
                low, high = params[feature_name]

                if feature_name == 'particle_count':
                    data[feature_name] = self._generate_particle_count(
                        low, high, n_samples
                    )
                else:
                    data[feature_name] = self._sample_from_range(
                        low, high, n_samples
                    )

            df = pd.DataFrame(data)
            all_data.append(df)
            all_labels.extend([label] * n_samples)
            all_levels.extend([self.LEVEL_MAP[label]] * n_samples)

        result = pd.concat(all_data, ignore_index=True)
        result['quality_label'] = all_labels
        result['quality_level'] = all_levels

        # 添加一些特征之间的相关性（让数据更真实）
        result = self._add_feature_correlations(result)

        return result

    def _add_feature_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加特征间相关性，使合成数据更接近真实物理规律
        """
        # 浑浊度越高，亮度越低（负相关）
        df['brightness'] = df['brightness'] - df['turbidity_index'] * 20
        df['brightness'] = df['brightness'].clip(10, 255)

        # 浑浊度越高，蓝绿比越低
        df['blue_green_ratio'] = df['blue_green_ratio'] - df['turbidity_index'] * 0.3
        df['blue_green_ratio'] = df['blue_green_ratio'].clip(0.1, 3.0)

        # 颗粒越多，边缘密度越大
        df['edge_density'] = df['edge_density'] + df['particle_count'] * 0.0005
        df['edge_density'] = df['edge_density'].clip(0.001, 0.6)

        # 纹理均匀度与浑浊度呈反比
        df['texture_homogeneity'] = df['texture_homogeneity'] - df['turbidity_index'] * 0.15
        df['texture_homogeneity'] = df['texture_homogeneity'].clip(0.1, 0.98)

        # 纹理熵与浑浊度呈正比
        df['texture_entropy'] = df['texture_entropy'] + df['turbidity_index'] * 1.5
        df['texture_entropy'] = df['texture_entropy'].clip(1.0, 9.5)

        return df

    def generate_balanced_dataset(self, samples_per_class: int = 300) -> pd.DataFrame:
        """生成平衡数据集"""
        return self.generate_samples(
            n_excellent=samples_per_class,
            n_good=samples_per_class,
            n_fair=samples_per_class,
            n_poor=samples_per_class,
        )

    def split_data(self, df: pd.DataFrame, test_size: float = 0.2,
                   random_state: int = 42):
        """分割训练集和测试集"""
        X = df[self.FEATURE_NAMES].values
        y = df['quality_label'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y
        )
        return X_train, X_test, y_train, y_test


def generate_default_dataset(save_path: str = None) -> pd.DataFrame:
    """生成默认数据集"""
    generator = WaterQualityDataGenerator()
    df = generator.generate_balanced_dataset(samples_per_class=500)

    if save_path:
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"数据集已保存至: {save_path}")
        print(f"数据形状: {df.shape}")
        print(f"等级分布:\n{df['quality_level'].value_counts()}")

    return df


if __name__ == '__main__':
    # 测试数据生成
    df = generate_default_dataset()
    print("\n特征统计:")
    print(df.describe())
    print("\n前5行:")
    print(df.head())
