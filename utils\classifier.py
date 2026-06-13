"""
水质分类器模块
===============
使用随机森林（RandomForest）模型对水体光学特征进行分类，
输出水质等级、可信度和风险等级。
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional


class WaterQualityClassifier:
    """
    水质分类器

    基于光学特征的水质智能评估：
    1. 提取26维光学特征
    2. RandomForest分类
    3. 输出水质等级 + 置信度 + 风险等级
    """

    LEVEL_MAP = {0: '优', 1: '良', 2: '中', 3: '差'}
    LEVEL_DESC = {
        '优': '水质清澈透明，无污染迹象，适合生活饮用水源',
        '良': '水质较好，轻微影响，适合渔业用水和景观用水',
        '中': '水质一般，中度污染，需关注治理，仅适合工业用水',
        '差': '水质较差，严重污染，需立即采取治理措施',
    }
    RISK_MAP = {
        '优': '低风险',
        '良': '低风险',
        '中': '中等风险',
        '差': '高风险',
    }
    RISK_COLOR_MAP = {
        '优': '#00CC66',    # 绿色
        '良': '#66B3FF',    # 蓝色
        '中': '#FFA500',    # 橙色
        '差': '#FF3333',    # 红色
    }
    LEVEL_COLOR_MAP = {
        '优': '#006633',
        '良': '#004C99',
        '中': '#996600',
        '差': '#990000',
    }

    def __init__(self, model_dir: str = None):
        """
        初始化分类器，加载预训练模型

        参数:
            model_dir: 模型文件目录，默认使用项目中的models目录
        """
        if model_dir is None:
            model_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'models'
            )

        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.metadata = None
        self.feature_names = None

        # 尝试加载模型
        self._load_models()

    def _load_models(self):
        """加载所有模型文件"""
        model_path = os.path.join(self.model_dir, 'water_quality_model.pkl')
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        metadata_path = os.path.join(self.model_dir, 'model_metadata.pkl')

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"模型文件不存在: {model_path}\n"
                "请先运行 'python train_model.py' 训练模型"
            )

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        self.feature_names = self.metadata.get('feature_names', None)
        print(f"模型加载成功: {type(self.model).__name__}")
        print(f"模型准确率: {self.metadata.get('accuracy', 'N/A')}")

    def predict(self, features: np.ndarray) -> Tuple[str, float, str]:
        """
        对单个样本进行水质分类

        参数:
            features: 26维光学特征向量

        返回:
            (水质等级, 可信度, 风险等级)
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先训练模型")

        # 确保输入是二维数组
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # 标准化特征
        features_scaled = self.scaler.transform(features)

        # 预测
        pred = self.model.predict(features_scaled)[0]
        proba = self.model.predict_proba(features_scaled)[0]

        # 获取等级和可信度
        level = self.LEVEL_MAP[int(pred)]
        confidence = float(proba[int(pred)])

        # 风险等级
        risk = self.RISK_MAP[level]

        return level, confidence, risk

    def predict_with_details(self, features: np.ndarray) -> dict:
        """
        详细预测结果（含所有类别的概率）

        参数:
            features: 26维光学特征向量

        返回:
            包含所有预测信息的字典
        """
        level, confidence, risk = self.predict(features)

        # 获取各类别概率
        if features.ndim == 1:
            features = features.reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        proba = self.model.predict_proba(features_scaled)[0]

        class_probs = {}
        for i, (label, name) in enumerate(self.LEVEL_MAP.items()):
            class_probs[name] = float(proba[i])

        return {
            'level': level,
            'level_code': [k for k, v in self.LEVEL_MAP.items() if v == level][0],
            'confidence': confidence,
            'risk': risk,
            'risk_color': self.RISK_COLOR_MAP[level],
            'level_color': self.LEVEL_COLOR_MAP[level],
            'description': self.LEVEL_DESC[level],
            'class_probabilities': class_probs,
            'all_levels': list(self.LEVEL_MAP.values()),
            'all_confidences': [float(p) for p in proba],
        }

    def batch_predict(self, features_batch: np.ndarray) -> list:
        """批量预测"""
        results = []
        for feat in features_batch:
            results.append(self.predict_with_details(feat))
        return results

    def get_feature_importance(self) -> pd.DataFrame:
        """获取特征重要性排序"""
        if self.metadata is None or 'feature_importance' not in self.metadata:
            return None

        importance = self.metadata['feature_importance']
        df = pd.DataFrame(
            list(importance.items()),
            columns=['特征名称', '重要性']
        )
        df = df.sort_values('重要性', ascending=False).reset_index(drop=True)
        df['排名'] = df.index + 1
        return df[['排名', '特征名称', '重要性']]

    def analyze_feature_contribution(self, features: np.ndarray) -> dict:
        """
        分析各特征对分类结果的贡献

        参数:
            features: 26维光学特征向量

        返回:
            贡献度分析字典
        """
        if self.feature_names is None:
            return {}

        if features.ndim == 1:
            features = features.reshape(1, -1)

        features_scaled = self.scaler.transform(features)

        # 获取决策路径
        estimators = self.model.estimators_
        feature_contrib = np.zeros(len(self.feature_names))

        for tree in estimators:
            # 获取当前样本在每棵树中的叶节点
            leaf_idx = tree.apply(features_scaled)[0]
            # 获取该叶节点的路径特征
            n_nodes = tree.tree_.node_count
            children_left = tree.tree_.children_left
            children_right = tree.tree_.children_right
            feature = tree.tree_.feature
            threshold = tree.tree_.threshold

            # 从根节点到叶节点的路径
            node = 0
            while node != leaf_idx:
                if feature[node] != -2:  # 非叶节点
                    feat_idx = feature[node]
                    contrib = (features_scaled[0, feat_idx] - threshold[node])
                    feature_contrib[feat_idx] += abs(contrib)
                    if features_scaled[0, feat_idx] <= threshold[node]:
                        node = children_left[node]
                    else:
                        node = children_right[node]
                else:
                    break

        feature_contrib = feature_contrib / len(estimators)

        # 排序并返回Top特征
        sorted_idx = np.argsort(feature_contrib)[::-1]
        top_features = []
        for idx in sorted_idx[:10]:
            if feature_contrib[idx] > 0:
                top_features.append({
                    'name': self.feature_names[idx],
                    'contribution': float(feature_contrib[idx]),
                })

        return {'top_contributing_features': top_features}


def get_classifier() -> WaterQualityClassifier:
    """获取默认分类器实例"""
    model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'models'
    )
    return WaterQualityClassifier(model_dir=model_dir)
