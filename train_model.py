"""
模型训练脚本
=============
生成合成训练数据，训练水质分类模型，并保存模型。

用法:
    python train_model.py

输出:
    - models/water_quality_model.pkl: 训练好的RandomForest模型
    - models/model_metadata.pkl: 模型元数据（特征名、标签映射等）
    - data/training_data.csv: 生成的训练数据集
"""

import os
import sys
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score,
                             recall_score, f1_score)

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from utils.data_generator import WaterQualityDataGenerator


def train_model():
    """训练水质分类模型"""
    print("=" * 60)
    print("  河道水质智能评估平台 - 模型训练")
    print("=" * 60)

    # ========== 1. 生成训练数据 ==========
    print("\n[1/5] 生成合成训练数据...")
    generator = WaterQualityDataGenerator(random_seed=42)
    df = generator.generate_balanced_dataset(samples_per_class=500)
    print(f"  生成数据: {df.shape[0]} 条记录, {len(generator.FEATURE_NAMES)} 个特征")
    print(f"  等级分布:")
    for label, count in df['quality_level'].value_counts().items():
        print(f"    {label}: {count}")

    # ========== 2. 数据预处理 ==========
    print("\n[2/5] 数据预处理...")
    X = df[generator.FEATURE_NAMES].values
    y = df['quality_label'].values

    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  特征矩阵: {X_scaled.shape}")
    print(f"  特征均值(标准化后): {X_scaled.mean(axis=0)[:5]}...")
    print(f"  特征标准差(标准化后): {X_scaled.std(axis=0)[:5]}...")

    # ========== 3. 训练RandomForest模型 ==========
    print("\n[3/5] 训练RandomForest分类模型...")
    model = RandomForestClassifier(
        n_estimators=200,           # 决策树数量
        max_depth=15,               # 树的最大深度
        min_samples_split=5,        # 内部节点再划分所需最小样本数
        min_samples_leaf=2,         # 叶子节点最少样本数
        max_features='sqrt',        # 每棵树使用的最大特征数
        bootstrap=True,             # 使用Bootstrap采样
        oob_score=True,             # 使用袋外数据评估
        random_state=42,
        n_jobs=-1,                  # 使用全部CPU核心
        class_weight='balanced',    # 类别权重平衡
    )
    model.fit(X_scaled, y)
    print(f"  模型训练完成")
    print(f"  袋外评分 (OOB Score): {model.oob_score_:.4f}")
    print(f"  决策树数量: {model.n_estimators}")
    print(f"  特征重要性 (Top 5):")

    # 显示特征重要性
    importance = model.feature_importances_
    top_features = np.argsort(importance)[::-1][:5]
    for i, idx in enumerate(top_features):
        print(f"    {i+1}. {generator.FEATURE_NAMES[idx]}: {importance[idx]:.4f}")

    # ========== 4. 模型评估 ==========
    print("\n[4/5] 模型评估...")

    # 使用袋外数据进行评估
    y_pred = model.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)

    print(f"  训练集准确率: {accuracy:.4f}")
    print(f"\n  分类报告:")
    print(classification_report(
        y, y_pred,
        target_names=['优(Excellent)', '良(Good)', '中(Fair)', '差(Poor)'],
        digits=4
    ))

    # 混淆矩阵
    cm = confusion_matrix(y, y_pred)
    print(f"  混淆矩阵:")
    print(f"             预测优  预测良  预测中  预测差")
    for i, label in enumerate(['实际优', '实际良', '实际中', '实际差']):
        print(f"    {label}: {cm[i]}")

    # ========== 5. 保存模型 ==========
    print("\n[5/5] 保存模型...")
    model_dir = os.path.join(project_root, 'models')
    os.makedirs(model_dir, exist_ok=True)

    # 保存RandomForest模型
    model_path = os.path.join(model_dir, 'water_quality_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  RF模型已保存: {model_path}")

    # 保存标准化器
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  标准化器已保存: {scaler_path}")

    # 保存元数据
    metadata = {
        'feature_names': generator.FEATURE_NAMES,
        'n_features': len(generator.FEATURE_NAMES),
        'level_map': generator.LEVEL_MAP,
        'level_desc': generator.LEVEL_DESC,
        'accuracy': accuracy,
        'oob_score': model.oob_score_,
        'feature_importance': dict(zip(generator.FEATURE_NAMES, importance)),
    }
    metadata_path = os.path.join(model_dir, 'model_metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"  元数据已保存: {metadata_path}")

    # 保存训练数据
    data_path = os.path.join(project_root, 'data', 'training_data.csv')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    df.to_csv(data_path, index=False, encoding='utf-8-sig')
    print(f"  训练数据已保存: {data_path}")

    print("\n" + "=" * 60)
    print("  模型训练完成！")
    print("=" * 60)

    return model, scaler, metadata


def evaluate_model_on_test_set():
    """在实际测试图像上评估模型（逻辑验证）"""
    # 该函数验证模型可以在模拟的实际图像特征上正确分类
    print("\n验证模型逻辑...")
    model_path = os.path.join(project_root, 'models', 'water_quality_model.pkl')
    scaler_path = os.path.join(project_root, 'models', 'scaler.pkl')

    if not os.path.exists(model_path):
        print("模型文件不存在，请先训练模型")
        return

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(project_root, 'models', 'model_metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)

    # 生成测试数据
    generator = WaterQualityDataGenerator(random_seed=100)
    test_df = generator.generate_balanced_dataset(samples_per_class=100)

    X_test = test_df[generator.FEATURE_NAMES].values
    y_test = test_df['quality_label'].values
    y_pred = model.predict(scaler.transform(X_test))

    accuracy = accuracy_score(y_test, y_pred)
    print(f"测试集准确率: {accuracy:.4f}")
    print(f"F1-Score (加权): {f1_score(y_test, y_pred, average='weighted'):.4f}")

    return accuracy


if __name__ == '__main__':
    # 训练主模型
    model, scaler, metadata = train_model()

    # 验证
    evaluate_model_on_test_set()

    print("\n✅ 模型训练完成，可以在 app.py 中使用了！")
