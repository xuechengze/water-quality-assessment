"""
报告生成器模块
==============
整合特征提取、分类器和LLM模块，生成完整的检测报告。
"""

import os
import json
import datetime
import numpy as np
from typing import Dict, Optional, Tuple

from utils.feature_extractor import WaterOpticalFeatureExtractor
from utils.classifier import WaterQualityClassifier
from utils.llm_interface import LLMReportGenerator, WaterQualityResult


class WaterQualityReportGenerator:
    """
    水质报告生成器
    整合光学特征提取、AI分类和大模型分析，生成完整检测报告
    """

    def __init__(self, classifier: WaterQualityClassifier = None,
                 llm_generator: LLMReportGenerator = None):
        """
        初始化报告生成器

        参数:
            classifier: 水质分类器实例
            llm_generator: LLM报告生成器实例
        """
        self.feature_extractor = WaterOpticalFeatureExtractor()
        self.classifier = classifier or WaterQualityClassifier()
        self.llm_generator = llm_generator
        self.template_mode = llm_generator is None or not llm_generator.is_available()

    def analyze(self, image: np.ndarray,
                location: str = "未知位置",
                water_body_type: str = "河道",
                enable_llm: bool = True) -> dict:
        """
        对水体图像进行完整分析

        参数:
            image: BGR格式的水体图像
            location: 监测地点
            water_body_type: 水体类型（河道/水库/池塘等）
            enable_llm: 是否启用大模型分析

        返回:
            完整的分析报告字典
        """
        # ========== 1. 提取光学特征 ==========
        features_dict = self.feature_extractor.extract_with_names(image)
        features_array = np.array(list(features_dict.values()))

        # ========== 2. 生成特征摘要 ==========
        feature_summary = self.feature_extractor.get_feature_summary(features_dict)

        # ========== 3. AI分类 ==========
        classification = self.classifier.predict_with_details(features_array)

        # ========== 4. 特征贡献分析 ==========
        contribution = self.classifier.analyze_feature_contribution(features_array)

        # ========== 5. 准备检测结果 ==========
        result = WaterQualityResult(
            level=classification['level'],
            level_code=classification['level_code'],
            confidence=classification['confidence'],
            risk=classification['risk'],
            risk_color=classification['risk_color'],
            description=classification['description'],
            class_probabilities=classification['class_probabilities'],
            feature_summary=feature_summary,
            image_description=self._generate_image_description(features_dict),
            location=location,
            water_body_type=water_body_type,
        )

        # ========== 6. 生成AI报告（可选） ==========
        ai_report = {}
        if enable_llm and self.llm_generator is not None:
            ai_report = self.llm_generator.generate_full_report(result)

        # ========== 7. 组装完整报告 ==========
        report = {
            'basic_info': {
                'location': location,
                'water_body_type': water_body_type,
                'detection_time': datetime.datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S'
                ),
                'method': '多模态AI与光学特征分析技术',
                'standard': '《地表水环境质量标准》(GB 3838-2002)',
            },
            'optical_features': features_dict,
            'feature_summary': feature_summary,
            'feature_contribution': contribution,
            'classification': classification,
            'ai_report': ai_report,
            'template_mode': self.template_mode,
        }

        return report

    def _generate_image_description(self, features: dict) -> str:
        """根据特征生成图像描述"""
        r, g, b = features['R_mean'], features['G_mean'], features['B_mean']
        brightness = features['brightness']
        turbidity = features['turbidity_index']

        # 判断水色
        if b > r and b > g:
            color_desc = "偏蓝色调"
        elif g > r and g > b:
            color_desc = "偏绿色调"
        elif r > b and r > g:
            color_desc = "偏黄褐色调"
        else:
            color_desc = "混合色调"

        # 判断透明度
        if brightness > 150:
            clarity = "高透明度"
        elif brightness > 100:
            clarity = "中等透明度"
        else:
            clarity = "低透明度"

        # 判断浑浊度
        if turbidity < 0.3:
            turbidity_desc = "低浑浊度"
        elif turbidity < 0.6:
            turbidity_desc = "中等浑浊度"
        else:
            turbidity_desc = "高浑浊度"

        return f"图像特征：{color_desc}，{clarity}，{turbidity_desc}。" \
               f"RGB均值({r:.0f},{g:.0f},{b:.0f})，亮度{brightness:.0f}。"

    def get_report_summary(self, report: dict) -> dict:
        """提取报告摘要"""
        classification = report['classification']
        basic_info = report['basic_info']
        has_llm = bool(report.get('ai_report', {}))

        summary = {
            'location': basic_info['location'],
            'time': basic_info['detection_time'],
            'water_type': basic_info['water_body_type'],
            'level': classification['level'],
            'confidence': classification['confidence'],
            'risk': classification['risk'],
            'description': classification['description'],
            'top_features': [],
            'has_llm_report': has_llm,
        }

        # 提取Top特征
        contribution = report.get('feature_contribution', {})
        if contribution and 'top_contributing_features' in contribution:
            top = contribution['top_contributing_features'][:3]
            summary['top_features'] = [
                {'name': f['name'], 'contribution': f['contribution']}
                for f in top
            ]

        return summary
