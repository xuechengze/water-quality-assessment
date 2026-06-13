"""
河道水质智能评估平台 - Streamlit 主应用
============================================
基于多模态AI与光学特征分析的河道水质智能评估系统

浙江省大学生物理实验与科技创新竞赛
职教赛道 - 一等奖参赛作品

使用方式:
    streamlit run app.py
"""

import os
import sys
import json
import datetime
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# 设置页面配置 - 必须在第一条
st.set_page_config(
    page_title="河道水质智能评估平台",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from utils.feature_extractor import WaterOpticalFeatureExtractor
from utils.classifier import WaterQualityClassifier
from utils.llm_interface import LLMReportGenerator, WaterQualityResult
from utils.report_generator import WaterQualityReportGenerator


# ============================================================
# 样式和UI辅助函数
# ============================================================

def load_css():
    """加载自定义CSS样式"""
    st.markdown("""
    <style>
    /* 全局样式 */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0066CC;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
        border-radius: 15px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .level-badge {
        display: inline-block;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-size: 1.5rem;
        font-weight: 700;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.2rem 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666;
    }
    .feature-bar {
        margin: 0.3rem 0;
    }
    .report-section {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #0066CC;
    }
    .risk-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 15px;
        font-size: 0.9rem;
        font-weight: 600;
        color: white;
    }
    .progress-container {
        width: 100%;
        background-color: #f0f0f0;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .progress-bar {
        height: 24px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-weight: 600;
        font-size: 0.8rem;
        line-height: 24px;
    }
    .footer {
        text-align: center;
        color: #888;
        font-size: 0.8rem;
        padding: 2rem 0 0 0;
        border-top: 1px solid #eee;
        margin-top: 2rem;
    }
    .innovation-tag {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        background: #E8F5E9;
        color: #2E7D32;
        margin: 0.2rem;
    }
    </style>
    """, unsafe_allow_html=True)


def get_level_color(level: str) -> str:
    """获取等级对应的颜色"""
    colors = {
        '优': '#00CC66',
        '良': '#3399FF',
        '中': '#FFA500',
        '差': '#FF3333',
    }
    return colors.get(level, '#999999')


def get_risk_color(risk: str) -> str:
    """获取风险等级对应颜色"""
    colors = {
        '低风险': '#00CC66',
        '中等风险': '#FFA500',
        '高风险': '#FF3333',
    }
    return colors.get(risk, '#999999')


# ============================================================
# 页面初始化
# ============================================================

@st.cache_resource
def init_system():
    """初始化系统组件（带缓存）"""
    with st.spinner("🔄 正在加载AI模型，请稍候..."):
        classifier = WaterQualityClassifier(
            model_dir=os.path.join(project_root, 'models')
        )
        report_gen = WaterQualityReportGenerator(classifier=classifier)
        return classifier, report_gen


@st.cache_resource
def init_llm():
    """初始化LLM组件"""
    # 检查环境变量
    api_key = os.environ.get("LLM_API_KEY", "")
    api_base = os.environ.get("LLM_API_BASE", "")

    if api_key and api_base:
        llm = LLMReportGenerator(
            api_key=api_key,
            api_base=api_base,
            model_name=os.environ.get("LLM_MODEL", "gpt-3.5-turbo"),
        )
        # 测试连接
        if llm.is_available():
            st.success("✅ 大模型连接成功，将生成专业级分析报告")
            return llm

    return None


def display_level_badge(level: str):
    """显示水质等级徽章"""
    color = get_level_color(level)
    bg_color = f"{color}22"
    border_color = color

    icon_map = {'优': '🌟', '良': '👍', '中': '⚠️', '差': '🚨'}

    st.markdown(f"""
    <div style="text-align:center; padding:1rem;">
        <div class="level-badge" style="background: linear-gradient(135deg, {color}, {color}dd);">
            {icon_map.get(level, '❓')} 水质等级：{level}
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_probability_bar(level: str, prob: float, max_prob: float):
    """显示单个概率条"""
    color = get_level_color(level)
    width_pct = prob * 100
    icon_map = {'优': '🌟', '良': '👍', '中': '⚠️', '差': '🚨'}

    # 高亮最高概率
    is_max = abs(prob - max_prob) < 0.001

    label_style = "font-weight:700;" if is_max else ""
    marker = " ◄" if is_max else ""

    st.markdown(f"""
    <div class="feature-bar">
        <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
            <span style="{label_style}">{icon_map.get(level, '')} {level}{marker}</span>
            <span style="{label_style}">{prob*100:.1f}%</span>
        </div>
        <div class="progress-container">
            <div class="progress-bar" style="width:{width_pct}%; background: {color};">
                {'' if width_pct < 8 else f'{prob*100:.1f}%'}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_risk_badge(risk: str):
    """显示风险等级标签"""
    color = get_risk_color(risk)
    st.markdown(
        f'<span class="risk-badge" style="background:{color};">{risk}</span>',
        unsafe_allow_html=True
    )


def display_feature_chart(features: dict):
    """显示光学特征雷达图"""
    import plotly.graph_objects as go

    # 选择关键特征展示
    key_features = [
        ('brightness', '亮度'),
        ('blue_green_ratio', '蓝绿比'),
        ('turbidity_index', '浑浊度'),
        ('texture_homogeneity', '纹理均匀度'),
        ('texture_energy', '纹理能量'),
    ]

    # 归一化到0-1范围
    norm_values = []
    labels = []
    for key, label in key_features:
        val = features.get(key, 0)
        labels.append(label)
        if key == 'brightness':
            norm_values.append(min(val / 200, 1.0))
        elif key == 'blue_green_ratio':
            norm_values.append(min(val / 2.0, 1.0))
        elif key == 'turbidity_index':
            norm_values.append(min(val, 1.0))
        elif key == 'texture_homogeneity':
            norm_values.append(min(val, 1.0))
        elif key == 'texture_energy':
            norm_values.append(min(val / 0.5, 1.0))

    # 闭合雷达图
    norm_values.append(norm_values[0])
    labels.append(labels[0])

    fig = go.Figure(data=go.Scatterpolar(
        r=norm_values,
        theta=labels,
        fill='toself',
        line=dict(color='#0066CC', width=2),
        fillcolor='rgba(0,102,204,0.2)',
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=10),
            ),
            bgcolor='rgba(0,0,0,0)',
        ),
        showlegend=False,
        height=300,
        margin=dict(l=40, r=40, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
    )
    st.plotly_chart(fig, use_container_width=True)


def display_feature_table(features: dict):
    """显示详细特征数据表格"""
    # 归类展示
    feature_groups = {
        'RGB颜色特征': ['R_mean', 'G_mean', 'B_mean', 'R_std', 'G_std', 'B_std'],
        'HSV颜色特征': ['H_mean', 'S_mean', 'V_mean', 'H_std', 'S_std', 'V_std'],
        '亮度与对比度': ['brightness', 'contrast'],
        '浑浊度指标': ['turbidity_ratio', 'turbidity_entropy', 'blue_green_ratio', 'turbidity_index'],
        '悬浮颗粒特征': ['edge_density', 'particle_count', 'particle_area_ratio', 'particle_mean_size'],
        '纹理特征': ['texture_contrast', 'texture_energy', 'texture_homogeneity', 'texture_entropy'],
    }

    display_names = {
        'R_mean': 'R均值', 'G_mean': 'G均值', 'B_mean': 'B均值',
        'R_std': 'R标准差', 'G_std': 'G标准差', 'B_std': 'B标准差',
        'H_mean': 'H均值', 'S_mean': 'S均值', 'V_mean': 'V均值',
        'H_std': 'H标准差', 'S_std': 'S标准差', 'V_std': 'V标准差',
        'brightness': '亮度', 'contrast': '对比度',
        'turbidity_ratio': '浑浊度比率', 'turbidity_entropy': '浑浊度熵',
        'blue_green_ratio': '蓝绿比', 'turbidity_index': '综合浑浊指数',
        'edge_density': '边缘密度', 'particle_count': '颗粒计数',
        'particle_area_ratio': '颗粒面积比', 'particle_mean_size': '颗粒平均大小',
        'texture_contrast': '纹理对比度', 'texture_energy': '纹理能量',
        'texture_homogeneity': '纹理均匀度', 'texture_entropy': '纹理熵',
    }

    # 格式化函数
    def fmt_val(v, name):
        if name == 'particle_count':
            return f"{int(v)}"
        elif name in ['edge_density', 'particle_area_ratio']:
            return f"{v:.4f}"
        elif name in ['texture_homogeneity', 'texture_energy']:
            return f"{v:.4f}"
        elif name in ['turbidity_ratio', 'blue_green_ratio', 'turbidity_index']:
            return f"{v:.4f}"
        elif name in ['H_mean', 'H_std']:
            return f"{v:.1f}"
        else:
            return f"{v:.2f}"

    for group_name, feat_names in feature_groups.items():
        with st.expander(f"📊 {group_name}", expanded=False):
            data = []
            for name in feat_names:
                if name in features:
                    data.append({
                        '特征名称': display_names.get(name, name),
                        '值': fmt_val(features[name], name),
                    })
            if data:
                st.table(pd.DataFrame(data))


def display_report_section(title: str, content: str, icon: str = "📋"):
    """显示报告章节"""
    with st.container():
        st.markdown(f"""
        <div class="report-section">
            <h3 style="margin:0 0 0.5rem 0; color:#0066CC;">{icon} {title}</h3>
            <div style="white-space:pre-wrap; font-family:'Microsoft YaHei',sans-serif;
                        line-height:1.6; color:#333;">
        """, unsafe_allow_html=True)
        st.markdown(content)
        st.markdown("</div>", unsafe_allow_html=True)


def display_innovation_section():
    """显示创新点说明"""
    with st.sidebar.expander("🎯 项目创新点", expanded=False):
        st.markdown("""
        **① 光学物理特征分析**
        基于OpenCV提取7类26维光学特征
        
        **② AI智能分类**
        RandomForest模型自动判断水质等级
        
        **③ 水质智能评估**
        综合光学特征与AI的评估体系
        
        **④ 河道巡检应用**
        面向实际巡检场景的实用工具
        
        **⑤ 低成本部署**
        普通摄像头+AI即可完成检测
        """)

    with st.sidebar.expander("🏆 技术特色", expanded=False):
        st.markdown("""
        - 纯视觉方案，无需传感器
        - 实时检测，秒级出结果
        - 离线可用，适配野外作业
        - 支持大模型增强分析
        - 政府级报告输出
        """)


# ============================================================
# 主页面
# ============================================================

def main():
    """主应用入口"""
    # 加载CSS
    load_css()

    # ========== 初始化 ==========
    if 'location' not in st.session_state:
        st.session_state.location = "浙江省杭州市西湖区"

    # 侧边栏
    with st.sidebar:
        st.markdown("## 💧 水质检测系统")
        st.markdown("---")

        # 监测信息
        st.markdown("### 📍 监测位置")
        st.info("📍 当前位置: " + st.session_state.location)

        # 常用监测点位快速选择
        preset_location = st.selectbox(
            "快速选择监测点",
            options=[
                "自定义输入",
                "浙江省杭州市西湖区",
                "浙江省杭州市钱塘江",
                "浙江省杭州市京杭大运河",
                "浙江省杭州市西溪湿地",
                "浙江省宁波市甬江",
                "浙江省温州市瓯江",
                "浙江省嘉兴市南湖",
                "浙江省湖州市太湖",
                "浙江省绍兴市鉴湖",
                "浙江省金华市婺江",
                "浙江省衢州市衢江",
                "浙江省舟山市近海",
                "浙江省台州市灵江",
                "浙江省丽水市瓯江上游",
            ],
            index=0,
        )

        if preset_location == "自定义输入":
            location = st.text_input(
                "手动输入监测位置",
                value=st.session_state.location,
                help="输入河道/水库/池塘的具体位置"
            )
        else:
            location = preset_location

        st.session_state.location = location

        water_type = st.selectbox(
            "水体类型",
            options=["河道", "水库", "池塘", "湖泊", "溪流", "其他"],
            index=0,
        )

        st.markdown("---")

        # LLM配置
        st.markdown("### 🤖 大模型配置（可选）")
        enable_llm = st.checkbox(
            "启用AI报告增强",
            value=False,
            help="启用后调用大模型生成政府级水质分析报告"
        )

        # LLM配置区域
        llm_config_expanded = enable_llm
        with st.expander("API配置", expanded=llm_config_expanded):
            llm_api_base = st.text_input(
                "API地址",
                value=os.environ.get("LLM_API_BASE", "http://localhost:1234/v1"),
                help="支持OpenAI兼容API或本地LM Studio"
            )
            llm_api_key = st.text_input(
                "API密钥",
                value=os.environ.get("LLM_API_KEY", "not-needed"),
                type="password",
            )
            llm_model = st.text_input(
                "模型名称",
                value=os.environ.get("LLM_MODEL", "qwen2.5-7b-instruct"),
            )

        st.markdown("---")
        display_innovation_section()

        st.markdown("---")
        st.markdown("""
        <div style="text-align:center; color:#888; font-size:0.8rem;">
            浙江省大学生物理实验与科技创新竞赛<br>
            职教赛道参赛作品
        </div>
        """, unsafe_allow_html=True)

    # ========== 主内容区域 ==========
    st.markdown("""
    <div class="main-header">
        🌊 河道水质智能评估平台
    </div>
    <div class="sub-header">
        基于多模态AI与光学特征分析 | 支持河道 · 水库 · 池塘水质检测
    </div>
    """, unsafe_allow_html=True)

    # 初始化系统
    try:
        classifier, report_gen = init_system()
    except FileNotFoundError as e:
        st.error(f"""
        ⚠️ **模型文件未找到！**
        
        {str(e)}
        
        请先在终端运行以下命令训练模型：
        ```bash
        cd river-water-quality-assessment
        python train_model.py
        ```
        """)
        st.stop()
    except Exception as e:
        st.error(f"⚠️ 系统初始化失败: {str(e)}")
        st.stop()

    # 初始化LLM（如果需要）
    llm_generator = None
    if enable_llm and llm_api_base and llm_api_key:
        with st.spinner("🤖 正在连接大模型..."):
            try:
                llm_generator = LLMReportGenerator(
                    api_key=llm_api_key,
                    api_base=llm_api_base,
                    model_name=llm_model,
                )
                if llm_generator.is_available():
                    st.sidebar.success("✅ 大模型已连接")
            except Exception as e:
                st.sidebar.warning(f"⚠️ 大模型连接失败，将使用模板报告: {str(e)[:50]}...")
                llm_generator = None

    # ========== 图片上传区域 ==========
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📷 上传水体图像")
        st.markdown("支持 JPG、PNG、JPEG 格式")

        upload_option = st.radio(
            "选择上传方式",
            ["📤 上传图片", "📸 拍摄照片"],
            horizontal=True,
        )

        uploaded_file = None
        if upload_option == "📤 上传图片":
            uploaded_file = st.file_uploader(
                "选择水体图片",
                type=['jpg', 'jpeg', 'png'],
                label_visibility="collapsed",
            )
        else:
            uploaded_file = st.camera_input(
                "拍摄水体照片",
                label_visibility="collapsed",
            )

    with col2:
        st.markdown("### 🔍 图片预览")
        if uploaded_file is not None:
            # 读取图片
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            st.image(image_rgb, caption="上传的水体图像", use_container_width=True)

            # 图像信息
            h, w = image_bgr.shape[:2]
            file_size = len(file_bytes) / 1024
            st.caption(f"分辨率: {w}×{h} | 文件大小: {file_size:.1f} KB")
        else:
            # 显示占位
            st.markdown("""
            <div style="
                border: 2px dashed #ccc;
                border-radius: 12px;
                height: 350px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: #fafafa;
                color: #999;
                font-size: 1.1rem;
            ">
                📸 请上传或拍摄水体图像<br>
                <small style="color:#bbb;">支持河道、水库、池塘等水体</small>
            </div>
            """, unsafe_allow_html=True)

    # ========== 分析操作 ==========
    if uploaded_file is not None:
        st.markdown("---")
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            analyze_btn = st.button(
                "🚀 开始水质检测分析",
                type="primary",
                use_container_width=True,
            )

        if analyze_btn:
            with st.spinner("🔬 正在分析水体光学特征..."):
                # 重新读取图片（file_bytes已被读取）
                file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
                image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                # 执行分析
                report = report_gen.analyze(
                    image=image_bgr,
                    location=location,
                    water_body_type=water_type,
                    enable_llm=(enable_llm and llm_generator is not None),
                )

                # 如果有LLM且刚未使用，重新生成报告
                if enable_llm and llm_generator is not None and not report.get('ai_report'):
                    report_gen.llm_generator = llm_generator
                    report = report_gen.analyze(
                        image=image_bgr,
                        location=location,
                        water_body_type=water_type,
                        enable_llm=True,
                    )

                # 保存结果到session state
                st.session_state['report'] = report
                st.session_state['analyzed'] = True
                st.rerun()

    # ========== 检测结果展示 ==========
    if st.session_state.get('analyzed', False) and 'report' in st.session_state:
        report = st.session_state['report']
        classification = report['classification']
        features = report['optical_features']
        ai_report = report.get('ai_report', {})

        st.markdown("---")
        st.markdown("## 📊 水质检测结果")

        # ========== 结果概览 ==========
        level = classification['level']
        confidence = classification['confidence']
        risk = classification['risk']

        # 顶部概览卡
        overview_cols = st.columns([1, 1, 1, 1])
        with overview_cols[0]:
            color = get_level_color(level)
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid {color};">
                <div class="metric-label">水质等级</div>
                <div class="metric-value" style="color:{color};">{level}</div>
            </div>
            """, unsafe_allow_html=True)

        with overview_cols[1]:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #0066CC;">
                <div class="metric-label">AI可信度</div>
                <div class="metric-value" style="color:#0066CC;">{confidence*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with overview_cols[2]:
            risk_color = get_risk_color(risk)
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid {risk_color};">
                <div class="metric-label">风险等级</div>
                <div class="metric-value" style="color:{risk_color};">{risk}</div>
            </div>
            """, unsafe_allow_html=True)

        with overview_cols[3]:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #666;">
                <div class="metric-label">监测时间</div>
                <div class="metric-value" style="font-size:1rem; color:#666;">
                    {report['basic_info']['detection_time'][:10]}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ========== 详细信息（两列布局） ==========
        detail_col1, detail_col2 = st.columns([3, 2])

        with detail_col1:
            # 水质等级大徽章
            display_level_badge(level)

            # 各类别概率
            st.markdown("### 📈 各类别概率分布")
            probs = classification['class_probabilities']
            max_prob = max(probs.values())
            for level_name in ['优', '良', '中', '差']:
                if level_name in probs:
                    display_probability_bar(level_name, probs[level_name], max_prob)

            # 等级描述
            st.markdown(f"""
            <div style="
                background: #F5F5F5;
                padding: 1rem;
                border-radius: 10px;
                margin: 0.5rem 0;
                border-left: 4px solid {get_level_color(level)};
            ">
                <strong>等级说明：</strong>{classification['description']}
            </div>
            """, unsafe_allow_html=True)

        with detail_col2:
            # 光学特征雷达图
            st.markdown("### 🎯 光学特征雷达图")
            display_feature_chart(features)

            # 特征贡献TOP
            contribution = report.get('feature_contribution', {})
            if contribution and 'top_contributing_features' in contribution:
                st.markdown("### 🔑 关键影响特征")
                top_features = contribution['top_contributing_features'][:5]
                for i, feat in enumerate(top_features):
                    feat_name = feat['name']
                    contrib_val = feat['contribution']
                    display_names = {
                        'R_mean': '红色通道均值', 'G_mean': '绿色通道均值',
                        'B_mean': '蓝色通道均值', 'brightness': '图像亮度',
                        'contrast': '图像对比度', 'turbidity_index': '综合浑浊指数',
                        'blue_green_ratio': '蓝绿通道比', 'edge_density': '边缘密度',
                        'particle_count': '颗粒计数', 'texture_homogeneity': '纹理均匀度',
                        'texture_entropy': '纹理熵', 'texture_contrast': '纹理对比度',
                        'turbidity_entropy': '浑浊熵值',
                    }
                    display_name = display_names.get(feat_name, feat_name)
                    bar_width = min(contrib_val / 5 * 100, 100)
                    st.markdown(f"""
                    <div class="feature-bar">
                        <div style="display:flex; justify-content:space-between; font-size:0.85rem;">
                            <span>{i+1}. {display_name}</span>
                            <span>{contrib_val:.2f}</span>
                        </div>
                        <div class="progress-container">
                            <div class="progress-bar" style="width:{bar_width}%; background:#0066CC; height:16px; line-height:16px; font-size:0.7rem;">
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # ========== 特征分析摘要 ==========
        st.markdown("---")
        st.markdown("### 🔬 光学特征分析摘要")
        st.markdown(f"""
        <div style="
            background: #F8F9FA;
            padding: 1.2rem;
            border-radius: 10px;
            font-family: 'Microsoft YaHei', sans-serif;
            line-height: 1.8;
        ">
        {report['feature_summary']}
        </div>
        """, unsafe_allow_html=True)

        # ========== 详细特征数据（折叠面板） ==========
        with st.expander("📊 查看全部26维光学特征数据", expanded=False):
            display_feature_table(features)

        # ========== AI分析报告 ==========
        if ai_report:
            st.markdown("---")
            st.markdown("## 📑 AI智能分析报告")
            st.markdown("""
            <div style="color:#888; font-size:0.9rem; margin-bottom:1rem;">
                🤖 本报告由大语言模型基于水质检测数据自动生成，达到政府环保监测报告水平
            </div>
            """, unsafe_allow_html=True)

            # 报告标签页
            report_tabs = st.tabs([
                "📋 水质分析报告",
                "🔍 污染原因分析",
                "⚠️ 风险预测",
                "💡 治理建议",
                "📈 未来趋势预测",
            ])

            tab_contents = [
                ("水质分析报告", ai_report.get('analysis_report', '')),
                ("污染原因分析", ai_report.get('pollution_analysis', '')),
                ("风险预测", ai_report.get('risk_prediction', '')),
                ("治理建议", ai_report.get('treatment_advice', '')),
                ("未来趋势预测", ai_report.get('trend_prediction', '')),
            ]

            for i, (title, content) in enumerate(tab_contents):
                with report_tabs[i]:
                    if content:
                        st.markdown(f"<div style='white-space:pre-wrap; font-family:monospace; background:#f8f9fa; padding:1.5rem; border-radius:10px;'>{content}</div>", unsafe_allow_html=True)
                    else:
                        st.info("暂无分析数据")
        else:
            st.markdown("---")
            st.markdown("### 🤖 AI智能报告")
            if enable_llm:
                st.warning("""
                ⚠️ 大模型连接可能存在问题。如需AI增强报告，请：
                1. 检查API地址和密钥是否正确
                2. 确保本地LM Studio已启动或API服务可用
                3. 或设置环境变量：LLM_API_KEY 和 LLM_API_BASE
                """)
            else:
                st.info("""
                💡 **启用AI报告增强可获得：**
                - 政府级水质分析报告
                - 污染原因专业分析
                - 风险预测与预警
                - 专业治理建议方案
                - 未来趋势预测
                
                请在侧边栏中启用"AI报告增强"并配置API。
                """)

        # ========== 报告导出 ==========
        st.markdown("---")
        st.markdown("### 📥 导出检测报告")

        export_col1, export_col2, export_col3 = st.columns([1, 1, 1])

        with export_col1:
            if st.button("📄 导出JSON报告", use_container_width=True):
                # 生成可序列化的报告
                export_report = {
                    'basic_info': report['basic_info'],
                    'classification': {
                        'level': classification['level'],
                        'confidence': classification['confidence'],
                        'risk': classification['risk'],
                        'description': classification['description'],
                        'class_probabilities': classification['class_probabilities'],
                    },
                    'feature_summary': report['feature_summary'],
                    'ai_report': {k: v for k, v in ai_report.items()} if ai_report else {},
                }
                report_json = json.dumps(export_report, ensure_ascii=False, indent=2)
                st.download_button(
                    label="⬇️ 下载JSON",
                    data=report_json,
                    file_name=f"水质报告_{location}_{datetime.date.today()}.json",
                    mime="application/json",
                    use_container_width=True,
                )

        with export_col2:
            if st.button("📊 导出特征数据", use_container_width=True):
                # 生成特征CSV
                features_df = pd.DataFrame([features])
                csv = features_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="⬇️ 下载CSV",
                    data=csv,
                    file_name=f"水质特征_{location}_{datetime.date.today()}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        with export_col3:
            if ai_report:
                # 合并所有报告
                full_report = f"""# 河道水质智能评估报告

## 基本信息
- 监测位置：{location}
- 水体类型：{water_type}
- 监测时间：{report['basic_info']['detection_time']}
- 水质等级：{level}
- AI可信度：{confidence*100:.1f}%
- 风险等级：{risk}

## 水质分析报告
{ai_report.get('analysis_report', '')}

## 污染原因分析
{ai_report.get('pollution_analysis', '')}

## 风险预测
{ai_report.get('risk_prediction', '')}

## 治理建议
{ai_report.get('treatment_advice', '')}

## 未来趋势预测
{ai_report.get('trend_prediction', '')}

---
报告生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                st.download_button(
                    label="⬇️ 下载完整报告",
                    data=full_report.encode('utf-8'),
                    file_name=f"水质完整报告_{location}_{datetime.date.today()}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

    # ========== 页脚 ==========
    st.markdown("""
    <div class="footer">
        🌊 河道水质智能评估平台 v1.0<br>
        浙江省大学生物理实验与科技创新竞赛 · 职教赛道
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 启动入口
# ============================================================

if __name__ == '__main__':
    main()
