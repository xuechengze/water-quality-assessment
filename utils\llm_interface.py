"""
大语言模型分析模块
==================
在完成水质检测后，自动调用大语言模型生成政府环保监测报告级别的分析。

功能：
1. 水质分析报告
2. 污染原因分析
3. 风险预测
4. 治理建议
5. 未来趋势预测

支持两种模式：
- 在线模式：调用 OpenAI 兼容 API（支持 ChatGPT / 本地 LM Studio 等）
- 离线模板模式：无需 API Key，直接生成高质量的模板报告
"""

import json
import os
import re
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class WaterQualityResult:
    """水质检测结果数据类"""
    level: str
    level_code: int
    confidence: float
    risk: str
    risk_color: str
    description: str
    class_probabilities: Dict[str, float]
    feature_summary: str = ""
    image_description: str = ""
    location: str = "未知位置"
    water_body_type: str = "河道"


class LLMReportGenerator:
    """
    LLM报告生成器
    生成政府环保监测报告级别的水质分析文档
    """

    def __init__(self, api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 model_name: str = "qwen2.5-7b-instruct"):
        """
        初始化LLM报告生成器

        参数:
            api_key: API密钥（如使用OpenAI或兼容API）
            api_base: API基础地址
                     - OpenAI: https://api.openai.com/v1
                     - LM Studio 本地: http://localhost:1234/v1
            model_name: 模型名称
        """
        self.api_key = api_key
        self.api_base = api_base
        self.model_name = model_name
        self._client = None

        # 尝试初始化API客户端
        if api_key and api_base:
            self._init_client()

    def _init_client(self):
        """初始化OpenAI兼容客户端"""
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
            )
            print(f"LLM客户端初始化成功: {self.api_base}")
        except ImportError:
            print("openai 库未安装，将使用离线模板模式")
            self._client = None
        except Exception as e:
            print(f"LLM客户端初始化失败: {e}，将使用离线模板模式")
            self._client = None

    def is_available(self) -> bool:
        """检查LLM是否可用"""
        return self._client is not None

    def generate_full_report(self, result: WaterQualityResult) -> Dict[str, str]:
        """
        生成完整的水质分析报告

        参数:
            result: 水质检测结果

        返回:
            包含各分析模块的字典
        """
        report = {}

        if self.is_available():
            # 在线模式：使用大模型生成
            report = self._generate_with_llm(result)
        else:
            # 离线模板模式：生成高质量模板报告
            report = self._generate_template_report(result)

        return report

    def _generate_with_llm(self, result: WaterQualityResult) -> Dict[str, str]:
        """使用大模型生成报告"""
        sections = {
            "analysis_report": "水质分析报告",
            "pollution_analysis": "污染原因分析",
            "risk_prediction": "风险预测",
            "treatment_advice": "治理建议",
            "trend_prediction": "未来趋势预测",
        }

        reports = {}

        for key, title in sections.items():
            prompt = self._build_prompt(key, result)

            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": self._get_system_prompt(key)
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.7,
                    max_tokens=2048,
                )
                reports[key] = response.choices[0].message.content
            except Exception as e:
                print(f"LLM生成 '{title}' 失败: {e}")
                reports[key] = self._generate_fallback(key, result)

        return reports

    def _get_system_prompt(self, section: str) -> str:
        """获取系统提示词"""
        base_prompt = """你是一名浙江省环境监测中心的高级环保工程师，\
具备十年的水质监测与评估经验。你正在编写一份提交给政府环保部门的 \
官方水质监测与评估报告。

写作要求：
1. 语言正式、专业、严谨，使用政府环保报告的标准格式
2. 包含具体数据分析和专业术语
3. 结论明确，建议可执行
4. 500-800字，使用中文
5. 不包含免责声明
6. 符合《地表水环境质量标准》(GB 3838-2002) 的技术规范
"""

        section_prompts = {
            "analysis_report": base_prompt + """
请撰写水质分析报告部分，包括：
1. 监测基本信息（时间、地点、水体类型）
2. 光学特征分析（颜色、浊度、透明度等）
3. 水质等级判定依据
4. 综合水质评估结论
""",
            "pollution_analysis": base_prompt + """
请撰写污染原因分析部分，包括：
1. 可能的污染源类型（工业、农业、生活等）
2. 基于光学特征的污染物推断
3. 污染程度评估
4. 污染扩散趋势分析
""",
            "risk_prediction": base_prompt + """
请撰写风险预测部分，包括：
1. 短期风险（1-3个月）
2. 中期风险（3-6个月）
3. 长期风险（6-12个月）
4. 对生态环境和人类健康的潜在影响
""",
            "treatment_advice": base_prompt + """
请撰写治理建议部分，包括：
1. 短期应急措施
2. 中期治理方案
3. 长期生态修复策略
4. 监测频率建议
5. 预期治理效果
""",
            "trend_prediction": base_prompt + """
请撰写未来趋势预测部分，包括：
1. 基于当前水质状况的趋势分析
2. 季节性变化影响
3. 不同治理方案下的预测情景
4. 水质改善时间预估
""",
        }
        return section_prompts.get(section, base_prompt)

    def _build_prompt(self, section: str, result: WaterQualityResult) -> str:
        """构建用户提示"""
        class_probs_str = ', '.join([
            f"{k}: {v*100:.1f}%"
            for k, v in result.class_probabilities.items()
        ])

        prompt = f"""
监测信息：
- 位置：{result.location}
- 水体类型：{result.water_body_type}
- 水质等级：{result.level}
- 可信度：{result.confidence*100:.1f}%
- 风险等级：{result.risk}

各类别概率：{class_probs_str}

光学特征分析：
{result.feature_summary}

{result.image_description}

请根据上述监测数据，撰写专业的 {section} 部分。
"""
        return prompt

    def _generate_template_report(self, result: WaterQualityResult) -> Dict[str, str]:
        """生成离线模板报告（高品质模板）"""
        level = result.level
        confidence_pct = result.confidence * 100
        location = result.location
        water_type = result.water_body_type

        # 根据不同等级生成报告
        reports = {
            "analysis_report": self._gen_analysis_report(
                result, level, confidence_pct, location, water_type
            ),
            "pollution_analysis": self._gen_pollution_analysis(result, level),
            "risk_prediction": self._gen_risk_prediction(result, level),
            "treatment_advice": self._gen_treatment_advice(result, level),
            "trend_prediction": self._gen_trend_prediction(result, level),
        }
        return reports

    def _gen_analysis_report(self, result, level, confidence, location, water_type):
        """生成水质分析报告"""
        level_descriptions = {
            '优': (
                "Ⅰ类 ~ Ⅱ类水质标准",
                "水体呈现天然的蓝绿色调，透明度高，无肉眼可见悬浮物",
                "水质优良，符合集中式生活饮用水地表水源地一级保护区标准",
                "水体清澈透明，光学特征表明悬浮物浓度极低，"
                "藻类含量处于正常水平，溶解氧充足"
            ),
            '良': (
                "Ⅱ类 ~ Ⅲ类水质标准",
                "水体轻微偏绿或偏蓝，透明度较好，有微量悬浮物",
                "水质良好，适用于集中式生活饮用水地表水源地二级保护区",
                "水体光学特征显示存在少量悬浮颗粒和微量有机物质，"
                "总体水质处于可接受水平，建议持续监测"
            ),
            '中': (
                "Ⅲ类 ~ Ⅳ类水质标准",
                "水体呈浅绿色或微黄色，透明度下降，可见悬浮物",
                "水质一般，适用于工业用水和一般景观水体",
                "水体浑浊度明显升高，光学特征表明存在一定程度的有机污染和"
                "营养盐超标，可能伴有藻类增殖趋势，需要重点关注"
            ),
            '差': (
                "Ⅳ类 ~ Ⅴ类水质标准",
                "水体呈黄褐色或深绿色，透明度差，悬浮物密集",
                "水质较差，仅适用于农业灌溉和一般工业冷却用水",
                "水体严重浑浊，光学特征显示悬浮物浓度超标，"
                "可能存在工业废水排放或生活污水直排，"
                "藻类大量繁殖，溶解氧可能严重不足，需立即采取治理措施"
            ),
        }

        std_level, appearance, usage, detail = level_descriptions.get(
            level, level_descriptions['中']
        )

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║              浙江省水体水质光学监测分析报告                      ║
╚══════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
一、监测基本信息
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

监测时间：{__import__('datetime').datetime.now().strftime('%Y年%m月%d日 %H:%M')}
监测地点：{location}
水体类型：{water_type}
监测方法：基于多模态AI与光学特征分析的水质智能评估技术
执行标准：《地表水环境质量标准》(GB 3838-2002)
监测仪器：高分辨率光学成像系统 + AI分析终端

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
二、光学特征分析
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 颜色特征分析
   {result.feature_summary.split(chr(10))[0] if chr(10) in result.feature_summary else ''}

2. 水体外观描述
   {appearance}

3. 光学参数综合分析
   - 综合浑浊指数：{result.class_probabilities.get('优', 0):.3f} ~ {result.class_probabilities.get('差', 0):.3f}
   - 检测可信度：{confidence:.1f}%
   - 对应水质标准：{std_level}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
三、水质等级判定
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

综合判定等级：{level}
判定依据：{usage}
详细分析：
{detail}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
四、综合评估结论
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

经多模态AI模型对水体光学特征进行综合分析，判定当前监测点位
水质等级为「{level}」，AI模型评估可信度为 {confidence:.1f}%。
{detail[:50] + '……' if len(detail) > 50 else detail}

建议相关部门根据本报告启动相应的水质管理措施。

报告生成时间：{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return report

    def _gen_pollution_analysis(self, result, level):
        """生成污染原因分析"""
        analyses = {
            '优': (
                "当前未检测到明显污染源",
                "水质保持天然状态，生态良好",
                "无需采取污染防治措施，继续保持常规监测即可",
                "无污染扩散风险",
            ),
            '良': (
                "可能存在轻微面源污染，如农业径流或大气沉降",
                "微量有机物质和悬浮颗粒物检测值略高于背景值",
                "建议排查上游农业活动和周边生活区排水情况",
                "污染程度低，自然降解能力充足",
            ),
            '中': (
                "可能存在以下污染源：\n"
                "  1) 农业面源污染：氮磷肥料流失导致水体富营养化趋势\n"
                "  2) 生活污水：周边村庄生活污水未完全处理排放\n"
                "  3) 底泥释放：内源污染释放营养盐\n"
                "  4) 工业排放：周边工业企业可能存在超标排放",
                "光学特征显示水体浑浊度显著升高，\n"
                "悬浮颗粒和有机质含量增加",
                "建议对上游3km范围内的排污口进行全面排查，\n"
                "重点检查工业企业和规模化养殖场",
                "污染有扩散趋势，如不及时控制，\n"
                "下游水质可能在3-6个月内进一步恶化",
            ),
            '差': (
                "污染源分析（高置信度）：\n"
                "  1) 工业废水直排：水体颜色异常（黄褐色），\n"
                "     特征光谱与工业废水排放高度吻合\n"
                "  2) 生活污水排放：悬浮物和有机物含量严重超标\n"
                "  3) 农业面源污染：氮磷富集导致严重富营养化\n"
                "  4) 内源污染：长期污染物积累导致的底泥二次释放\n"
                "  5) 可能存在的非法排污口",
                "光学特征综合异常：浊度严重超标，\n"
                "颜色偏离天然水体光谱范围，\n"
                "悬浮颗粒浓度高且粒径分布异常",
                "建议立即启动应急排查程序，\n"
                "对上游5km范围内所有可能的排污源进行地毯式排查",
                "污染物正在扩散，\n"
                "若不立即采取控制措施，将严重影响下游水源地和生态系统",
            ),
        }

        sources, indicators, actions, spread = analyses.get(level, analyses['中'])
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                   污染原因分析报告                              ║
╚══════════════════════════════════════════════════════════════╝

一、污染源识别
{ sources }

二、污染指示特征
根据光学特征分析：{indicators}

三、建议排查措施
{ actions }

四、污染扩散趋势
{ spread }
"""

    def _gen_risk_prediction(self, result, level):
        """生成风险预测"""
        risks = {
            '优': {
                'short': '无显著风险，水质维持优良状态',
                'mid': '如持续保护，水质可长期维持在优良水平',
                'long': '生态稳定性好，可持续提供优质水源',
                'health': '对人体健康无风险，可直接作为饮用水源',
            },
            '良': {
                'short': '低风险，但需关注季节性水质波动',
                'mid': '如不加强保护，可能在雨季后面临水质下降风险',
                'long': '需要建立长效监测机制，防止水质退化',
                'health': '对人体健康风险较低，建议煮沸后饮用',
            },
            '中': {
                'short': '中等风险，水质可能在1-3个月内继续恶化',
                'mid': '若不干预，3-6个月内可能恶化至Ⅳ类水质',
                'long': '持续恶化可能导致水体功能严重退化',
                'health': '存在一定健康风险，不建议直接接触水体',
            },
            '差': {
                'short': '高风险！水质可能在水华暴发期急剧恶化',
                'mid': '极度风险！若不治理，6个月内可能发生大规模水华',
                'long': '可能导致水生生态系统崩溃，恢复成本极高',
                'health': '严重健康风险！禁止接触水体，'
                         '可能含有致病菌和有毒物质',
            },
        }

        r = risks.get(level, risks['中'])
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                     水质风险预测报告                            ║
╚══════════════════════════════════════════════════════════════╝

一、短期风险预测（1-3个月）
风险等级：{result.risk}
评估：{r['short']}

二、中期风险预测（3-6个月）
{r['mid']}

三、长期风险预测（6-12个月）
{r['long']}

四、生态环境与健康风险评估
{r['health']}

五、综合风险等级：{result.risk}
建议监测频率：
- {'每月一次' if level == '优' else '每两周一次' if level == '良' else '每周一次' if level == '中' else '每日一次'}
"""

    def _gen_treatment_advice(self, result, level):
        """生成治理建议"""
        advices = {
            '优': {
                'emergency': '无需应急措施',
                'mid_term': '加强日常保护，防止污染源侵入',
                'long_term': '建立水源保护区，实施生态补偿机制',
                'monitor': '每月常规监测一次',
                'effect': '维持现状即可',
            },
            '良': {
                'emergency': '加强上游污染源巡查，控制面源污染',
                'mid_term': '实施生态缓冲带建设，减少农业面源输入',
                'long_term': '推进流域综合治理，建立水质预警系统',
                'monitor': '每两周监测一次',
                'effect': '3-6个月可维持或改善水质',
            },
            '中': {
                'emergency': '1) 立即排查上游排污口\n'
                             '2) 对重点污染源实施限排措施\n'
                             '3) 增加曝气设备，提升水体溶解氧',
                'mid_term': '1) 实施底泥疏浚工程\n'
                            '2) 建设人工湿地净化系统\n'
                            '3) 控制外源营养盐输入\n'
                            '4) 实施生态补水',
                'long_term': '1) 建设完整的水质自动监测站网\n'
                             '2) 推进流域产业结构调整\n'
                             '3) 实施全流域生态修复工程',
                'monitor': '每周监测一次',
                'effect': '6-12个月可恢复至良等级',
            },
            '差': {
                'emergency': '【紧急】立即启动以下措施：\n'
                             '1) 封堵所有可能的非法排污口\n'
                             '2) 启动应急曝气设备，防止鱼类死亡\n'
                             '3) 投放净水药剂（如PAC/PAM）进行应急处理\n'
                             '4) 下达限制用水通知，确保下游供水安全\n'
                             '5) 成立应急指挥部，启动多部门联动机制',
                'mid_term': '1) 实施全面的底泥清淤工程\n'
                            '2) 建设大型污水处理设施\n'
                            '3) 实施生态调水工程，加速水体置换\n'
                            '4) 投放微生物制剂进行生物修复\n'
                            '5) 重建水生植物群落',
                'long_term': '1) 彻底排查并整治所有污染源\n'
                             '2) 实施全流域水环境综合治理\n'
                             '3) 建立智慧水环境管理平台\n'
                             '4) 推进产业转型升级\n'
                             '5) 建立长效生态补偿机制',
                'monitor': '每日监测，关键时期实时监测',
                'effect': '1-2年可恢复至中等级，3-5年可达到良等级',
            },
        }

        a = advices.get(level, advices['中'])
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                     水质治理建议方案                            ║
╚══════════════════════════════════════════════════════════════╝

一、短期应急措施
{a['emergency']}

二、中期治理方案（3-6个月）
{a['mid_term']}

三、长期生态修复策略（6-24个月）
{a['long_term']}

四、监测频率建议
{a['monitor']}

五、预期治理效果
{a['effect']}
"""

    def _gen_trend_prediction(self, result, level):
        """生成未来趋势预测"""
        trends = {
            '优': (
                "在当前保护力度下，水质可长期维持优良状态。"
                "建议持续加强水源保护，防止工业化进程带来的污染风险。",
                "夏秋季节（6-10月）水温升高，可能出现短期藻类滋生，"
                "但不会影响总体水质。冬季水质最为稳定。",
                "情景一（积极）：加强保护，实施生态补偿 → 持续优良\n"
                "情景二（中性）：维持现状 → 保持优良\n"
                "情景三（消极）：周边开发增加 → 可能下降至良等级",
                "无改善需求，保持现状即可"
            ),
            '良': (
                "当前水质处于可接受水平，但存在下降隐患。"
                "若不加强管理，可能在1-2年内降至中等水平。",
                "夏秋季节藻类活性增强，水质可能短暂下降。"
                "雨季面源污染增加，需重点关注暴雨后的水质变化。",
                "情景一（积极）：开展生态修复 → 6个月内可能提升至优\n"
                "情景二（中性）：维持现状 → 长期维持良\n"
                "情景三（消极）：忽视保护 → 12个月后可能降至中",
                "如采取适当措施，预计6个月可见改善"
            ),
            '中': (
                "当前水质呈下降趋势，预警信号明显。"
                "主要压力来自持续输入的污染负荷和日益严重的内源污染。",
                "夏季高温期（7-9月）存在水华暴发风险。"
                "枯水期（11-2月）水体自净能力下降，污染浓度相对升高。",
                "情景一（积极）：积极治理 → 12个月内回升至良\n"
                "情景二（中性）：部分治理 → 水质维持中等\n"
                "情景三（消极）：不采取行动 → 6个月后降至差",
                "积极治理下12个月可见明显改善"
            ),
            '差': (
                "当前水质处于严重恶化通道。"
                "若不立即采取强有力的治理措施，水体生态功能将在短期内崩溃。",
                "夏季高温期极大概率暴发大规模蓝藻水华，"
                "可能导致鱼类大量死亡和严重的异味问题。"
                "秋季可能伴有黑臭现象。",
                "情景一（积极）：立即全面治理 → 12个月后恢复至中\n"
                "情景二（中性）：部分应急处理 → 维持差到中之间\n"
                "情景三（消极）：不采取行动 → 水体完全丧失功能",
                "立即治理下18个月可见初步改善，"
                "完全恢复至良需要3-5年"
            ),
        }

        trend, seasonal, scenarios, timeline = trends.get(level, trends['中'])
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                   水质未来趋势预测报告                          ║
╚══════════════════════════════════════════════════════════════╝

一、总体趋势分析
{trend}

二、季节性变化影响
{seasonal}

三、不同情景下的水质预测
{scenarios}

四、水质改善时间预估
{timeline}

五、建议
建议相关部门根据本预测报告，提前制定应对方案，优化水资源管理策略。
"""
