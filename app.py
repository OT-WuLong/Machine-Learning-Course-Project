"""
机器学习课设 —— Streamlit 可视化交互面板（应用层）
===================================================
启动方式：streamlit run app.py
访问地址：http://localhost:8501

功能页面：
  1. 🎯 个体预测 —— 勾选特征 → 滑块/输入数值 → 实时预测
  2. 📈 总体分析 —— 热力图 / 渐进效果对比 / 数据统计
  3. 🔍 模型可视化 —— 决策树结构 / 模型评估图 / 混淆矩阵

设计特点：
  - 顶部导航栏（无侧边栏）
  - 自由勾选特征，未勾选的用训练集均值填充
  - 滑块+数字输入框双向同步
  - 始终使用 5 特征完整模型，按信息完整度显示精度
"""

import os
import streamlit as st
import numpy as np
import pandas as pd
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils import score_to_grade
warnings.filterwarnings('ignore', category=FutureWarning)

# ================================================================
# 路径配置
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')

# ================================================================
# 页面基础配置
# ================================================================
st.set_page_config(
    page_title="学生成绩预测系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",  # 侧边栏默认折叠
)

# 隐藏侧边栏和顶栏（Streamlit 默认的侧边栏容器用 CSS 隐藏）
st.markdown("""
<style>
    section[data-testid="stSidebar"] {display: none;}
    header[data-testid="stHeader"] {display: none;}
</style>
""", unsafe_allow_html=True)

# 设置 matplotlib 中文字体（确保图表正常显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ================================================================
# 全局加载：模型、配置、数据
# ================================================================
# @st.cache_resource / @st.cache_data 是 Streamlit 的缓存机制，
# 避免每次页面重绘都重新加载文件，提升性能。

@st.cache_resource
def load_models():
    """加载 4 组渐进式模型（2特征 ~ 5特征）"""
    models = {}
    for n in [2, 3, 4, 5]:
        models[f'a_{n}'] = joblib.load(f'{MODELS_DIR}/model_a_{n}feat.pkl')
        models[f'b_{n}'] = joblib.load(f'{MODELS_DIR}/model_b_{n}feat.pkl')
    return models

@st.cache_data
def load_config():
    """加载特征顺序和班级特征配置"""
    with open(f'{OUTPUTS_DIR}/feature_config.json', 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def load_data():
    """加载清洗后数据和渐进效果对比表"""
    df = pd.read_csv(f'{DATA_DIR}/cleaned_data.csv')
    results = pd.read_csv(f'{DATA_DIR}/progressive_results.csv')
    # 计算各班级占比（用于「全部」选项的编码）
    with open(f'{OUTPUTS_DIR}/feature_config.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    class_feats = cfg['class_features']
    class_props = {}
    total = len(df)
    for cls_col in class_feats:
        cls_name = cls_col.replace('班级_', '')
        class_props[cls_name] = df[cls_col].sum() / total
    return df, results, class_props

# 执行全局加载
models = load_models()
config = load_config()
df_clean, prog_results, class_props = load_data()

# ================================================================
# 全局常量定义
# ================================================================

# 特征显示名称（固定顺序，与训练时的 FEATURE_ORDER 一致）
# 注意：模型训练时用的是同样顺序的 5 个核心特征
FEATURE_ORDER = ['线上平时成绩', '线上章测试', '线上期末考', '线下互动', '线下期末考']

# 班级特征名（从配置中读取，如 ['班级_Sheet2', ..., '班级_Sheet7']）
CLASS_FEATURES = config['class_features']

# 班级选项（下拉框用中文显示）
CLASS_OPTIONS = ['全部', '1班', '2班', '3班', '4班', '5班', '6班', '7班']
# 中文班级名 → 内部 Sheet 名的映射（用于匹配模型中的班级特征）
CLASS_SHEET_MAP = {f'{i}班': f'Sheet{i}' for i in range(1, 8)}

# 特征信息：取值范围、步长、说明（用于滑块和数字输入框）
FEATURE_INFO = {
    '线上平时成绩': {'min': 0, 'max': 50, 'step': 1, 'desc': '平时作业、课堂表现等'},
    '线上章测试':   {'min': 0, 'max': 10, 'step': 1, 'desc': '各章节测验成绩'},
    '线上期末考':   {'min': 0, 'max': 40, 'step': 1, 'desc': '线上部分期末考试'},
    '线下互动':    {'min': 0, 'max': 100, 'step': 1, 'desc': '课堂互动、出勤等'},
    '线下期末考':  {'min': 0, 'max': 100, 'step': 1, 'desc': '线下期末考试'},
}

# 成绩等级划分（用于模型 B 的等级预测输出）
GRADE_ORDER = ['优(90-100)', '良(80-89)', '中(70-79)', '差(<70)']

# 各特征训练集均值（从 cleaned_data.csv 动态计算，避免硬编码）
_feature_cols = ['线上平时成绩', '线上章测试', '线上期末考', '线下互动', '线下期末考']
FEATURE_MEANS = {feat: df_clean[feat].mean() for feat in _feature_cols}

# ================================================================
# 低分预警 & 教学建议 相关配置
# ================================================================
# 各特征百分位阈值（从训练数据中计算得出，用于判断学生成绩是否偏低）
# P25 = 第 25 百分位，低于此值视为"薄弱"
# 满分 = 该特征最高可能得分
FEATURE_THRESHOLDS = {
    '线上平时成绩': {'p25': 40.0, 'p50': 45.0, 'max': 50,
                   'advice_weak': '线上平时成绩偏低（低于25%同学），建议加强日常作业提交质量和课堂参与度'},
    '线上章测试':   {'p25': 10.0, 'p50': 10.0, 'max': 10,
                   'advice_weak': '线上章测试失分较多，建议回顾章节知识点并重新完成测试'},
    '线上期末考':   {'p25': 38.0, 'p50': 38.5, 'max': 40,
                   'advice_weak': '线上期末考试成绩不理想，建议重点复习线上课程内容'},
    '线下互动':    {'p25': 60.0, 'p50': 70.6, 'max': 100,
                   'advice_weak': '线下互动参与度偏低（低于25%同学），建议积极出勤并参与课堂讨论'},
    '线下期末考':   {'p25': 48.0, 'p50': 60.0, 'max': 100,
                   'advice_weak': '线下期末考试成绩偏低，建议加强线下课程复习和练习'},
}

# ================================================================
# 辅助函数
# ================================================================

# score_to_grade 已提取至 utils.py


def generate_suggestions(scores_dict, checked_feats, pred_score, reg_grade, dt_grade):
    """
    根据学生输入的特征值和预测结果，生成低分预警和教学干预建议。

    参数：
        scores_dict : dict — 用户输入的各特征值
        checked_feats : set — 用户勾选的特征名
        pred_score : float — 模型A预测的连续分数
        reg_grade : str — 由回归分数推算的等级（模型A）
        dt_grade : str — 决策树直接预测的等级（模型B）

    返回：
        warnings : list[str] — 预警信息列表
        suggestions : list[str] — 干预建议列表
    """
    warnings_list = []
    suggestions_list = []

    # ---------- 低分预警 ----------
    # 仅当两个模型都认为成绩较差时才触发，避免模型分歧时弹出矛盾预警
    if pred_score < 70 and reg_grade == GRADE_ORDER[3] and dt_grade == GRADE_ORDER[3]:
        warnings_list.append(
            f"🚨 低分预警：预测总成绩仅 {pred_score:.1f} 分，"
            f"两个模型均判定为「差」，存在不及格风险！"
        )
    elif pred_score < 70 and reg_grade == GRADE_ORDER[3] and dt_grade != GRADE_ORDER[3]:
        # 模型分歧：回归认为差，决策树不认为差 → 提示而非预警
        warnings_list.append(
            f"⚠️ 模型分歧：线性回归预测 {pred_score:.1f} 分（{reg_grade}），"
            f"但决策树预测为{dt_grade}，建议综合参考"
        )

    # ---------- 教学干预建议 ----------
    # 遍历各个特征，检查已知特征中哪些低于 P25（薄弱环节）
    for feat in FEATURE_ORDER:
        if feat in checked_feats and feat in FEATURE_THRESHOLDS:
            threshold = FEATURE_THRESHOLDS[feat]
            actual_val = scores_dict.get(feat, 0)
            # 该特征低于 25% 同学 → 薄弱环节，生成建议
            if actual_val < threshold['p25']:
                suggestions_list.append(f"📌 {threshold['advice_weak']}（当前 {actual_val:.0f}/{threshold['max']}）")

    # 如果所有勾选的特征都高于 P25 但两个模型都判定较差，给出笼统建议
    if not suggestions_list and reg_grade == GRADE_ORDER[3] and dt_grade == GRADE_ORDER[3]:
        suggestions_list.append("📌 学生各项成绩均处于中等以上，但总成绩偏低，建议关注综合应用能力")

    return warnings_list, suggestions_list


def build_input_vector(scores_dict, checked_feats, selected_class_display):
    """
    构建预测用的特征向量。

    参数：
        scores_dict : dict — 用户输入的各特征值
        checked_feats : set — 用户勾选的特征名集合
        selected_class_display : str — 选择的班级（中文显示）

    返回：
        numpy.ndarray, shape (1, n_features)

    策略：
        - 勾选的特征：使用用户输入的实际值
        - 未勾选的特征：用训练集均值填充
        - 「全部」班级：所有班级特征设为 1，综合所有班级影响
        - 具体班级：对应班级特征为 1，其余为 0
    """
    vec = []
    # --- 核心特征部分 ---
    for feat in FEATURE_ORDER:
        if feat in checked_feats:
            vec.append(scores_dict.get(feat, FEATURE_MEANS[feat]))
        else:
            vec.append(FEATURE_MEANS[feat])  # 均值填充

    # --- 班级特征部分 ---
    if selected_class_display == '全部':
        # 「全部」= 使用各类别在训练集中的占比，实现真正的"平均"效果
        # （训练时 drop_first=True，Sheet1 全0为参照组，全1是训练集中不存在的非法输入）
        for cls_col in CLASS_FEATURES:
            cls_name = cls_col.replace('班级_', '')
            vec.append(class_props.get(cls_name, 0.0))
    else:
        selected_sheet = CLASS_SHEET_MAP.get(selected_class_display, 'Sheet1')
        for cls_col in CLASS_FEATURES:
            cls_sheet = cls_col.replace('班级_', '')
            vec.append(1.0 if selected_sheet == cls_sheet else 0.0)
    return np.array(vec).reshape(1, -1)


def show_data_overview():
    """在页面标题下方显示数据概览条"""
    st.markdown(
        f"<div style='display:flex;gap:2rem;padding:0.5rem 0;color:#666;font-size:0.9rem;'>"
        f"<span>📊 总样本数：{len(df_clean)}</span>"
        f"<span>🔢 核心特征：{len(FEATURE_ORDER)}个</span>"
        f"<span>🏫 班级特征：{len(CLASS_FEATURES)}个</span>"
        f"<span>📐 模型A：多元线性回归</span>"
        f"<span>🌲 模型B：决策树分类</span>"
        f"</div>",
        unsafe_allow_html=True
    )


# ================================================================
# 页面一：个体预测
# ================================================================
def page_predict():
    """
    个体预测页面：
      1. 用户通过勾选框选择要使用的特征（至少 2 个）
      2. 每个特征对应一个滑块 + 数字输入框（双向同步）
      3. 选择班级后点击「开始预测」
      4. 模型 A 输出预测分数，模型 B 输出等级和概率
      5. 显示当前信息完整度和估计精度
    """
    st.header("🎯 个体成绩预测（自由选特征）")
    st.markdown("""
    > 勾选你拥有的学生信息，至少 **勾选 2 个特征** 即可开始预测。
    > 未勾选的特征会用训练集均值填充，信息越多预测越准。
    """)
    show_data_overview()
    st.markdown("---")

    # 左右两栏布局：左 = 输入区，右 = 结果区
    col1, col2 = st.columns([1, 1.2], gap="large")

    # ---------- 左栏：输入区 ----------
    with col1:
        st.subheader("📝 选择要输入的学生信息")

        # 初始化勾选状态（默认勾选前 2 个：线上平时成绩 + 线上章测试）
        if 'checked_feats' not in st.session_state:
            st.session_state.checked_feats = set(FEATURE_ORDER[:2])

        # 5 个并排的勾选框
        cols_check = st.columns(len(FEATURE_ORDER))
        for i, feat in enumerate(FEATURE_ORDER):
            with cols_check[i]:
                st.checkbox(f"{feat}", value=feat in st.session_state.checked_feats,
                           key=f"ck_{i}")

        # 从勾选框状态重构 checked_feats
        st.session_state.checked_feats = {feat for i, feat in enumerate(FEATURE_ORDER)
                                          if st.session_state.get(f'ck_{i}', i < 2)}

        n_checked = len(st.session_state.checked_feats)

        # 提示信息
        if n_checked < 2:
            st.warning(f"⚠️ 已勾选 {n_checked} 个特征，至少勾选 2 个才能预测")
        elif n_checked == 5:
            st.success("✅ 已勾选全部5个特征，进行完整预测")
        else:
            st.info(f"📌 已勾选 {n_checked} 个特征，{5 - n_checked} 个未勾选的特征将用均值填充")

        st.markdown("---")
        st.subheader("📝 输入数值")

        # 每个特征一行：滑块（左）+ 数字输入框（右），双向同步
        # 同步原理：滑块改变时 on_change 更新数字框的 session_state，
        # 数字框改变时 on_change 更新滑块的 session_state。
        scores = {}
        for i, feat in enumerate(FEATURE_ORDER):
            info = FEATURE_INFO[feat]
            enabled = feat in st.session_state.checked_feats
            default_val = info['max'] // 2

            col_s, col_n = st.columns([3, 1])
            with col_s:
                st.slider(f"{feat}", info['min'], info['max'],
                         default_val, step=info['step'],
                         help=info['desc'], disabled=not enabled,
                         key=f"sl_{i}",
                         on_change=lambda i=i: st.session_state.__setitem__(
                             f'nm_{i}', st.session_state[f'sl_{i}']
                         ))
            with col_n:
                st.number_input("", info['min'], info['max'],
                               default_val, step=info['step'],
                               disabled=not enabled,
                               key=f"nm_{i}", label_visibility="collapsed",
                               on_change=lambda i=i: st.session_state.__setitem__(
                                   f'sl_{i}', st.session_state[f'nm_{i}']
                               ))
            scores[feat] = st.session_state.get(f'nm_{i}', default_val)

        # 班级选择
        selected_class = st.selectbox("选择班级", options=CLASS_OPTIONS, index=0)

        # 预测按钮（少于 2 个特征时禁用）
        predict_btn = st.button("🚀 开始预测", type="primary", use_container_width=True,
                               disabled=n_checked < 2)

    # ---------- 右栏：结果区 ----------
    with col2:
        n = n_checked
        # 获取完整 5 特征模型的精度作为基准
        max_r2_full = prog_results[prog_results['特征数'] == 5]['模型A_R2'].values[0]
        max_acc_full = prog_results[prog_results['特征数'] == 5]['模型B_准确率'].values[0]
        # 用信息完整度（已用特征数/5）估算当前精度
        info_ratio = n / 5
        # 从渐进效果表中动态读取最小 R²（2特征），避免硬编码
        min_r2 = prog_results[prog_results['特征数'] == 2]['模型A_R2'].values[0]
        est_r2 = min_r2 + (max_r2_full - min_r2) * ((n - 2) / 3)  # 线性插值

        st.subheader(f"📋 预测结果（已勾选 {n} 个特征）")
        st.markdown(f"""
        <div style="margin-bottom:10px">
            <span style="font-size:0.9rem;color:#666;">信息完整度：</span>
            <span style="font-size:1.1rem;font-weight:bold;color:#2ecc71;">{n}/5</span>
            <span style="font-size:0.9rem;color:#666;margin-left:10px;">（估计 R² ≈ {est_r2:.2f}）</span>
        </div>
        """, unsafe_allow_html=True)

        st.progress(info_ratio, text=f"信息完整度：{info_ratio:.0%}")

        # 执行预测
        if predict_btn and n_checked >= 2:
            # 构建特征向量（未勾选的特征自动用均值填充）
            input_vec = build_input_vector(scores, st.session_state.checked_feats, selected_class)

            # 始终使用 5 特征模型（最准确，未勾选特征用均值填充后也能正常工作）
            pred_score = models['a_5'].predict(input_vec)[0]
            pred_score = np.clip(pred_score, 0, 100)  # 限制在合理范围
            pred_grade = models['b_5'].predict(input_vec)[0]
            pred_proba = models['b_5'].predict_proba(input_vec)[0]
            class_names = models['b_5'].classes_

            # 获取完整模型的 MAE 作为参考指标
            mae_full = prog_results[prog_results['特征数'] == 5]['模型A_MAE'].values[0]
            # 以回归分数（R²=0.94）推算等级，比决策树分类更稳定可靠
            reg_grade = score_to_grade(pred_score)

            # 并排展示两个模型的结果卡片
            res_c1, res_c2 = st.columns(2)
            with res_c1:
                st.markdown("##### 模型A：线性回归")
                st.markdown(f"""
                <div style="background-color:#e8f5e9;padding:20px;border-radius:10px;text-align:center;">
                    <h2 style="color:#2e7d32;margin:0;">{pred_score:.1f}</h2>
                    <p style="color:#555;">预测总成绩（分）</p>
                    <p style="color:#999;font-size:0.8rem;">对应等级：{reg_grade} | MAE = {mae_full:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            with res_c2:
                st.markdown("##### 模型B：决策树")
                st.markdown(f"""
                <div style="background-color:#e3f2fd;padding:20px;border-radius:10px;text-align:center;">
                    <h2 style="color:#1565c0;margin:0;">{pred_grade}</h2>
                    <p style="color:#555;">决策树分类等级</p>
                    <p style="color:#999;font-size:0.8rem;">5特征准确率 = {max_acc_full:.2%}</p>
                </div>
                """, unsafe_allow_html=True)

            # 各等级预测概率的条形图
            st.markdown("##### 各等级预测概率")
            fig, ax = plt.subplots(figsize=(8, 2.5))
            colors = ['#4caf50' if p == max(pred_proba) else '#bdbdbd' for p in pred_proba]
            ax.barh(class_names, pred_proba, color=colors, edgecolor='white')
            ax.set_xlim(0, 1)
            ax.set_xlabel('概率')
            for bar, prob in zip(ax.containers[0], pred_proba):
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                        f'{prob:.1%}', va='center', fontsize=11)
            plt.tight_layout()
            st.pyplot(fig)

            # ================================================================
            # 智能学情预警 + 教学干预建议（拓展功能）
            # ================================================================
            # 用两个模型的等级共同判断预警，避免模型分歧时弹出矛盾预警
            warnings_list, suggestions_list = generate_suggestions(
                scores, st.session_state.checked_feats, pred_score, reg_grade, pred_grade
            )

            # 低分预警（红色高亮）
            if warnings_list:
                for w in warnings_list:
                    st.markdown(f"""
                    <div style="background-color:#ffebee;border:2px solid #e53935;
                                border-radius:10px;padding:15px;margin:10px 0;">
                        <span style="color:#c62828;font-size:1.1rem;font-weight:bold;">{w}</span>
                    </div>
                    """, unsafe_allow_html=True)

            # 教学干预建议
            if suggestions_list:
                with st.expander("💡 教学干预建议", expanded=True):
                    for s in suggestions_list:
                        st.markdown(s)

            # 输入特征汇总表
            used_feats = sorted(st.session_state.checked_feats,
                              key=lambda f: FEATURE_ORDER.index(f))
            input_summary = pd.DataFrame({
                '特征': used_feats,
                '输入值': [scores[f] for f in used_feats],
                '满分': [FEATURE_INFO[f]['max'] for f in used_feats]
            })
            st.dataframe(input_summary, use_container_width=True, hide_index=True)
        else:
            st.info("👈 请在左侧输入学生信息，然后点击「开始预测」")

        # 可展开的精度对比表
        with st.expander("📊 查看各特征数预测精度对比"):
            st.dataframe(prog_results[['特征数', '名称', '模型A_R2', '模型A_MAE', '模型B_准确率']],
                        use_container_width=True, hide_index=True)


# ================================================================
# 页面二：总体分析
# ================================================================
def page_analysis():
    """
    总体分析页面，三个子标签页：
      1. 相关性热力图 —— 展示各特征间的 Pearson 相关系数
      2. 渐进预测效果 —— 展示特征数增加对精度的影响
      3. 数据统计 —— 描述性统计 + 成绩分布直方图 + 等级分布
    """
    st.header("📈 总体数据分析")
    show_data_overview()
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["相关性热力图", "渐进预测效果", "数据统计"])

    # ---- 标签页 1：相关性热力图 ----
    with tab1:
        st.subheader("特征相关性热力图")
        # 选择 5 个核心特征 + 2 个计算总成绩 + 1 个目标变量
        corr_cols = FEATURE_ORDER + ['线上总成绩', '线下总成绩', '总成绩(线上+线下)']
        available_cols = [c for c in corr_cols if c in df_clean.columns]
        st.caption(f"参与热力图的特征（{len(available_cols)}个）：{'、'.join(available_cols)}")
        corr_matrix = df_clean[available_cols].corr()

        # Seaborn 热力图
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                    ax=ax, cbar_kws={'label': 'Pearson相关系数'})
        ax.set_title('各特征与总成绩的 Pearson 相关性热力图', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("""
        **读图方法**：
        - 看目标：关注 **总成绩(线上+线下)** 所在行/列，颜色越红表示与该特征相关性越强
        - 看颜色：红色 = 正相关，蓝色 = 负相关，颜色越深相关性越强
        - 看特征之间：若两个不同特征间 r > 0.9，可能存在多重共线性
        - **线下期末考**（r = 0.81）和 **线上平时成绩**（r = 0.72）与总成绩相关性最强
        """)

    # ---- 标签页 2：渐进预测效果 ----
    with tab2:
        st.subheader("渐进预测效果对比")
        try:
            st.image(f'{FIGURES_DIR}/progressive_comparison.png', use_container_width=True)
        except FileNotFoundError:
            st.error("渐进对比图未找到，请先运行 train_models.py")
        st.markdown("""
        **解读**：
        - **R² 和准确率** 随特征增多而提升，**MAE** 随特征增多而降低
        - 2 个特征 → 5 个特征：R² 从 0.69 → 0.94，信息越完整预测越准确
        """)
        st.dataframe(prog_results[['特征数', '名称', '模型A_R2', '模型A_MAE', '模型B_准确率']],
                    use_container_width=True, hide_index=True)

    # ---- 标签页 3：数据统计 ----
    with tab3:
        st.subheader("数据集统计信息")
        stat_cols = FEATURE_ORDER + ['总成绩(线上+线下)']
        stat_cols = [c for c in stat_cols if c in df_clean.columns]
        st.dataframe(df_clean[stat_cols].describe().round(2), use_container_width=True)

        # 总成绩分布直方图
        st.subheader("总成绩分布")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(df_clean['总成绩(线上+线下)'], bins=20, color='#42a5f5', edgecolor='white', alpha=0.8)
        ax.axvline(df_clean['总成绩(线上+线下)'].mean(), color='red', linestyle='--',
                   linewidth=2, label=f"均值: {df_clean['总成绩(线上+线下)'].mean():.1f}")
        ax.set_xlabel('总成绩'); ax.set_ylabel('人数')
        ax.set_title('总成绩分布直方图'); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig)

        # 成绩等级分布柱状图（使用副本避免修改缓存的 df_clean）
        st.subheader("成绩等级分布")
        df_display = df_clean.copy()
        df_display['等级'] = df_display['总成绩(线上+线下)'].apply(score_to_grade)
        grade_counts = df_display['等级'].value_counts().reindex(GRADE_ORDER)
        fig, ax = plt.subplots(figsize=(8, 4))
        colors_grade = ['#4caf50', '#8bc34a', '#ff9800', '#f44336']
        bars = ax.bar(GRADE_ORDER, grade_counts.values, color=colors_grade, edgecolor='white')
        ax.set_ylabel('人数'); ax.set_title('成绩等级分布')
        for bar, count in zip(bars, grade_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(count), ha='center', fontsize=12)
        plt.tight_layout(); st.pyplot(fig)


# ================================================================
# 页面三：模型可视化
# ================================================================
def page_visual():
    """
    模型可视化页面，三个子标签页：
      1. 决策树结构图 —— 完整的分支规则可视化
      2. 模型A评估图 —— 预测vs真实散点图、残差分布、系数柱状图
      3. 模型B评估图 —— 混淆矩阵热力图、特征重要性图
    """
    st.header("🔍 模型可视化")
    st.markdown("> 以下图表基于完整5特征模型展示")
    show_data_overview()
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["决策树结构", "模型A评估图", "模型B评估图"])

    # ---- 标签页 1：决策树结构 ----
    with tab1:
        st.subheader("决策树分支规则图")
        try:
            # 由 train_models.py 生成的完整决策树图像
            st.image(f'{FIGURES_DIR}/model_b_tree_structure.png', use_container_width=True)
        except FileNotFoundError:
            st.error("决策树结构图未找到，请先运行 train_models.py")
        st.markdown("""
        **读图方法**：
        - 从根节点（顶部）开始，根据分裂条件判断走左分支还是右分支
        - 到达叶节点后，该节点的主导类别即为预测结果
        - 基尼系数越低（接近 0），节点纯度越高
        """)

    # ---- 标签页 2：模型A评估图 ----
    with tab2:
        st.subheader("模型A：线性回归评估图")
        col1, col2 = st.columns(2)
        with col1:
            try:
                st.image(f'{FIGURES_DIR}/model_a_evaluation.png', use_container_width=True)
            except FileNotFoundError:
                st.error("评估图未找到，请先运行 train_models.py")
            st.markdown("**散点图**：散点越贴近对角线预测越准确；**残差图**：残差应在 0 附近随机分布")
        with col2:
            try:
                st.image(f'{FIGURES_DIR}/model_a_coefficients.png', use_container_width=True)
            except FileNotFoundError:
                st.error("系数图未找到，请先运行 train_models.py")
            st.markdown("**系数图**：绿色 = 正相关，红色 = 负相关，条越长影响越大")

        # 模型A关键指标
        r2_full = prog_results[prog_results['特征数'] == 5]['模型A_R2'].values[0]
        mae_full = prog_results[prog_results['特征数'] == 5]['模型A_MAE'].values[0]
        rmse_full = prog_results[prog_results['特征数'] == 5]['模型A_RMSE'].values[0]
        mc = st.columns(3)
        mc[0].metric("R² (决定系数)", f"{r2_full:.4f}")
        mc[1].metric("MAE (平均绝对误差)", f"{mae_full:.4f}")
        mc[2].metric("RMSE (均方根误差)", f"{rmse_full:.4f}")

    # ---- 标签页 3：模型B评估图 ----
    with tab3:
        st.subheader("模型B：决策树评估图")
        col1, col2 = st.columns(2)
        with col1:
            try:
                st.image(f'{FIGURES_DIR}/model_b_confusion_matrix.png', use_container_width=True)
            except FileNotFoundError:
                st.error("混淆矩阵图未找到，请先运行 train_models.py")
            st.markdown("**混淆矩阵**：对角线 = 正确预测，颜色越深数字越大")
        with col2:
            try:
                st.image(f'{FIGURES_DIR}/model_b_feature_importance.png', use_container_width=True)
            except FileNotFoundError:
                st.error("特征重要性图未找到，请先运行 train_models.py")
            st.markdown("**特征重要性**：线下期末考（0.47）是最重要的分裂特征")

        # 模型B关键指标
        acc_full = prog_results[prog_results['特征数'] == 5]['模型B_准确率'].values[0]
        depth = models['b_5'].get_depth()
        leaves = models['b_5'].get_n_leaves()
        mc2 = st.columns(3)
        mc2[0].metric("准确率 (Accuracy)", f"{acc_full:.4f}")
        mc2[1].metric("模型深度", f"{depth}")
        mc2[2].metric("叶节点数", f"{leaves}")

    # 双模型对比总结
    st.markdown("---")
    st.subheader("⚖️ 双模型对比总结")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **模型A：多元线性回归**
        - ✅ 预测精细到具体分数，R² = 0.94
        - ✅ 可解释性强（权重系数直观）
        - ⚠️ 假设特征与目标呈线性关系
        """)
    with c2:
        st.markdown("""
        **模型B：决策树分类**
        - ✅ 等级划分更符合教学管理场景，准确率 85.4%
        - ✅ 无需线性假设
        - ⚠️ 信息少时准确率偏低（2特征仅55.3%）
        """)


# ================================================================
# 顶部导航栏 + 页面路由
# ================================================================
# 使用 3 个按钮实现顶部导航，无侧边栏。
# st.session_state.page 记录当前页面状态，点击按钮切换。
# ================================================================

if 'page' not in st.session_state:
    st.session_state.page = 'predict'  # 默认页面：个体预测

# 3 个等宽按钮作为导航栏，第 4 列占位（空白，保持右对齐）
nav_cols = st.columns([1, 1, 1, 6], vertical_alignment="center")

with nav_cols[0]:
    if st.button("🎯 个体预测", use_container_width=True,
                 type="primary" if st.session_state.page == 'predict' else "secondary"):
        st.session_state.page = 'predict'
        st.rerun()
with nav_cols[1]:
    if st.button("📈 总体分析", use_container_width=True,
                 type="primary" if st.session_state.page == 'analysis' else "secondary"):
        st.session_state.page = 'analysis'
        st.rerun()
with nav_cols[2]:
    if st.button("🔍 模型可视化", use_container_width=True,
                 type="primary" if st.session_state.page == 'visual' else "secondary"):
        st.session_state.page = 'visual'
        st.rerun()

st.markdown("---")  # 导航栏与页面内容的分隔线

# 根据 st.session_state.page 渲染对应页面
if st.session_state.page == 'predict':
    page_predict()
elif st.session_state.page == 'analysis':
    page_analysis()
elif st.session_state.page == 'visual':
    page_visual()
