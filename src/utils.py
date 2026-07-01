"""
机器学习课设 —— 公共工具模块
==============================
供 data_preprocess.py、train_models.py、app.py 共同引用，
避免函数重复定义。
"""

# 成绩等级划分（全局常量）
GRADE_ORDER = ['优(90-100)', '良(80-89)', '中(70-79)', '差(<70)']


def score_to_grade(score):
    """将连续分数 [0,100] 映射为等级标签"""
    if score >= 90:
        return GRADE_ORDER[0]
    elif score >= 80:
        return GRADE_ORDER[1]
    elif score >= 70:
        return GRADE_ORDER[2]
    else:
        return GRADE_ORDER[3]
