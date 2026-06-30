"""
机器学习课设 —— 模型训练脚本
===============================
训练两种模型（渐进式，4 个版本）：
  模型 A：多元线性回归（预测连续分数）
  模型 B：决策树分类（预测成绩等级：优/良/中/差）

渐进式含义：按课程进度顺序，分别用 2、3、4、5 个核心特征训练模型，
模拟「信息越完整预测越准」的教学场景。每个版本附带对应的班级特征。

输出：4 组模型文件（.pkl）+ 效果对比图 + 特征配置文件
"""

import os
import numpy as np
import pandas as pd
import warnings
import joblib
import json
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             accuracy_score)
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from utils import score_to_grade, GRADE_ORDER
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================================================================
# 1. 加载预处理后的数据
# ================================================================
# cleaned_data.csv 是 data_preprocess.py 的输出，
# 包含完整清洗 + 独热编码后的数据。
# selected_features.txt 记录最终入选的 11 个特征名。
# ================================================================
print("=" * 60)
print("[1] 加载预处理后的数据")
print("=" * 60)

df = pd.read_csv(f'{BASE_DIR}/cleaned_data.csv')
with open(f'{BASE_DIR}/selected_features.txt', 'r', encoding='utf-8') as f:
    all_feature_names = [line.strip() for line in f.readlines()]

print(f"全部特征数: {len(all_feature_names)}")
print(f"全部特征: {all_feature_names}")

# 核心特征（5 个原始得分），按课程进度顺序排列
# 这个顺序很重要：2特征模型用前2个，3特征用前3个...
FEATURE_ORDER = ['线上平时成绩', '线上章测试', '线上期末考', '线下互动', '线下期末考']

# 班级独热编码特征（始终附带，不参与渐进选择）
CLASS_FEATURES = [c for c in all_feature_names if c.startswith('班级_')]

print(f"\n核心特征顺序: {FEATURE_ORDER}")
print(f"班级特征: {CLASS_FEATURES}")

# ================================================================
# 2. 按特征数量渐进训练（2特征 → 3特征 → 4特征 → 5特征）
# ================================================================
# 对每个特征数版本，分别训练模型 A 和模型 B。
# 训练数据的 X = N个核心特征 + 全部班级特征。
# ================================================================
FEATURE_SETS = {
    2: {'name': '2特征-基础预测', 'feats': FEATURE_ORDER[:2]},
    3: {'name': '3特征-中期预测', 'feats': FEATURE_ORDER[:3]},
    4: {'name': '4特征-进阶预测', 'feats': FEATURE_ORDER[:4]},
    5: {'name': '5特征-完整预测', 'feats': FEATURE_ORDER[:5]},
}

target_col = '总成绩(线上+线下)'

# 成绩等级划分（全局常量，定义在 utils.py 中）

results_summary = []  # 收集各版本评估结果，最后汇总对比

for n_feat, info in FEATURE_SETS.items():
    print("\n" + "=" * 60)
    print(f"[{info['name']}] ({n_feat}个核心特征 + {len(CLASS_FEATURES)}个班级特征 = {n_feat+len(CLASS_FEATURES)}维)")
    print("=" * 60)

    # 选择当前版本所需的特征列
    selected = info['feats'] + CLASS_FEATURES
    X = df[selected].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---- 模型 A：线性回归 ----
    # 最小二乘法拟合，无需额外超参数
    model_a = LinearRegression()
    model_a.fit(X_train, y_train)
    y_pred_a = model_a.predict(X_test)

    r2 = r2_score(y_test, y_pred_a)
    mae = mean_absolute_error(y_test, y_pred_a)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_a))

    print(f"  模型A - R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    print(f"  系数: {pd.DataFrame({'特征': selected, '权重': np.round(model_a.coef_, 4)}).to_string(index=False)}")

    joblib.dump(model_a, f'{BASE_DIR}/model_a_{n_feat}feat.pkl')

    # ---- 模型 B：决策树分类 ----
    # 将连续分数转为等级标签，训练分类器
    y_train_cls = np.array([score_to_grade(s) for s in y_train])
    y_test_cls = np.array([score_to_grade(s) for s in y_test])

    # GridSearchCV 自动搜索最优超参数，防止过拟合
    # max_depth=7 搭配 min_samples_leaf≥5，兼顾准确率与泛化能力
    param_grid = {
        'max_depth': [5, 7, 10],               # 树的最大深度
        'min_samples_split': [10, 20],          # 内部节点再划分所需最小样本数
        'min_samples_leaf': [5, 10],            # 叶节点最少样本数
    }
    dt_base = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    grid_search = GridSearchCV(dt_base, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train_cls)

    model_b = grid_search.best_estimator_
    y_pred_b = model_b.predict(X_test)
    accuracy = accuracy_score(y_test_cls, y_pred_b)

    print(f"  模型B - 准确率: {accuracy:.4f}, 最佳参数: {grid_search.best_params_}")
    print(f"  特征重要性:\n{pd.DataFrame({'特征': selected, '重要性': model_b.feature_importances_}).to_string(index=False)}")

    joblib.dump(model_b, f'{BASE_DIR}/model_b_{n_feat}feat.pkl')

    # 记录本版本结果
    results_summary.append({
        '特征数': n_feat,
        '名称': info['name'],
        '模型A_R2': round(r2, 4),
        '模型A_MAE': round(mae, 4),
        '模型A_RMSE': round(rmse, 4),
        '模型B_准确率': round(accuracy, 4),
        '模型B_最佳参数': str(grid_search.best_params_),
    })

# ================================================================
# 2.5 生成模型评估图（5特征完整模型）
# ================================================================
# 生成 5 张评估图供 app.py「模型可视化」页面展示：
#   1. model_a_evaluation.png     预测vs真实散点图 + 残差分布
#   2. model_a_coefficients.png   回归系数柱状图
#   3. model_b_confusion_matrix.png 混淆矩阵热力图
#   4. model_b_tree_structure.png   决策树结构图
#   5. model_b_feature_importance.png 特征重要性图
# ================================================================
print("\n" + "=" * 60)
print("【生成模型评估图（5特征模型）】")
print("=" * 60)

# 取最后一轮（5特征）的变量
sel5 = FEATURE_ORDER + CLASS_FEATURES
model_a5 = model_a  # 循环结束后的 model_a 就是 5 特征版本
model_b5 = model_b

# ---- 图1: 模型A评估（散点图 + 残差分布）----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 左: 预测 vs 真实
ax1.scatter(y_test, y_pred_a, alpha=0.6, edgecolors='k', linewidth=0.5, color='#42a5f5')
lims = [min(y_test.min(), y_pred_a.min()), max(y_test.max(), y_pred_a.max())]
ax1.plot(lims, lims, 'r--', linewidth=2, label='理想线 (y=x)')
ax1.set_xlabel('真实总成绩'); ax1.set_ylabel('预测总成绩')
ax1.set_title(f'模型A：预测 vs 真实（R²={r2:.4f}）')
ax1.legend(); ax1.grid(True, alpha=0.3)

# 右: 残差分布
residuals = y_test - y_pred_a
ax2.hist(residuals, bins=20, color='#ef5350', edgecolor='white', alpha=0.8)
ax2.axvline(0, color='black', linestyle='--', linewidth=1.5)
ax2.set_xlabel('残差（真实 - 预测）'); ax2.set_ylabel('频数')
ax2.set_title('模型A：残差分布'); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{BASE_DIR}/model_a_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  model_a_evaluation.png [OK]")

# ---- 图2: 回归系数柱状图 ----
coef_df = pd.DataFrame({'特征': sel5, '系数': model_a5.coef_})
coef_df['abs'] = coef_df['系数'].abs()
coef_df = coef_df.sort_values('abs', ascending=True)
colors_coef = ['#4caf50' if c >= 0 else '#ef5350' for c in coef_df['系数']]

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(coef_df['特征'], coef_df['系数'], color=colors_coef, edgecolor='white')
ax.set_xlabel('回归系数'); ax.set_title('模型A：各特征回归系数')
ax.axvline(0, color='black', linewidth=0.8)
for i, (feat, val) in enumerate(zip(coef_df['特征'], coef_df['系数'])):
    ax.text(val + (0.02 if val >= 0 else -0.02), i, f'{val:.3f}',
            va='center', ha='left' if val >= 0 else 'right', fontsize=9)
plt.tight_layout()
plt.savefig(f'{BASE_DIR}/model_a_coefficients.png', dpi=150, bbox_inches='tight')
plt.close()
print("  model_a_coefficients.png [OK]")

# ---- 图3: 混淆矩阵 ----
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_cls, y_pred_b, labels=GRADE_ORDER)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[g[:2] for g in GRADE_ORDER],
            yticklabels=[g[:2] for g in GRADE_ORDER], ax=ax)
ax.set_xlabel('预测等级'); ax.set_ylabel('真实等级')
ax.set_title(f'模型B：混淆矩阵（准确率={accuracy:.4f}）')
plt.tight_layout()
plt.savefig(f'{BASE_DIR}/model_b_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  model_b_confusion_matrix.png [OK]")

# ---- 图4: 决策树结构 ----
from sklearn.tree import plot_tree
fig, ax = plt.subplots(figsize=(30, 15))
plot_tree(model_b5, feature_names=sel5,
          class_names=[g[:4] for g in GRADE_ORDER],
          filled=True, rounded=True, fontsize=8, ax=ax)
ax.set_title('模型B：决策树结构')
plt.tight_layout()
plt.savefig(f'{BASE_DIR}/model_b_tree_structure.png', dpi=100, bbox_inches='tight')
plt.close()
print("  model_b_tree_structure.png [OK]")

# ---- 图5: 特征重要性 ----
imp_df = pd.DataFrame({'特征': sel5, '重要性': model_b5.feature_importances_})
imp_df = imp_df.sort_values('重要性', ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(imp_df['特征'], imp_df['重要性'], color='#2196f3', edgecolor='white')
ax.set_xlabel('特征重要性'); ax.set_title('模型B：决策树特征重要性')
for i, (feat, val) in enumerate(zip(imp_df['特征'], imp_df['重要性'])):
    ax.text(val + 0.005, i, f'{val:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(f'{BASE_DIR}/model_b_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  model_b_feature_importance.png [OK]")

# ================================================================
# 3. 渐进效果对比 与 可视化
# ================================================================
# 生成对比表格和折线图，展示随特征数增加模型精度的变化趋势。
# 这个图在 app.py 的「渐进预测效果」标签页中展示。
# ================================================================
print("\n" + "=" * 60)
print("【渐进预测效果对比】")
print("=" * 60)

summary_df = pd.DataFrame(results_summary)
print(summary_df.to_string(index=False))
summary_df.to_csv(f'{BASE_DIR}/progressive_results.csv', index=False, encoding='utf-8-sig')

# 绘制双轴折线图：左轴 R² + 准确率，右轴 MAE
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
feat_counts = [r['特征数'] for r in results_summary]
r2_values = [r['模型A_R2'] for r in results_summary]
acc_values = [r['模型B_准确率'] for r in results_summary]

ax1.plot(feat_counts, r2_values, 'o-', color='#2ecc71', linewidth=2, markersize=8, label='模型A R²')
ax1.plot(feat_counts, acc_values, 's-', color='#3498db', linewidth=2, markersize=8, label='模型B 准确率')
ax1.set_xlabel('核心特征数')
ax1.set_ylabel('得分')
ax1.set_title('预测效果随特征数变化')
ax1.set_xticks(feat_counts)
ax1.legend()
ax1.grid(True, alpha=0.3)
for i, (r2_v, acc_v) in enumerate(zip(r2_values, acc_values)):
    ax1.annotate(f'{r2_v:.3f}', (feat_counts[i], r2_v), textcoords="offset points",
                 xytext=(0, 10), ha='center', fontsize=9, color='#2ecc71')
    ax1.annotate(f'{acc_v:.3f}', (feat_counts[i], acc_v), textcoords="offset points",
                 xytext=(0, -15), ha='center', fontsize=9, color='#3498db')

ax2 = axes[1]
mae_values = [r['模型A_MAE'] for r in results_summary]
ax2.plot(feat_counts, mae_values, 'o-', color='#e74c3c', linewidth=2, markersize=8)
ax2.set_xlabel('核心特征数')
ax2.set_ylabel('MAE')
ax2.set_title('模型A误差随特征数变化')
ax2.set_xticks(feat_counts)
ax2.grid(True, alpha=0.3)
for i, v in enumerate(mae_values):
    ax2.annotate(f'{v:.2f}', (feat_counts[i], v), textcoords="offset points",
                 xytext=(0, 10), ha='center', fontsize=9, color='#e74c3c')

plt.tight_layout()
plt.savefig(f'{BASE_DIR}/progressive_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n渐进对比图已保存: progressive_comparison.png")

# 保存特征配置供 app.py 使用
feature_config = {
    'feature_order': FEATURE_ORDER,
    'class_features': CLASS_FEATURES,
}
with open(f'{BASE_DIR}/feature_config.json', 'w', encoding='utf-8') as f:
    json.dump(feature_config, f, ensure_ascii=False, indent=2)
print("特征配置已保存: feature_config.json")

print("\n全部模型训练完成！")
print("\n输出模型文件:")
for n in [2, 3, 4, 5]:
    print(f"  - model_a_{n}feat.pkl  (线性回归 {n}特征)")
    print(f"  - model_b_{n}feat.pkl  (决策树 {n}特征)")
