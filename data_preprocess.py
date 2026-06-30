"""
机器学习课设 —— 完整数据预处理流程
====================================
该脚本负责从原始 Excel 读取数据，完成以下任务：
  1. 合并 7 个表单 → 统一 DataFrame
  2. 训练/测试集划分（8:2）—— 在预处理之前划分，防止数据泄露
  3. 缺失值识别与中位数填充（仅用训练集中位数）
  4. IQR 异常值检测与 Winsorize 截尾处理（仅用训练集参数）
  5. 班级类别特征的 One-Hot 独热编码
  6. Pearson 相关系数计算与特征筛选
  7. 保存清洗后数据和特征列表

输出文件供 train_models.py 和 app.py 使用。
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ================================================================
# 路径配置：使用相对路径（相对于脚本所在目录）
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, '分数数据集.xlsx')

# ================================================================
# 第1步：读取所有 7 个表单并合并
# ================================================================
# 原始 Excel 有 7 个 Sheet，每个代表一个班级的成绩数据。
# 每个 Sheet 的结构相同：
#   第 1-3 行：合并单元格的表头（无实际数据）
#   第 4 行起：实际学生数据（含公式计算值）
# 我们跳过前 3 行，从第 4 行开始提取数值行。
# ================================================================
print("=" * 70)
print("【第1步】读取并合并所有表单数据")
print("=" * 70)

# sheet_name=None → 返回 {sheet名: DataFrame} 字典
all_sheets = pd.read_excel(FILE_PATH, sheet_name=None, header=None, engine='openpyxl')
print(f"共找到 {len(all_sheets)} 个表单: {list(all_sheets.keys())}")

# 列名映射（与原始 Excel 列结构对应）
# 原始列结构：学号 | 线上平时成绩 | 线上章测试 | 线上期末考 | 线上总成绩
#             | 线下互动 | 线下期末考 | 线下总成绩
#             | 总成绩(线上+线下) | 总成绩(平时44%+期末考56%)
#             | 平时成绩 | 期末成绩
COL_NAMES = {
    0: '学号',
    1: '线上平时成绩',
    2: '线上章测试',
    3: '线上期末考',
    4: '线上总成绩',
    5: '线下互动',
    6: '线下期末考',
    7: '线下总成绩',
    8: '总成绩(线上+线下)',       # ← 目标变量（模型预测的目标）
    9: '总成绩(平时44%+期末考56%)',
    10: '平时成绩',
    11: '期末成绩',
}

all_data = []  # 收集所有表单的有效数据行
for sheet_name, sheet_data in all_sheets.items():
    data_rows = []
    # 从第 4 行（索引 3）开始扫描
    for i in range(3, len(sheet_data)):
        row = sheet_data.iloc[i]
        row_data = row.iloc[:12]  # 只取前 12 列
        # 检查前 3 个成绩列是否有至少一个有效正数 → 判定为有效数据行
        # （原代码仅检查第1列，可能遗漏该列为0但其他列有数据的行）
        try:
            vals = [float(row_data.iloc[j]) for j in [1, 2, 3]]
            if any(pd.notna(v) and v > 0 for v in vals):
                data_rows.append(row_data)
        except (ValueError, TypeError):
            # 非数值（如 'XXX' 占位符）跳过
            pass

    if not data_rows:
        continue

    df_sheet = pd.DataFrame(data_rows).reset_index(drop=True)
    df_sheet = df_sheet.rename(columns=COL_NAMES)
    # 强制转数值，无法转换的置 NaN
    for col in df_sheet.columns:
        df_sheet[col] = pd.to_numeric(df_sheet[col], errors='coerce')
    # 删除缺失值过多的行（至少 6 个非空值才保留）
    df_sheet = df_sheet.dropna(thresh=6)
    # 标记来源班级
    df_sheet['来源表单'] = sheet_name
    all_data.append(df_sheet)

# 合并所有表单
df = pd.concat(all_data, ignore_index=True)

# 删除全零行（Excel 中可能的分隔行）
zero_cols = ['线上平时成绩', '线上章测试', '线上期末考']
df = df[(df[zero_cols] != 0).any(axis=1)]

print(f"合并后总数据量: {len(df)} 条")
print(f"各表单分布:\n{df['来源表单'].value_counts().sort_index().to_string()}")

# ================================================================
# 第2步：划分训练/测试集（在预处理之前！防止数据泄露）
# ================================================================
# 关键：缺失值填充的中位数、异常值截尾的 IQR 边界等统计量
# 只能从训练集计算，然后应用到测试集。
# 如果先用全量数据计算再划分，测试集信息会通过这些统计量
# "泄露"到训练过程中，导致评估指标偏乐观。
# ================================================================
print("\n" + "=" * 70)
print("【第2步】划分训练/测试集（8:2）")
print("=" * 70)

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

print(f"训练集: {len(df_train)} 条")
print(f"测试集: {len(df_test)} 条")

# 定义后续分析会用到的特征列和目标列
feature_cols_raw = ['线上平时成绩', '线上章测试', '线上期末考',
                    '线下互动', '线下期末考']
target_col = '总成绩(线上+线下)'  # 模型预测的目标

# ================================================================
# 第3步：缺失值识别与填充（仅用训练集中位数）
# ================================================================
# 策略：从训练集计算中位数，同时应用到训练集和测试集。
# 中位数相比均值对异常值更鲁棒，适合成绩数据。
# ================================================================
print("\n" + "=" * 70)
print("【第3步】缺失值识别与填充（仅用训练集中位数）")
print("=" * 70)

# 打印训练集缺失值统计
print("\n训练集缺失值统计:")
print(df_train.isnull().sum().to_string())

# 逐列检查：用训练集中位数填充训练集和测试集
numeric_cols = df_train.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    if df_train[col].isnull().sum() > 0 or df_test[col].isnull().sum() > 0:
        median_val = df_train[col].median()  # 仅从训练集计算
        n_train_miss = df_train[col].isnull().sum()
        n_test_miss = df_test[col].isnull().sum()
        df_train[col] = df_train[col].fillna(median_val)
        df_test[col] = df_test[col].fillna(median_val)
        print(f"  -> {col} 训练集中位数={median_val:.2f}，"
              f"填充训练集{n_train_miss}个 + 测试集{n_test_miss}个缺失值")

total_miss = df_train.isnull().sum().sum() + df_test.isnull().sum().sum()
print(f"\n填充后缺失值总数: {total_miss}")

# ================================================================
# 第4步：异常值检测（IQR 方法，仅用训练集参数）
# ================================================================
# 使用经典的 IQR（四分位距）方法检测异常值：
#   下界 = Q1 - 1.5 × IQR
#   上界 = Q3 + 1.5 × IQR
# 超出上下界的值视为异常值，做 Winsorize 截尾（替换为边界值）。
# Q1/Q3 仅从训练集计算，然后应用到两个集合。
# 注意：若 IQR = 0（取值高度集中如「线上章测试」几乎全员满分），跳过检测。
# ================================================================
print("\n" + "=" * 70)
print("【第4步】异常值检测（IQR 方法，仅用训练集参数）")
print("=" * 70)

outlier_report = []
for col in feature_cols_raw:
    # 仅从训练集计算 Q1、Q3、IQR
    Q1 = df_train[col].quantile(0.25)
    Q3 = df_train[col].quantile(0.75)
    IQR = Q3 - Q1

    # 跳过 IQR=0 的特征（如线上章测试几乎全员满分）
    if IQR == 0:
        outlier_report.append({
            '特征': col,
            '下界': '—',
            '上界': '—',
            '异常值数': 0,
            '占比': '0%（IQR=0，跳过）'
        })
        print(f"\n  注意: {col} 的 IQR=0（取值高度集中），跳过异常值检测")
        continue

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # 统计训练集异常值
    outliers_train = df_train[(df_train[col] < lower) | (df_train[col] > upper)]
    # 统计测试集异常值
    outliers_test = df_test[(df_test[col] < lower) | (df_test[col] > upper)]

    outlier_report.append({
        '特征': col,
        '下界': round(lower, 2),
        '上界': round(upper, 2),
        '异常值数': len(outliers_train),
        '占比': f"{len(outliers_train)/len(df_train)*100:.1f}%"
    })

    # Winsorize 截尾：用训练集的边界同时处理训练集和测试集
    df_train[col] = df_train[col].clip(lower=lower, upper=upper)
    df_test[col] = df_test[col].clip(lower=lower, upper=upper)

outlier_df = pd.DataFrame(outlier_report)
print(outlier_df.to_string(index=False))

# 同样对目标变量做异常值检测（仅用训练集参数）
Q1y = df_train[target_col].quantile(0.25)
Q3y = df_train[target_col].quantile(0.75)
IQRy = Q3y - Q1y
if IQRy > 0:
    lower_y, upper_y = Q1y - 1.5 * IQRy, Q3y + 1.5 * IQRy
    y_outliers_train = df_train[(df_train[target_col] < lower_y) | (df_train[target_col] > upper_y)]
    y_outliers_test = df_test[(df_test[target_col] < lower_y) | (df_test[target_col] > upper_y)]
    print(f"\n目标变量（{target_col}）异常值检测:")
    print(f"  正常范围: [{lower_y:.1f}, {upper_y:.1f}]")
    print(f"  训练集异常值: {len(y_outliers_train)}（{len(y_outliers_train)/len(df_train)*100:.1f}%）")
    print(f"  测试集异常值: {len(y_outliers_test)}（{len(y_outliers_test)/len(df_test)*100:.1f}%）")
    df_train[target_col] = df_train[target_col].clip(lower=lower_y, upper=upper_y)
    df_test[target_col] = df_test[target_col].clip(lower=lower_y, upper=upper_y)
    print("  已进行 Winsorize 截尾处理")

# ================================================================
# 第5步：类别特征独热编码（One-Hot Encoding）
# ================================================================
# 「来源表单」列有 7 个类别（Sheet1~Sheet7），是类别特征。
# 使用 pd.get_dummies 转为 6 个二进制列（drop_first=True 避免多重共线性），
# Sheet1 作为参照组（全 0），其他班级各自对应一列 1。
# ================================================================
print("\n" + "=" * 70)
print("【第5步】类别特征独热编码（One-Hot Encoding）")
print("=" * 70)

# 合并后统一编码（确保训练集和测试集拥有相同的独热编码列）
df_combined = pd.concat([df_train, df_test], ignore_index=True)

print(f"类别特征「来源表单」取值: {df_combined['来源表单'].unique()}")
print(f"类别数: {df_combined['来源表单'].nunique()}")

# 独热编码：7 个类别 → 6 个二进制特征
ohe = pd.get_dummies(df_combined['来源表单'], prefix='班级', drop_first=True).astype(int)
df_combined = pd.concat([df_combined, ohe], axis=1)
df_combined = df_combined.drop(columns=['来源表单'])  # 原始类别列已转为数值，删除

print(f"\n独热编码新增 {len(ohe.columns)} 个特征:")
print(list(ohe.columns))
print(f"当前总列数: {df_combined.shape[1]}")

# 重新拆分为训练集和测试集
df_train = df_combined.iloc[:len(df_train)].copy()
df_test = df_combined.iloc[len(df_train):].copy().reset_index(drop=True)

# ================================================================
# 第6步：相关性分析与特征选择（仅用训练集计算相关系数）
# ================================================================
# 候选特征由两部分组成：
#   1. 5 个原始得分特征（线上平时成绩、线上章测试、线上期末考、
#      线下互动、线下期末考）
#   2. 6 个班级独热编码特征（班级_Sheet2~班级_Sheet7）
# 剔除计算列（线上总成绩、线下总成绩等）避免数据泄露。
#
# 筛选策略：
#   - 原始得分：保留 |Pearson r| ≥ 0.3 的特征（中等以上相关性）
#   - 班级特征：全部保留（类别特征不适合用线性相关系数筛选）
# ================================================================
print("\n" + "=" * 70)
print("【第6步】相关性分析（Pearson）与特征选择（仅用训练集）")
print("=" * 70)

# 构造候选特征列表
candidate_features = (
    ['线上平时成绩', '线上章测试', '线上期末考',
     '线下互动', '线下期末考']  # 5个原始得分
    + list(ohe.columns)                     # 6个班级独热编码
)

# 逐个计算与目标变量的 Pearson 相关系数（仅用训练集）
corr_list = []
for col in candidate_features:
    r = df_train[col].corr(df_train[target_col])
    if pd.notna(r):
        corr_list.append({'特征': col, 'Pearson相关系数': round(r, 4)})
    else:
        corr_list.append({'特征': col, 'Pearson相关系数': 'NaN（无波动）'})

corr_df = pd.DataFrame(corr_list).sort_values('Pearson相关系数', ascending=False)
print("\n候选特征与「总成绩(线上+线下)」的 Pearson 相关系数（从高到低）:")
print(corr_df.to_string(index=False))

# 执行特征筛选
selected_features = []
for _, row in corr_df.iterrows():
    r = row['Pearson相关系数']
    if isinstance(r, str):  # NaN 特征无法计算相关系数，跳过
        continue
    col_name = row['特征']
    # 班级特征全部保留，不按相关系数筛选
    if col_name.startswith('班级_'):
        selected_features.append(col_name)
    # 原始得分按 |r| >= 0.3 筛选（中等以上相关性）
    elif abs(r) >= 0.3:
        selected_features.append(col_name)

print(f"\n>>> 筛选结果：|r| >= 0.3，共选出 {len(selected_features)} 个特征")
print(f"    {selected_features}")

# ================================================================
# 第7步：构建最终数据集并保存
# ================================================================
# 合并训练集和测试集，保存清洗后的完整数据。
# train_models.py 会自行从 cleaned_data.csv 重新划分。
# ================================================================
print("\n" + "=" * 70)
print("【第7步】构建最终数据集并保存")
print("=" * 70)

# 合并处理后的训练集和测试集
df_final = pd.concat([df_train, df_test], ignore_index=True)

X = df_final[selected_features].copy()
y = df_final[target_col].copy()

assert X.isnull().sum().sum() == 0, "特征矩阵仍有缺失值！"
print(f"最终特征矩阵 X: {X.shape}")
print(f"目标变量 y: {y.shape}")
print(f"总成绩范围: {y.min():.1f} ~ {y.max():.1f}，均值: {y.mean():.1f}")

# 保存清洗后完整数据（供 train_models.py 和 app.py 使用）
df_final.to_csv(os.path.join(BASE_DIR, 'cleaned_data.csv'),
                index=False, encoding='utf-8-sig')

# 保存所选特征名列表（供 train_models.py 和 app.py 读取）
with open(os.path.join(BASE_DIR, 'selected_features.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(selected_features))

print(f"\n{'=' * 70}")
print("【最终输出文件】")
print(f"{'=' * 70}")
print(f"  cleaned_data.csv       清洗后完整数据（含独热编码）")
print(f"  selected_features.txt  所选特征名列表")

print(f"\n{'=' * 70}")
print("【数据摘要】")
print(f"{'=' * 70}")
print(f"  样本总数:  {len(df_final)}")
print(f"  特征总数:  {len(selected_features)}")
print(f"  训练集:    {len(df_train)} 条")
print(f"  测试集:    {len(df_test)} 条")
print(f"  所选特征:  {selected_features}")
print(f"\n  数据统计:")
print(X.describe().round(2).to_string())
