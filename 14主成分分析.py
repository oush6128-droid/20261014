import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# ===================== 【已配置你的路径】输入输出文件 =====================
input_file = r"C:\Users\千里走单骑\Desktop\大论文复现\数据复现输出目录\聚类成分提取\标准化汇总原.xlsx"
output_file = r"C:\Users\千里走单骑\Desktop\大论文复现\数据复现输出目录\聚类成分提取\主成分分析结果1.xlsx"

# ===================== 1. 读取数据与预处理 =====================
df = pd.read_excel(input_file)

# 提取特征列（自动排除"车辆ID"和"第几次"两列）
feature_cols = [col for col in df.columns if col not in ["车辆ID", "第几次"]]
X = df[feature_cols].values  # 特征矩阵

# ===================== 2. 执行PCA分析 =====================
pca = PCA(n_components=None)  # 保留所有主成分
pca_result = pca.fit_transform(X)  # 主成分得分矩阵

# ===================== 3. 计算特征值、贡献率、累积贡献率 =====================
eigenvalues = pca.explained_variance_  # 特征值
explain_ratio = pca.explained_variance_ratio_  # 方差贡献率
cumulative_ratio = np.cumsum(explain_ratio)  # 累积方差贡献率

# 生成特征值与贡献率汇总表
pca_summary = pd.DataFrame({
    "主成分": [f"PC{i+1}" for i in range(len(eigenvalues))],
    "特征值(方差)": eigenvalues.round(6),
    "贡献率(%)": (explain_ratio * 100).round(4),
    "累积贡献率(%)": (cumulative_ratio * 100).round(4)
})

# ===================== 4. 计算前7个主成分的载荷矩阵 =====================
# 载荷矩阵 = 特征向量 × √特征值
eigen_vectors = pca.components_.T  # 特征向量（列为主成分）
loadings = eigen_vectors * np.sqrt(eigenvalues)[:, np.newaxis]

# 提取前7个主成分的载荷
top7_loadings = pd.DataFrame(
    loadings[:, :7].round(6),
    index=feature_cols,
    columns=[f"PC{i+1}" for i in range(7)]
)

# ===================== 5. 计算所有样本前7个主成分的得分 =====================
top7_scores = pd.DataFrame(
    pca_result[:, :7].round(6),
    columns=[f"PC{i+1}得分" for i in range(7)]
)

# 关联车辆ID和测试次数信息
top7_scores = pd.concat([
    df[["车辆ID", "第几次"]].reset_index(drop=True),
    top7_scores
], axis=1)

# ===================== 6. 结果保存到Excel =====================
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    pca_summary.to_excel(writer, sheet_name="1_特征值与贡献率", index=False)
    top7_loadings.to_excel(writer, sheet_name="2_前7主成分载荷矩阵")
    top7_scores.to_excel(writer, sheet_name="3_前7主成分得分", index=False)

# ===================== 7. 控制台输出提示 =====================
print("=" * 60)
print("PCA分析完成！")
print(f"结果已保存至：{output_file}")
print("\n包含3个工作表：")
print("  1. 1_特征值与贡献率（所有主成分的特征值、贡献率、累积贡献率）")
print("  2. 2_前7主成分载荷矩阵（16个特征在PC1-PC7上的载荷）")
print("  3. 3_前7主成分得分（4255个样本的PC1-PC7得分，含车辆ID和测试次数）")
print("=" * 60)