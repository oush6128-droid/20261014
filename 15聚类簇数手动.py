import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings

warnings.filterwarnings('ignore')

# --- 1. 配置参数 ---
# 请直接使用你提供的路径
input_path = r'C:\Users\千里走单骑\Desktop\大论文复现\数据复现输出目录\聚类成分提取\降维后数据.xlsx'
output_path = r'C:\Users\千里走单骑\Desktop\大论文复现\数据复现输出目录\聚类成分提取\聚类效果量化结果5.xlsx'

# --- 2. 数据加载 (假设数据已标准化) ---
print("正在读取数据...")
try:
    # 读取Excel
    df = pd.read_excel(input_path, sheet_name='3_前7主成分得分')  # 请确认Sheet名称是否完全一致

    # 提取特征列 (假设PC1到PC7是第3列到第9列)
    # 如果列名是 'PC1得分', 'PC2得分'... 请使用 iloc 或 loc 精确选择
    feature_columns = df.columns[2:9]  # 选取第3列(C)到第9列(I)
    X = df[feature_columns].values

    print(f"数据加载成功！样本数: {X.shape[0]}, 特征数: {X.shape[1]}")

except Exception as e:
    print(f"数据读取失败: {e}")
    print("请检查文件路径或Sheet名称是否正确。")
    exit()

# --- 3. 参数设置 ---
# 定义聚类簇数范围 (2-9类)
n_clusters_range = range(2, 10)

# 初始化存储列表
results = []

# --- 4. 核心循环：算法对比 ---
print("\n开始聚类评估计算...")

for n_clusters in n_clusters_range:
    # --- 模型 1: K-means++ ---
    # n_init=10 是默认值，保证了K-means++的稳定性
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)

    # 计算 K-means 指标
    kmeans_sil = silhouette_score(X, kmeans_labels)
    kmeans_ch = calinski_harabasz_score(X, kmeans_labels)

    # --- 模型 2: 谱聚类 (Spectral Clustering) ---
    # 注意：谱聚类计算复杂度高，对于大样本(>1000)可能较慢。
    # affinity='nearest_neighbors' 比 'rbf' 更适合高维数据且更稳定
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='nearest_neighbors',
        n_neighbors=min(10, n_clusters),  # 防止邻居数大于样本数
        random_state=42,
        assign_labels='kmeans'  # 使用kmeans优化标签分配
    )
    spectral_labels = spectral.fit_predict(X)

    # 计算 谱聚类 指标
    spectral_sil = silhouette_score(X, spectral_labels)
    spectral_ch = calinski_harabasz_score(X, spectral_labels)

    # 存储结果
    results.append({
        'K': n_clusters,
        'Kmeans_Silhouette': kmeans_sil,
        'Kmeans_CH': kmeans_ch,
        'Spectral_Silhouette': spectral_sil,
        'Spectral_CH': spectral_ch
    })

# --- 5. 结果整理与导出 ---
results_df = pd.DataFrame(results)
results_df.set_index('K', inplace=True)

# 导出详细结果到Excel
with pd.ExcelWriter(output_path) as writer:
    results_df.to_excel(writer, sheet_name='量化指标结果')
    # 也可以将原始数据写入另一个Sheet，方便查阅
    df.to_excel(writer, sheet_name='原始数据', index=False)

print(f"\n结果已保存至: {output_path}")

# --- 6. 可视化分析 ---
plt.figure(figsize=(16, 6))

# --- 子图 1: 轮廓系数 ---
plt.subplot(1, 2, 1)
plt.plot(results_df.index, results_df['Kmeans_Silhouette'], 'o-', color='#1f77b4', label='K-means++', linewidth=2,
         markersize=8)
plt.plot(results_df.index, results_df['Spectral_Silhouette'], 's-', color='#ff7f0e', label='Spectral Clustering',
         linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.title('Silhouette Score Comparison', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# --- 子图 2: CH 分数 ---
plt.subplot(1, 2, 2)
plt.plot(results_df.index, results_df['Kmeans_CH'], 'o-', color='#1f77b4', label='K-means++', linewidth=2, markersize=8)
plt.plot(results_df.index, results_df['Spectral_CH'], 's-', color='#ff7f0e', label='Spectral Clustering', linewidth=2,
         markersize=8)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Calinski-Harabasz Index', fontsize=12)
plt.title('CH Index Comparison', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# --- 7. 最终决策与打印 ---
print("\n" + "=" * 60)
print("最优聚类方案分析报告")
print("=" * 60)

# 寻找最优解 (基于轮廓系数，因为它对簇的分离度衡量更直观)
best_k_kmeans = results_df['Kmeans_Silhouette'].idxmax()
best_sil_kmeans = results_df['Kmeans_Silhouette'].max()

best_k_spectral = results_df['Spectral_Silhouette'].idxmax()
best_sil_spectral = results_df['Spectral_Silhouette'].max()

print(f"🔍 K-means++ 最佳簇数: K = {best_k_kmeans} (轮廓系数: {best_sil_kmeans:.4f})")
print(f"🔍 谱聚类 最佳簇数: K = {best_k_spectral} (轮廓系数: {best_sil_spectral:.4f})")

if best_sil_kmeans > best_sil_spectral:
    print(f"\n结论: 推荐使用 K-means++ 算法，设定聚类簇数 K = {best_k_kmeans}。")
else:
    print(f"结论: 推荐使用 Spectral Clustering (谱聚类) 算法，设定聚类簇数 K = {best_k_spectral}。")