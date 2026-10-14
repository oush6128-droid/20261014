import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

# ================= 路径配置 =================
input_path = r'C:\Users\千里走单骑\Desktop\大论文复现\数据复现输出目录\聚类成分提取\降维后数据.xlsx'
output_path = r'C:\Users\千里走单骑\Desktop\大论文复现\数据复现输出目录\聚类成分提取\聚类效果量化结果4.xlsx'

# 1. 读取数据（已经是标准化后的 PCA 得分）
df = pd.read_excel(input_path)
X = df[['PC1得分', 'PC2得分', 'PC3得分', 'PC4得分', 'PC5得分', 'PC6得分', 'PC7得分']].values

# 2. 定义簇数范围
k_range = range(2, 11)
results = []

print("计算中...")

for k in k_range:
    # ---------- K-means++ ----------
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    labels_kmeans = kmeans.fit_predict(X)
    sil_kmeans = silhouette_score(X, labels_kmeans)
    ch_kmeans = calinski_harabasz_score(X, labels_kmeans)

    # ---------- 谱聚类 (affinity='nearest_neighbors') ----------
    # 为了加快速度，可以手动指定 n_neighbors，这里使用默认值 10
    spectral = SpectralClustering(n_clusters=k, affinity='nearest_neighbors',
                                  random_state=42, n_neighbors=10)
    labels_spectral = spectral.fit_predict(X)
    sil_spectral = silhouette_score(X, labels_spectral)
    ch_spectral = calinski_harabasz_score(X, labels_spectral)

    results.append([k, sil_kmeans, ch_kmeans, sil_spectral, ch_spectral])
    print(f"k = {k} 完成")

# 3. 保存结果
result_df = pd.DataFrame(results, columns=['k', 'K-means++_Silhouette', 'K-means++_CH',
                                           'Spectral_Silhouette', 'Spectral_CH'])
result_df.to_excel(output_path, index=False)

print("\n结果已保存至:", output_path)
print("\n===== 结果预览 =====")
print(result_df.round(4))