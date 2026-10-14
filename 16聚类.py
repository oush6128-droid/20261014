import pandas as pd
from sklearn.cluster import KMeans

# ===================== 路径（你的路径）=====================
input_path = r'C:\Users\千里走单骑\Desktop\大论文复现\数据复现输出目录\聚类成分提取\降维后数据.xlsx'
output_path = r'C:\Users\千里走单骑\Desktop\大论文复现\数据复现输出目录\聚类成分提取\降维后数据_3类聚类结果.xlsx'

# ===================== 读取数据 =====================
df = pd.read_excel(input_path)

# 聚类用的特征（PC1~PC7得分）
X = df[['PC1得分', 'PC2得分', 'PC3得分', 'PC4得分', 'PC5得分', 'PC6得分', 'PC7得分']].values

# ===================== K-means++ 聚类（固定3类）=====================
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
df['聚类类别'] = kmeans.fit_predict(X)  # 直接新增一列

# ===================== 导出Excel =====================
df.to_excel(output_path, index=False)

print("聚类完成！已生成 3 分类结果")
print("文件保存到：", output_path)
print("\n前10行类别预览：")
print(df[['PC1得分', 'PC2得分', '聚类类别']].head(10))