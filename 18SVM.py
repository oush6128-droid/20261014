
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib  # 用于保存和加载模型
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc)
from sklearn.decomposition import PCA
import os  # 用于创建保存目录

# ===================== 2. 加载你的Excel数据集（已按你的路径修改！） =====================
# 你的本地Excel文件路径（直接复制你的路径，注意转义符）
data_path = r"C:\Users\千里走单骑\Desktop\大论文复现\数据复现输出目录\聚类成分提取\SVM\SVM训练数据_保留(0-2类各972条).xlsx"

# 读取Excel文件（默认读取第一个工作表，若你的数据在其他sheet，需加 sheet_name="工作表名"）
df = pd.read_excel(data_path)

# 关键：确认特征列和标签列（请根据你的Excel表格调整列名！）
# 假设你的Excel中：前7列是PCA降维后的特征（列名如feature1~feature7），最后一列是标签（列名如label）
# 若列名不同，需替换成你的实际列名，例如：X = df[['速度均值', '加速度标准差', ...]].values
X = df.iloc[:, 1:7].values   # 第2~7列（索引1到6）作为特征
y = df.iloc[:, 7].values     # 第8列（索引7）作为标签
# 7:3划分训练集/测试集（分层抽样，保证每类样本比例）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 特征标准化（SVM模型必须步骤，避免量纲影响）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 训练集拟合并标准化
X_test_scaled = scaler.transform(X_test)        # 测试集用训练集的标准化参数

# ===================== 3. SVM模型训练（网格搜索调参） =====================
# 论文常用参数搜索范围（可根据需要调整）
param_grid_svm = {
    'C': [1, 10, 100, 1000],        # 惩罚系数（控制模型复杂度）
    'kernel': ['linear', 'rbf'],     # 核函数（线性/径向基）
    'gamma': ['scale', 'auto']       # RBF核的gamma参数（默认scale即可）
}

# 网格搜索 + 5折交叉验证（找最优参数）
grid_svm = GridSearchCV(
    estimator=SVC(probability=True, random_state=42),  # probability=True：保留概率输出（用于ROC）
    param_grid=param_grid_svm,
    cv=5,                            # 5折交叉验证
    scoring='accuracy',              # 以准确率为评价指标
    n_jobs=None,                       # 调用所有CPU核心加速
    verbose=1                        # 显示训练过程
)

# 训练模型
print("="*50)
print("开始SVM模型训练（网格搜索调参）...")
grid_svm.fit(X_train_scaled, y_train)

# 输出最优结果
best_svm = grid_svm.best_estimator_  # 最优模型
print(f"\nSVM最优参数：{grid_svm.best_params_}")
print(f"SVM交叉验证最优准确率：{round(grid_svm.best_score_, 4)}")
print("="*50)

# ===================== 4. 模型评估（输出论文所需指标） =====================
def evaluate_model(model, X_test, y_test, model_name):
    """评估模型性能，输出准确率、精确率等指标"""
    y_pred = model.predict(X_test)
    # 计算多分类评价指标（weighted：按样本数加权，适合不均衡样本）
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # 输出结果
    print(f"\n【{model_name} 测试集评估结果】")
    print(f"准确率（Accuracy）: {acc:.4f}")
    print(f"精确率（Precision）: {precision:.4f}")
    print(f"召回率（Recall）: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print("\n分类报告：")
    print(classification_report(y_test, y_pred, target_names=["激进自信型", "谨慎效率型", "保守跟随型"], digits=4))
    return y_pred

# 评估SVM模型（核心）
y_pred_svm = evaluate_model(best_svm, X_test_scaled, y_test, "SVM最优模型")

# （可选）评估对比模型（KNN/随机森林，用于论文对比）
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
evaluate_model(knn, X_test_scaled, y_test, "KNN模型")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
evaluate_model(rf, X_test_scaled, y_test, "随机森林模型")

# ===================== 5. 绘制论文图表（混淆矩阵+ROC曲线） =====================
# 5.1 绘制混淆矩阵
def plot_confusion_matrix(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["激进自信型", "谨慎效率型", "保守跟随型"]  # 和你的类别对应
    )
    disp.plot(cmap=plt.cm.Blues, values_format='d')  # values_format='d'：显示整数
    plt.title(f'{model_name} 混淆矩阵', fontsize=12)
    plt.xticks(rotation=0)  # 避免标签重叠
    plt.tight_layout()
    # 保存图片（路径可修改）
    plt.savefig(f"C:/Users/千里走单骑/Desktop/大论文复现/SVM混淆矩阵.png", dpi=300)
    plt.show()

# 5.2 绘制多分类ROC曲线
# ===================== 正确的多分类 ROC 曲线 =====================
def plot_multiclass_roc(model, X_test, y_test):
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    import numpy as np
    import matplotlib.pyplot as plt

    # 你的标签是 0 1 2，这里写死正确类别
    classes = [0, 1, 2]
    n_classes = len(classes)

    # 标签二值化
    y_test_bin = label_binarize(y_test, classes=classes)
    y_score = model.predict_proba(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(7, 6))
    colors = ['red', 'blue', 'green']
    labels = ["激进自信型", "谨慎效率型", "保守跟随型"]

    for i, color in enumerate(colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{labels[i]} (AUC = {roc_auc[i]:.3f})')

    plt.plot([0, 0], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (FPR)')
    plt.ylabel('真正率 (TPR)')
    plt.title('SVM 驾驶风格分类 ROC 曲线')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("SVM_ROC曲线_最终版.png", dpi=300)
    plt.show()

# 生成图表
plot_confusion_matrix(best_svm, X_test_scaled, y_test, "SVM最优模型")
plot_multiclass_roc(best_svm, X_test_scaled, y_test)

# ===================== 6. 保存训练好的模型（核心新增功能） =====================
# 创建模型保存目录（避免路径不存在报错）
save_dir = r"C:\Users\千里走单骑\Desktop\大论文复现\模型保存"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  # 若目录不存在，自动创建

# 保存3个关键文件（后续调用需同时加载）
1. # 保存SVM最优模型
joblib.dump(best_svm, os.path.join(save_dir, "SVM最优模型.pkl"))

2. # 保存特征标准化器（后续预测新数据需用同一标准化器）
joblib.dump(scaler, os.path.join(save_dir, "特征标准化器.pkl"))

3. # 保存网格搜索结果（可选，方便后续查看参数）
joblib.dump(grid_svm, os.path.join(save_dir, "SVM网格搜索结果.pkl"))

print(f"\n模型已保存至：{save_dir}")
print("保存文件包括：SVM最优模型.pkl、特征标准化器.pkl、SVM网格搜索结果.pkl")

# ===================== 7. 调用已保存的模型（后续使用时直接运行这部分） =====================
def load_svm_model(model_path, scaler_path):
    """加载已保存的SVM模型和标准化器"""
    # 加载模型
    loaded_svm = joblib.load(model_path)
    # 加载标准化器
    loaded_scaler = joblib.load(scaler_path)
    print("\n模型加载成功！")
    return loaded_svm, loaded_scaler

