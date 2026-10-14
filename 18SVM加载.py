import numpy as np
import joblib  # 必须导入，用来加载模型
def load_svm_model(model_path, scaler_path):
    """
    加载保存好的SVM模型和标准化器
    """
    svm_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return svm_model, scaler


# ==============================================
# 主程序：加载模型 + 预测新数据
# ==============================================
if __name__ == "__main__":
    # 模型路径（你自己的真实路径）
    model_path = r"C:\Users\千里走单骑\Desktop\大论文复现\模型保存\SVM最优模型.pkl"
    scaler_path = r"C:\Users\千里走单骑\Desktop\大论文复现\模型保存\特征标准化器.pkl"

    # 加载模型（现在不会报错了）
    loaded_svm, loaded_scaler = load_svm_model(model_path, scaler_path)
    print("模型与标准化器加载成功！")

    # 示例：新数据预测（7维特征）
    new_data = np.array([[17.6597, 0.0773, 33.6914, 1.9094, -0.004811478, -0.000551887]])

    # 数据标准化 + 预测
    new_data_scaled = loaded_scaler.transform(new_data)
    predicted_class = loaded_svm.predict(new_data_scaled)
    predicted_prob = loaded_svm.predict_proba(new_data_scaled)

    # 输出结果
    print(f"\n新数据预测结果：")
    print(f"预测驾驶风格类别：{predicted_class[0]}（1=激进自信，0=谨慎效率，2=保守跟随）")
    print(f"各类别概率：{np.round(predicted_prob[0], 4)}")