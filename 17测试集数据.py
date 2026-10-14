import pandas as pd
import os

INPUT_PATH = r"C:\Users\千里走单骑\Desktop\大论文复现\数据复现输出目录\聚类成分提取\SVM\SVM模型训练.xlsx"
SAMPLE_SIZE = 972  # 每类保留数量



def main():
    # 1. 读取数据
    df = pd.read_excel(INPUT_PATH)
    print(f"原始数据总量：{len(df)}")

    # 2. 只保留类别 1、2、3
    df_target = df[df['聚类类别'].isin([0, 1, 2])].copy()
    print(f"有效类别（0/1/2）总量：{len(df_target)}")
    print("类别分布：")
    print(df_target['聚类类别'].value_counts().sort_index())

    # 3. 每类抽 972 条
    keep_list = []
    drop_list = []

    for c in [0, 1, 2]:
        subset = df_target[df_target['聚类类别'] == c]
        if len(subset) > SAMPLE_SIZE:
            keep = subset.sample(n=SAMPLE_SIZE, random_state=42)
            drop = subset.drop(keep.index)
        else:
            keep = subset
            drop = pd.DataFrame(columns=df.columns)

        keep_list.append(keep)
        drop_list.append(drop)
        print(f"类别 {c} → 保留 {len(keep)} 条，剔除 {len(drop)} 条")

    # 合并
    df_keep = pd.concat(keep_list, ignore_index=True)
    df_drop = pd.concat(drop_list, ignore_index=True)

    # 4. 保存到同一目录
    out_dir = os.path.dirname(INPUT_PATH)
    keep_path = os.path.join(out_dir, "SVM训练数据_保留(0-2类各972条).xlsx")
    drop_path = os.path.join(out_dir, "SVM训练数据_剔除.xlsx")

    df_keep.to_excel(keep_path, index=False)
    df_drop.to_excel(drop_path, index=False)

    print("\n✅ 处理完成！")
    print(f"📄 保留文件：{keep_path}")
    print(f"📄 剔除文件：{drop_path}")
    print(f"最终保留：{len(df_keep)} 条（3类 × 972）")
    print(f"最终剔除：{len(df_drop)} 条")


if __name__ == "__main__":
    main()