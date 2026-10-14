import csv

input_path = r'C:\Users\千里走单骑\Desktop\大论文复现\20260325曾锦鹤大论文与关键数据\第三次轨迹数据\1845-1预处理.csv'
output_path = r'D:\1TEST20260510\指标\1845.csv'

with open(input_path, 'r', encoding='utf-8') as f_in:
    reader = csv.DictReader(f_in)
    fieldnames = reader.fieldnames
    rows_kept = []
    for row in reader:
        ts = float(row['时间戳'])
        if ts <= 500:
            rows_kept.append(row)

with open(output_path, 'w', encoding='utf-8', newline='') as f_out:
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows_kept)

print(f'保留 {len(rows_kept)} 行')
