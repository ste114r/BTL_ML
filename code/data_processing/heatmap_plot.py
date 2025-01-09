import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Đọc file dữ liệu từ đường dẫn được cung cấp
data = pd.read_csv('pro_data/train_clean.csv')

# Tính ma trận tương quan
correlation_matrix = data.corr(numeric_only=True)

# Vẽ heatmap của ma trận tương quan
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
plt.title("Correlation Heatmap")
plt.savefig('pro_data/heatmap.png',dpi=300, bbox_inches='tight')
plt.show()

# Vẽ boxplot cho tất cả các cột số trong dữ liệu
plt.figure(figsize=(14, 8))
sns.boxplot(data=data.select_dtypes(include=["float64", "int64"]))
plt.title("Boxplot of Numeric Features", fontsize=16)
plt.xticks(rotation=45)
plt.savefig('pro_data/boxplot.png',dpi=300, bbox_inches='tight')
plt.show()