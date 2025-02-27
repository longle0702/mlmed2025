import matplotlib.pyplot as plt
import pandas as pd
from p_prepare_data import P_Data
import seaborn as sns

data = P_Data()
counts = pd.Series(data.p_label).value_counts().sort_index()

plt.figure(figsize=(10, 6))
sns.barplot(x=counts.index, y=counts.values, palette="viridis")

plt.xlabel("Labels")
plt.ylabel("Samples")
plt.title("Class Distribution in PTB Dataset")
plt.show()

counts2 = pd.Series(data.p_train_label).value_counts().sort_index()
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=counts2.index, y=counts2.values, palette="viridis")
for i, value in enumerate(counts2.values):
    ax.text(i, value + max(counts2.values)*0.01, f'{value}', ha='center', va='bottom')

plt.xlabel("Labels")
plt.ylabel("Samples")
plt.title("Class Distribution in PTB Train Dataset")
plt.tight_layout(pad=3.0)
plt.show()

plt.figure(figsize=(20, 7))
for i, (id, name) in enumerate(data.classes.items(), 1):
    idx = data.p_label[data.p_label == name].index[0]
    plt.subplot(1, 2, i)
    plt.plot(data.p_data.iloc[idx])
    plt.title(f'{name}')

plt.tight_layout(pad=3.0)
plt.show()