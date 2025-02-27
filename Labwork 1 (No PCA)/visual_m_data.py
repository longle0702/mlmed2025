import matplotlib.pyplot as plt
import pandas as pd
from m_prepare_data import M_Data
import seaborn as sns

data = M_Data()
counts = pd.Series(data.m_label).value_counts().sort_index()

plt.figure(figsize=(22, 6))
sns.barplot(x=counts.index, y=counts.values, palette="viridis")
plt.tight_layout(pad=3.0)

plt.xlabel("Labels")
plt.ylabel("Samples")
plt.title("Class Distribution in MIT-BIH Dataset")
plt.show()

counts2 = pd.Series(data.m_train_label).value_counts().sort_index()
plt.figure(figsize=(22, 6))
ax = sns.barplot(x=counts2.index, y=counts2.values, palette="viridis")
for i, value in enumerate(counts2.values):
    ax.text(i, value + max(counts2.values)*0.01, f'{value}', ha='center', va='bottom')

plt.xlabel("Labels")
plt.ylabel("Samples")
plt.title("Class Distribution in MIT-BIH Train Dataset")
plt.tight_layout(pad=3.0)
plt.show()

plt.figure(figsize=(20, 15))
for i, (id, name) in enumerate(data.classes.items(), 1):
    idx = data.m_label[data.m_label == name].index[0]
    plt.subplot(3, 2, i)
    plt.plot(data.m_data.iloc[idx])
    plt.title(f'{name}')

plt.tight_layout(pad=3.0)
plt.show()


