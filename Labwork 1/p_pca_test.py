from p_prepare_data import P_Data
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

num_components = 35
explained_variances = []
data = P_Data()
for i in range(1, num_components + 1):
    pca = PCA(n_components=i)
    pca.fit(data.p_train)
    explained_variances.append(sum(pca.explained_variance_ratio_))

plt.figure(figsize=(50, 8))
plt.plot(range(1, num_components + 1), explained_variances, marker='o', linestyle='-')

for i, var in enumerate(explained_variances, start=1):
    plt.text(i, var + 0.01, f"{var:.3f}", ha='center', color='black')

plt.title('Optimal Number PCA in Ptbdb Dataset')
plt.xlabel('Number of PCs')
plt.ylabel('Cumulative Explained Variance')
plt.xticks(range(1, num_components + 1))
plt.yticks([0.1 * i for i in range(11)])
plt.grid(True)
plt.show()