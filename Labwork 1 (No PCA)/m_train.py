from m_prepare_data import M_Data
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data = M_Data()
m_train = data.m_train_data
m_test = data.m_test_data
for i in [100, 200, 300]:
    model = RandomForestClassifier(n_estimators=i, random_state=42)
    model.fit(m_train, data.m_train_label)
    m_pred = model.predict(m_test)
    def normalize_confusion_matrix(cm, norm='true'):
        if norm == 'true':
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        elif norm == 'pred':
            cm_normalized = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        elif norm == 'all':
            cm_normalized = cm.astype('float') / cm.sum()
        else:
            raise ValueError("Unknown normalization type. Use 'true', 'pred', or 'all'.")
        
        return cm_normalized

    print('='*100)
    print()
    print(f"Classification Report at Number Estimators = {i}")
    print(classification_report(data.m_test_label, m_pred))

    cm = confusion_matrix(data.m_test_label, m_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix at Number Estimators = {i}')
    plt.savefig(f'confusion_matrix_n_estimators_{i}.png')
    plt.show()

    sns.heatmap(normalize_confusion_matrix(cm, norm='true'), annot=True, fmt=".2f", cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Normalized Confusion Matrix at Number Estimators = {i}')
    plt.savefig(f'normalized_confusion_matrix_n_estimators_{i}.png')
    plt.show()