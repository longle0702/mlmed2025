import kagglehub

path = kagglehub.dataset_download("hoinhi/hc18-split", force_download=True)
path2 = r'/home/long/longdata/mlmed/prac2'
print("Path to dataset files:", path)

import shutil
shutil.move(path, path2)
print('ok')