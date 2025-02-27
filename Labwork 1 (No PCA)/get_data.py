import kagglehub
import shutil
path = kagglehub.dataset_download("shayanfazeli/heartbeat")
print("Path to dataset files:", path)
path2=r'D:\USTH\Nam Ba\ml in med\Labwork 1'
shutil.move(path, path2)