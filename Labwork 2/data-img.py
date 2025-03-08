import os
import shutil

train_mask = r'/home/long/longdata/mlmed/prac2/1/training_mask'
train_set = r'/home/long/longdata/mlmed/prac2/1/training_set'
val_mask = r'/home/long/longdata/mlmed/prac2/1/val_mask'
val_set = r'/home/long/longdata/mlmed/prac2/1/val_set'

total_mask = r'/home/long/longdata/mlmed/prac2/1/total_mask'
total_set = r'/home/long/longdata/mlmed/prac2/1/total_set'
os.makedirs(total_mask, exist_ok=True)
os.makedirs(total_set, exist_ok=True)

def i_like_to_move_it_move_it(src, dst):
    for name in os.listdir(src):
        src_file = os.path.join(src, name)
        dst_file = os.path.join(dst, name)
        shutil.move(src_file, dst_file)

i_like_to_move_it_move_it(train_mask, total_mask)
i_like_to_move_it_move_it(val_mask, total_mask)

i_like_to_move_it_move_it(train_set, total_set)
i_like_to_move_it_move_it(val_set, total_set)
