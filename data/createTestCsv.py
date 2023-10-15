import csv
import os

with open('test.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    DataSets = "data"
    for i in range(0,3):
        target_dir_color_F = DataSets + "/color/"
        target_dir_depth_F = DataSets + "/depth/"
        target_dir_depth_B_SMPL = DataSets + "/smpldepth/"
        p = str(i) + ".png"
        target_path1 = target_dir_color_F + p
        target_path2 = target_dir_depth_F + p
        target_path3 = target_dir_depth_B_SMPL + p
        writer.writerow([target_path1, target_path2, target_path3])
