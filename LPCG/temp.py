import os

txt_filepath = "./high_acc/label_2"
txt_list = os.listdir(txt_filepath)
imgs_list = [val for val in txt_list if val.find(".txt") > -1]
imgs_list = sorted(imgs_list, key=lambda x: int(x[:-4]))
for sub_path in imgs_list:
    if sub_path.find("txt") > -1 and int(sub_path[:-4]) >= 33530:
        txt_path = os.path.join(txt_filepath, sub_path)
        os.remove(txt_path)