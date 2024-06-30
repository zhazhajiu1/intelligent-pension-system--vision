import os


def rename_images_in_folder(person_dir):
    if not os.path.isdir(person_dir):
        print(f"{person_dir} 不是一个有效的目录")
        return

    # 获取人物姓名
    person_name = os.path.basename(person_dir)

    count = 1
    for img_file in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_file)
        if os.path.isfile(img_path):
            # 构建新的文件名
            new_img_name = f"{person_name}{count}.jpg"
            new_img_path = os.path.join(person_dir, new_img_name)
            # 重命名文件
            os.rename(img_path, new_img_path)
            count += 1


# 给定的文件夹路径
folder_to_rename = 'data/staff/qjx'

# 执行重命名操作
rename_images_in_folder(folder_to_rename)
