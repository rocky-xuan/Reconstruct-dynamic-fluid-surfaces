import os
def run(filename):
    Dataset_paths = os.listdir(filename)
    for Dataset_path in Dataset_paths:
        choose_Dataset_path = os.path.join(filename, Dataset_path) #当前选择的Dataset绝对路径
        depth_gt_folder = os.path.join(choose_Dataset_path, 'GT', 'depth_map')
        img_gt_folder = os.path.join(choose_Dataset_path, 'GT', 'render')
        reference_gt_folder = os.path.join(choose_Dataset_path, 'GT', 'reference')
        img_calibration_path = sorted(os.listdir(reference_gt_folder))
        run_serial = sorted(os.listdir(depth_gt_folder))
        for i in range(len(run_serial)):
            print("当前的run_serial为：" + run_serial[i]) #run_serial[i] 对应 当前测试的具体编号数目
            img_calibration_run_path = img_calibration_path[i]
            depth_gt_run_folder = os.path.join(depth_gt_folder, run_serial[i])
            img_gt_run_folder = os.path.join(img_gt_folder, run_serial[i])
            # print('depth_gt_run_file')
            depth_gt_run_file = sorted(os.listdir(depth_gt_run_folder))  # 对应depth_gt 文件名列表
            # print(depth_gt_run_file)
            img_gt_run_file = sorted(os.listdir(img_gt_run_folder))  # 对应img_gt 文件名列表
            # print('img_gt_run_file')
            # print(img_gt_run_file)
            for j in range(len(depth_gt_run_file)):
                # print("当前运行文件为：" + depth_gt_run_file[j])
                depth_gt_running_file = depth_gt_run_file[j] #对应当前运行的depth名(str)
                img_gt_running_file = img_gt_run_file[j] #对应当前运行的图像名(str)
                print(os.path.join(depth_gt_run_folder,img_gt_running_file))





if __name__ == '__main__':
    # filename = 'Dataset'
    # Dataset_path = os.listdir(filename)
    #
    # #需要遍历列表，选择当前数据集
    # choose_Dataset_path = os.path.join(filename,Dataset_path[0])
    # print("choose_Dataset_path")
    # print(choose_Dataset_path)
    #
    # depth_gt_folder = os.path.join(choose_Dataset_path,'GT','depth_map')
    # img_gt_folder = os.path.join(choose_Dataset_path,'GT','render')
    #
    # print("run_serial")
    # run_serial = sorted(os.listdir(depth_gt_folder))
    # print(run_serial)
    #
    # # 需要便利列表选择 run_serial
    # reference_gt_folder = os.path.join(choose_Dataset_path,'GT','reference')
    # print("img_calibration_path")
    # img_calibration_path = sorted(os.listdir(reference_gt_folder))
    # print(img_calibration_path)
    #
    # depth_gt_run_folder = os.path.join(depth_gt_folder,run_serial[0])
    # img_gt_run_folder = os.path.join(img_gt_folder,run_serial[0])
    # print('depth_gt_run_file')
    # depth_gt_run_file = sorted(os.listdir(depth_gt_run_folder)) #对应depth_gt 文件名列表
    # print(depth_gt_run_file)
    # img_gt_run_file = sorted(os.listdir(img_gt_run_folder))  #对应img_gt 文件名列表
    # print('img_gt_run_file')
    # print(img_gt_run_file)
    #
    # #此时需要便利当前运行的一张图像和depth
    run(filename='../Dataset')



