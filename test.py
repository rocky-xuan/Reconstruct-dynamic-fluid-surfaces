import torch
import torch.optim as optim
import torch.nn as nn
import time
import pandas as pd
import numpy as np
from Model.inr import INR
from Model.unet_model import UNet
import cv2
import os
from utils.loss import FlowLoss,DepthLoss
from torchsummary import summary
from Option.flow import args as flow_args
from Option.depth import args as depth_args
from utils.utils import img2tensor, save_result, warping, log, warp2d,save_depth_result,depth_to_normal,raytracing,make_numpy_grid,mse,\
    ssim,RMSE,Abs_Rel
def weight_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear,nn.BatchNorm2d)):
            nn.init.xavier_uniform_(m.weight)

def flow_generate(img_calibration,img_GT,args):
    """
    本函数用于大批量数据集测试时生成变形场
    :param img_calibration: 标定物
    :param img_GT: 实拍图像
    :param args: 参数
    :return: 变形场flow(tensor_gpu) img_MSE img_SSIM
    """
    device = torch.device("cuda:" + args.gpu_id if torch.cuda.is_available() else "cpu")
    img_cal = cv2.cvtColor(img_calibration, cv2.COLOR_BGR2RGB)
    h, w, _ = img_cal.shape
    img_cal = img2tensor(img_cal,device)
    img_gt = cv2.cvtColor(img_GT, cv2.COLOR_BGR2RGB)
    img_gt = img2tensor(img_gt,device)
    B, C, H, W = img_cal.size()

    # define generator
    if args.module == 'INR':
        generate_model = INR(args).to(device)
    elif args.module == 'U-NET':
        generate_model = UNet(C,2).to(device)
    # print(next(generate_model.parameters()).device)
    # summary(generate_model, (C, H, W),device='cuda:'+ args.gpu_id)

    # define optimizer
    optimizer = optim.Adam(generate_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                                gamma=args.gamma) if args.lr_decay != 0 else None

    # define loss
    loss_model = FlowLoss(args)

    # iterative optimize
    for i in range(args.total_iter + 1):
        flow_formal = generate_model(img_cal)

        # warp
        flow_x = flow_formal[:, 0].view(H, W).permute(1, 0)
        flow_y = flow_formal[:, 1].view(H, W).permute(1, 0)
        flow = torch.stack((flow_x, flow_y), dim=0).view(-1, 2, H, W)
        img_warp = warp2d(img_cal, flow)

        # optimize
        loss, img_dif_loss, regular_loss, edge_loss = loss_model(img_warp, img_gt, flow)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if args.pattern == 'train':
            if i % args.iter_print == 0:
                print("step: %d, lr: %.4f, img_dif_loss: %.8f, regular_loss: %.8f, edge_loss: %.8f" % (
                    i, optimizer.param_groups[0]['lr'], img_dif_loss.item(), regular_loss.item(),edge_loss.item()))
        else:
            if i % args.iter_print == 0:
                print("step: %d, lr: %.4f, img_dif_loss: %.8f, regular_loss: %.8f, edge_loss: %.8f" % (
                    i, optimizer.param_groups[0]['lr'], img_dif_loss.item(), regular_loss.item(),edge_loss.item()))
    flow_save = flow.clone()
    # save result
    img_out = torch.clip(img_warp.detach(), max=1.0, min=0)
    vis_warp = make_numpy_grid(img_out)
    vis = (vis_warp * 255.).astype(np.uint8)
    vis = vis[:, :, ::-1]
    warp_img = vis.copy()
    img_MSE = mse(vis, img_GT)
    img_SSIM = ssim(vis, img_GT)
    return flow_save , warp_img, img_MSE, img_SSIM

def depth_generate(flow_input,depth_gt,args):
    device = torch.device("cuda:" + args.gpu_id if torch.cuda.is_available() else "cpu")
    flag = 0
    B, C, H, W = flow_input.size()
    # define generator
    if args.module == 'INR':
        generate_model = INR(args).to(device)
    elif args.module == 'U-NET':
        generate_model = UNet(C, 1).to(device)
    # summary(generate_model, (C, H, W),device=device)

    # define optimizer
    optimizer = optim.Adam(generate_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                                gamma=args.gamma) if args.lr_decay != 0 else None

    # define loss
    loss_model = DepthLoss(args)

    # iterative optimize
    i = 1
    count = 0
    while i < args.total_iter+1:
    # for i in range(args.total_iter+1):

        depth = generate_model(flow_input).view(-1, 1, H, W)
        normal = depth_to_normal(depth, args)
        flow_pred = raytracing(depth, normal, args, [H, W])
        # optimize
        loss = loss_model(flow_pred, flow_input, depth)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        i = i + 1
        if i == 800:
            cos_cal = torch.cosine_similarity(flow_pred,flow_input,dim=2)
            print(cos_cal.mean().item())
            if cos_cal.mean().item() <= 0.5 and count <=3 :
                count += 1
                generate_model = UNet(C, 1).to(device)
                i = 1
            if count > 3:
                flag = 1
                break

        if args.pattern == 'train':
            if i % args.iter_print == 0:
                print("step: %d, lr: %.4f, loss: %.8f" % (
                    i, optimizer.param_groups[0]['lr'], loss.item(),))


        else:
            if i % args.iter_print == 0:
                print("step: %d, lr: %.4f, loss: %.8f" % (
                    i, optimizer.param_groups[0]['lr'], loss.item(),))

    depth_save = depth.clone()
    # save result
    flow_COS = torch.cosine_similarity(flow_input,flow_pred,dim = 2).mean()
    if depth_gt!= None:
        depth_COS = torch.cosine_similarity(torch.from_numpy(depth_gt).to(device),depth,dim = 2).mean()
        depth_RMSE = RMSE(depth_gt,depth)
        depth_abs_rel = Abs_Rel(depth_gt,depth)
    return depth_save,flow_COS, flag

def log(obj, save_path, filename = 'log.txt'):
    print(obj)
    with open(os.path.join(save_path, filename), 'a') as f:
        print(obj, file = f)

def save_result_ForDataset(depth_result,  Dataset_path, run_serial,  j):
    """
    本函数用于大批量数据集时使用，用于保存最终的流体表面重建结果（只保留流体是深度信息.npy文件以及折射图像）
    :param depth_result: tensor_gpu
    :param warp_img: RGB图像
    :param Dataset_path: str (当前测试的Dataset) ——绝对路径
    :parameter run_serial: str (当前测试的具体编号数目) —— 对应列表中的一个str
    :parameter current_runimg_path: 当前运行的图像名(str) ——对应的是列表中的一个str
    :parameter current_depth_path：当前运行的depth名(str) ——对应的是列表中的一个str
    :return:
    """

    #完成tensor_gpu 到 numpy 的转换
    depth_save = depth_result.cpu().detach().numpy()


    #在DataSet_STEP下创建 result文件夹（ 包括depth和warp_img子文件夹 ）
    depth_result_folder = Dataset_path + '/result2/depth'
    if not(os.path.exists(depth_result_folder)):
        os.makedirs(depth_result_folder)

    #在两个子文件夹下根据当前运行的编号创建对应的结果文件夹，保存结果
    depth_serial_folder = depth_result_folder + '/' + run_serial
    if not(os.path.exists(depth_serial_folder)):
        os.makedirs(depth_serial_folder)
    np.save(depth_serial_folder + '/' + str(j),depth_save)


def run(filename,flow_args,depth_args):
    Dataset_paths = os.listdir(filename)
    for Dataset_path in Dataset_paths:
        choose_Dataset_path = os.path.join(filename, Dataset_path)  # 当前选择的Dataset绝对路径
        img_gt_folder = os.path.join(choose_Dataset_path,  'render')
        reference_gt_folder = os.path.join(choose_Dataset_path, 'reference')
        img_calibration_path = sorted(os.listdir(reference_gt_folder))
        run_serial = sorted(os.listdir(img_gt_folder))
        for i in range(len(run_serial)):
            print("当前的run_serial为：" + run_serial[i])  # run_serial[i] 对应 当前测试的具体编号数目
            img_calibration_run_path = img_calibration_path[i] #标定物的路径
            img_gt_run_folder = os.path.join(img_gt_folder, run_serial[i])
            # print('depth_gt_run_file')
            # print(depth_gt_run_file)
            img_gt_run_file = sorted(os.listdir(img_gt_run_folder))  # 对应img_gt 文件名列表
            data = {'img_MSE': [], 'img_SSIM': [], 'flow_COS': [], 'depth_RMSE': [], 'depth_abs_rel': [],
                    'depth_COS': []}

            # print('img_gt_run_file')
            # print(img_gt_run_file)
            j = 0
            while j < len(img_gt_run_file):
                # print("当前运行文件为：" + depth_gt_run_file[j])
                img_gt_running_file = img_gt_run_file[j]  # 对应当前运行的图像名(str)
                print(img_gt_running_file)
                img_calibration = cv2.imread(os.path.join(reference_gt_folder,img_calibration_run_path)) #标定物
                img_gt = cv2.imread(os.path.join(img_gt_run_folder,img_gt_running_file))
                flow_save , warp_img, img_MSE, img_SSIM = flow_generate(img_calibration,img_gt,flow_args)
                print(img_SSIM)
                depth_save, flow_COS, flag  = depth_generate(flow_save.data,None,depth_args)

                if flag == 1:
                    continue
                save_result_ForDataset(depth_save, choose_Dataset_path, run_serial[i],  j)
                j += 1




if __name__ == '__main__':
    """
    此处实现的是多张图片的运行结果
    输入图片文件夹的组织形式为：波形-扭曲程度-标定物
    """
    run('/home3/hyx/sys', flow_args, depth_args)
