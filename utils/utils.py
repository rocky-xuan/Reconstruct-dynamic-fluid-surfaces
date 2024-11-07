import random
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn.functional as tf
import torchvision.transforms as tvt
from torchvision import transforms, utils
import os
import math
import shutil
import cv2
import matplotlib.pyplot as plt
from torchvision import utils
from Option.flow import args
from sklearn.metrics import r2_score
import torch

device = torch.device("cuda:" + args.gpu_id if torch.cuda.is_available() else "cpu")

def log(obj, save_path, filename = 'log.txt'):
    print(obj)
    with open(os.path.join(save_path, filename), 'a') as f:
        print(obj, file = f)

def warp2d(img, flow):
    img_size = img.shape[-1]
    H, W = img_size, img_size
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, H, W)
    yy = yy.view(1, H, W)
    grid = torch.cat((xx, yy), 0).float()  # [2, H, W]
    grid = grid.repeat(flow.shape[0], 1, 1, 1)  # [bs, 2, H, W]
    if img.is_cuda:
        grid = grid.to(device)
    grid = torch.autograd.Variable(grid, requires_grad=False) + flow
    grid[grid < 0] = 0
    grid[grid > img_size - 1] = img_size - 1
    grid = 2.0 * grid / (img_size - 1) - 1.0  # max(W-1,1)
    grid = grid.permute(0, 2, 3, 1)
    output = F.grid_sample(img, grid, mode='bilinear', align_corners=True)

    return output

def warping(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    """

    B, C, H, W = x.size()
    grid = meshgrid(B, C, H, W)
    vgrid = grid + flo

    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, mode='bilinear', align_corners=True)


    return output

def draw_grid(flow,fig_name):
    plt.figure(figsize = (128, 128))

    step_size = 10
    x_map = flow[0, :, :].detach().cpu().numpy()
    h = x_map.shape[1]
    x_map = x_map[::-1, :]
    y_map = flow[1, :, :].detach().cpu().numpy()
    y_map = y_map[::-1, :]
    w = y_map.shape[1]
    ys = np.arange(0, h, step_size)
    xs = np.arange(0, w, step_size)
    for xx in range(0, w, step_size):
        plt.plot(xx + x_map[ys, xx], ys, 'black')
    for yy in range(0, h, step_size):
        plt.plot(xs, yy + y_map[yy, xs], 'black')
    plt.savefig(fig_name)

def img2tensor(img, device):
    img = np.expand_dims(img.transpose((2, 0, 1)), 0)  # 将图片转换为C*H*W,并在最开始增加一个B维度
    img = img.astype(np.float32) / 255
    img = torch.from_numpy(img).to(device)
    return img

def sobel_tensor(img):
    #sobel_tensor运算
    img_pad = F.pad(img, [1, 1, 1, 1], mode='replicate')
    sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                            [-2.0, 0.0, 2.0],
                            [-1.0, 0.0, 1.0]]).repeat([1, 1, 1, 1])
    sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                            [0.0, 0.0, 0.0],
                            [1.0, 2.0, 1.0]]).repeat([1, 1, 1, 1])
    if img.is_cuda:
        sobel_x = sobel_x.to(device)
        sobel_y = sobel_y.to(device)

    dx = F.conv2d(img_pad, sobel_x)
    dy = F.conv2d(img_pad, sobel_y)

    return dx, dy

def make_numpy_grid(tensor_data):
    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data)
    vis = np.array(vis.cpu()).transpose((1, 2, 0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis

def grid_coordinate(shape, range = None, flatten = True):
    """
    为图像建立以网格形式描述的像素位置变化(X、Y方向各一个)
    :param shape: H * W (输入图像的尺寸)
    :param range: 默认将位置变化归一化到 -1~1之间，如需改变范围可以手动输入[min，max]的形式
    :param flatten: True 则输出 HW * 2,false 则输出(H * W * 2)
    :return:
    """
    grids = []
    for index, item in enumerate(shape):
        if range is None:
            min, max = -1, 1  # -1~1有利于后续的傅里叶编码
        else:
            min, max = range
        r = (max - min) / (2 * item)
        grid = min + r + (2 * r) * torch.arange(item).float()
        grids.append(grid)
    ret = torch.stack(torch.meshgrid(*grids), dim = -1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def input_mapping(grid, L = 6):
    """
    本函数实现的是对输入进行傅里叶域的扩充
    :param x: 输入像素的坐标对 (HW * 2)
    :param L: 傅里叶编码的扩充项数
    :return: 编码后的结果
    """
    gamma = []
    gamma.append(grid)
    for l in range(L):
        gamma.append(torch.sin(2 ** l * np.pi * grid))
        gamma.append(torch.cos(2 ** l * np.pi * grid))
    return torch.cat(gamma, dim = -1)

def meshgrid(B, C, H, W):
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().to(device)
    return grid

def snell_refraction(normal, s1, n1, n2):
    """
    :param normal:法线信息
    :param s1:
    :param n1:折射参数
    :param n2:折射参数
    :return:
    """
    this_normal = normal
    term_1 = torch.cross(this_normal, torch.cross(-this_normal, s1))
    term_temp = torch.cross(this_normal, s1)
    n_sq = (n1 / n2) ** 2
    term_2_temp = torch.abs(1.0 - n_sq * torch.sum(torch.multiply(term_temp, term_temp), dim=1))
    term_2 = torch.sqrt(term_2_temp + 1e-5)
    term_3 = torch.stack([term_2, term_2, term_2], dim=1)
    s2 = torch.subtract((n1 / n2) * term_1, torch.multiply(this_normal, term_3))
    s2_length = torch.sqrt(torch.square(s2[:, 0:1, :, :]) +
                           torch.square(s2[:, 1:2, :, :]) +
                           torch.square(s2[:, 2:3, :, :]))
    s2_n = s2 / s2_length
    return s2_n

def depth_to_normal(y_depth_pred,args):
    step = args.step

    zx, zy = sobel_tensor(y_depth_pred)

    zx = zx * step      # b 1 h w
    zy = zy * step

    normal_ori = torch.cat([-zx, -zy, torch.ones_like(y_depth_pred)], dim=1)
    new_normal = torch.sqrt(torch.square(zx) + torch.square(zy) + 1)
    normal_pred = normal_ori / new_normal

    normal_pred = (normal_pred + 1) / 2

    return normal_pred

def raytracing(depth_pred, normal_pred, args,shape_origin):
    #折射
    image_size = shape_origin[-1]
    step = args.step
    n1 = args.n1
    n2 = args.n2

    # depth_min = scale[:, :, 0:1, 0:1]
    # depth_max = scale[:, :, 0:1, 1:2]
    # normal_min = scale[:, :, 0:1, 2:3]
    # normal_max = scale[:, :, 0:1, 3:4]
    #
    # depth = denorm(depth_pred, depth_min, depth_max)
    # normal = denorm(normal_pred, normal_min, normal_max)

    normal = normal_pred * 2 - 1

    depth = torch.squeeze(depth_pred, dim=1)

    s1 = torch.zeros([depth.shape[0], 1, image_size, image_size], requires_grad=False)
    s11 = -1 * torch.ones([depth.shape[0], 1, image_size, image_size], requires_grad=False)

    assigned_s1 = torch.cat([s1, s1, s11], dim=1)
    if normal.is_cuda:
        assigned_s1 = assigned_s1.to(device)
    assigned_s1 = assigned_s1.detach()
    s2 = snell_refraction(normal, assigned_s1, n1, n2)

    x_c_ori, y_c_ori, z_ori = s2[:, 0:1, :, :], s2[:, 1:2, :, :], s2[:, 2:3, :, :]

    z_ori[torch.abs(z_ori) < 1e-5] = 1e-5

    z_ori = torch.squeeze(z_ori)
    amplify = torch.divide(depth, z_ori)

    x_c = torch.multiply(amplify, torch.squeeze(x_c_ori)) * step
    y_c = torch.multiply(amplify, torch.squeeze(y_c_ori)) * step

    flow = torch.stack([x_c, y_c], dim=1)
    #print(flow.shape)
    return flow
def RMSE(numpy1, tensor2):
    input_1 = numpy1
    input_2 = tensor2.cpu().detach().numpy()
    rmse = (input_1 - input_2) ** 2
    rmse = np.sqrt(rmse.mean())
    return rmse

def min_max_normalization(array):
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = array/max_val
    return normalized_array

def Abs_Rel(numpy1, tensor2):
    #input1 为真值
    input_1 = numpy1
    input_2 = tensor2.cpu().detach().numpy()
    abs_rel = np.mean(np.abs(input_1-input_2)/input_1)

    return abs_rel
def R2_score(y_true,y_pred):

    y_pred = y_pred.cpu().detach().numpy()

    # 计算R2score
    r2 = r2_score(y_true, y_pred)
    return r2jias


def mse(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    return mse

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def ssim(y_true , y_pred):
    u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_true)
    std_pred = np.sqrt(var_pred)
    c1 = np.square(0.01*7)
    c2 = np.square(0.03*7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim/denom

def save_result(img_gt, img, img_out, flow, save_path, i,args):
    vis_in = make_numpy_grid(img)
    h, w, _ = vis_in.shape
    vis_warp = make_numpy_grid(img_out)
    vis = vis_warp
    vis = (vis * 255.).astype(np.uint8)
    vis = vis[:, :, ::-1]
    cv2.imwrite(os.path.join(save_path, str(i).zfill(5) + '_vis.jpg'), vis)



    plt.figure(figsize=(10, 10))
    step_size = 10
    x_map = flow[0, 0, :, :].detach().cpu().numpy()
    x_map = x_map[::-1, :]
    y_map = flow[0, 1, :, :].detach().cpu().numpy()
    y_map = y_map[::-1, :]
    ys = np.arange(0, h, step_size)
    xs = np.arange(0, w, step_size)
    for xx in range(0, w, step_size):
        plt.plot(xx + x_map[ys, xx], ys, 'black')
    for yy in range(0, h, step_size):
        plt.plot(xs, yy + y_map[yy, xs], 'black')
    plt.savefig(os.path.join(save_path, str(i).zfill(5) + '_grid.png'))
    plt.close()

    if i == args.total_iter:
        torch.save(flow.to(torch.device('cpu')),save_path + '/flow.pth') # 1 B H W
        log('MSE:' + str(mse(vis, img_gt)),save_path)
        log('SSIM:' + str(ssim(vis, img_gt)),save_path)

def save_depth_result(depth, flow_pred, flow_orign,depth_gt,  save_path, i,args):
    #保存depth以及flow_pred
    vis_flow = make_numpy_grid(flow_pred)
    #flow_pred为拟合的函数，此处导出保存的为矩阵
    vis = vis_flow
    # print(vis.shape)
    vis = vis[:, :, ::-1]
    # np.save(os.path.join(save_path, str(i).zfill(5) + '_vis.npy'), vis)
    flow_in = make_numpy_grid(flow_orign)
    depth_pred = depth.cpu().detach().numpy()






    if i == args.total_iter:
        torch.save(depth.to(torch.device('cpu')),save_path + '/depth.pth') # 1 B H W
        log('MSE:' + str(mse(vis, flow_in)),save_path)
        # log('PSNR:' + str(psnr(vis, flow_in)),save_path)
        log('SSIM:' + str(ssim(vis, flow_in)),save_path)
        log('CosSim:' + str(torch.cosine_similarity(flow_pred,flow_orign,dim = 2).mean()),save_path)
        log('KL:'+ str(F.kl_div(flow_pred.softmax(dim=-1).log(), flow_orign.softmax(dim=-1), reduction='mean')),save_path)
        if depth_gt.any() != None:
            log('when given depth_gt:',save_path)
            log('MSE:' + str(mse(depth_pred, depth_gt)),save_path)
            # log('PSNR:' + str(psnr(depth_pred, depth_gt)),save_path)
            log('SSIM:' + str(ssim(depth_pred, depth_gt)),save_path)
            log('CosSim:' + str(torch.cosine_similarity(depth, torch.tensor(depth_gt).to(device), dim=2).mean()),save_path)



def save_result_ForDataset(depth_result, warp_img, Dataset_path, run_serial, current_runimg_path, current_depth_path):
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
    img_result_folder = Dataset_path + '/result2/warp_img'
    if not(os.path.exists(depth_result_folder)):
        os.makedirs(depth_result_folder)
    if not(os.path.exists(img_result_folder)):
        os.makedirs(img_result_folder)

    #在两个子文件夹下根据当前运行的编号创建对应的结果文件夹，保存结果
    depth_serial_folder = depth_result_folder + '/' + run_serial
    img_serial_folder = img_result_folder + '/' + run_serial
    if not(os.path.exists(depth_serial_folder)):
        os.makedirs(depth_serial_folder)
    if not(os.path.exists(img_serial_folder)):
        os.makedirs(img_serial_folder)
    np.save(depth_serial_folder + '/' + current_depth_path,depth_save)
    cv2.imwrite(img_serial_folder + '/' + current_runimg_path, warp_img)









class MakeCutouts(torch.nn.Module):
    def __init__(self, cut_size: int, num_cutouts: int, cutout_size_power: float = 1.0, use_augs: bool = False):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = num_cutouts
        self.cut_pow = cutout_size_power
        if use_augs:
            self.augs = tvt.Compose([
                tvt.RandomHorizontalFlip(p=0.5),
                tvt.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                tvt.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                tvt.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                tvt.RandomPerspective(distortion_scale=0.4, p=0.7),
                tvt.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                tvt.RandomGrayscale(p=0.15),
                tvt.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            ])
        else:
            self.augs = tvt.Compose([])

    def forward(self, input: torch.Tensor):
        side_y, side_x = input.shape[2:4]
        max_size = min(side_y, side_x)
        min_size = min(side_y, side_x, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, side_x - size + 1, ())
            offsety = torch.randint(0, side_y - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            if random.random() > 0.5:
                cutout = self.augs(cutout)
            cutout = tf.adaptive_avg_pool2d(cutout, self.cut_size)
            cutouts.append(cutout)

        return torch.cat(cutouts)


