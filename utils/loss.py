import torch
import torch.nn as nn
import utils
import torch.nn.functional
import torch.nn.functional as F
from Option.flow import args
device = torch.device("cuda:" + args.gpu_id if torch.cuda.is_available() else "cpu")
def tailor_for_tensor(img_input,tailor_size):
    """
    规定此处的img_input为C*H*W
    """
    img_output = img_input[:,:,tailor_size:-tailor_size,tailor_size:-tailor_size]
    return img_output

class KLLoss(torch.nn.Module):

    def __init__(self):
        super(KLLoss,self).__init__()

    def forward(self,flow,warp_origin):
        """计算KL散度"""
        kl = F.kl_div(flow.softmax(dim = -1).log(), warp_origin.softmax(dim = -1), reduction = 'sum')

        return kl

class CossimLoss(torch.nn.Module):

    def __init__(self):
        super(CossimLoss,self).__init__()

    def forward(self,flow,warp_origin):
        """计算KL散度"""
        cossim = torch.cosine_similarity(warp_origin,flow,dim = 2).mean()
        cossim = 1/(cossim*cossim)

        return cossim



class EdgeLoss(nn.Module):

    def __init__(self):
        super(EdgeLoss,self).__init__()

    def sobel_tensor(self,img):
        # sobel_tensor运算
        img_pad = F.pad(img, [1, 1, 1, 1], mode = 'replicate')
        sobel_x = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype = torch.float, requires_grad = False).view(1, 1, 3, 3).repeat(
            [1, 3, 1, 1])
        sobel_y = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype = torch.float, requires_grad = False).view(1, 1, 3, 3).repeat(
            [1, 3, 1, 1])
        if img.is_cuda:
            sobel_x = sobel_x.to(device)
            sobel_y = sobel_y.to(device)

        dx = F.conv2d(img_pad, sobel_x)
        dy = F.conv2d(img_pad, sobel_y)
        return dx, dy

    def Laplace_tensor(self,img):
        # Laplace运算
        img_pad = F.pad(img, [1, 1, 1, 1], mode = 'replicate')
        Laplace_operator = torch.tensor([[-1.0, 1.0, 1.0],
                                         [1.0, -8.0, 1.0],
                                         [1.0, 1.0, 1.0]], requires_grad = False).repeat([1, 3, 1, 1])
        if img.is_cuda:
            Laplace_operator = Laplace_operator.to(device)

        edge = F.conv2d(img_pad, Laplace_operator)
        return edge

    def data_normal(self,orign_data):
        orign_data_clone = orign_data.clone()
        d_min = orign_data_clone.min()
        if d_min < 0:
            orign_data_clone =orign_data_clone + torch.abs(d_min)
            d_min = orign_data_clone.min()
        d_max = orign_data_clone.max()
        dst = d_max - d_min
        normal_data = (orign_data_clone - d_min).true_divide(dst)
        return normal_data

    def conv_operator(self, img, kernel):
        if kernel == 'sobel':
            dx, dy = self.sobel_tensor(img)
            dx = self.data_normal(dx)
            dy = self.data_normal(dy)
            return dx,dy
        elif kernel == 'laplace':
            edge = self.Laplace_tensor(img)
            edge = self.data_normal(edge)
            return edge


    def forward(self,img_gt,img_warp,kernel):
        img_gt_clone = img_gt.clone()
        img_warp_clone = img_warp.clone()
        if kernel == 'sobel':
            dx1,dy1 =self.conv_operator(img_gt_clone,kernel)
            dx2,dy2 =self.conv_operator(img_warp_clone,kernel)
            loss_function = nn.MSELoss()
            mse1 = loss_function(dx1,dx2)
            mse2 = loss_function(dy1,dy2)
            mse = mse1 + mse2
            return mse
        if kernel == 'laplace':
            edge1 = self.conv_operator(img_gt_clone,kernel)
            edge2 =self.conv_operator(img_warp_clone,kernel)
            loss_function = nn.MSELoss()
            mse = loss_function(edge1,edge2)
            return mse



class FlowLoss(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.MSELoss = nn.MSELoss()
        self.Edgeloss = EdgeLoss()
    def forward(self,img_gt,img_warp,flow):
        img_gt_clone = img_gt.clone()
        img_warp_clone = img_warp.clone()
        img_warp_clone = tailor_for_tensor(img_warp_clone,10)
        img_gt_clone = tailor_for_tensor(img_gt_clone,10)
        regular_loss = self.args.weight_regular_loss * torch.pow(flow, 2).sum()
        img_dif_loss = self.args.weight_MSE_loss * self.MSELoss(img_gt_clone,img_warp_clone)
        edge_loss = self.args.weight_edge_loss * self.Edgeloss(img_gt_clone,img_warp_clone,'laplace')
        loss = regular_loss + img_dif_loss + edge_loss
        # loss = img_dif_loss
        return loss,img_dif_loss,regular_loss,edge_loss

class DepthLoss(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.MSELoss = nn.MSELoss()
    def forward(self,flow_pred,flow_former,depth):
        # kl_loss = self.KLLoss(flow_pred,flow_former)
        mse_loss = self.MSELoss(flow_pred,flow_former)
        # cos_loss = self.CosLoss(flow_pred,flow_former)
        return mse_loss

class VAELoss(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.MSELoss = nn.MSELoss()
    def forward(self,flow_pred, flow_former, mu, log_var):
        KLD = 0.5 * torch.sum(torch.exp(log_var) + torch.pow(mu, 2) - 1. - log_var)
        mse = self.MSELoss(flow_pred,flow_former)
        loss = KLD
        return loss,KLD,mse