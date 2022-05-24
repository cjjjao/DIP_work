import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
import torchvision.models as models

class PEF_Loss(nn.Module):
    def __init__(self):
        super(PEF_Loss, self).__init__()
        self.model = models.vgg16(pretrained=True)
        self.mse_loss = nn.MSELoss(reduction='mean')
    def forward(self,x,x_g):
        with torch.no_grad():
            conv_op = nn.Conv2d(1, 1, 3, padding= 1, stride= 4, bias=False)
            # 定义sobel算子参数
            '''sobel_kernel_h = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype='float32').reshape((1, 1, 3, 3))
            sobel_kernel_v = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32').reshape((1, 1, 3, 3))
            sobel_kernel_l = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype='float32').reshape((1, 1, 3, 3))
            sobel_kernel_r = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype='float32').reshape((1, 1, 3, 3))
            sobel_kernel_h = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype='float32').reshape((1, 1, 3, 3))
            sobel_kernel_v = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype='float32').reshape((1, 1, 3, 3))
            sobel_kernel_l = np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]], dtype='float32').reshape((1, 1, 3, 3))
            sobel_kernel_r = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]], dtype='float32').reshape((1, 1, 3, 3))
            # 将sobel算子转换为适配卷积操作的卷积核
            # 给卷积操作的卷积核赋值
            conv_op.weight.data = torch.from_numpy(sobel_kernel_h)
            edge_detect_h = conv_op(Variable(x_g))
            conv_op.weight.data = torch.from_numpy(sobel_kernel_v)
            edge_detect_v = conv_op(Variable(x_g))
            conv_op.weight.data = torch.from_numpy(sobel_kernel_l)
            edge_detect_l = conv_op(Variable(x_g))
            conv_op.weight.data = torch.from_numpy(sobel_kernel_r)
            edge_detect_r = conv_op(Variable(x_g))
            edge_detect = (edge_detect_h + edge_detect_v + edge_detect_l + edge_detect_r).repeat(1,64,1,1).cuda()'''
            laps_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32').reshape((1, 1, 3, 3))
            conv_op.weight.data = torch.from_numpy(laps_kernel)
            edge_detect = conv_op(Variable(x_g)).repeat(1,64,1,1).cuda()
            #self.visualize(edge_detect)

        vgg_layer = [5, 7, 7, 7]
        index_cur = 5
        loss = 0
        self.model.eval()
        for i in vgg_layer:
            x = self.model.features[index_cur:index_cur + i](x)
            edge_detect = self.model.features[index_cur:index_cur + i](edge_detect)
            loss += self.mse_loss(x,edge_detect)
            index_cur += i
        return loss

    def visualize(self,featmap):
        #select_id = [0,32,4,36,8,40,12,44]
        b,c,h,w = featmap.shape[0],featmap.shape[1],featmap.shape[2],featmap.shape[3]
        featmap = featmap[0].data.cpu().numpy()
        for id in range(1):
            mask = featmap[id].reshape(-1,h * w)
            mask = mask - np.min(mask, axis=1)
            mask = mask / np.max(mask, axis=1)
            # mask = 1 - mask
            mask = mask.reshape(h, w)
            cam_img = np.uint8(255 * mask)
            cam_img = cv2.resize(cam_img, (144, 288))
            path = './heatmap'
            cv2.imwrite(path + '/' + str(id) + '.jpg',cam_img)
