import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

class RX01Data(Dataset):
    def __init__(self, data_dir, transform=None, colorIndex=None, thermalIndex=None):
        data_dir = '/media/data/cja/cja_proj/Datasets/RX2_07/train/' #'/data/cja/cja_proj/Datasets/RX2_07/train/' #'/media/data/cja/cja_proj/Datasets/RX2_07/train/'
        # Load training images (path) and labels
        train_color_image = np.load(data_dir + 'train_rgb_resized_img288.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label288.npy')

        train_thermal_image = np.load(data_dir + 'train_ir_resized_img288.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label288.npy')

        # BGR to RGB
        self.train_color_image = train_color_image
        self.train_thermal_image = train_thermal_image
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):
        img11, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img22, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        '''img22 = img22[:, :, 1]
        img22 = img22[..., np.newaxis]

        img11 = rgb2gray(img11)
        img11 = img11[..., np.newaxis]

        padding = np.zeros_like(img11)
        img11 = np.concatenate((img11, padding), axis=2).astype(np.uint8)
        img22 = np.concatenate((padding, img22), axis=2).astype(np.uint8)'''
        img1 = self.transform(img11)
        img2 = self.transform(img22)

        return {'img_v': img1, 'img_t': img2, 'target_v': target1, 'target_t': target2}#,'img11':img11,'img22':img22}

    def __len__(self):
        return len(self.train_color_label)

class SYSUData(Dataset):
    def __init__(self, data_dir, transform=None, colorIndex=None, thermalIndex=None, transform_gray=None,transform_ori=None):
        
        # Load training images (path) and labels
        train_color_image = np.load(os.path.join(data_dir, 'train_rgb_resized_img.npy'))
        self.train_color_label = np.load(os.path.join(data_dir, 'train_rgb_resized_label.npy'))

        train_thermal_image = np.load(os.path.join(data_dir, 'train_ir_resized_img.npy'))
        self.train_thermal_label = np.load(os.path.join(data_dir, 'train_ir_resized_label.npy'))
        
        # BGR to RGB
        self.train_color_image = train_color_image
        self.train_thermal_image = train_thermal_image
        self.transform = transform
        self.transform_gray = transform_gray
        self.transform_ori = transform_ori
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img_v, target_v = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img_t, target_t = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img_v = self.transform(img_v)
        img_t = self.transform(img_t)

        img_gv = self.transform_gray(img_v)
        img_gt = self.transform_gray(img_t)

        img_v = self.transform_ori(img_v)
        img_t = self.transform_ori(img_t)


        return {'img_v': img_v, 'img_t': img_t, 'img_gv':img_gv, 'img_gt':img_gt, 'target_v': target_v, 'target_t': target_t}

    def __len__(self):
        return len(self.train_color_label)
        
class RegDBData(Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex=None, thermalIndex=None):
        # Load training images (path) and labels
        train_color_list   = os.path.join(data_dir, 'idx/train_visible_{}'.format(trial)+'.txt')
        train_thermal_list = os.path.join(data_dir, 'idx/train_thermal_{}'.format(trial)+'.txt')

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)
        
        train_color_image = []
        for i in range(len(color_img_file)):
   
            img = Image.open(data_dir+ color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image) 
        
        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir+ thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)
        
        # BGR to RGB
        self.train_color_image = train_color_image  
        self.train_color_label = train_color_label
        
        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label
        
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img_v, target_v = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img_t, target_t = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img_v = self.transform(img_v)
        img_t = self.transform(img_t)

        return {'img_v': img_v, 'img_t': img_t, 'target_v': target_v, 'target_t': target_t}

    def __len__(self):
        return len(self.train_color_label)
        
class TestData(Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size=(144,288),mode = 0):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            '''if mode == 1:
                pix_array = rgb2gray(pix_array)
                pix_array = pix_array[..., np.newaxis]
                padding = np.zeros_like(pix_array)
                pix_array = np.concatenate((pix_array, padding), axis=2).astype(np.uint8)
            elif mode == 2:
                pix_array = pix_array[:, :, 1]
                pix_array = pix_array[..., np.newaxis]
                padding = np.zeros_like(pix_array)
                pix_array = np.concatenate((padding,pix_array), axis=2).astype(np.uint8)'''
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.test_image[index], self.test_label[index]
        img1 = img
        img = self.transform(img)
        return {'img': img, 'target': target,'img1':img1}

    def __len__(self):
        return len(self.test_image)
         
def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label


def GenIdx( train_color_label, train_thermal_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k,v in enumerate(train_color_label) if v==unique_label_color[i]]
        color_pos.append(tmp_pos)
        
    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k,v in enumerate(train_thermal_label) if v==unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)
    return color_pos, thermal_pos
    
def GenCamIdx(gall_img, gall_label, mode):
    if mode =='indoor':
        camIdx = [1,2]
    else:
        camIdx = [1,2,4,5]
    gall_cam = []
    for i in range(len(gall_img)):
        gall_cam.append(int(gall_img[i][-10]))
    
    sample_pos = []
    unique_label = np.unique(gall_label)
    for i in range(len(unique_label)):
        for j in range(len(camIdx)):
            id_pos = [k for k,v in enumerate(gall_label) if v==unique_label[i] and gall_cam[k]==camIdx[j]]
            if id_pos:
                sample_pos.append(id_pos)
    return sample_pos
    
def ExtractCam(gall_img):
    gall_cam = []
    for i in range(len(gall_img)):
        cam_id = int(gall_img[i][-10])
        # if cam_id ==3:
            # cam_id = 2
        gall_cam.append(cam_id)
    
    return np.array(gall_cam)
    
class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, num_pos, batchSize, epoch):        
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)
        
        N = np.maximum(len(train_color_label), len(train_thermal_label)) 
        for j in range(int(N/(batchSize*num_pos))+1):
            batch_idx = np.random.choice(uni_label, batchSize, replace = False)  
            for i in range(batchSize):
                sample_color  = np.random.choice(color_pos[batch_idx[i]], num_pos)
                sample_thermal = np.random.choice(thermal_pos[batch_idx[i]], num_pos)
                
                if j ==0 and i==0:
                    index1= sample_color
                    index2= sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))
        
        self.index1 = index1
        self.index2 = index2
        self.N  = N
        
    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N     

def process_query_RX01(data_path, mode='all', relabel=False):
    data_path = '/media/data/cja/cja_proj/Datasets/RT2_01/query/'#'/data/cja/cja_proj/Datasets/RX2_07/query/'#'/media/data/cja/cja_proj/Datasets/RX2_07/query/'
    if mode == 'all':
        ir_cameras = ['cam2']
    elif mode == 'indoor':
        ir_cameras = ['cam2']

    files_ir = []

    for cam in ir_cameras:
        lab_dir = os.path.join(data_path, cam)
        ids = ["%04d" % int(id) for id in os.listdir(lab_dir)]

    for id in sorted(ids):
        for cam in ir_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_ir.extend(new_files)
    query_img = []
    query_id = []
    query_cam = []
    for img_path in files_ir:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)
    return query_img, np.array(query_id), np.array(query_cam)


def process_gallery_RX01(data_path, mode='all', trial=0, relabel=False):
    data_path = '/media/data/cja/cja_proj/Datasets/RT2_01/gallery/'#'/data/cja/cja_proj/Datasets/RX2_07/gallery/' #'/media/data/cja/cja_proj/Datasets/RX2_07/gallery/'
    random.seed(trial)

    if mode == 'all':
        rgb_cameras = ['cam1']
    elif mode == 'indoor':
        rgb_cameras = ['cam1']

    files_rgb = []
    for cam in rgb_cameras:
        lab_dir = os.path.join(data_path, cam)
        ids = ["%04d" % int(id) for id in os.listdir(lab_dir)]
    for id in sorted(ids):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_rgb.append(random.choice(new_files))
    gall_img = []
    gall_id = []
    gall_cam = []
    for img_path in files_rgb:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        gall_img.append(img_path)
        gall_id.append(pid)
        gall_cam.append(camid)
    return gall_img, np.array(gall_id), np.array(gall_cam)

def process_query_sysu(data_path, mode = 'all', relabel=False):
    if mode== 'all':
        ir_cameras = ['cam3','cam6']
    elif mode =='indoor':
        ir_cameras = ['cam3','cam6']
    
    file_path = os.path.join(data_path,'exp/test_id.txt')
    files_rgb = []
    files_ir = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in ir_cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_ir.extend(new_files)
    query_img = []
    query_id = []
    query_cam = []
    for img_path in files_ir:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)
    return query_img, np.array(query_id), np.array(query_cam)

def process_gallery_sysu(data_path, mode = 'all', trial = 0, relabel=False):
    
    random.seed(trial)
    
    if mode== 'all':
        rgb_cameras = ['cam1','cam2','cam4','cam5']
    elif mode =='indoor':
        rgb_cameras = ['cam1','cam2']
        
    file_path = os.path.join(data_path,'exp/test_id.txt')
    files_rgb = []
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_rgb.append(random.choice(new_files))
    gall_img = []
    gall_id = []
    gall_cam = []
    for img_path in files_rgb:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        gall_img.append(img_path)
        gall_id.append(pid)
        gall_cam.append(camid)
    return gall_img, np.array(gall_id), np.array(gall_cam)
    
def process_test_regdb(img_dir, trial = 1, modal = 'visible'):
    if modal=='visible':
        input_data_path = img_dir + 'idx/test_visible_{}'.format(trial) + '.txt'
    elif modal=='thermal':
        input_data_path = img_dir + 'idx/test_thermal_{}'.format(trial) + '.txt'
    
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, np.array(file_label)
