import glob
import socket
import timeit
from collections import OrderedDict
from datetime import datetime

import scipy.misc as sm
# PyTorch includes
import torch.optim as optim
# Tensorboard include
from tensorboardX import SummaryWriter
from torch.nn.functional import upsample
from torch.utils.data import DataLoader
from torchvision import transforms

import dataloaders.pascal as pascal
import dataloaders.sbd as sbd
import networks.deeplab_resnet as resnet
from dataloaders import custom_transforms as tr
# Custom includes
from dataloaders.combine_dbs import CombineDBs as combine_dbs
from dataloaders.helpers import *
from layers.loss import class_balanced_cross_entropy_loss

import os.path

from torch.utils.data import DataLoader

import dataloaders.pascal as pascal
from evaluation.eval import eval_one_result
import argparse
parser = argparse.ArgumentParser(description='PyTorch DEXTNET TESTING')

parser.add_argument('--model_path', type=str)
parser.add_argument('--save_dir_res', type=str)
parser.add_argument('--res_dir',type=str)
args = parser.parse_args()

# Setting parameters
use_sbd = False
nEpochs = 100  # Number of epochs for training
resume_epoch = 100  # Default is 0, change if want to resume


p = OrderedDict()  # Parameters to include in report
classifier = 'psp'  # Head classifier to use
useTest = 1  # See evolution of the test set when training?
nTestInterval = 10  # Run on test set every nTestInterval epochs
snapshot = 20  # Store a model every snapshot epochs
relax_crop = 50  # Enlarge the bounding box by relax_crop pixels
nInputChannels = 4  # Number of input channels (RGB + heatmap of extreme points)
zero_pad_crop = True  # Insert zero padding when cropping the image




composed_transforms_ts = transforms.Compose([
    tr.CropFromMask(crop_elems=('image', 'gt'), relax=relax_crop, zero_pad=zero_pad_crop),
    tr.FixedResize(resolutions={'gt': None, 'crop_image': (512, 512), 'crop_gt': (512, 512)}),
    tr.ExtremePoints(sigma=10, pert=30, elem='crop_gt', num_pts=50, type='polygon', vis=False),
    tr.ToImage(norm_elem='extreme_points'),
    tr.ConcatInputs(elems=('crop_image', 'extreme_points')),
    tr.ToTensor()])
db_test = pascal.VOCSegmentation(split='val', transform=composed_transforms_ts, retname=True)
testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=8)

modelName = 'dextr_pascal'
net = resnet.resnet101(1, pretrained=True, nInputChannels=nInputChannels, classifier=classifier)

net = torch.nn.DataParallel(net.cuda())

# net = net.cuda()
print("Initializing weights from: {}".format(
        args.model_path))
net.load_state_dict(
        torch.load(os.path.join(args.model_path, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth'),
                   map_location=lambda storage, loc: storage))
import ipdb
ipdb.set_trace()
save_dir_res = os.path.join(args.save_dir_res, args.res_dir)

if not os.path.exists(save_dir_res):
    os.makedirs(save_dir_res)
    print('Testing Network')
    net.eval()
    import time
    begin = time.time()
    with torch.no_grad():
        # Main Testing Loop
        for ii, sample_batched in enumerate(testloader):

            inputs, gts, metas = sample_batched['concat'], sample_batched['gt'], sample_batched['meta']

            # Forward of the mini-batch
            inputs = inputs.cuda()

            outputs = net.forward(inputs)
            outputs = upsample(outputs, size=(512, 512), mode='bilinear', align_corners=True)
            outputs = outputs.to(torch.device('cpu'))

            if ii % 100 == 0:
                print('Batch {} LEN {}, cost_time {}'.format(ii, len(testloader), int(time.time()-begin)))
                begin = time.time()
            for jj in range(int(inputs.size()[0])):
                pred = np.transpose(outputs.data.numpy()[jj, :, :, :], (1, 2, 0))
                pred = 1 / (1 + np.exp(-pred))
                pred = np.squeeze(pred)
                gt = tens2image(gts[jj, :, :, :])
                bbox = get_bbox(gt, pad=relax_crop, zero_pad=zero_pad_crop)
                result = crop2fullmask(pred, bbox, gt, zero_pad=zero_pad_crop, relax=relax_crop)

                # Save the result, attention to the index jj
                sm.imsave(os.path.join(save_dir_res, metas['image'][jj] + '-' + metas['object'][jj] + '.png'), result)



exp_root_dir = './'

method_names = []
method_names.append(args.save_dir_res)



    # Dataloader
dataset = pascal.VOCSegmentation(transform=None, retname=True)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

# Iterate through all the different methods
for method in method_names:
    results_folder = os.path.join(exp_root_dir, method, args.res_dir)

    filename = os.path.join(exp_root_dir, 'eval_results', method.replace('/', '-') + '.txt')
    if not os.path.exists(os.path.join(exp_root_dir, 'eval_results')):
        os.makedirs(os.path.join(exp_root_dir, 'eval_results'))

    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            val = float(f.read())
    else:
        print("Evaluating method: {}".format(method))
        jaccards = eval_one_result(dataloader, results_folder, mask_thres=0.8)
        val = jaccards["all_jaccards"].mean()

    # Show mean and store result
    print("Result for {:<80}: {}".format(method, str.format("{0:.1f}", 100*val)))
    with open(filename, 'w') as f:
        f.write(str(val))
