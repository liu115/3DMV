
import argparse
import os, sys, inspect, time
import glob
import random
import torch
import torchnet as tnt
import numpy as np
import itertools

import util
import data_util
from model import Model3d

# ENET_TYPES = {'scannet': (41, [0.496342, 0.466664, 0.440796], [0.277856, 0.28623, 0.291129])}  #classes, color mean/std 

# params
parser = argparse.ArgumentParser()
# data paths

parser.add_argument('--model_dir', required=True, help='path to validate models')
parser.add_argument('--train_data_list', required=True, help='path to file list of h5 train data')
parser.add_argument('--val_data_list', default='', help='path to file list of h5 val data')
parser.add_argument('--output', default='./logs', help='folder to output model checkpoints')
# parser.add_argument('--data_path_2d', required=True, help='path to 2d train data')
parser.add_argument('--class_weight_file', default='', help='path to histogram over classes')
# train params
parser.add_argument('--num_classes', default=42, help='#classes')
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
parser.add_argument('--max_epoch', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum, default=0.9')
# parser.add_argument('--num_nearest_images', type=int, default=3, help='#images')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay, default=0.0005')
parser.add_argument('--retrain', default='', help='model to load')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
# parser.add_argument('--model2d_type', default='scannet', help='which enet (scannet)')
# parser.add_argument('--model2d_path', required=True, help='path to enet model')
# parser.add_argument('--use_proxy_loss', dest='use_proxy_loss', action='store_true')
# 2d/3d
parser.add_argument('--selected_input_channel', nargs='+', type=int, required=True)
parser.add_argument('--voxel_size', type=float, default=0.05, help='voxel size (in meters)')
parser.add_argument('--grid_dimX', type=int, default=31, help='3d grid dim x')
parser.add_argument('--grid_dimY', type=int, default=31, help='3d grid dim y')
parser.add_argument('--grid_dimZ', type=int, default=62, help='3d grid dim z')
parser.add_argument('--depth_min', type=float, default=0.4, help='min depth (in meters)')
parser.add_argument('--depth_max', type=float, default=4.0, help='max depth (in meters)')
# scannet intrinsic params
parser.add_argument('--intrinsic_image_width', type=int, default=640, help='2d image width')
parser.add_argument('--intrinsic_image_height', type=int, default=480, help='2d image height')
parser.add_argument('--fx', type=float, default=577.870605, help='intrinsics')
parser.add_argument('--fy', type=float, default=577.870605, help='intrinsics')
parser.add_argument('--mx', type=float, default=319.5, help='intrinsics')
parser.add_argument('--my', type=float, default=239.5, help='intrinsics')

parser.set_defaults(use_proxy_loss=False)
opt = parser.parse_args()
print(opt)

# specify gpu
os.environ['CUDA_VISIBLE_DEVICES']=str(opt.gpu)

# Parse global parameters
grid_dims = [opt.grid_dimX, opt.grid_dimY, opt.grid_dimZ]
column_height = opt.grid_dimZ
batch_size = opt.batch_size
grid_centerX = opt.grid_dimX // 2
grid_centerY = opt.grid_dimY // 2

# create model
num_classes = opt.num_classes
model = Model3d(num_classes, grid_dims, len(opt.selected_input_channel))
print(model)

# create loss
criterion_weights = torch.ones(num_classes) 
if opt.class_weight_file:
    # Load loss weights for classes base on the label counts
    criterion_weights = util.read_class_weights_from_file(opt.class_weight_file, num_classes, True)
for c in range(num_classes):
    if criterion_weights[c] > 0:
        criterion_weights[c] = 1 / np.log(1.2 + criterion_weights[c])
print(criterion_weights.numpy())
criterion = torch.nn.CrossEntropyLoss(criterion_weights).cuda()

# move to gpu
model = model.cuda()
criterion = criterion.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

# data files
train_files = util.read_lines_from_file(opt.train_data_list)
val_files = [] if not opt.val_data_list else util.read_lines_from_file(opt.val_data_list)
print('#train files = ', len(train_files))
print('#val files = ', len(val_files))

_SPLITTER = ','
confusion_val = tnt.meter.ConfusionMeter(num_classes)

def test(epoch, iter, log_file, val_file):
    test_loss = []
    model.eval()
    start = time.time()

    selected_input_channel = opt.selected_input_channel
    volumes, labels = data_util.load_hdf5_data(val_file, num_classes, selected_input_channel)
    
    volumes = volumes.permute(0, 1, 4, 3, 2)
    labels = labels.permute(0, 1, 4, 3, 2)
    labels = labels[:, 0, :, grid_centerX, grid_centerY]  # center columns as targets
    num_samples = volumes.shape[0]

    # shuffle
    indices = torch.randperm(num_samples).long().split(batch_size)
    # remove last mini-batch so that all the batches have equal size
    indices = indices[:-1]

    with torch.no_grad():
        mask = torch.cuda.LongTensor(batch_size*column_height)

        for t,v in enumerate(indices):
            targets = labels[v].cuda()
            # valid targets
            mask = targets.view(-1).data.clone()
            for k in range(num_classes):
                if criterion_weights[k] == 0:
                    mask[mask.eq(k)] = 0
            maskindices = mask.nonzero().squeeze()
            if len(maskindices.shape) == 0:
                continue

            # 2d/3d
            input3d = volumes[v].cuda()
            output = model(input3d)
            loss = criterion(output.view(-1, num_classes), targets.view(-1))
            test_loss.append(loss.item())
            
            # confusion
            y = output.data
            y = y.view(y.nelement() // y.size(2), num_classes)[:, :-1]
            _, predictions = y.max(1)
            predictions = predictions.view(-1)
            k = targets.data.view(-1)
            confusion_val.add(torch.index_select(predictions, 0, maskindices), torch.index_select(k, 0, maskindices))

    end = time.time()
    took = end - start
    evaluate_confusion(confusion_val, test_loss, epoch, iter, took, 'Test', log_file)
    return test_loss


def evaluate_confusion(confusion_matrix, loss, epoch, iter, time, which, log_file):
    '''
    confusion_matrix: a torchnet meter ConfusionMeter
    '''
    conf = confusion_matrix.value()  # return average precision for each class, shape: (1, K)
    total_correct = 0
    valids = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        num = conf[c,:].sum()
        valids[c] = -1 if num == 0 else float(conf[c][c]) / float(num)
        total_correct += conf[c][c]
    instance_acc = -1 if conf.sum() == 0 else float(total_correct) / float(conf.sum())
    avg_acc = -1 if np.all(np.equal(valids, -1)) else np.mean(valids[np.not_equal(valids, -1)])
    log_file.write(_SPLITTER.join([str(f) for f in [epoch, iter, torch.mean(torch.Tensor(loss)), avg_acc, instance_acc, time]]) + '\n')
    log_file.flush()

    print('{} Epoch: {}\tIter: {}\tLoss: {:.6f}\tAcc(inst): {:.6f}\tAcc(avg): {:.6f}\tTook: {:.2f}'.format(
        which, epoch, iter, torch.mean(torch.Tensor(loss)).item(), instance_acc, avg_acc, time))


def main():
    if not os.path.exists(opt.output):
        os.makedirs(opt.output)

    has_val = len(val_files) > 0
    if has_val:
        log_file_val = open(os.path.join(opt.output, 'log_val.csv'), 'w')
        log_file_val.write(_SPLITTER.join(['epoch', 'iter', 'loss','avg acc', 'instance acc', 'time']) + '\n')
        log_file_val.flush()

    for model_path in glob.glob(os.path.join(opt.model_dir, 'model-epoch-*.pth')):
        model.load_state_dict(torch.load(model_path))

        val_loss = []
        start_time = time.time()
        for val_file in val_files:
            loss = test(0, 0, log_file_val, val_file)
            val_loss.extend(loss)
        took = time.time() - start_time
        evaluate_confusion(confusion_val, val_loss, 0, 0, took, model_path, log_file_val)
        confusion_val.reset()
    log_file_val.close()

    # # start training
    # print('starting training...')
    # iter = 0
    # num_files_per_val = 10
    # for epoch in range(opt.max_epoch):
    #     train_loss = []
    #     # train2d_loss = []
    #     # val2d_loss = []
    #     # go thru shuffled train files
    #     train_file_indices = torch.randperm(len(train_files))
    #     for k in range(len(train_file_indices)):
    #         print('Epoch: {}\tFile: {}/{}\t{}'.format(epoch, k, len(train_files), train_files[train_file_indices[k]]))
    #         loss, iter = train(epoch, iter, log_file, train_files[train_file_indices[k]])
    #         train_loss.extend(loss)
    #         if has_val and k % num_files_per_val == 0:
    #             val_index = torch.randperm(len(val_files))[0]
    #             loss = test(epoch, iter, log_file_val, val_files[val_index])
    #             val_loss.extend(loss)
    #     evaluate_confusion(confusion, train_loss, epoch, iter, -1, 'Train', log_file)
    #     if has_val:
    #         evaluate_confusion(confusion_val, val_loss, epoch, iter, -1, 'Test', log_file_val)

    #     torch.save(model.state_dict(), os.path.join(opt.output, 'model-epoch-%s.pth' % epoch))
    #     confusion.reset()
    #     confusion_val.reset()
    # log_file.close()
    # if has_val:
    #     log_file_val.close()


if __name__ == '__main__':
    main()


