import argparse
import os, sys, inspect, time
import random
import torch
import torchnet as tnt
import torch.nn.functional as F
import numpy as np

import util
import data_util
from model import Model3d

# ENET_TYPES = {'scannet': (41, [0.496342, 0.466664, 0.440796], [0.277856, 0.28623, 0.291129])}  #classes, color mean/std 

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--scene_list', required=True, help='path to file list of scenes to test')
parser.add_argument('--model_path', required=True, help='path to model')
# parser.add_argument('--data_path_2d', required=True, help='path to 2d data')
parser.add_argument('--data_path_3d', required=True, help='path to 3d data')
parser.add_argument('--has_gt', type=int, default=0, help='test scenes have gt to evaluate against')
parser.add_argument('--output_path', default='./output', help='output path')
# test params
parser.add_argument('--num_classes', default=42, help='#classes')
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
# parser.add_argument('--num_nearest_images', type=int, required=True, help='#images')
parser.add_argument('--model2d_type', default='scannet', help='which enet (scannet)')
#parser.add_argument('--test_2d_model', dest='test_2d_model', action='store_true')
# parser.add_argument('--model2d_orig_path', required=True, help='path to model')
# 2d/3d 
parser.add_argument('--selected_input_channel',
                    nargs='+', type=int, required=True)
parser.add_argument('--voxel_size', type=float,
                    default=0.05, help='voxel size (in meters)')
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

parser.set_defaults(train_2d_model=False)
opt = parser.parse_args()
# assert opt.model2d_type in ENET_TYPES
print(opt)

# specify gpu
os.environ['CUDA_VISIBLE_DEVICES']=str(opt.gpu)

# create camera intrinsics
# input_image_dims = [328, 256]
# proj_image_dims = [41, 32]
# intrinsic = util.make_intrinsic(opt.fx, opt.fy, opt.mx, opt.my)
# intrinsic = util.adjust_intrinsic(intrinsic, [opt.intrinsic_image_width, opt.intrinsic_image_height], proj_image_dims)
# intrinsic = intrinsic.cuda()
grid_dims = [opt.grid_dimX, opt.grid_dimY, opt.grid_dimZ]
column_height = opt.grid_dimZ
# num_images = opt.num_nearest_images
grid_padX = opt.grid_dimX // 2
grid_padY = opt.grid_dimY // 2
# color_mean = ENET_TYPES[opt.model2d_type][1]
# color_std = ENET_TYPES[opt.model2d_type][2]

#TODO READ THIS FROM FILE INSTEAD OF HARDCODING
print('warning: using hard-coded scannet label set')
valid_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
gt_class_dict = np.array([0] + valid_classes)

# create model
num_classes = opt.num_classes
model = Model3d(num_classes, grid_dims, len(opt.selected_input_channel))
model.load_state_dict(torch.load(opt.model_path))
print(model)

# move to gpu
model = model.cuda()
model.eval()

# data files
scenes = util.read_lines_from_file(opt.scene_list)
print('#scenes = ', len(scenes))
if opt.has_gt:
    print('evaluating test scenes')
else:
    print('running model over test scenes (no evaluation)')


_SPLITTER = ','
def evaluate_prediction(scene_occ, scene_label, output):
    print(scene_occ.shape, scene_label.shape, output.shape)
    mask = np.equal(scene_occ[0], 1)
    output[np.logical_not(mask)] = 0

    # Inst_acc should mask out label == 0
    mask = np.logical_and(mask, np.not_equal(scene_label, num_classes-1))
    mask = np.logical_and(mask, np.not_equal(scene_label, 0))
    num_wrong = np.count_nonzero(scene_label.astype(np.int32)[mask] - output.astype(np.int32)[mask])
    inst_num_occ = np.count_nonzero(mask)
    inst_num_correct = inst_num_occ - num_wrong
    # class stats
    class_num_correct = np.zeros(num_classes)
    class_num_occ = np.zeros(num_classes)
    class_num_union = np.zeros(num_classes)
    for c in range(num_classes):
        if not c in valid_classes:
            continue
        mask = np.equal(scene_label, c)
        if np.any(mask):
            class_num_occ[c] = np.count_nonzero(mask)
            num_wrong = np.count_nonzero(scene_label.astype(np.int32)[mask] - output.astype(np.int32)[mask])
            class_num_correct[c] = class_num_occ[c] - num_wrong
            
            legal_ann = np.logical_and(np.not_equal(scene_label, num_classes-1), np.not_equal(scene_label, 0))
            tp = np.logical_and(np.equal(output, c), np.equal(scene_label, c))
            fp = np.logical_and(np.equal(output, c), np.logical_and(legal_ann, np.not_equal(scene_label, c)))
            fn = np.logical_and(np.not_equal(output, c), np.equal(scene_label, c))

            assert tp.astype(int).sum() == class_num_correct[c]
            class_num_union[c] = (tp.astype(int).sum() + fp.astype(int).sum() + fn.astype(int).sum())

    print('instance acc = ', float(inst_num_correct)/float(inst_num_occ))
    class_acc = np.divide(class_num_correct, class_num_occ)
    class_iou = np.divide(class_num_correct, class_num_union)
    print('class_acc', np.nanmean(class_acc), class_acc)
    print('class_iou', np.nanmean(class_iou), class_iou)
    return {'instance_num_correct':inst_num_correct, 'instance_num_total':inst_num_occ,'class_num_correct':class_num_correct, 'class_num_total':class_num_occ, 'class_num_union': class_num_union}


def test(scene_name, eval_file):
    print('scene', scene_name)
    scene_file = os.path.join(opt.data_path_3d, scene_name)
    # scene_image_file = os.path.join(opt.data_path_3d, scene_name + '.image')
    if not os.path.exists(scene_file):
        print(scene_file, os.path.exists(scene_file))
        # print(scene_image_file, os.path.exists(scene_image_file))
        raise

    scene_occ, scene_label = data_util.load_scene(scene_file, num_classes, opt.has_gt)
    
    # Padding X-Y
    padSize = ((0, 0), (0, 0), (grid_padX, grid_padX), (grid_padY, grid_padY))
    scene_occ = np.pad(scene_occ, padSize, 'constant', constant_values=0)
    if opt.has_gt:
        padSize = ((0, 0), (grid_padX, grid_padX), (grid_padY, grid_padY))
        scene_label = np.pad(scene_label, padSize, 'constant', constant_values=0)

    # Trim height (Z-dim)
    if scene_occ.shape[1] > column_height:
        scene_occ = scene_occ[:, :column_height, :, :]
        if opt.has_gt:
            scene_label = scene_label[:column_height, :, :]
    scene_occ_sz = scene_occ.shape[1:]
    
    # transform label from 0-20 to 0-40
    if opt.has_gt:
        scene_label = gt_class_dict[scene_label]
        print(scene_label.shape)

    # Mask volume channels
    selected_input_channel = np.array(opt.selected_input_channel)
    scene_occ = scene_occ[selected_input_channel, :, :, :]

    # B , C , Z , X , Y
    input_occ = torch.cuda.FloatTensor(
        1, len(opt.selected_input_channel), grid_dims[2], grid_dims[1], grid_dims[0])
    output_probs = np.zeros([num_classes, scene_occ_sz[0], scene_occ_sz[1], scene_occ_sz[2]])

    # go thru all columns
    for y in range(grid_padY, scene_occ_sz[1] - grid_padY):
        for x in range(grid_padX, scene_occ_sz[2] - grid_padX):
            input_occ.fill_(0)
            input_occ[0, :, :scene_occ_sz[0], :, :] = torch.from_numpy(
                scene_occ[:, :, y-grid_padY:y+grid_padY+1, x-grid_padX:x+grid_padX+1]
            )
            
            
            # cur_frame_ids = frame_ids[:, y, x][np.greater_equal(frame_ids[:, y, x], 0)]
            # if len(cur_frame_ids) < num_images or torch.sum(input_occ[0, 0, :,grid_padY,grid_padX]) == 0:
            #     continue
            # for k in range(num_images):
            #     depth_image[k] = depth_images[cur_frame_ids[k]]
            #     color_image[k] = color_images[cur_frame_ids[k]]
            #     pose[k] = poses[cur_frame_ids[k]]
            #     world_to_grid[k] = torch.from_numpy(world_to_grids[y, x])

            # proj_mapping = [projection.compute_projection(d, c, t) for d, c, t in zip(depth_image, pose, world_to_grid)]
            # if None in proj_mapping: #invalid sample
                #print('(invalid sample)')
                # continue
            # proj_mapping = list(zip(*proj_mapping))
            # proj_ind_3d = torch.stack(proj_mapping[0])
            # proj_ind_2d = torch.stack(proj_mapping[1])
            # imageft_fixed = model2d_fixed(torch.autograd.Variable(color_image))
            # imageft = model2d_trainable(imageft_fixed)
            input3d = torch.autograd.Variable(input_occ)
            out = model(input3d)
            # out = model(torch.autograd.Variable(input_occ), imageft, torch.autograd.Variable(proj_ind_3d), torch.autograd.Variable(proj_ind_2d), grid_dims)
            output = out.data[0].permute(1, 0)
            output_probs[:, :, y, x] = output[:, :scene_occ_sz[0]]
        sys.stdout.write('\r[ %d | %d ]' % (y + 1, scene_occ_sz[1] - grid_padY))
        sys.stdout.flush()
    sys.stdout.write('\n')

    pred_label = np.argmax(output_probs, axis=0)
    mask = np.equal(scene_occ[0], 1)
    pred_label[np.logical_not(mask)] = 0
    util.write_array_to_file(pred_label.astype(np.uint8), os.path.join(opt.output_path, scene_name + '.npy'))
    eval_scene = None
    if opt.has_gt:
        eval_scene = evaluate_prediction(scene_occ, scene_label, pred_label)
    return eval_scene


def main():
    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)
    eval_file = None
    if opt.has_gt:
        eval_file = open(os.path.join(opt.output_path, 'eval.csv'), 'w')
        header_fields = ['scene']
        for c in valid_classes:
            header_fields.append('#corr class ' + str(c))
        for c in valid_classes:
            header_fields.append('#total class ' + str(c))
        for c in valid_classes:
            header_fields.append('#union class ' + str(c))
        header_fields.extend(['instance #corr', 'instance #total'])
        eval_file.write(_SPLITTER.join(header_fields) + '\n')

    # start testing
    inst_total_correct = 0
    inst_total_occ = 0
    class_total_correct = np.zeros(num_classes)
    class_total_occ = np.zeros(num_classes)
    class_total_union = np.zeros(num_classes)
    for scene in scenes:
        stats = test(scene, eval_file)
        if opt.has_gt:
            inst_total_correct += stats['instance_num_correct']
            inst_total_occ += stats['instance_num_total']
            class_total_correct += stats['class_num_correct']
            class_total_occ += stats['class_num_total']
            class_total_union += stats['class_num_union']
            fields = [scene]
            for c in valid_classes:
                fields.append(stats['class_num_correct'][c])
            for c in valid_classes:
                fields.append(stats['class_num_total'][c])
            for c in valid_classes:
                fields.append(stats['class_num_union'][c])
            fields.extend([stats['instance_num_correct'], stats['instance_num_total']])
            eval_file.write(_SPLITTER.join([str(f) for f in fields]) + '\n')
    
    if opt.has_gt:
        # summary stats
        instance_acc = float(inst_total_correct)/float(inst_total_occ)
        class_acc = np.divide(class_total_correct, class_total_occ)
        class_iou = np.divide(class_total_correct, class_total_union)
        # summary stats header
        fields = ['SUMMARY']
        for c in valid_classes:
            fields.append('class ' + str(c))
        eval_file.write(_SPLITTER.join([str(f) for f in fields]) + '\n')
        fields = ['%acc']
        for c in valid_classes:
            fields.append(class_acc[c])
        eval_file.write(_SPLITTER.join([str(f) for f in fields]) + '\n')
        fields = ['iou']
        for c in valid_classes:
            fields.append(class_iou[c])
        eval_file.write(_SPLITTER.join([str(f) for f in fields]) + '\n')
        fields = ['instance acc', str(instance_acc)]
        eval_file.write(_SPLITTER.join([str(f) for f in fields]) + '\n')
        eval_file.close()


if __name__ == '__main__':
    main()


