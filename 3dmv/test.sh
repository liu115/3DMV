python test.py \ 
    --gpu 3 \
    --selected_input_channel 0 \
    --scene_list val_scene_list \
    --data_path_3d ../data/voxel_scenes/val \
    --has_gt \
    --model_path ./train_3d_geo/model-epoch-15.pth
