python train.py \
    --gpu 3 \
    --selected_input_channel 0 \
    --train_data_list ./train_list.txt \
    --val_data_list ./val_list.txt \
    --class_weight_file ../../../../scannetv2_train_data/counts.txt \
    --output ./train_3d_geo \
    --batch_size 32

