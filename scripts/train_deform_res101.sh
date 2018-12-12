export PYTHONPATH=/home/chenyifeng/TensorflowModelZoo/
nohup python3 TensorflowModelZoo/trainings/train_dfm_res101.py --valid_per_epoch 1 \
	--batch_size 8  --max_iter_epoch 8 --end_learning_rate 1e-5 \
	--summaries_dir volume/TF_Logs/TensorflowModelZoo/resnet101_deform/ \
	--last_ckpt volume/TF_Models/atrain/TensorflowModelZoo/resnet101_deform/ \
	--next_ckpt volume/TF_Models/atrain/TensorflowModelZoo/resnet101_deform/ \
	--offset_learning_ratio 20 --weight_learning_rate 1e-3 >resnet_deform_101.log 2>&1 &