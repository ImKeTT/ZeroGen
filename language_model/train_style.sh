CUDA_VISIBLE_DEVICES=0 python train.py\
    --model_name gpt2\
    --task flickr_humor\
    --train_path /mnt/data0/tuhq21/dataset/FlickrStyle/humor/humor_train6k.txt\
    --dev_path /mnt/data0/tuhq21/dataset/FlickrStyle/humor/humor_test1k.txt\
    --test_path /mnt/data0/tuhq21/dataset/FlickrStyle/humor/humor_test1k.txt\
    --add_eos_token_to_data True\
    --margin 0.5\
    --max_len 128\
    --number_of_gpu 1\
    --batch_size_per_gpu 25\
    --gradient_accumulation_steps 4\
    --effective_batch_size 100\
    --total_steps 6000\
    --print_every 500\
    --save_every 500\
    --learning_rate 1e-5 &&\
CUDA_VISIBLE_DEVICES=0 python train.py\
    --model_name gpt2\
    --task flickr_romantic\
    --train_path /mnt/data0/tuhq21/dataset/FlickrStyle/romantic/romantic_train6k.txt\
    --dev_path /mnt/data0/tuhq21/dataset/FlickrStyle/romantic/romantic_test1k.txt\
    --test_path /mnt/data0/tuhq21/dataset/FlickrStyle/romantic/romantic_test1k.txt\
    --add_eos_token_to_data True\
    --margin 0.5\
    --max_len 128\
    --number_of_gpu 1\
    --batch_size_per_gpu 25\
    --gradient_accumulation_steps 4\
    --effective_batch_size 100\
    --total_steps 6000\
    --print_every 500\
    --save_every 500\
    --learning_rate 1e-5