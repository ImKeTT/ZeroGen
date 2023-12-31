CUDA_VISIBLE_DEVICES=0 python train.py\
    --task visnews\
    --model_name /mnt/data0/tuhq21/news_writer/ctgsrc/language_model/output/simctg_visnews_dev_ppl_35.904\
    --train_path /mnt/data0/tuhq21/dataset/visnews/origin/train_bert_4topics_caps.txt\
    --dev_path /mnt/data0/tuhq21/dataset/visnews/origin/val_bert_4topics_caps.txt\
    --test_path /mnt/data0/tuhq21/dataset/visnews/origin/val_bert_4topics_caps.txt\
    --add_eos_token_to_data True\
    --margin 0.5\
    --max_len 128\
    --number_of_gpu 1\
    --batch_size_per_gpu 25\
    --gradient_accumulation_steps 4\
    --effective_batch_size 100\
    --total_steps 50000\
    --print_every 2000\
    --save_every 2000\
    --learning_rate 3e-5