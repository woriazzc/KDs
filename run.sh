# train teacher
python -u main.py --dataset=gowalla --backbone=bpr --train_teacher

# KD
python -u main.py --dataset=gowalla --backbone=bpr --model=rrd