# train teacher
python -u main.py --dataset=foursquare --backbone=bpr --train_teacher --teacher embedding_dim=800 --cfg wd=0.01 --suffix teacher
python -u main.py --dataset=foursquare --backbone=lightgcn --train_teacher --teacher embedding_dim=2000 --cfg wd=1e-3 num_layers=1 --suffix teacher

python -u main.py --dataset=citeulike --backbone=bpr --train_teacher --teacher embedding_dim=400 --cfg wd=1e-3 --suffix teacher
python -u main.py --dataset=citeulike --backbone=lightgcn --train_teacher --teacher embedding_dim=2000 --cfg wd=1e-7 num_layers=3 --suffix teacher

python -u main.py --dataset=gowalla --backbone=bpr --train_teacher
python -u main.py --dataset=gowalla --backbone=lightgcn --train_teacher

#from scratch
python -u main.py --dataset=foursquare --backbone=bpr --train_teacher --teacher embedding_dim=10 --cfg wd=0.01 --suffix student
python -u main.py --dataset=foursquare --backbone=lightgcn --train_teacher --teacher embedding_dim=10 --cfg wd=0.01 num_layers=1 --suffix student

python -u main.py --dataset=citeulike --backbone=bpr --train_teacher --teacher embedding_dim=20 --cfg wd=1e-3 --suffix student
python -u main.py --dataset=citeulike --backbone=lightgcn --train_teacher --teacher embedding_dim=20 --cfg wd=1e-3 num_layers=3 --suffix student

# KD
python -u main.py --dataset=citeulike --backbone=bpr --model=nkd


python -u main.py --dataset=foursquare --backbone=bpr --model=de
python -u main.py --dataset=foursquare --backbone=bpr --model=cpd
python -u main.py --dataset=foursquare --backbone=bpr --model=nkd
python -u main.py --dataset=foursquare --backbone=bpr --model=graphd
python -u main.py --dataset=foursquare --backbone=bpr --model=filterd

python -u main.py --dataset=foursquare --backbone=lightgcn --model=nkd


python -u main.py --dataset=gowalla --backbone=bpr --model=rrd