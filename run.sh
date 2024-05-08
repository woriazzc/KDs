# train teacher
python -u main.py --dataset=foursquare --backbone=bpr --train_teacher --teacher embedding_dim=800 --cfg wd=0.01 --suffix teacher
python -u main.py --dataset=foursquare --backbone=lightgcn --train_teacher --teacher embedding_dim=2000 num_layers=1 --cfg wd=1e-3 --suffix teacher

python -u main.py --dataset=citeulike --backbone=bpr --train_teacher --teacher embedding_dim=400 --cfg wd=1e-3 --suffix teacher
python -u main.py --dataset=citeulike --backbone=lightgcn --train_teacher --teacher embedding_dim=2000 num_layers=3 --cfg wd=1e-7 --suffix teacher

python -u main.py --dataset=gowalla --backbone=bpr --train_teacher --teacher embedding_dim=300 --cfg wd=1e-4 --suffix teacher
python -u main.py --dataset=gowalla --backbone=lightgcn --train_teacher --teacher embedding_dim=2000 num_layers=4 --cfg wd=1e-6 --suffix teacher

python -u main.py --dataset=music --backbone=bpr --train_teacher --suffix teacher

python -u main.py --dataset=books --backbone=bpr --train_teacher --suffix teacher

python -u main.py --dataset=yelp --backbone=bpr --train_teacher --suffix teacher --teacher embedding_dim=300 --cfg wd=1e-4 --suffix teacher
python -u main.py --dataset=yelp --backbone=lightgcn --train_teacher --suffix teacher

#from scratch
python -u main.py --dataset=foursquare --backbone=bpr --model=scratch --student embedding_dim=10 --cfg wd=0.01 --suffix student
python -u main.py --dataset=foursquare --backbone=lightgcn --model=scratch --student embedding_dim=10 num_layers=1 --cfg wd=0.01 --suffix student

python -u main.py --dataset=citeulike --backbone=bpr --model=scratch --student embedding_dim=20 --cfg wd=1e-3 --suffix student
python -u main.py --dataset=citeulike --backbone=lightgcn --model=scratch --student embedding_dim=20 num_layers=3 --cfg wd=1e-3 --suffix student
python -u main.py --dataset=citeulike --backbone=simplex --model=scratch --student embedding_dim=20 --suffix student

python -u main.py --dataset=gowalla --backbone=bpr --model=scratch --student embedding_dim=20 --cfg wd=1e-3 --suffix student
python -u main.py --dataset=gowalla --backbone=lightgcn --model=scratch --student embedding_dim=20 num_layers=4 --cfg wd=1e-7 --suffix student

python -u main.py --dataset=yelp --backbone=bpr --model=scratch --student embedding_dim=20 --cfg wd=1e-3 --suffix student

# KD
# For HetComp, you need to pre-save teacher checkpoints through:
python -u main.py --dataset=citeulike --backbone=bpr --train_teacher --teacher embedding_dim=400 --cfg wd=1e-3 --no_log --ckpt_interval=50
python -u main.py --dataset=citeulike --backbone=bpr --model=hetcomp --run_all
python -u main.py --dataset=citeulike --backbone=bpr --model=de --run_all

python -u main.py --dataset=citeulike --backbone=bpr --model=nkd
python -u main.py --dataset=citeulike --backbone=bpr --model=graphd
python -u main.py --dataset=citeulike --backbone=bpr --model=filterd
python -u main.py --dataset=citeulike --backbone=bpr --model=fd
python -u main.py --dataset=citeulike --backbone=bpr --model=gdcp

python -u main.py --dataset=citeulike --backbone=lightgcn --model=graphd --cfg ablation=True --suffix ablation
python -u main.py --dataset=citeulike --backbone=lightgcn --model=graphd


python -u main.py --dataset=foursquare --backbone=bpr --model=de
python -u main.py --dataset=foursquare --backbone=bpr --model=cpd
python -u main.py --dataset=foursquare --backbone=bpr --model=nkd
python -u main.py --dataset=foursquare --backbone=bpr --model=graphd
python -u main.py --dataset=foursquare --backbone=bpr --model=filterd

python -u main.py --dataset=foursquare --backbone=lightgcn --model=nkd


python -u main.py --dataset=gowalla --backbone=bpr --model=graphd