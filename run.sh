"""
For Recommendation Models
"""
# train teacher
python -u main.py --dataset=foursquare --backbone=bpr --train_teacher --teacher embedding_dim=800 --cfg wd=0.01 --suffix teacher
python -u main.py --dataset=foursquare --backbone=lightgcn --train_teacher --teacher embedding_dim=2000 num_layers=1 --cfg wd=1e-3 --suffix teacher

python -u main.py --dataset=citeulike --backbone=bpr --train_teacher --teacher embedding_dim=400 --cfg wd=1e-3 --suffix teacher
python -u main.py --dataset=citeulike --backbone=lightgcn --train_teacher --teacher embedding_dim=2000 num_layers=3 --cfg wd=1e-7 --suffix teacher
python -u main.py --dataset=citeulike --backbone=simplex --train_teacher --teacher embedding_dim=500 --cfg wd=1e-5 num_ns=2000 --suffix teacher
python -u main.py --dataset=citeulike --backbone=xsimgcl --train_teacher --suffix teacher

python -u main.py --dataset=gowalla --backbone=bpr --train_teacher --teacher embedding_dim=300 --cfg wd=1e-4 --suffix teacher
python -u main.py --dataset=gowalla --backbone=lightgcn --train_teacher --teacher embedding_dim=2000 num_layers=4 --cfg wd=1e-6 --suffix teacher
python -u main.py --dataset=gowalla --backbone=simplex --train_teacher --teacher embedding_dim=1000 --cfg wd=1e-8 num_ns=2000 --suffix teacher

python -u main.py --dataset=yelp --backbone=bpr --train_teacher --suffix teacher --teacher embedding_dim=300 --cfg wd=1e-4 --suffix teacher
python -u main.py --dataset=yelp --backbone=lightgcn --train_teacher --teacher embedding_dim=1000 num_layers=3 --cfg wd=1e-7 --suffix teacher
python -u main.py --dataset=yelp --backbone=simplex --train_teacher --teacher embedding_dim=500 --cfg wd=1e-8 num_ns=1000 --suffix teacher

# from scratch
python -u main.py --dataset=foursquare --backbone=bpr --model=scratch --student embedding_dim=10 --cfg wd=0.01 --suffix student
python -u main.py --dataset=foursquare --backbone=lightgcn --model=scratch --student embedding_dim=10 num_layers=1 --cfg wd=0.01 --suffix student

python -u main.py --dataset=citeulike --backbone=bpr --model=scratch --student embedding_dim=20 --cfg wd=1e-3 --suffix student
python -u main.py --dataset=citeulike --backbone=lightgcn --model=scratch --student embedding_dim=20 num_layers=3 --cfg wd=1e-3 --suffix student
python -u main.py --dataset=citeulike --backbone=simplex --model=scratch --student embedding_dim=20 --suffix student
python -u main.py --dataset=citeulike --backbone=xsimgcl --model=scratch --suffix student

python -u main.py --dataset=gowalla --backbone=bpr --model=scratch --student embedding_dim=20 --cfg wd=1e-3 --suffix student
python -u main.py --dataset=gowalla --backbone=lightgcn --model=scratch --student embedding_dim=20 num_layers=4 --cfg wd=1e-7 --suffix student
python -u main.py --dataset=gowalla --backbone=simplex --model=scratch --student embedding_dim=20 --suffix student

python -u main.py --dataset=yelp --backbone=bpr --model=scratch --student embedding_dim=20 --cfg wd=1e-3 --suffix student
python -u main.py --dataset=yelp --backbone=lightgcn --model=scratch --student embedding_dim=20 num_layers=3 --cfg wd=1e-4 --suffix student
python -u main.py --dataset=yelp --backbone=simplex --model=scratch --student embedding_dim=20 --suffix student

# KD
# For HetComp, you need to pre-save teacher checkpoints through:
python -u main.py --dataset=citeulike --backbone=bpr --train_teacher --teacher embedding_dim=400 --cfg wd=1e-3 --no_log --ckpt_interval=50
python -u main.py --dataset=citeulike --backbone=bpr --model=hetcomp
python -u main.py --dataset=citeulike --backbone=bpr --model=de
python -u main.py --dataset=citeulike --backbone=bpr --model=rrd
python -u main.py --dataset=citeulike --backbone=bpr --model=cpd
python -u main.py --dataset=citeulike --backbone=bpr --model=nkd
python -u main.py --dataset=citeulike --backbone=bpr --model=graphd
python -u main.py --dataset=citeulike --backbone=bpr --model=filterd
python -u main.py --dataset=citeulike --backbone=bpr --model=fd
python -u main.py --dataset=citeulike --backbone=bpr --model=gdcp
python -u main.py --dataset=citeulike --backbone=bpr --model=preld
python -u main.py --dataset=citeulike --backbone=bpr --model=vkd

python -u main.py --dataset=citeulike --backbone=lightgcn --model=graphd --cfg ablation=True --suffix ablation
python -u main.py --dataset=citeulike --backbone=lightgcn --model=graphd
python -u main.py --dataset=citeulike --backbone=lightgcn --model=de
python -u main.py --dataset=citeulike --backbone=lightgcn --model=cpd
python -u main.py --dataset=citeulike --backbone=lightgcn --model=rrd


python -u main.py --dataset=foursquare --backbone=bpr --model=de
python -u main.py --dataset=foursquare --backbone=bpr --model=cpd
python -u main.py --dataset=foursquare --backbone=bpr --model=nkd
python -u main.py --dataset=foursquare --backbone=bpr --model=graphd
python -u main.py --dataset=foursquare --backbone=bpr --model=filterd

python -u main.py --dataset=foursquare --backbone=lightgcn --model=nkd


python -u main.py --dataset=gowalla --backbone=bpr --model=de
python -u main.py --dataset=gowalla --backbone=bpr --model=rrd
python -u main.py --dataset=gowalla --backbone=bpr --model=cpd
python -u main.py --dataset=gowalla --backbone=bpr --model=graphd

python -u main.py --dataset=gowalla --backbone=lightgcn --model=graphd


python -u main.py --dataset=yelp --backbone=bpr --model=de
python -u main.py --dataset=yelp --backbone=bpr --model=cpd
python -u main.py --dataset=yelp --backbone=bpr --model=rrd



"""
For CTR Prediction Models
"""
# train teacher
python -u main_ctr.py --dataset=criteo --backbone=crossnet --train_teacher --suffix teacher
python -u main_ctr.py --dataset=criteo --backbone=dnn --train_teacher --suffix teacher
python -u main_ctr.py --dataset=criteo --backbone=dcnv2 --train_teacher --suffix teacher
python -u main_ctr.py --dataset=criteo --backbone=cin --train_teacher --suffix teacher
python -u main_ctr.py --dataset=criteo --backbone=xdeepfm --train_teacher --suffix teacher
python -u main_ctr.py --dataset=criteo --backbone=autoint --train_teacher --suffix teacher
python -u main_ctr.py --dataset=criteo --backbone=eulernet --train_teacher --suffix teacher
python -u main_ctr.py --dataset=criteo --backbone=gatecrossnet --train_teacher --suffix teacher

# from scratch
python -u main_ctr.py --dataset=criteo --backbone=crossnet --model=scratch --suffix student --no_save
python -u main_ctr.py --dataset=criteo --backbone=dnn --model=scratch --suffix student --no_save
python -u main_ctr.py --dataset=criteo --backbone=dcnv2 --model=scratch --suffix student --no_save
python -u main_ctr.py --dataset=criteo --backbone=gatecrossnet --model=scratch --suffix student --no_save

# KD
python -u main_ctr.py --dataset=criteo --backbone=crossnet --model=hetd
python -u main_ctr.py --dataset=criteo --backbone=crossnet --model=rkd
python -u main_ctr.py --dataset=criteo --backbone=crossnet --model=bced
python -u main_ctr.py --dataset=criteo --backbone=crossnet --model=fitnet
python -u main_ctr.py --dataset=criteo --backbone=crossnet --model=clid
python -u main_ctr.py --dataset=criteo --backbone=crossnet --model=paird
python -u main_ctr.py --dataset=criteo --backbone=crossnet --model=anyd
python -u main_ctr.py --dataset=criteo --backbone=crossnet --model=adad
python -u main_ctr.py --dataset=criteo --backbone=crossnet --model=ofa

python -u main_ctr.py --dataset=criteo --backbone=dnn --model=bced
python -u main_ctr.py --dataset=criteo --backbone=dnn --model=fitnet
python -u main_ctr.py --dataset=criteo --backbone=dnn --model=hetd

python -u main_ctr.py --dataset=criteo --backbone=dcnv2 --model=bced
python -u main_ctr.py --dataset=criteo --backbone=dcnv2 --model=clid
python -u main_ctr.py --dataset=criteo --backbone=dcnv2 --model=hetd

python -u main_ctr.py --dataset=criteo --backbone=cin --model=fitnet
python -u main_ctr.py --dataset=criteo --backbone=cin --model=hetd
python -u main_ctr.py --dataset=criteo --backbone=cin --model=warmup

python -u main_ctr.py --dataset=criteo --backbone=gatecrossnet --model=fitnet
python -u main_ctr.py --dataset=criteo --backbone=gatecrossnet --model=bced