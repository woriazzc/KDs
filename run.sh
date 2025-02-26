"""
For Recommendation Models
"""
# train teacher
python -u main.py --task=rec --dataset=foursquare --S_backbone=bpr --train_teacher --teacher embedding_dim=800 --cfg wd=0.01 --suffix teacher
python -u main.py --task=rec --dataset=foursquare --S_backbone=lightgcn --train_teacher --teacher embedding_dim=2000 num_layers=1 --cfg wd=1e-3 --suffix teacher

python -u main.py --task=rec --dataset=citeulike --S_backbone=bpr --train_teacher --teacher embedding_dim=400 --cfg wd=1e-3 --suffix teacher
python -u main.py --task=rec --dataset=citeulike --S_backbone=lightgcn --train_teacher --suffix teacher --postsave
python -u main.py --task=rec --dataset=citeulike --S_backbone=simplex --train_teacher --teacher embedding_dim=500 --cfg wd=1e-5 num_ns=2000 --suffix teacher
python -u main.py --task=rec --dataset=citeulike --S_backbone=xsimgcl --train_teacher --suffix teacher
python -u main.py --task=rec --dataset=citeulike --S_backbone=sccf --train_teacher --suffix teacher
python -u main.py --task=rec --dataset=citeulike --S_backbone=hstu --train_teacher --suffix teacher --postsave

python -u main.py --task=rec --dataset=gowalla --S_backbone=bpr --train_teacher --teacher embedding_dim=300 --cfg wd=1e-4 --suffix teacher
python -u main.py --task=rec --dataset=gowalla --S_backbone=lightgcn --train_teacher --suffix teacher --postsave
python -u main.py --task=rec --dataset=gowalla --S_backbone=simplex --train_teacher --teacher embedding_dim=1000 --cfg wd=1e-8 num_ns=2000 --suffix teacher
python -u main.py --task=rec --dataset=gowalla --S_backbone=hstu --train_teacher --suffix teacher --postsave

python -u main.py --task=rec --dataset=yelp --S_backbone=bpr --train_teacher --suffix teacher --teacher embedding_dim=300 --cfg wd=1e-4 --suffix teacher
python -u main.py --task=rec --dataset=yelp --S_backbone=lightgcn --train_teacher --suffix teacher --postsave
python -u main.py --task=rec --dataset=yelp --S_backbone=simplex --train_teacher --teacher embedding_dim=500 --cfg wd=1e-8 num_ns=1000 --suffix teacher
python -u main.py --task=rec --dataset=yelp --S_backbone=hstu --train_teacher --suffix teacher --postsave

# from scratch
python -u main.py --task=rec --dataset=foursquare --S_backbone=bpr --model=scratch --student embedding_dim=10 --cfg wd=0.01 --suffix student
python -u main.py --task=rec --dataset=foursquare --S_backbone=lightgcn --model=scratch --student embedding_dim=10 num_layers=1 --cfg wd=0.01 --suffix student

python -u main.py --task=rec --dataset=citeulike --S_backbone=bpr --model=scratch --student embedding_dim=20 --cfg wd=1e-3 --suffix student
python -u main.py --task=rec --dataset=citeulike --S_backbone=lightgcn --model=scratch --student embedding_dim=20 num_layers=3 --cfg wd=1e-3 --suffix student
python -u main.py --task=rec --dataset=citeulike --S_backbone=simplex --model=scratch --student embedding_dim=20 --suffix student
python -u main.py --task=rec --dataset=citeulike --S_backbone=xsimgcl --model=scratch --suffix student
python -u main.py --task=rec --dataset=citeulike --S_backbone=hstu --model=scratch --suffix student

python -u main.py --task=rec --dataset=gowalla --S_backbone=bpr --model=scratch --student embedding_dim=20 --cfg wd=1e-3 --suffix student
python -u main.py --task=rec --dataset=gowalla --S_backbone=lightgcn --model=scratch --student embedding_dim=20 num_layers=4 --cfg wd=1e-7 --suffix student
python -u main.py --task=rec --dataset=gowalla --S_backbone=simplex --model=scratch --student embedding_dim=20 --suffix student
python -u main.py --task=rec --dataset=gowalla --S_backbone=hstu --model=scratch --suffix student

python -u main.py --task=rec --dataset=yelp --S_backbone=bpr --model=scratch --student embedding_dim=20 --cfg wd=1e-3 --suffix student
python -u main.py --task=rec --dataset=yelp --S_backbone=lightgcn --model=scratch --student embedding_dim=20 num_layers=3 --cfg wd=1e-4 --suffix student
python -u main.py --task=rec --dataset=yelp --S_backbone=simplex --model=scratch --student embedding_dim=20 --suffix student
python -u main.py --task=rec --dataset=yelp --S_backbone=hstu --model=scratch --suffix student

# KD
# For HetComp, you need to pre-save teacher checkpoints through:
python -u main.py --task=rec --dataset=citeulike --S_backbone=bpr --T_backbone=bpr --train_teacher --teacher embedding_dim=400 --cfg wd=1e-3 --no_log --ckpt_interval=50
python -u main.py --task=rec --dataset=citeulike --S_backbone=bpr --T_backbone=bpr --model=hetcomp
python -u main.py --task=rec --dataset=citeulike --S_backbone=bpr --T_backbone=bpr --model=de
python -u main.py --task=rec --dataset=citeulike --S_backbone=bpr --T_backbone=bpr --model=rrd
python -u main.py --task=rec --dataset=citeulike --S_backbone=bpr --T_backbone=bpr --model=dcd
python -u main.py --task=rec --dataset=citeulike --S_backbone=bpr --T_backbone=bpr --model=cpd
python -u main.py --task=rec --dataset=citeulike --S_backbone=bpr --T_backbone=bpr --model=nkd
python -u main.py --task=rec --dataset=citeulike --S_backbone=bpr --T_backbone=bpr --model=graphd
python -u main.py --task=rec --dataset=citeulike --S_backbone=bpr --T_backbone=bpr --model=filterd
python -u main.py --task=rec --dataset=citeulike --S_backbone=bpr --T_backbone=bpr --model=fd
python -u main.py --task=rec --dataset=citeulike --S_backbone=bpr --T_backbone=bpr --model=gdcp
python -u main.py --task=rec --dataset=citeulike --S_backbone=bpr --T_backbone=bpr --model=preld
python -u main.py --task=rec --dataset=citeulike --S_backbone=bpr --T_backbone=bpr --model=vkd
python -u main.py --task=rec --dataset=citeulike --S_backbone=bpr --T_backbone=bpr --model=kd
python -u main.py --task=rec --dataset=citeulike --S_backbone=bpr --T_backbone=bpr --model=mkd

python -u main.py --task=rec --dataset=citeulike --S_backbone=lightgcn --T_backbone=lightgcn --model=graphd --cfg ablation=True --suffix ablation
python -u main.py --task=rec --dataset=citeulike --S_backbone=lightgcn --T_backbone=lightgcn --model=graphd
python -u main.py --task=rec --dataset=citeulike --S_backbone=lightgcn --T_backbone=lightgcn --model=de
python -u main.py --task=rec --dataset=citeulike --S_backbone=lightgcn --T_backbone=lightgcn --model=cpd
python -u main.py --task=rec --dataset=citeulike --S_backbone=lightgcn --T_backbone=lightgcn --model=rrd


python -u main.py --task=rec --dataset=foursquare --S_backbone=bpr --T_backbone=bpr --model=de
python -u main.py --task=rec --dataset=foursquare --S_backbone=bpr --T_backbone=bpr --model=cpd
python -u main.py --task=rec --dataset=foursquare --S_backbone=bpr --T_backbone=bpr --model=nkd
python -u main.py --task=rec --dataset=foursquare --S_backbone=bpr --T_backbone=bpr --model=graphd
python -u main.py --task=rec --dataset=foursquare --S_backbone=bpr --T_backbone=bpr --model=filterd

python -u main.py --task=rec --dataset=foursquare --S_backbone=lightgcn --T_backbone=lightgcn --model=nkd


python -u main.py --task=rec --dataset=gowalla --S_backbone=bpr --T_backbone=bpr --model=de
python -u main.py --task=rec --dataset=gowalla --S_backbone=bpr --T_backbone=bpr --model=rrd
python -u main.py --task=rec --dataset=gowalla --S_backbone=bpr --T_backbone=bpr --model=cpd
python -u main.py --task=rec --dataset=gowalla --S_backbone=bpr --T_backbone=bpr --model=graphd

python -u main.py --task=rec --dataset=gowalla --S_backbone=lightgcn --T_backbone=lightgcn --model=graphd


python -u main.py --task=rec --dataset=yelp --S_backbone=bpr --T_backbone=bpr --model=de
python -u main.py --task=rec --dataset=yelp --S_backbone=bpr --T_backbone=bpr --model=cpd
python -u main.py --task=rec --dataset=yelp --S_backbone=bpr --T_backbone=bpr --model=rrd



"""
For Multi-Modal Prediction Models
"""
# train teacher
python -u main.py --task=mm --dataset=baby --S_backbone=bm3 --train_teacher --suffix teacher
python -u main.py --task=mm --dataset=baby --S_backbone=vbpr --train_teacher --suffix teacher
python -u main.py --task=mm --dataset=baby --S_backbone=mgcn --train_teacher --suffix teacher

python -u main.py --task=mm --dataset=clothing --S_backbone=bm3 --train_teacher --suffix teacher

# from scratch
python -u main.py --task=mm --dataset=baby --S_backbone=bpr --model=scratch --suffix student

python -u main.py --task=mm --dataset=clothing --S_backbone=bpr --model=scratch --suffix student

# KD
python -u main.py --task=mm --dataset=baby --S_backbone=bpr --T_backbone=bm3 --model=freqmm
python -u main.py --task=mm --dataset=baby --S_backbone=bpr --T_backbone=bm3 --model=fitnet



"""
For CTR Prediction Models
"""
# train teacher
python -u main.py --task=ctr --dataset=criteo --S_backbone=crossnet --train_teacher --suffix teacher
python -u main.py --task=ctr --dataset=criteo --S_backbone=dnn --train_teacher --suffix teacher
python -u main.py --task=ctr --dataset=criteo --S_backbone=dcnv2 --train_teacher --suffix teacher
python -u main.py --task=ctr --dataset=criteo --S_backbone=cin --train_teacher --suffix teacher
python -u main.py --task=ctr --dataset=criteo --S_backbone=xdeepfm --train_teacher --suffix teacher
python -u main.py --task=ctr --dataset=criteo --S_backbone=autoint --train_teacher --suffix teacher
python -u main.py --task=ctr --dataset=criteo --S_backbone=eulernet --train_teacher --suffix teacher
python -u main.py --task=ctr --dataset=criteo --S_backbone=gatecrossnet --train_teacher --suffix teacher

# from scratch
python -u main.py --task=ctr --dataset=criteo --S_backbone=crossnet --model=scratch --suffix student --no_save
python -u main.py --task=ctr --dataset=criteo --S_backbone=dnn --model=scratch --suffix student --no_save
python -u main.py --task=ctr --dataset=criteo --S_backbone=dcnv2 --model=scratch --suffix student --no_save
python -u main.py --task=ctr --dataset=criteo --S_backbone=gatecrossnet --model=scratch --suffix student --no_save

# KD
python -u main.py --task=ctr --dataset=criteo --S_backbone=crossnet --T_backbone=crossnet --model=hetd
python -u main.py --task=ctr --dataset=criteo --S_backbone=crossnet --T_backbone=crossnet --model=rkd
python -u main.py --task=ctr --dataset=criteo --S_backbone=crossnet --T_backbone=crossnet --model=bced
python -u main.py --task=ctr --dataset=criteo --S_backbone=crossnet --T_backbone=crossnet --model=fitnet
python -u main.py --task=ctr --dataset=criteo --S_backbone=crossnet --T_backbone=crossnet --model=clid
python -u main.py --task=ctr --dataset=criteo --S_backbone=crossnet --T_backbone=crossnet --model=paird
python -u main.py --task=ctr --dataset=criteo --S_backbone=crossnet --T_backbone=crossnet --model=anyd
python -u main.py --task=ctr --dataset=criteo --S_backbone=crossnet --T_backbone=crossnet --model=adad
python -u main.py --task=ctr --dataset=criteo --S_backbone=crossnet --T_backbone=crossnet --model=ofa

python -u main.py --task=ctr --dataset=criteo --S_backbone=dnn --T_backbone=dnn --model=bced
python -u main.py --task=ctr --dataset=criteo --S_backbone=dnn --T_backbone=dnn --model=fitnet
python -u main.py --task=ctr --dataset=criteo --S_backbone=dnn --T_backbone=dnn --model=hetd

python -u main.py --task=ctr --dataset=criteo --S_backbone=dcnv2 --T_backbone=dcnv2 --model=bced
python -u main.py --task=ctr --dataset=criteo --S_backbone=dcnv2 --T_backbone=dcnv2 --model=clid
python -u main.py --task=ctr --dataset=criteo --S_backbone=dcnv2 --T_backbone=dcnv2 --model=hetd

python -u main.py --task=ctr --dataset=criteo --S_backbone=cin --T_backbone=cin --model=fitnet
python -u main.py --task=ctr --dataset=criteo --S_backbone=cin --T_backbone=cin --model=hetd
python -u main.py --task=ctr --dataset=criteo --S_backbone=cin --T_backbone=cin --model=warmup

python -u main.py --task=ctr --dataset=criteo --S_backbone=gatecrossnet --T_backbone=gatecrossnet --model=fitnet
python -u main.py --task=ctr --dataset=criteo --S_backbone=gatecrossnet --T_backbone=gatecrossnet --model=bced