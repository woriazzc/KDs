import os
import time
import numpy as np
from copy import deepcopy

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from parse import *
from dataset import CTRDataset
from evaluation import Evaluator
import modeling.backbone as backbone
import modeling.KD as KD
from utils import seed_all, avg_dict, Logger

def main(args):
    # Dataset
    trainset = CTRDataset(args.dataset, 'train')
    validset = CTRDataset(args.dataset, 'valid')
    testset = CTRDataset(args.dataset, 'test')
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    args.field_dims = trainset.field_dims
    args.early_stop_metric = "AUC"

    # Backbone
    all_backbones = [e.lower() for e in dir(backbone)]
    if args.backbone.lower() in all_backbones:
        all_teacher_args, all_student_args = deepcopy(args), deepcopy(args)
        all_teacher_args.__dict__.update(teacher_args.__dict__)
        all_student_args.__dict__.update(student_args.__dict__)
        Teacher = getattr(backbone, dir(backbone)[all_backbones.index(args.backbone.lower())])(all_teacher_args).cuda()
        Student = getattr(backbone, dir(backbone)[all_backbones.index(args.backbone.lower())])(all_student_args).cuda()
    else:
        logger.log(f'Invalid backbone {args.backbone}.')
        raise(NotImplementedError, f'Invalid backbone {args.backbone}.')

    if args.model.lower() == "scratch":
        if args.train_teacher:
            model = KD.Scratch(args, Teacher).cuda()
        else:
            model = KD.Scratch(args, Student).cuda()
    else:
        T_path = os.path.join("checkpoints", args.dataset, args.backbone, f"scratch-{teacher_args.embedding_dim}", "BEST_EPOCH.pt")
        Teacher.load_state_dict(torch.load(T_path))
        all_models = [e.lower() for e in dir(KD)]
        if args.model.lower() in all_models:
            model = getattr(KD, dir(KD)[all_models.index(args.model.lower())])(args, Teacher, Student).cuda()
        else:
            logger.log(f'Invalid model {args.model}.')
            raise(NotImplementedError, f'Invalid model {args.model}.')

    # Optimizer
    optimizer = optim.Adam(model.get_params_to_update())

    # Evaluator
    evaluator = Evaluator(args)
    best_model, best_epoch = deepcopy(model.param_to_save), -1
    ckpts = []

    # Test Teacher first
    if args.model.lower() != "scratch":
        logger.log('-' * 40 + "Teacher" + '-' * 40, pre=False)
        tmp_evaluator = Evaluator(args)
        tmp_model = KD.Scratch(args, Teacher).cuda()

        is_improved, early_stop, eval_results, elapsed = tmp_evaluator.evaluate_while_training(tmp_model, -1, train_loader, valid_loader, test_loader)
        Evaluator.print_final_result(logger, tmp_evaluator.eval_dict)
        logger.log('-' * 88, pre=False)

    for epoch in range(args.epochs):
        logger.log(f'Epoch [{epoch + 1}/{args.epochs}]')
        tic1 = time.time()

        logger.log("Model's personal time...")
        model.do_something_in_each_epoch(epoch)

        epoch_loss = []
        logger.log('Training...')
        
        for idx, (features, labels) in enumerate(train_loader):
            features = features.cuda()      # batch_size, F
            labels = labels.cuda()  # batch_size
            
            # Forward Pass
            model.train()
            loss = model(features, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss)

        epoch_loss = torch.mean(torch.stack(epoch_loss)).item()

        toc1 = time.time()
        
        # evaluation
        if epoch % args.eval_period == 0:
            logger.log("Evaluating...")
            is_improved, early_stop, eval_results, elapsed = evaluator.evaluate_while_training(model, epoch, train_loader, valid_loader, test_loader)
            evaluator.print_result_while_training(logger, epoch_loss, eval_results, is_improved=is_improved, train_time=toc1-tic1, test_time=elapsed)
            if early_stop:
                break
            if is_improved:
                best_model = deepcopy(model.param_to_save)
                best_epoch = epoch
        
        # save intermediate checkpoints
        if not args.no_save and args.ckpt_interval != -1 and epoch % args.ckpt_interval == 0 and epoch != 0:
            ckpts.append(deepcopy(model.param_to_save))
    
    eval_dict = evaluator.eval_dict
    Evaluator.print_final_result(logger, eval_dict)
    if not args.no_save:
        embedding_dim = Teacher.embedding_dim if args.train_teacher else Student.embedding_dim
        save_dir = os.path.join("checkpoints", args.dataset, args.backbone, f"{args.model.lower()}-{embedding_dim}")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(best_model, os.path.join(save_dir, "BEST_EPOCH.pt"))
        for idx, ckpt in enumerate(ckpts):
            if (idx + 1) * args.ckpt_interval >= best_epoch:
                break
            torch.save(ckpt, os.path.join(save_dir, f"EPOCH_{(idx + 1) * args.ckpt_interval}.pt"))

    return eval_dict


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    logger = Logger(args, args.no_log)
    args.task = 'ctr'

    if args.run_all:
        args_copy = deepcopy(args)
        eval_dicts = []
        for seed in range(5):
            args = deepcopy(args_copy)
            args.seed = seed
            seed_all(args.seed)
            logger.log_args(teacher_args, "TEACHER")
            if not args.train_teacher:
                logger.log_args(student_args, "STUDENT")
            logger.log_args(args, "ARGUMENTS")
            eval_dicts.append(main(args))
        
        avg_eval_dict = avg_dict(eval_dicts)

        logger.log('=' * 60)
        Evaluator.print_final_result(logger, avg_eval_dict, prefix="avg ")
    else:
        logger.log_args(teacher_args, "TEACHER")
        if not args.train_teacher:
            logger.log_args(student_args, "STUDENT")
        logger.log_args(args, "ARGUMENTS")
        seed_all(args.seed)
        main(args)
