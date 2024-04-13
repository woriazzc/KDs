import os
import time
import numpy as np
from copy import deepcopy

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from parse import *
from dataset import load_data, implicit_CF_dataset, implicit_CF_dataset_test
from evaluation import Evaluator
import modeling.backbone as backbone
import modeling.KD as KD
from utils import seed_all, avg_dict, Logger

def main(args):
    # Dataset
    num_users, num_items, train_pairs, valid_pairs, test_pairs, train_dict, valid_dict, test_dict, train_matrix, user_pop, item_pop = load_data(args.dataset)
    trainset = implicit_CF_dataset(args.dataset, num_users, num_items, train_pairs, train_matrix, train_dict, user_pop, item_pop, args.num_ns)
    testset = implicit_CF_dataset_test(num_users, num_items, valid_dict, test_dict)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    # Backbone
    all_backbones = [e.lower() for e in dir(backbone)]
    if args.backbone.lower() in all_backbones:
        Teacher = getattr(backbone, dir(backbone)[all_backbones.index(args.backbone.lower())])(trainset, teacher_args).cuda()
        Student = getattr(backbone, dir(backbone)[all_backbones.index(args.backbone.lower())])(trainset, student_args).cuda()
    else:
        logger.log('Invalid backbone model.')
        raise(NotImplementedError, 'Invalid backbone model.')

    if args.train_teacher:
        model = KD.Scratch(args, Teacher).cuda()
    else:
        T_path = os.path.join("checkpoints", args.dataset, args.backbone, f"{teacher_args.embedding_dim}.pt")
        Teacher.load_state_dict(torch.load(T_path))
        all_models = [e.lower() for e in dir(KD)]
        if args.model.lower() in all_models:
            model = getattr(KD, dir(KD)[all_models.index(args.model.lower())])(args, Teacher, Student).cuda()
        else:
            logger.log('Invalid backbone model.')
            raise(NotImplementedError, 'Invalid backbone model.')

    # Optimizer
    optimizer = optim.Adam(model.get_params_to_update())

    # Evaluator
    evaluator = Evaluator(args)
    best_model, save_path = deepcopy(model.save())

    for epoch in range(args.epochs):
        logger.log(f'Epoch [{epoch + 1}/{args.epochs}]')
        tic1 = time.time()
        logger.log('Negative sampling...')
        train_loader.dataset.negative_sampling()

        logger.log("Model's personal time...")
        model.do_something_in_each_epoch(epoch)

        epoch_loss = []
        logger.log('Training...')
        
        for idx, (batch_user, batch_pos_item, batch_neg_item) in enumerate(train_loader):
            batch_user = batch_user.cuda()
            batch_pos_item = batch_pos_item.cuda()
            batch_neg_item = batch_neg_item.cuda()
            
            # Forward Pass
            model.train()
            loss = model(batch_user, batch_pos_item, batch_neg_item)

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
            is_improved, early_stop, eval_results, elapsed = evaluator.evaluate_while_training(model, epoch, train_loader, testset)
            evaluator.print_result_while_training(logger, epoch_loss, eval_results, is_improved=is_improved, train_time=toc1-tic1, test_time=elapsed)
            if early_stop:
                break
            if is_improved:
                best_model, save_path = deepcopy(model.save())
    
    eval_dict = evaluator.eval_dict
    Evaluator.print_final_result(logger, eval_dict)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(best_model, save_path)

    return eval_dict


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    logger = Logger(args, args.no_log)

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
        Evaluator.print_final_result(logger, avg_eval_dict)
    else:
        logger.log_args(teacher_args, "TEACHER")
        if not args.train_teacher:
            logger.log_args(student_args, "STUDENT")
        logger.log_args(args, "ARGUMENTS")
        seed_all(args.seed)
        main(args)
