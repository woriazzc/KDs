import os
import gc
import time
from copy import deepcopy

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from modules.dataset import load_cf_data, implicit_CF_dataset, implicit_CF_dataset_test, implicit_SR_dataset
from modules.evaluation import Evaluator
from modeling import backbone
from modeling import KD


def main(args, teacher_args, student_args, logger):
    # Dataset
    num_users, num_items, train_pairs, valid_pairs, test_pairs, train_dict, valid_dict, test_dict, train_matrix, user_pop, item_pop = load_cf_data(args.dataset)
    no_neg_sampling = getattr(teacher_args, "no_neg_sampling", False) if args.train_teacher else getattr(student_args, "no_neg_sampling", False)
    trainset = implicit_CF_dataset(args.dataset, num_users, num_items, train_pairs, train_matrix, train_dict, user_pop, item_pop, args.num_ns, args.neg_sampling_on_all, no_neg_sampling)
    validset = implicit_CF_dataset_test(num_users, num_items, valid_dict)
    testset = implicit_CF_dataset_test(num_users, num_items, test_dict)
    if args.S_backbone.lower() in ["hstu"]:
        max_sequence_len = teacher_args.max_sequence_len if args.train_teacher else student_args.max_sequence_len
        trainset = implicit_SR_dataset(implicit_CF_dataset(args.dataset, num_users, num_items, train_pairs, train_matrix, train_dict, user_pop, item_pop, args.num_ns, args.neg_sampling_on_all, no_neg_sampling), 
                                       max_sequence_len)
    if args.T_backbone.lower() in ["hstu"] and not args.preload:
        trainset_T = implicit_SR_dataset(implicit_CF_dataset(args.dataset, num_users, num_items, train_pairs, train_matrix, train_dict, user_pop, item_pop, args.num_ns, args.neg_sampling_on_all, no_neg_sampling), 
                                         teacher_args.max_sequence_len)
        validset_T = implicit_CF_dataset_test(num_users, num_items, valid_dict)
        testset_T = implicit_CF_dataset_test(num_users, num_items, test_dict)
    else:
        trainset_T = implicit_CF_dataset(args.dataset, num_users, num_items, train_pairs, train_matrix, train_dict, user_pop, item_pop, args.num_ns, args.neg_sampling_on_all, no_neg_sampling)
        validset_T = implicit_CF_dataset_test(num_users, num_items, valid_dict)
        testset_T = implicit_CF_dataset_test(num_users, num_items, test_dict)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    # Backbone
    all_backbones = [e.lower() for e in dir(backbone)]
    if args.S_backbone.lower() in all_backbones:
        all_teacher_args, all_student_args = deepcopy(args), deepcopy(args)
        all_teacher_args.__dict__.update(teacher_args.__dict__)
        all_student_args.__dict__.update(student_args.__dict__)
        if args.preload:
            Teacher = backbone.Prediction(trainset_T, all_teacher_args).cuda()
        else:
            Teacher = getattr(backbone, dir(backbone)[all_backbones.index(args.T_backbone.lower())])(trainset_T, all_teacher_args).cuda()
        if not args.train_teacher:
            Student = getattr(backbone, dir(backbone)[all_backbones.index(args.S_backbone.lower())])(trainset, all_student_args).cuda()
    else:
        logger.log(f'Invalid backbone {args.S_backbone}.')
        raise(NotImplementedError, f'Invalid backbone {args.S_backbone}.')

    if args.model.lower() == "scratch":
        if args.train_teacher:
            model = KD.Scratch(args, Teacher).cuda()
        else:
            model = KD.Scratch(args, Student).cuda()
    else:
        T_path = os.path.join("checkpoints", args.dataset, args.T_backbone, f"scratch-{teacher_args.embedding_dim}")
        if args.preload: T_path = os.path.join(T_path, "BEST_SCORE_MAT.pt")
        else: T_path = os.path.join(T_path, "BEST_EPOCH.pt")
        Teacher.load_state_dict(torch.load(T_path, weights_only=True))
        all_models = [e.lower() for e in dir(KD)]
        if args.model.lower() in all_models:
            if args.model.lower() == "mrrd":
                model = getattr(KD, dir(KD)[all_models.index(args.model.lower())])(args, Teacher, Student, valid_dict, test_dict).cuda()
            else:
                model = getattr(KD, dir(KD)[all_models.index(args.model.lower())])(args, Teacher, Student).cuda()
        else:
            logger.log(f'Invalid model {args.model}.')
            raise(NotImplementedError, f'Invalid model {args.model}.')

    # Optimizer
    optimizer = optim.Adam(model.get_params_to_update())

    # Evaluator
    evaluator = Evaluator(args)
    if not args.no_save:
        best_model, best_epoch = deepcopy(model.param_to_save), -1
        ckpts = []
        if args.postsave:
            best_score_mat = deepcopy(model.score_mat_to_save)
            score_mats = []

    # Test Teacher first
    if args.model.lower() != "scratch":
        logger.log('-' * 40 + "Teacher" + '-' * 40, pre=False)
        tmp_evaluator = Evaluator(args)
        tmp_model = KD.Scratch(args, Teacher).cuda()
        train_loader_T = DataLoader(trainset_T, batch_size=args.batch_size, shuffle=True)
        is_improved, early_stop, eval_results, elapsed = tmp_evaluator.evaluate_while_training(tmp_model, -1, train_loader_T, validset_T, testset_T)
        Evaluator.print_final_result(logger, tmp_evaluator.eval_dict)
        logger.log('-' * 88, pre=False)
        del train_loader_T, trainset_T, validset_T, testset_T
        gc.collect()

    for epoch in range(args.epochs):
        logger.log(f'Epoch [{epoch + 1}/{args.epochs}]')
        tic1 = time.time()
        logger.log('Negative sampling...')
        if hasattr(train_loader.dataset, "negative_sampling"):
            train_loader.dataset.negative_sampling()

        logger.log("Model's personal time...")
        model.do_something_in_each_epoch(epoch)

        epoch_loss, epoch_base_loss, epoch_kd_loss = [], [], []
        logger.log('Training...')
        
        for idx, data in enumerate(train_loader):
            # Forward Pass
            model.train()
            try:
                data = [data.cuda()]
            except:
                for i in range(len(data)):
                    try: data[i] = data[i].cuda()
                    except: pass
            loss, base_loss, kd_loss = model(*data)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.detach())
            epoch_base_loss.append(base_loss)
            epoch_kd_loss.append(kd_loss)

        epoch_loss = torch.mean(torch.stack(epoch_loss)).item()
        epoch_base_loss = torch.mean(torch.stack(epoch_base_loss)).item()
        epoch_kd_loss = torch.mean(torch.stack(epoch_kd_loss)).item()

        toc1 = time.time()
        
        # evaluation
        if epoch % args.eval_period == 0:
            logger.log("Evaluating...")
            is_improved, early_stop, eval_results, elapsed = evaluator.evaluate_while_training(model, epoch, train_loader, validset, testset)
            evaluator.print_result_while_training(logger, epoch_loss, epoch_base_loss, epoch_kd_loss, eval_results, is_improved=is_improved, train_time=toc1-tic1, test_time=elapsed)
            if early_stop:
                break
            if not args.no_save and is_improved:
                best_model = deepcopy(model.param_to_save)
                best_epoch = epoch
                if args.postsave:
                    best_score_mat = deepcopy(model.score_mat_to_save)
        
        # save intermediate checkpoints
        if not args.no_save and args.ckpt_interval != -1 and epoch % args.ckpt_interval == 0 and epoch != 0:
            ckpts.append(deepcopy(model.param_to_save))
            score_mats.append(deepcopy(model.score_mat_to_save))
    
    eval_dict = evaluator.eval_dict
    Evaluator.print_final_result(logger, eval_dict)
    if not args.no_save:
        backbone_name = args.T_backbone if args.train_teacher else args.S_backbone
        embedding_dim = Teacher.embedding_dim if args.train_teacher else Student.embedding_dim
        args.suffix = '' if args.suffix == "teacher" else args.suffix
        save_dir = os.path.join("checkpoints", args.dataset, backbone_name, f"{args.model.lower()}-{embedding_dim}" + ('_' if args.suffix != '' else '') + args.suffix)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(best_model, os.path.join(save_dir, "BEST_EPOCH.pt"))
        if args.postsave:
            torch.save(best_score_mat, os.path.join(save_dir, "BEST_SCORE_MAT.pt"))
        for idx in range(len(ckpts)):
            if (idx + 1) * args.ckpt_interval >= best_epoch:
                break
            torch.save(ckpts[idx], os.path.join(save_dir, f"EPOCH_{(idx + 1) * args.ckpt_interval}.pt"))
            torch.save(score_mats[idx], os.path.join(save_dir, f"EPOCH_{(idx + 1) * args.ckpt_interval}_SCORE_MAT.pt"))

    return eval_dict
