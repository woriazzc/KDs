import os
from copy import deepcopy

from parse import *
from evaluation import Evaluator
from utils import seed_all, avg_dict, Logger
import runners


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    logger = Logger(args, args.no_log)
    if args.task == "rec": main = runners.main_rec
    elif args.task == "mm": main = runners.main_mm
    elif args.task == "ctr": main = runners.main_ctr
    else: raise ValueError(f"Unexpected task {args.task}. Please select from [rec, mm, ctr]")

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
            eval_dicts.append(main(args, teacher_args, student_args, logger))
        
        avg_eval_dict = avg_dict(eval_dicts)

        logger.log('=' * 60)
        Evaluator.print_final_result(logger, avg_eval_dict, prefix="avg ")
    else:
        logger.log_args(teacher_args, "TEACHER")
        if not args.train_teacher:
            logger.log_args(student_args, "STUDENT")
        logger.log_args(args, "ARGUMENTS")
        seed_all(args.seed)
        main(args, teacher_args, student_args, logger)
