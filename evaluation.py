from copy import deepcopy
import time
import numpy as np

import torch
import torch.utils.data as data

from utils import to_np
from utils.metric import Precision, Recall, NDCG, get_labels

METRIC2FUNC = {'Recall': Recall, 'NDCG': NDCG, 'Precision': Precision}

class Evaluator:
    def __init__(self, args):
        self.K_list = args.K_list
        self.K_max = max(self.K_list)
        self.early_stop_patience = args.early_stop_patience
        self.early_stop_metric = args.early_stop_metric
        self.early_top_K = args.early_stop_K
        self.metrics = ['Recall', 'NDCG']
        
        self.METRICS_DICT = {metric: {K: 0. for K in self.K_list} for metric in self.metrics}
        self.eval_dict = self.get_eval_template()

    def get_eval_template(self):
        """
        {
            'early_stop': 0, 'early_stop_max': 0, 'final_epoch': 0,
            'best_result': {
                                'NDCG': {5: -1., 10: -1.}, 
                                'Recall': {5: -1., 10: -1.}
                            },
            'final_result': {
                                'NDCG': {5: -1., 10: -1.}, 
                                'Recall': {5: -1., 10: -1.}
                            }
        }
        """
        eval_dict = {'early_stop': 0,  'early_stop_max': self.early_stop_patience, 'final_epoch': 0}
        for mode in ['best_result', 'final_result']:
            eval_dict[mode] = {}
            for metric in self.metrics:
                eval_dict[mode][metric] = {}
                for K in self.K_list:
                    eval_dict[mode][metric][K] = -1.
        return eval_dict

    def evaluate(self, model, train_loader, test_dataset):
        """
        {
            'valid': {
                        'NDCG': {5: -1., 10: -1.}, 
                        'Recall': {5: -1., 10: -1.}
                    },
            'test': {
                        'NDCG': {5: -1., 10: -1.}, 
                        'Recall': {5: -1., 10: -1.}
                    }
        }
        """
        eval_results = {'test': deepcopy(self.METRICS_DICT), 'valid':deepcopy(self.METRICS_DICT)}
        
        train_dict = train_loader.dataset.train_dict
        valid_dict = test_dataset.valid_dict
        test_dict = test_dataset.test_dict
        num_users = train_loader.dataset.num_users
        num_items = train_loader.dataset.num_items

        model.eval()
        test_loader = data.DataLoader(list(test_dict.keys()), batch_size=train_loader.batch_size)
        topK_items = torch.zeros((num_users, self.K_max), dtype=torch.long)
        for batch_user in test_loader:
            score_mat = model.get_ratings(batch_user)
            for idx, user in enumerate(batch_user):
                pos = train_dict[user.item()]
                score_mat[idx, pos] = -1e10
            _, sorted_mat = torch.topk(score_mat, k=self.K_max, dim=1)
            topK_items[batch_user, :] = sorted_mat.detach().cpu()
        
        for mode in ['valid', 'test']:
            if mode == 'valid':
                gt_mat = valid_dict
            elif mode == 'test':
                gt_mat = test_dict

            num_targets = np.zeros(num_users, dtype=int)
            labels = np.zeros((num_users, self.K_max), dtype=int)
            for test_user in gt_mat:
                num_targets[test_user] = len(gt_mat[test_user])
                labels[test_user, :] = get_labels(to_np(topK_items[test_user]), gt_mat[test_user], num_items)
            
            for metric in self.metrics:
                func = METRIC2FUNC[metric]
                for K in self.K_list:
                    eval_results[mode][metric][K] = func(labels[:, :K], num_targets)

        return eval_results


    def evaluate_while_training(self, model, epoch, train_loader, test_dataset):
        model.eval()
        with torch.no_grad():
            tic = time.time()
            eval_results = self.evaluate(model, train_loader, test_dataset)
            toc = time.time()
            metric = self.early_stop_metric
            K = self.early_top_K
            is_improved = False

            if self.eval_dict['early_stop'] < self.eval_dict['early_stop_max']:
                if self.eval_dict['best_result'][metric][K] < eval_results['valid'][metric][K]:
                    self.eval_dict['best_result'] = eval_results['valid']
                    self.eval_dict['final_result'] = eval_results['test']
                    is_improved = True
                    self.eval_dict['final_epoch'] = epoch

            if not is_improved:
                self.eval_dict['early_stop'] += 1
            else:
                self.eval_dict['early_stop'] = 0
            
            if (self.eval_dict['early_stop'] >= self.eval_dict['early_stop_max']):
                early_stop = True
            else:
                early_stop = False

            return is_improved, early_stop, eval_results, toc-tic

    def print_result_while_training(self, logger, train_loss, eval_results, is_improved=False, train_time=0., test_time=0.):
        """print evaluation results while training

        Parameters
        ----------
        train_loss : float
        eval_results : dict
            summarizes the evaluation results
        is_improved : bool, optional
            is the result improved compared to the last best results, by default False
        train_time :float, optional
            elapsed time for training, by default 0.
        test_time : float, optional
            elapsed time for test, by default 0.
        """
        logger.log('\tTrain Loss: {:.4f}, Elapsed: train {:.2f} test {:.2f}{}'.format(train_loss, train_time, test_time, " *" if is_improved else ""))

        for mode in ['valid', 'test']:
            logger.log('\t', mode, end='')
            for metric in self.metrics:
                for K in self.K_list:
                    result = eval_results[mode][metric][K]
                    logger.log('{}@{}: {:.5f} '.format(metric, K, result), pre=False, end='')
            logger.log()

    @classmethod
    def print_final_result(self, logger, eval_dict):
        """print final result after the training

        Parameters
        ----------
        eval_dict : dict
        """
        logger.log('-'*30)

        for mode in ['valid', 'test']:
            if mode == 'valid':
                key = 'best_result'
            else:
                key = 'final_result'

            logger.log(mode, end='')
            for metric in eval_dict[key].keys():
                for K in eval_dict[key][metric].keys():
                    result = eval_dict[key][metric][K]
                    logger.log(' {}@{}: {:.5f}'.format(metric, K, result), pre=False, end='')
            logger.log()
