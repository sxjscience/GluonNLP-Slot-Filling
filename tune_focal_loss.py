import subprocess, os
import numpy as np

__EXPPATH__ = os.path.realpath(os.path.join(os.path.dirname(__file__), 'tune_exp'))


of = open('tune_focal_exp.txt', 'w')
of.write('alpha, gamma, atis-slot-mean, atis-slot-std, atis-intent-mean, atis-intent-std, '
         'snips-slot-mean, snips-slot-std, snips-intent-mean, snips-intent-std\n')
for alpha in [5, 10, 20, 50]:
    for gamma in [1.0, 1.5, 2.0]:
        save_dir = __EXPPATH__
        avg_results = []
        for dataset in ['atis', 'snips']:
            dataset_results = []
            slot_f1_l = []
            intent_acc_l = []
            possible_seeds = [123, 231, 321]
            for seed in possible_seeds:
                subprocess.call(['python3', 'demo.py', '--gpu', '0',
                                 '--dataset', dataset,
                                 '--seed', seed,
                                 '--use-focal',
                                 '--focal-alpha', '{}'.format(alpha),
                                 '--focal-gamma', '{}'.format(gamma),
                                 '--save-dir', save_dir])
                # Read results
                with open(os.path.join(save_dir, 'test_error.txt'), 'r') as f:
                    slot_f1, intent_acc = f.read().strip().split()
                    slot_f1 = float(slot_f1)
                    intent_acc = float(intent_acc)
                    slot_f1_l.append(slot_f1)
                    intent_acc_l.append(intent_acc)
            avg_results.extend([np.mean(slot_f1_l), np.std(slot_f1_l),
                                np.mean(intent_acc_l), np.std(intent_acc_l)])
        of.write('{:g}, {:g}, '.format(alpha, gamma)
                 + ', '.join(['{:.2f}'.format(ele * 100) for ele in avg_results]) + '\n')
        of.flush()
of.close()
