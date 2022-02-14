"""Example Commands
clear && CUDA_VISIBLE_DEVICES=3 python main_generalization.py --num-domains 2 --algorithm ERM 

clear && CUDA_VISIBLE_DEVICES=4 python main_generalization.py --num-domains 2 --algorithm GroupDRO 

clear && CUDA_VISIBLE_DEVICES=5 python main_generalization.py --num-domains 2 --algorithm IRM 

clear && CUDA_VISIBLE_DEVICES=6 python main_generalization.py --num-domains 2 --algorithm CORAL 

clear && CUDA_VISIBLE_DEVICES=7 python main_generalization.py --num-domains 2 --algorithm CDANN 


care: self.group_indices[groups_local][:50]
"""
MEASURE_SUBSET_INFLUENCE = False

SUBJECT_LIST = ['cat', 'dog']
DOMAINS_TO_GROUPS = {
            0: {'cat':['cat(indoor)'], 'dog':['dog(indoor)']},
            1: {'cat':['cat(outdoor)'], 'dog':['dog(outdoor)']},
}

# local imports 
import algorithms
import hparams_registry
import misc

import argparse
import pickle 
import collections
import json
import os
import random
import sys
import time
import uuid
import PIL
import torch
import torchvision
import torch.utils.data
import shutil
import time
import warnings
import logging
import pickle 
import pathlib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from collections import Counter, defaultdict
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def node_str_to_subject_tag(node_str):
    tag = node_str.split('(')[-1][:-1]
    subject_str = node_str.split('(')[0] 
    return subject_str, tag

class SubsetShiftDatasetManager():
    def __init__(self, args):
        self.args = args
        ##################################
        # Meta Data Loading 
        ##################################
        meta_data_path = os.path.join( args.data, 'imageID_to_group.pkl')
        with open(meta_data_path, "rb") as pkl_file:
            imageID_to_group = pickle.load( pkl_file ) # defaultdict(set)

        ##################################
        # Train Dataset - Data loading code
        ##################################
        traindir = os.path.join(args.data, 'train')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        # print(train_dataset.samples) # list of tuples 
        self.train_dataset = train_dataset
        
        ##################################
        # Build reversed meta-data array; build group vocab
        ##################################
        group_to_idx = defaultdict(list) # reverse mapping 
        for data_idx, sample in enumerate(train_dataset.samples):
            image_path, target = sample
            imageID = image_path.split('/')[-1].split('.')[0] # image_path = IMAGE_DATA_FOLDER + imageID + '.jpg'
            for groups_local in imageID_to_group[imageID]:
                group_to_idx[groups_local].append(data_idx)
        self.group_indices = group_to_idx

        tmp = [ (x[0], len(x[1])) for x in group_to_idx.items()]
        tmp = sorted(tmp, key=lambda i: i[1], reverse=True)
        print("train_dataset.samples reverse:", tmp)
        groups_local_vocab = [x[0] for x in tmp]
        self.groups_local_vocab = groups_local_vocab
        self.init_sample_group()

        return


    def init_sample_group(self):
        self.MAX_ITERATIONS = 81
        self.subjects_list = SUBJECT_LIST
        self.domain_to_groups = DOMAINS_TO_GROUPS

        print('self.domain_to_groups', self.domain_to_groups)

        return

    def get_train_dataset(self):
        return self.train_dataset

    
    def __len__(self):
        ##################################
        # Infinite!
        ##################################
        # return self.num_batches
        return self.MAX_ITERATIONS

    def __iter__(self):
        for iteration in range(self.MAX_ITERATIONS):
            batch_size = self.args.batch_size
            sampled_ids_all = []
            for domain_idx in range(self.args.num_domains):
                for subject_str in self.subjects_list:
                    groups_local = np.random.choice(self.domain_to_groups[domain_idx][subject_str])
                    sampled_ids = np.random.choice(
                            self.group_indices[groups_local], 
                            size=batch_size,
                            replace=len(self.group_indices[groups_local]) <= batch_size, # False if the group is larger than the sample size
                            p=None)
                    sampled_ids_all.append(sampled_ids)

            sampled_ids_all = np.concatenate(sampled_ids_all)
            yield sampled_ids_all



def save_influence_results(subset_influence_batch_results, batch_id, args):
    print('subset_influence_batch_results saved! batch_id[{}]'.format(batch_id) )
    with open(os.path.join( args.output_dir, 'subset_influence_results.pkl'), "wb") as pkl_file:
        pickle.dump(
            subset_influence_batch_results, 
            pkl_file, 
        )
    return


##################################
# A simple training dataset wrapper 
# for knowing the imageIDs
##################################
class TrainingDataset_Wrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.loader = dataset.loader
        self.samples = dataset.samples
        self.transform = dataset.transform

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        imageID = image_path.split('/')[-1].split('.')[0] # image_path = IMAGE_DATA_FOLDER + imageID + '.jpg'
        sample = self.loader(image_path)
        sample = self.transform(sample)
        return sample, imageID, target

    def __len__(self):
        return len(self.samples)

def report_every_set_acc(my_dataset, args, split='val'):

    ##################################
    # 1. Load pkl that stores the prediction scores 
    # And the pkl that stores the meta-dataset-structure
    ##################################
    with open(os.path.join( args.output_dir, 'model_validate_dump.pkl'), "rb") as pkl_file:
        load_data = pickle.load( pkl_file )
        target_all = load_data['target_all']
        pred_score_all = load_data['pred_score_all']

    ##################################
    # Load Meta-Data
    ##################################
    meta_data_path = os.path.join( args.data, 'imageID_to_group.pkl')
    with open(meta_data_path, "rb") as pkl_file:
        imageID_to_group = pickle.load( pkl_file ) # defaultdict(set)

    ##################################
    # 2. Sanity check datapoint order with ground truth labels.     
    # my_dataset.targets: list 
    # my_dataset.samples: list to 2-tuples
    # target_all: np.array
    # pred_score_all: np.array
    ##################################
    assert len(pred_score_all) == len(my_dataset.samples)

    ##################################
    # 3. iterate through all data points, and collect prediction scores, and labels. 
    ##################################
    group_to_preds = defaultdict(lambda: defaultdict(list))
    for idx, sample in enumerate(my_dataset.samples):
        image_path, target = sample
        imageID = image_path.split('/')[-1].split('.')[0] # image_path = IMAGE_DATA_FOLDER + imageID + '.jpg'
        assert target_all[idx] == target
        for groups_local in set(imageID_to_group[imageID]):
            group_to_preds[groups_local]['target'].append(target) 
            group_to_preds[groups_local]['pred_score'].append(pred_score_all[idx]) 

    ##################################
    # 4. reduce collected data by acc, or auc. 
    # And report each group in the sorted order of acc. 
    ##################################
    group_accs = list()
    for groups_local in sorted(group_to_preds.keys()):
        groups_local_target = np.array(group_to_preds[groups_local]['target'])
        groups_local_pred_scores = np.array(group_to_preds[groups_local]['pred_score'])
        groups_local_pred_labels = (groups_local_pred_scores > 0.5)
        group_accs.append(
            (groups_local, accuracy_score(groups_local_target, groups_local_pred_labels), len(groups_local_target))
            # name, acc, group_size 
        )
    
    ##################################
    # 5. report each group in the sorted order of acc. 
    ##################################
    group_accs.sort(key=lambda x: x[1], reverse=True)
    for tup in group_accs:
        groups_local, acc, group_size = tup
        info_str = "accuracy {:.3f} \t size: {} \t {}".format(acc, group_size, groups_local)
        print(info_str)
        logging.info(info_str)

    return 
    

def validate(val_loader, model, criterion, args, dumpResult,
    get_grads=False, algorithm=None,
    ):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    ##################################
    # Also, output the confusion matrix 
    ##################################
    nb_classes = args.num_classes
    my_confusion_matrix = torch.zeros(nb_classes, nb_classes)

    ##################################
    # Fields to be stored for postprocessing 
    ##################################
    target_all = []
    pred_score_all = [] 

    val_grad_list = []

    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        images = images.to(device)
        target = target.to(device)

        with torch.no_grad():
            # compute output
            output = model(images)
            loss = criterion(output, target)

        if get_grads:
            all_grads = algorithm.calculate_gradient_for_influence([(images, target)])
            val_grad_list.extend(all_grads)

        ##################################
        # Confusion Matrix
        ##################################
        _, preds = torch.max(output, 1)
        for t, p in zip(target.view(-1), preds.view(-1)):
                my_confusion_matrix[t.long(), p.long()] += 1

        acc1 = accuracy(output, target, topk=(1, ))[0]
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        ##################################
        # For analysis
        ##################################
        output_scores = torch.nn.functional.softmax(output, dim=-1)
        positive_scores = output_scores[:,1]

        target_all.append( target.cpu().numpy() )
        pred_score_all.append( positive_scores.cpu().numpy() )

    target_all = np.concatenate( target_all, axis=0)
    pred_score_all = np.concatenate( pred_score_all, axis=0)

    dump_result_dict = {
                "target_all": target_all, 
                "pred_score_all": pred_score_all, 
                'val_grad_list': val_grad_list
                }
    if dumpResult is True:
        with open(os.path.join( args.output_dir, 'model_validate_dump.pkl'), "wb") as pkl_file:
            pickle.dump(
                dump_result_dict, 
                pkl_file, 
            )
    
    # a large analysis here 
    pred_label = (pred_score_all>0.5)
    print("accuracy {:.3f}".format(accuracy_score(target_all, pred_label)),
    '\t',
    "roc_auc_score {:.3f}".format(roc_auc_score(target_all, pred_score_all)), 
    )
    print("confusion_matrix\n{}".format(confusion_matrix(target_all, pred_label)))
    print("classification_report\n{}".format(classification_report(target_all, pred_label)))

    # TODO: this should also be done with the ProgressMeter
    # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
    #       .format(top1=top1, top5=top5))
    print('VAL * Acc@1 {top1.avg:.3f}'
            .format(top1=top1))

    # if is_main_process():
    logging.info("accuracy {:.3f}".format(accuracy_score(target_all, pred_label)))
    logging.info(
        "roc_auc_score {:.3f}".format( roc_auc_score(target_all, pred_score_all) )
    )
    logging.info("confusion_matrix\n{}".format(confusion_matrix(target_all, pred_label)))
    logging.info("classification_report\n{}".format(classification_report(target_all, pred_label)))
    logging.info('VAL * Acc@1 {top1.avg:.3f}'
        .format(top1=top1))



    # TODO: this should also be done with the ProgressMeter
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

    return top1.avg, dump_result_dict


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', args=None): 
    torch.save(state, os.path.join( args.output_dir, filename) )
    if is_best:
        shutil.copyfile(
            os.path.join( args.output_dir, filename), 
            os.path.join( args.output_dir, 'model_best.pth.tar')
            )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        ##################################
        # Save to logging 
        ##################################
        logging.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



##################################
# wrapper function for val data loader 
##################################
def get_val_loader(dataset_dir, args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ret_dataset = datasets.ImageFolder(
            os.path.join(args.data, dataset_dir),
            transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    ret_loader = torch.utils.data.DataLoader(
        ret_dataset,
        batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    return ret_loader, ret_dataset
    

def main_generalization():
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data', metavar='DIR', default='/data/MetaShift/MetaShift-subpopulation-shift', 
                        help='path to dataset')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--num-domains', default=2, type=int,
                        metavar='N',
                        help='number of domains '
                        )
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--num-classes', default=2, type=int, metavar='N',
                        help='number of meta tasks used (default: 2, binary classification)')
    parser.add_argument('--log-prefix', default='', type=str, 
                        help='prefix to the log file (default: (none))')
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    args = parser.parse_args()

    ##################################
    # Logging Setup
    ##################################
    os.makedirs(args.output_dir, exist_ok=True)
    log_name = args.log_prefix + args.algorithm + '_out.txt'
    sys.stdout = misc.Tee(os.path.join(args.output_dir, log_name))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ##################################
    # Fancy training dtaset 
    ##################################
    my_dataset_manager = SubsetShiftDatasetManager(args)
    train_dataset = my_dataset_manager.get_train_dataset()
    train_dataset_wrap=TrainingDataset_Wrapper(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset_wrap,
                shuffle=None,
                sampler=None,
                collate_fn=None, # by default use None 
                batch_sampler=my_dataset_manager,
                drop_last=False,
                )
    del train_dataset, train_dataset_wrap

    ##################################
    # EMR, out of domain val acc, containing other. 
    ##################################
    val_out_of_domain_loader, val_out_of_domain_dataset = get_val_loader(dataset_dir='val_out_of_domain', args=args)

    ##################################
    # Initialize the model 
    ##################################
    hparams = hparams_registry.default_hparams(args.algorithm, 'MetaDataset')
    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(
        (3, 224, 224), # Input Shape 
        args.num_classes, # Binary Classification 
        args.num_domains,# Number of Domains
        hparams, # Hyper parameters generated by domainbed. 
        )



    algorithm.to(device)
    # model for compatible validation 
    model = torch.nn.Sequential(algorithm.featurizer, algorithm.classifier) # algorithm.network


    if MEASURE_SUBSET_INFLUENCE: 
        subset_influence_batch_results = {
            'DOMAINS_TO_GROUPS': DOMAINS_TO_GROUPS,
            'sample_schedule': [], # append imageIDs 
            'target': [], # overwrite  -- val
            'batch_results': [], # append results from each batch 
            'train_batch_grads': [], 
            'val_all_grads': [], 
        }
    
    for batch_id, train_batch in enumerate(train_loader):
        ##################################
        # One train step
        ##################################
        (images, ImageIDs, target) = train_batch
        images = images.to(device)
        target = target.to(device)

        assert len(images) == 2 * args.num_domains * args.batch_size

        minibatches_list = []
        for domain_id in range(args.num_domains):
            x_slice = (images[2 * domain_id * args.batch_size:2 * (domain_id+1) * args.batch_size])
            y_slice = (target[2 * domain_id * args.batch_size:2 * (domain_id+1) * args.batch_size])
            minibatches_list.append( (x_slice, y_slice) )
        step_vals = algorithm.update(minibatches_list, unlabeled=None)
        print('step_vals', step_vals)


        criterion = torch.nn.CrossEntropyLoss() # For compatible 


        if MEASURE_SUBSET_INFLUENCE and batch_id % 20 == 0:
            train_batch_grads = algorithm.calculate_gradient_for_influence(minibatches_list)
            subset_influence_batch_results['sample_schedule'].append(ImageIDs)
            subset_influence_batch_results['train_batch_grads'].append(train_batch_grads)

            print('out-of-domain val')
            logging.info('out-of-domain val')
            acc1, dump_result_dict = validate(val_out_of_domain_loader, model, criterion, args, dumpResult=True, get_grads=True, algorithm=algorithm)


            subset_influence_batch_results['batch_results'].append(dump_result_dict['pred_score_all'])
            subset_influence_batch_results['target'] = dump_result_dict['target_all']
            subset_influence_batch_results['val_all_grads'].append(dump_result_dict['val_grad_list'])




        if batch_id % 20 == 0:
            print('Iteration:', batch_id)
            ##################################
            # Each Epoch: Save periodically or at the end
            ##################################
            if MEASURE_SUBSET_INFLUENCE:
                save_influence_results(subset_influence_batch_results, batch_id, args)


            print('out-of-domain val')
            logging.info('out-of-domain val')
            acc1, _ = validate(val_out_of_domain_loader, model, criterion, args, dumpResult=True)
            # Report every-group acc, worst-set acc 
            report_every_set_acc(val_out_of_domain_dataset, args)
    
    return



if __name__ == "__main__":
    main_generalization()

