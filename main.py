import argparse
import copy
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from codinit import initialize
from sklearn import metrics as sklearn_metrics
from tqdm import tqdm

import wandb
from data_utils.loader import get_data
from rnn import RNNModel, Ensemble
from focal_loss import FocalLoss

warnings.filterwarnings('ignore')

global LOGGER   # Initialized at the bottom


def log_dic(dic, step):
    for key, value in dic.items():
        LOGGER.record(key, value)
    LOGGER.dump(step=step)


class Trainer:
    def __init__(
            self,
            create_model,
            learning_rate,
            save_dir,
            device='cuda:0',
            seq_len=10,
            classes_dic=None,
            resume_from=None
        ):
        self.device = device
        self.save_dir = save_dir
        self.seq_len = seq_len
        self.loss_fn = self.define_loss_fn()
        self.classes_dic = classes_dic
        self.start_step = 0.

        if resume_from is None or len(resume_from) == 1:
            self.model = create_model().to(self.device)
            self.optimiser = self.define_optimiser(self.model, learning_rate)

            if resume_from is not None:
                self.model.module.load_state_dict(torch.load(resume_from[0], map_location=torch.device(self.device)))
                # Models are stored as *epoch_itr.pt
                if len(s := os.path.basename(resume_from[0]).split('_')) > 1:
                    self.start_step = int(s[-1].strip('.pt'))

        else:
            # Ensemble - will only be testing
            ensemble = []
            for path in resume_from:
                model = create_model().to(self.device)
                model.module.load_state_dict(torch.load(path, map_location=torch.device(self.device)))
                ensemble.append(model)
            self.model = Ensemble(*ensemble)

    def define_loss_fn(self):
        return FocalLoss(alpha=None, gamma=3)
        #return nn.CrossEntropyLoss(size_average=False)

    def define_optimiser(self, model, learning_rate):
        return optim.Adam([
            {'params': model.module.share.parameters()},
            {'params': model.module.lstm.parameters(), 'lr': learning_rate},
            {'params': model.module.fc.parameters(), 'lr': learning_rate},
        ], lr=learning_rate / 10)

    def compute_metrics(self, preds, labels, prepend=None, ignore_labels=None):
        if ignore_labels is not None:
            preds = [p for p, l in zip(preds, labels) if l not in ignore_labels]
            labels = [l for l in labels if l not in ignore_labels]

        preds, labels = np.array(preds), np.array(labels)
        classes = list(sorted(set(preds).union(set(labels))))

        # Global
        accuracy = np.mean(preds == labels)
        recall = sklearn_metrics.recall_score(labels, preds, labels=classes, average='macro')
        precision = sklearn_metrics.precision_score(labels, preds, labels=classes, average='macro')
        jaccard = sklearn_metrics.jaccard_score(labels, preds, labels=classes, average='macro')
        # Per class
        class_precision = sklearn_metrics.precision_score(labels, preds, labels=classes, average=None)
        class_recall = sklearn_metrics.recall_score(labels, preds, labels=classes, average=None)

        # Metrics dictionary
        if prepend is None:
            prepend = ''
        metrics = {
                f'{prepend}accuracy': accuracy,
                f'{prepend}recall': recall,
                f'{prepend}precision': precision,
                f'{prepend}jaccard': jaccard,
        }
        # TODO: Handle case where classes_dic is None
        val_or_zero = lambda lst, idx: lst[classes.index(idx)] if idx in classes else 0
        for name, idx in self.classes_dic.items():
            metrics[f'{prepend}precision_{name}'] = val_or_zero(class_precision, idx)
            metrics[f'{prepend}recall_{name}'] = val_or_zero(class_recall, idx)
        return metrics

    def smooth_list(self, original, trans_dic={7: 200}, trans_default=10):
        """Ignore transitions that are less than a specific length"""
        trans_fn = lambda x: trans_dic.get(x, trans_default)

        corrected, buf, last, pot = [], 0, None, None
        for i, x in enumerate(original):
            if last is None:
                last = x

            if x == last:
                corrected.append(x)
                if buf > 0:
                    corrected += [last] * buf
                    buf = 0
            else:
                if pot is None:
                    pot = x
                elif pot != x:
                    # Reset buffer
                    corrected += [last] * (buf+1)
                    buf = 0
                    pot = None
                    continue

                if buf < trans_fn(x) and i != len(original) - 1:
                    buf += 1
                elif buf < trans_fn(x) and i == len(original) - 1:
                    corrected += [last] * (buf+1)
                else:
                    corrected += original[i-buf:i+1]
                    last = None
                    buf = 0
        assert len(original) == len(corrected)
        return corrected

    def analysis(self, data_loader):
        labels = []
        for i, (X, y) in enumerate(tqdm(data_loader)):
            labels += list(y.cpu().numpy())

        labels = self.smooth_list(labels)

        # Minimum length of a single phase
        lengths = {}
        cnt, last = 0, None
        buf = 0
        for i, l in enumerate(labels):
            if last is None:
                last = l
            cnt += 1
            if l != last or i == len(labels)-1:
                if buf < 0:
                    buf += 1
                else:
                    lengths[l] = lengths.get(l, []) + [cnt-buf]
                    last = None
                    cnt = buf
            elif buf > 0:
                buf = 0

        for k, v in sorted(lengths.items()):
            print(f'{k}: {np.sum(v)/len(labels)}')

    def test(self, test_loader, prepend):
        # Test
        self.model.eval()
        total_loss = 0.
        all_labels, all_preds = [], []
        for i, (X, y) in enumerate(tqdm(test_loader)):
            X = X.to(self.device)
            y = y.to(self.device)

            # Labels
            y = y[(self.seq_len - 1)::self.seq_len]

            # Get preds
            X = X.view(-1, self.seq_len, 3, 224, 224)
            y_probs = self.model(X)
            y_probs = y_probs[self.seq_len - 1::self.seq_len]
            _, y_hat = torch.max(y_probs, 1)

            # Save
            total_loss += self.loss_fn(y_probs, y).item()
            all_labels += list(y.cpu().numpy())
            all_preds += list(y_hat.cpu().numpy())

        self.model.train()

        # Smooth out preds
        all_preds = self.smooth_list(all_preds)
        all_labels = self.smooth_list(all_labels)

        # Metrics
        metrics = self.compute_metrics(
            all_preds,
            all_labels,
            prepend,
            [7]
        )
        metrics[f'{prepend}loss'] = total_loss / len(all_labels)
        return metrics

    def train(
            self,
            epochs,
            train_loader_fn,
            val_loader,
            print_every=10000
        ):
        step, tic = 0, time.time()
        best_test_acc, best_test_loss = 0., np.inf
        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loader = train_loader_fn()
            for i, (X, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
                if step < self.start_step:
                    step += 1
                    continue       # just skip to resume step, want to resume exact state of data

                X = X.to(self.device)
                y = y.to(self.device)

                # Labels
                y = y[(self.seq_len - 1)::self.seq_len]

                # Forward pass
                X = X.view(-1, self.seq_len, 3, 224, 224)
                y_hat = self.model(X)
                y_hat = y_hat[(self.seq_len - 1)::self.seq_len]

                # Backward pass
                loss = self.loss_fn(y_hat, y)
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                # Print
                if step % print_every == 0:
                    preds = torch.argmax(y_hat, 1)
                    acc = torch.mean(preds == y, dtype=torch.float32)
                    metrics = {
                        'train/loss': loss.item(),
                        'train/accuracy': acc.item(),
                        'time/epoch': epoch,
                        'time/step': step,
                        'time/time(m)': (time.time()-tic)/60
                    }
                    log_dic(metrics, step)
                    torch.save(self.model.module.state_dict(), os.path.join(self.save_dir, f'{epoch}_{step}.pt'))

                step += 1

            if step < self.start_step:
                continue       # just skip to resume step

            # Validate
            with torch.no_grad():
                metrics = self.test(val_loader, 'test/')
            best_test_acc = max(best_test_acc, metrics['test/accuracy'])
            best_test_loss = min(best_test_loss, metrics['test/loss'])
            metrics.update({
                'time/epoch': epoch,
                'time/step': step,
                'time/time(m)': (time.time()-tic)/60,
                'best/test_accuracy': best_test_acc,
                'best/test_loss': best_test_loss
            })
            log_dic(metrics, step)

            torch.save(self.model.module.state_dict(), os.path.join(self.save_dir, f'end_{epoch}_{step}.pt'))


def main(config):
    # Load data
    train_dataset, val_dataset, classes_dic = get_data('./train_val_paths_labels.pkl')
    # Define model
    create_model = lambda : nn.DataParallel(RNNModel())
    # Train
    trainer = Trainer(
        create_model,
        config.learning_rate,
        config.save_dir,
        classes_dic=classes_dic,
        resume_from=config.resume_from
    )
    if config.only_analysis:
        print('Analysing')
        with torch.no_grad():
            metrics = trainer.analysis(val_dataset)
    elif config.only_test:
        print('Testing')
        with torch.no_grad():
            metrics = trainer.test(val_dataset, 'test/')
        log_dic(metrics, 0)
    else:
        print('Training')
        trainer.train(config.epochs, train_dataset, val_dataset)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project', default='surgical_detection')
    parser.add_argument('-n', '--name', default=None)
    parser.add_argument('-g', '--group', default=None)
    parser.add_argument('-s', '--seed', default=None)
    parser.add_argument('-bd', '--base_dir', default='.')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-rf', '--resume_from', type=str, nargs='*', default=None)
    parser.add_argument('-ot', '--only_test', action='store_true')
    parser.add_argument('-oa', '--only_analysis', action='store_true')
    args = parser.parse_args()

    global LOGGER
    _, LOGGER = initialize(
        parser,
        ignore_keys=['project', 'name', 'group', 'seed', 'base_dir', 'epochs'],
        path_keys=['resume_from']
    )
    main(wandb.config)
