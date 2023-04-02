import argparse
import copy
import math
import random
import time
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.utils.data import dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator

from models import TransformerModel
from comet import get_comet_experiment
from comet_ml import Artifact

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_batch(source: Tensor, i: int, bptt: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target


def batchify(data: Tensor, bsz: int) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)


def get_data(batch_size):

    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(
        map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
        """Converts raw text into a flat Tensor."""
        data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long)
                for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    # train_iter was "consumed" by the process of building the vocab,
    # so we have to create it again
    train_iter, val_iter, test_iter = WikiText2()
    train_data = data_process(train_iter)
    val_data = data_process(val_iter)
    test_data = data_process(test_iter)

    # shape [seq_len, batch_size]
    train_data = batchify(train_data, batch_size)
    val_data = batchify(val_data, batch_size)
    test_data = batchify(test_data, batch_size)

    return train_data, val_data, test_data, len(vocab)


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


def train(model: nn.Module, train_data, bptt, criterion, ntokens, optimizer, scheduler, epoch) -> None:
    global best_ppl, cur_loss, ppl, lr
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 100
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(bptt).to(device)

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i, bptt)
        seq_len = data.size(0)
        if seq_len != bptt:  # only on last batch
            src_mask = src_mask[:seq_len, :seq_len]
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            best_ppl = min(best_ppl, ppl)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.6f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()


def evaluate(model: nn.Module, eval_data: Tensor, bptt, ntokens, criterion) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i, bptt)
            seq_len = data.size(0)
            if seq_len != bptt:
                src_mask = src_mask[:seq_len, :seq_len]
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += seq_len * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)


# https://github.com/pytorch/pytorch/issues/32651#issuecomment-643791116
class MySummaryWriter(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        logdir = self._get_file_writer().get_logdir()

        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)


class Config:
    def __init__(self) -> None:
        # gbo
        self.gbo_p = 0.1
        self.gbo_posemb_p = 0.0
        self.gbo_on = True

        # transformer
        self.batch_size = 32  # training
        self.eval_batch_size = 32
        self.bptt = 128  # backpropogation in time, i.e. seq len of training batches
        self.emsize = 384  # embedding dimension
        self.d_hid = 384  # dimension of the feedforward network model in nn.TransformerEncoder
        self.nlayers = 6  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        self.nhead = 8  # number of heads in nn.MultiheadAttention
        self.pos_dropout = 0.25  # positional dropout probability
        self.enc_do = 0.35  # encoder layer dropout
        self.use_pos_embed = True

        self.SEED = 9999

        # learning rate
        self.momentum = 0.025  # sgd
        self.nesterov = True  # sgd
        self.lr = 5  # step lr
        self.gamma = 0.5  # step lr
        self.lr_step_size = 25  # step lr

        self.epochs = 200


# globals
best_ppl = float('inf')
cur_loss = 0
ppl = 0
lr = 0


def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data, val_data, test_data, ntokens = get_data(config.batch_size)

    model = TransformerModel(ntoken=ntokens, d_model=config.emsize,
                             nhead=config.nhead, d_hid=config.d_hid, nlayers=config.nlayers,
                             block_size=config.bptt, use_pos_embed=config.use_pos_embed, pos_dropout=config.pos_dropout,
                             enc_dropout=config.enc_do,
                             use_gbo=config.gbo_on, gbo_p=config.gbo_p, gbo_posemb_p=config.gbo_posemb_p).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, nesterov=config.nesterov)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_step_size, gamma=config.gamma)

    best_val_loss = float('inf')
    best_val_ppl = float('inf')
    best_model = None

    hyperparam_comment = f"lrss={config.lr_step_size},gbo={config.gbo_on},gbop={config.gbo_p},gbopep={config.gbo_posemb_p},sql={config.bptt},pos={config.use_pos_embed},do={config.pos_dropout},edo={config.enc_do}"
    writer = MySummaryWriter(comment=hyperparam_comment)

    experiment = get_comet_experiment()
    experiment.log_parameters(vars(config))

    try:
        for epoch in range(1, config.epochs + 1):
            epoch_start_time = time.time()

            with experiment.train():
                train(model, train_data, config.bptt, criterion, ntokens, optimizer, scheduler, epoch)
                experiment.log_metric('train/loss', cur_loss, epoch)
                experiment.log_metric('train/ppl', ppl, epoch)

            with experiment.test():
                val_loss = evaluate(model, val_data, config.bptt, ntokens, criterion)
                val_ppl = math.exp(val_loss)
                best_val_ppl = min(best_val_ppl, val_ppl)
                elapsed = time.time() - epoch_start_time
                print('-' * 89)
                print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
                      f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
                print('-' * 89)

                experiment.log_metric('val/loss', val_loss, epoch)
                experiment.log_metric('val/ppl', val_ppl, epoch)

            experiment.log_metric('lr', lr, epoch)

            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('PPL/val', val_ppl, epoch)
            writer.add_scalar('Hyperparms/LR', lr, epoch)
            writer.add_scalar('Loss/train', cur_loss, epoch)
            writer.add_scalar('PPL/train', ppl, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model)

            scheduler.step()
    except KeyboardInterrupt:
        print('Captured Keyboard Interrupt')

    with experiment.test():
        test_loss = evaluate(best_model, test_data, config.bptt, ntokens, criterion)
        test_ppl = math.exp(test_loss)
        print('=' * 89)
        print(f'| End of training | test loss {test_loss:5.2f} | '
              f'test ppl {test_ppl:8.2f}')
        print('=' * 89)

        experiment.log_metric('test/loss', test_loss, epoch)
        experiment.log_metric('test/ppl', test_ppl, epoch)

        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('PPL/test', test_ppl, epoch)

        metrics = {
            'hparam/ppl_train': best_ppl,
            'hparam/ppl_val': best_val_ppl,
            'hparam/ppl_test': test_ppl
        }

        experiment.log_metrics(metrics)

        writer.add_hparams(config.__dict__, metrics)

        print(config.__dict__)
        print(metrics)

    torch.save(best_model.state_dict(), 'best_model.pth')
    artifact = Artifact(name='Model File', artifact_type='model')
    artifact.add('best_model.pth')

    experiment.log_artifact(artifact)

    experiment.end()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('gbo_posemb_p', type=float)
    # TODO the rest

    args = parser.parse_args()

    config = Config()
    config.gbo_posemb_p = args.gbo_posemb_p

    print(args)

    main(config)
