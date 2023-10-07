import argparse
from torch.utils.data import DataLoader
from model import NIT_Registration, neuron_data_pytorch
import torch
import math
import time
import os
import numpy as np

def evaluate_ppl(model, dev_data_loader):
    """ Evaluate perplexity on dev sentences
    @param model (NMT): NMT Model
    @param dev_data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (batch size)
    @returns ppl (perplixty on dev sentences)
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        #for pt1_batch, pt2_batch, match_batch in batch_iter(dev_data, batch_size=batch_size):
        for batch_idx, data_batch in enumerate(dev_data_loader):
            #for pt_batch, match_dict in train_data.batch_iter():
            pt_batch = data_batch['pt_batch']
            match_dict = data_batch['match_dict']
        #for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            #loss = -model(src_sents, tgt_sents).sum()
            loss, _ = model(pt_batch, match_dict=match_dict, mode='train')

            cum_loss += loss['loss'].item()
            tgt_word_num_to_predict = loss['num']  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl


if __name__ == "__main__":
    # train the model.
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--log_every", default=50, type=int)
    parser.add_argument("--valid_niter", default=500, type=int,
                        help="perform validation after how many iterations")
    parser.add_argument("--model_path", default="../model", type=str)
    parser.add_argument("--lr_decay", default=0.5, type=float, help="learning rate decay")
    parser.add_argument("--max_num_trial", default=10, type=int,
                        help="terminate training after how many trials")
    parser.add_argument("--patience", default=5, type=int,
                        help="wait for how many iterations to decay learning rate")
    parser.add_argument("--clip_grad", default=1.0, type=float)
    parser.add_argument("--max_epoch", default=500, type=int)
    parser.add_argument("--n_hidden", default=48, type=int)
    parser.add_argument("--n_layer", default=8, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--stn_lam", default=1, type=float)
    parser.add_argument("--fstn_lam", default=1, type=float)
    parser.add_argument("--p_rotate", default=1, type=int)
    parser.add_argument("--f_trans", default=1, type=int)
    parser.add_argument("--cuda", default=1, type=int)
    parser.add_argument("--train_path", default="../Data/train", type=str)
    parser.add_argument("--eval_path", default="../Data/test", type=str)
    parser.add_argument("--data_mode", default="all", type=str)
    parser.add_argument("--model_idx", default=627, type=int)
    parser.add_argument("--lamb_entropy", default=0.1, type=float)
    parser.add_argument("--model_name", default='nitReg', type=str)
    parser.add_argument("--use_pretrain", default=0, type=int)
    tic = time.time()
    args = parser.parse_args()
    print('n_hidden:{}\n'.format(args.n_hidden))
    print('n_layer:{}\n'.format(args.n_layer))
    print('learn rate:{}\n'.format(args.lr))
    print('data mode:{}\n'.format(args.data_mode))
    cuda = args.cuda

    # loading the data
    train_data = neuron_data_pytorch(args.train_path, batch_sz=args.batch_size, shuffle=True, rotate=True, mode=args.data_mode)
    dev_data = neuron_data_pytorch(args.eval_path, batch_sz=args.batch_size, shuffle=False, rotate=True, mode=args.data_mode)


    train_data_loader = DataLoader(train_data, shuffle=False, num_workers=1, collate_fn=train_data.custom_collate_fn)
    dev_data_loader = DataLoader(dev_data, shuffle=False, num_workers=1, collate_fn=dev_data.custom_collate_fn)

    device = torch.device("cuda:0" if cuda else "cpu")


    model = NIT_Registration(input_dim=3, n_hidden=args.n_hidden, n_layer=args.n_layer, p_rotate=args.p_rotate,
                feat_trans=args.f_trans)
    if args.use_pretrain:
        pretrain_path = '../model/model.bin'
        params = torch.load(pretrain_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(params['state_dict'])


    model_name = '{}_nh{}_nl{}_ft{}_data{}_elam_{}_{}.bin'.format(args.model_name, args.n_hidden, args.n_layer,
                                                                  args.f_trans, args.data_mode, args.lamb_entropy,
                                                                  args.model_idx)
    model_save_path = os.path.join(args.model_path, model_name)
    model.train()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()

    while True:
        epoch += 1
        batch_num = train_data.batch_num
        current_iter = 0

        if train_data_loader.dataset.shuffle:
            train_data_loader.dataset.shuffle_batch()
        #for pt1_batch, pt2_batch, match_batch in batch_iter(train_data, batch_size=args.batch_size, shuffle=True):
        for batch_idx, data_batch in enumerate(train_data_loader):
            #for pt_batch, match_dict in train_data.batch_iter():
            pt_batch = data_batch['pt_batch']
            match_dict = data_batch['match_dict']
            #print('batch to batch time:{}'.format(time.time() - tic))
            #tic = time.time()
            current_iter += 1
            train_iter += 1

            optimizer.zero_grad()
            batch_size = len(pt_batch)

            batch_loss, _ = model(pt_batch, match_dict=match_dict, ref_idx=data_batch['ref_i'], mode='train')
             #batch_loss = example_losses.sum()
            loss = batch_loss['loss'] / batch_loss['num'] + args.stn_lam * batch_loss['reg_stn'] + \
                   args.fstn_lam * batch_loss['reg_fstn'] + args.lamb_entropy * batch_loss['loss_entropy'] / batch_loss['num_unlabel']
            loss.backward()
            # print('batch loss:{}'.format(batch_loss['loss'] / batch_loss['num']))
            # print('reg_stn:{}'.format(batch_loss['reg_stn'].item()))
            # print('reg_fstn:{}'.format(batch_loss['reg_fstn'].item()))

            # clip gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()

            batch_losses_val = batch_loss['loss'].item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            # omitting leading `<s>`
            tgt_words_num_to_predict = batch_loss['num']
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % args.log_every == 0:
                print('epoch %d (%d / %d), iter %d, avg. loss %.2f, avg. ppl %.2f '
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' %
                      (epoch, current_iter, batch_num, train_iter,
                       report_loss / report_examples,
                       math.exp(report_loss / report_tgt_words),
                       cum_examples,
                       report_tgt_words / (time.time() - train_time),
                       time.time() - begin_time))

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # perform validation
            if train_iter % args.valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                      cum_loss / cum_examples,
                      np.exp(cum_loss / cum_tgt_words),
                      cum_examples))

                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...')

                # compute dev. ppl and bleu
                dev_ppl = evaluate_ppl(model, dev_data_loader)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl))

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('epoch %d, iter %d: save currently the best model to [%s]' %
                          (epoch, train_iter, model_save_path))
                    model.save(model_save_path)
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
                elif patience < args.patience:
                    patience += 1
                    print('hit patience %d' % patience)

                    if patience == args.patience:
                        num_trial += 1
                        print('hit #%d trial' % num_trial)
                        if num_trial == args.max_num_trial:
                            print('early stop!')
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]['lr'] * args.lr_decay
                        print('load previously best model and decay learning rate to %f' % lr)

                        # load model
                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers')
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0

            if epoch == args.max_epoch:
                print('reached maximum number of epochs!')
                exit(0)
            #print('finish one batch:{}'.format(time.time() - tic))
    #train_data

    print('Total Run time:{}'.format(time.time()-tic))