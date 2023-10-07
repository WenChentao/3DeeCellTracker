# -*- coding: utf-8 -*-

# This code is for using transformer to learn a registration between different
# neuron configurattion.

from data_prep import compile_neuron_data, find_match
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
import numpy as np
import math
import pickle
import argparse
import time
import glob
import os
from model_utils import NIT_Registration, NIT

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

def add_random(pts, shuffle=True, sigma=3):
    # TODO: add randomness into the point sets( add outlier and delete some points, add Gaussian noise)
    # add some neuron
    num_neuron_ori = len(pts)
    num_add = np.random.randint(20, size=1)[0]
    add_idx = np.random.randint(num_neuron_ori, size=num_add)
    add_neu = pts[add_idx, :] + np.random.randn(num_add, pts.shape[1]) * sigma
    add_neu[:, -1] = -1
    pts = np.vstack((pts, add_neu))

    # miss some neurons.
    remain_idx = np.where(np.random.rand(len(pts)) > 0.1)[0]
    pts = pts[remain_idx, :]

    if shuffle:
        np.random.shuffle(pts)
    return pts

def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (pt1_path, pt2_path)): list of tuples containing two point sets to be matched.
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        pt1_batch = list()
        pt2_batch = list()
        match_batch = list()
        for e in examples:
            # first 3 elements is x y z, 4th one is the label.
            pt1 = add_random(np.load(e[0]))
            pt2 = add_random(np.load(e[1]))
            pt1_batch.append(pt1[:, :3])
            pt2_batch.append(pt2[:, :3])
            match_batch.append(find_match(pt1[:, 3], pt2[:, 3]))
        yield pt1_batch, pt2_batch, match_batch


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k*self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointTransFeat(nn.Module):
    """
    This class transform the point(rotation) and features.
    """
    def __init__(self, rotate=False, feature_transform=False, input_dim=3, hidden_d=64):
        super(PointTransFeat, self).__init__()
        self.hidden_d = hidden_d
        self.rotate = rotate
        if rotate:
            self.stn = STNkd(k=input_dim)

        self.conv1 = torch.nn.Conv1d(input_dim, hidden_d, 1)
        self.bn1 = nn.BatchNorm1d(hidden_d)

        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=hidden_d)

    def forward(self, x):
        n_pts = x.size()[2]
        if self.rotate:
            trans = self.stn(x)
            self.reg_stn = self.feature_transform_regularizer(trans)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
        else:
            self.reg_stn = 0

        x = self.bn1(self.conv1(x))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
            self.reg_fstn = self.feature_transform_regularizer(trans_feat)
        else:
            self.reg_fstn = 0

        return x

    def feature_transform_regularizer(self, trans):
        d = trans.size()[1]
        I = torch.eye(d)[None, :, :]
        if trans.is_cuda:
            I = I.cuda()
        loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
        return loss

class NNR(nn.Module):
    """ Simple Neural Neuron Registration Model:
        - Transformer
    """
    def __init__(self, input_dim, n_hidden, n_layer=6, cuda=True, p_rotate=False, feat_trans=False):
        """ Init Model

        """
        super(NNR, self).__init__()

        self.p_rotate = p_rotate
        self.feat_trans = feat_trans
        n_hidden *= 2

        self.input_dim = input_dim
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.cuda = cuda

        # Linear Layer with bias), project 3d coordinate into hidden dimension.
        self.point_f = PointTransFeat(rotate=self.p_rotate, feature_transform=feat_trans,
                                          input_dim=input_dim, hidden_d=n_hidden)

        self.enc_l = nn.TransformerEncoderLayer(d_model=n_hidden, nhead=8)
        self.model = nn.TransformerEncoder(self.enc_l, n_layer)

        self.device = torch.device("cuda:0" if cuda else "cpu")

    def encode(self, pts_padded, pts_length):
        # pts_padded should be of (b, num_pts, 3)
        #pts_proj = self.h_projection(pts_padded)
        pts_proj = self.point_f(pts_padded.transpose(2, 1))
        pts_proj = pts_proj.transpose(2, 1)

        mask = self.generate_sent_masks(pts_proj, pts_length)
        # add the src_key_mask need to test.
        pts_encode = self.model(pts_proj.transpose(dim0=0, dim1=1), src_key_padding_mask=mask)

        return pts_encode.transpose(dim0=0, dim1=1)

    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths) -> torch.Tensor:
        """ Generate sentence masks for encoder hidden states.

        @param enc_hiddens (Tensor): encodings of shape (b, src_len, 2*h), where b = batch size,
                                     src_len = max source length, h = hidden size.
        @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.

        @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                    where src_len = max source length, h = hidden size.
        """
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float).bool()
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = True
        return enc_masks.to(self.device)

    def forward(self, pts, match_dict=None, mode='train'):
        """ Take a mini-batch of "source" and "target" points, compute the encoded hidden code for each
        neurons.
        @param pts1 (List[List[x, y, z]]): list of source points set
        @param pts2 (List[List[x, y, z]]): list of target points set

        @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each example in the input batch. Here b = batch size.
        """
        # Compute sentence lengths
        pts_lengths = [len(s) for s in pts]
        #pts2_lengths = [len(s) for s in pts2]

        # pts_padded should be of dimension (
        pts_padded = self.to_input_tensor(pts)
        #pts2_padded = self.to_input_tensor(pts2)

        pts_encode = self.encode(pts_padded, pts_lengths)
        #pts2_encode = self.encode(pts2_padded, pts2_lengths)
        # pts_encode is of size (b, n_pt, n_dim)
        n_io = int(self.n_hidden / 2)
        pts_encode_i = pts_encode[:, :, :n_io]
        pts_encode_o = pts_encode[:, :, n_io:]
        batch_sz = pts_encode.size(0)
        loss = 0
        num_pt = 0
        output_pairs = dict()

        # Here we can see if only use part of it.
        for i_pt in range(batch_sz):
            pts_encode_single = pts_encode_i[i_pt:i_pt+1, :, :]
            pts_encode_single = pts_encode_single.expand_as(pts_encode_o).transpose(dim0=1, dim1=2)
            # sim_m is of size (b, n_pt, n_pt(copy of ith one))
            sim_m = torch.bmm(pts_encode_o, pts_encode_single)
            sim_m = sim_m[:, :, :pts_lengths[i_pt]]
            p_m = F.log_softmax(sim_m, dim=2)
            if (mode == 'train') or (mode == 'all'):
                for i_ref in range(batch_sz):
                    if i_ref < i_pt:
                        i_1 = i_ref
                        i_2 = i_pt
                        col1 = 0
                        col2 = 1
                    else:
                        i_1 = i_pt
                        i_2 = i_ref
                        col1 = 1
                        col2 = 0
                    match = match_dict[str(i_1) + '_' + str(i_2)]
                    if len(match) > 0:
                        match_ref = match[:, col1]
                        match_pt = match[:, col2]

                        log_p = p_m[i_ref, match_ref, match_pt]
                        loss -= log_p.sum()
                        num_pt += len(match_pt)
            elif (mode == 'eval') or (mode == 'all'):
                if i_pt == 0:
                    output_pairs['p_m'] = p_m
                    # choose the matched pairs for worm1
                    paired_idx = torch.argmax(p_m, dim=1)
                    output_pairs['paired_idx'] = paired_idx
                    # TODO:
                    # pick the maxima value

        loss_dict = dict()
        loss_dict['loss'] = loss
        loss_dict['num'] = num_pt
        if self.p_rotate:
            loss_dict['reg_stn'] = self.point_f.reg_stn
        if self.feat_trans:
            loss_dict['reg_fstn'] = self.point_f.reg_fstn

        return loss_dict, output_pairs


    def to_input_tensor(self, pts, pad_pt=[0, 0, 0]):
        sents_padded = []
        max_len = max(len(s) for s in pts)
        for s in pts:
            padded = [pad_pt] * max_len
            padded[:len(s)] = s
            sents_padded.append(padded)
        #sents_var = torch.tensor(sents_padded, dtype=torch.long, device=self.device)
        sents_var = torch.tensor(sents_padded, dtype=torch.float, device=self.device)
        return sents_var #torch.t(sents_var)


    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NNR(**args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        params = {
            'args': dict(input_dim=self.input_dim,
                         n_hidden=self.n_hidden,
                         n_layer=self.n_layer,
                         cuda=self.cuda),
            'state_dict': self.state_dict()
        }

        torch.save(params, path)

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

class neuron_data_pytorch(Dataset):
    """
    This class is to compile neuron data from different worms
    """
    def __init__(self, path, batch_sz, shuffle, rotate=False, mode='all', ref_idx=0, show_name=False, shuffle_pt = True, tmp_path= None):
        """
        Initialize parameters.
        :param path: the path for all worms
        :param batch_sz: batch_size
        :param shuffle : whether to shuffle the data
        """
        self.path = path
        self.mode = mode
        self.batch_sz = batch_sz
        self.shuffle = shuffle

        self.rotate = rotate
        self.ref_idx = ref_idx
        self.show_name = show_name
        self.shuffle_pt = shuffle_pt

        # set the temp_plate
        if tmp_path is not None:
            # set the ref_idx to 0
            self.ref_idx = 0
            self.tmp_path = tmp_path
            self.load_path(path, batch_sz-1)
        else:
            self.tmp_path = tmp_path
            self.load_path(path, batch_sz)



    def load_path(self, path, batch_sz):
        """
        This function get the folder names + file names(volume) together. Set the index for further use
        :param batch_sz: batch size
        :return:
        """
        if self.mode == 'all':
            self.folders = glob.glob(os.path.join(path, '*/'))
        elif self.mode == 'real':
            self.folders = glob.glob(os.path.join(path, 'real_*/'))
        elif self.mode == 'syn':
            self.folders = glob.glob(os.path.join(path, 'syn_*/'))

        # files in each folder is a list
        all_files = list()
        bundle_list = list()
        num_data = 0

        for folder_idx, folder in enumerate(self.folders):
            if self.mode == 'all':
                volume_list = glob.glob1(folder, '*.npy')
            elif self.mode == 'real':
                volume_list = glob.glob1(folder, 'real_*.npy')
            elif self.mode == 'syn':
                volume_list = glob.glob1(folder, 'syn_*.npy')

            num_volume = len(volume_list)
            num_data += num_volume
            all_files.append(volume_list)

            if batch_sz > num_volume:
                bundle_list.append([folder_idx, 0, num_volume])
            else:
                for i in range(0, num_volume, batch_sz):
                    end_idx = i + batch_sz
                    if end_idx > num_volume:
                        end_idx = num_volume
                        start_idx = num_volume - batch_sz
                    else:
                        start_idx = i

                    bundle_list.append([folder_idx, start_idx, end_idx])

        self.all_files = all_files
        self.bundle_list = bundle_list
        self.batch_num = len(bundle_list)
        print('total volumes:{}'.format(num_data))
        if self.shuffle:
            self.shuffle_batch()

    def __len__(self):
        return self.batch_num

    def shuffle_batch(self):
        """
        This function shuffles every element in all_files list(volume) and bundle list(order of batch)
        :return:
        """
        for volumes_list in self.all_files:
            np.random.shuffle(volumes_list)

        np.random.shuffle(self.bundle_list)

    def __getitem__(self, item):
        return self.bundle_list[item]


    def load_pt(self, pt_name):
        pt = np.load(pt_name)
        if self.shuffle_pt:
            np.random.shuffle(pt)
        if self.rotate:
            theta = np.random.rand(1)[0] * 2 * np.pi
            r_m = [[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], \
                   [0, 0, 1]]
            pt[:, :3] = np.matmul(pt[:, :3], r_m)
        return pt

    def custom_collate_fn(self, bundle):
        bundle = bundle[0]
        pt_batch = list()
        label_batch = list()
        pt_name_list = list()


        if self.tmp_path is not None:
            if self.show_name:
                pt_name_list.append(self.tmp_path)
            pt = self.load_pt(self.tmp_path)
            pt_batch.append(pt[:, :3])
            label_batch.append(pt[:, 3])

        for volume_idx in range(bundle[1], bundle[2]):
            pt_name = os.path.join(self.folders[bundle[0]], self.all_files[bundle[0]][volume_idx])
            if self.show_name:
                pt_name_list.append(pt_name)

            pt = self.load_pt(pt_name)
            pt_batch.append(pt[:, :3])
            label_batch.append(pt[:, 3])

        match_dict = dict()

        ref_i = self.ref_idx
        for i in range(len(label_batch)):
            match_dict[i], match_dict['unlabel_{}'.format(i)] = find_match(label_batch[i], label_batch[ref_i])
            # get the unlabelled neuron
            #match_dict['unlabel_{}'.format(i)] = np.where(label_batch[i] == -1)[0]
            # get the outlier neuron
            match_dict['outlier_{}'.format(i)] = np.where(label_batch[i] == -2)[0]

        # for i in range(len(label_batch)):
        #     for j in range(i, len(label_batch)):
        #         if i == j:
        #             match_dict[str(i) + '_' + str(j)] = np.array([[tmp, tmp] for tmp in range(len(label_batch[i]))])
        #
        #         else:
        #             match_dict[str(i) + '_' + str(j)] = find_match(label_batch[i], label_batch[j])

        data_batch = dict()
        data_batch['pt_batch'] = pt_batch
        data_batch['match_dict'] = match_dict
        data_batch['pt_label'] = label_batch
        data_batch['ref_i'] = ref_i
        if self.show_name:
            data_batch['pt_names'] = pt_name_list
        return data_batch



if __name__ == "__main__":
    # train the model.
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--log_every", default=50, type=int)
    parser.add_argument("--valid_niter", default=500, type=int,
                        help="perform validation after how many iterations")
    parser.add_argument("--model_path", default="/projects/LEIFER/Xinwei/github/NeuronNet/model", type=str)
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
    parser.add_argument("--train_path", default="/projects/LEIFER/Xinwei/github/NeuronNet/pts_id/train", type=str)
    parser.add_argument("--eval_path", default="/projects/LEIFER/Xinwei/github/NeuronNet/pts_id/test", type=str)
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
    # with open(args.data_path, 'rb') as fp:
    #     train_data = pickle.load(fp)
    # dev_data = train_data
    train_data_loader = DataLoader(train_data, shuffle=False, num_workers=1, collate_fn=train_data.custom_collate_fn)
    dev_data_loader = DataLoader(dev_data, shuffle=False, num_workers=1, collate_fn=dev_data.custom_collate_fn)

    device = torch.device("cuda:0" if cuda else "cpu")

    if args.model_name == 'nitReg':
        model = NIT_Registration(input_dim=3, n_hidden=args.n_hidden, n_layer=args.n_layer, p_rotate=args.p_rotate,
                    feat_trans=args.f_trans)
        if args.use_pretrain:
            pretrain_path = '/scratch/gpfs/xinweiy/github/NeuronNet/model/reg_nh128_nl6_ft0_dataall_elam_0.1_627.bin'
            params = torch.load(pretrain_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(params['state_dict'])

    else:
        model = NIT(input_dim=3, n_hidden=args.n_hidden, n_layer=args.n_layer, p_rotate=args.p_rotate,
                    feat_trans=args.f_trans)
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







