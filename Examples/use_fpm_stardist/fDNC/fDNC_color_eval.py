from DNC_predict import pre_matt
import torch
import argparse
import glob
import os
import pickle
import numpy as np
from model import NIT_Registration, find_match
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


def get_batch(mov_list, temp_p, batch_sz=32):

    mov_sz = batch_sz - 1
    num_i = len(mov_list) // mov_sz + 1
    temp = pre_matt(temp_p)

    for i_iter in range(num_i):
        if i_iter < num_i:
            cur_list = mov_list[i_iter*mov_sz:(i_iter+1)*mov_sz]
        else:
            cur_list = mov_list[i_iter*mov_sz:]

        pt_batch = list()
        label_batch = list()
        color_batch = list()
        pt_name_list = list()

        pt_name_list.append(temp['f_name'])
        pt = temp['pts']

        #np.random.shuffle(pt)

        pt_batch.append(pt[:, :3])
        label_batch.append(pt[:, 3])
        color_batch.append(temp['color'])

        for mov_f in cur_list:
            mov = pre_matt(mov_f)
            pt_name_list.append(mov['f_name'])
            pt = mov['pts']

            #np.random.shuffle(pt)

            pt_batch.append(pt[:, :3])
            label_batch.append(pt[:, 3])
            color_batch.append(mov['color'])

        match_dict = dict()
        ref_i = 0
        for i in range(len(label_batch)):
            match_dict[i], match_dict['unlabel_{}'.format(i)] = find_match(label_batch[i], label_batch[ref_i])
            # get the outlier neuron
            match_dict['outlier_{}'.format(i)] = np.where(label_batch[i] == -2)[0]

        data_batch = dict()
        data_batch['pt_batch'] = pt_batch
        data_batch['color'] = color_batch
        data_batch['match_dict'] = match_dict
        data_batch['pt_label'] = label_batch
        data_batch['ref_i'] = 0
        data_batch['pt_names'] = pt_name_list
        yield data_batch


def match_color_norm(x_cs, y_cs):
    # x_cs = np.log(1 + np.copy(x_cs)[:, :3])
    # y_cs = np.log(1 + np.copy(y_cs)[:, :3])
    x_cs = np.copy(x_cs)[:, :3]
    y_cs = np.copy(y_cs)[:, :3]

    x_c_norm = x_cs / np.sum(x_cs, axis=1, keepdims=True)
    y_c_norm = y_cs / np.sum(y_cs, axis=1, keepdims=True)
    x_c_norm = np.clip(x_c_norm, 1e-5, None)
    y_c_norm = np.clip(y_c_norm, 1e-5, None)

    y_c_log = np.log(y_c_norm)
    x_c_log = np.log(x_c_norm)
    y_H = np.sum(y_c_log * y_c_norm, axis=1)

    color_m = np.sum(x_c_log[np.newaxis, :, :] * y_c_norm[:, np.newaxis, :], axis=2) - y_H[:, np.newaxis]

    return color_m



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--model_path",
                        default="../model/model.bin",
                        type=str)
    parser.add_argument("--eval_path", default="../Data/test_neuropal_our", type=str)
    parser.add_argument("--save", default=0, type=int)
    parser.add_argument("--save_p", default="../results/match_results", type=str)
    parser.add_argument("--cuda", default=1, type=int)
    parser.add_argument("--n_hidden", default=128, type=int)
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--method", default='hung', type=str)
    parser.add_argument("--show_name", default=0, type=int)
    parser.add_argument("--ref_idx", default=0, type=int)
    parser.add_argument("--conf_thd", default=0.00, type=float)
    parser.add_argument("--temp_p", default='../Data/test_neuropal_all_features/20191213_142710.data', type=str)
    parser.add_argument("--mov_p", default='../Data/test_neuropal_all_features/mov', type=str)
    args = parser.parse_args()


    mov_fs = glob.glob(os.path.join(args.mov_p, "*.data"))
    with open(args.temp_p, 'rb') as fp:
        data = pickle.load(fp)
        fp.close()

    # load model
    model = NIT_Registration(input_dim=3, n_hidden=args.n_hidden, n_layer=args.n_layer, cuda=args.cuda,
                             p_rotate=0, feat_trans=0)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    params = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(params['state_dict'])
    model = model.to(device)

    num_hit_all = num_hit_all_pos = 0
    num_match_all = num_match_all_pos = 0

    num_hit_list = [0] * 100
    num_pair = 0
    save_idx = 0

    color_diff_gt = 0
    color_diff_rand = 0
    color_num = 0

    batch_gen = get_batch(mov_fs, args.temp_p, args.batch_size)

    acc_list = list()
    acc_list_pos = list()

    thd_ratio_list_c = list()
    thd_ratio_list = list()


    for data_batch in batch_gen:
        model.eval()
        #print('batch:{}'.format(batch_idx))

        pt_batch = data_batch['pt_batch']
        match_dict = data_batch['match_dict']
        label = data_batch['pt_label']
    #for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
        #loss = -model(src_sents, tgt_sents).sum()
        with torch.no_grad():
            _, output_pairs = model(pt_batch, match_dict=match_dict, ref_idx=data_batch['ref_i'], mode='eval')
        num_worm = output_pairs['p_m'].size(0)
        # p_m is the match of worms to the worm0
        for i in range(0, num_worm):
            if i == data_batch['ref_i']:
                continue

            p_m = output_pairs['p_m'][i].detach().cpu().numpy()
            num_neui = len(pt_batch[i])
            p_m = p_m[:num_neui, :]
            #color_m = match_color_multiple(data_batch['color'][data_batch['ref_i']], data_batch['color'][i])
            color_m = match_color_norm(data_batch['color'][data_batch['ref_i']], data_batch['color'][i]) * 60

            p_m_pos = np.copy(p_m)
            p_m = p_m[:, :-1] + color_m * 1



            if args.method == 'hung':
                row, col = linear_sum_assignment(-p_m)
            if args.method == 'max':
                col = np.argmax(p_m, axis=1)
                row = np.array(range(num_neui))

            row_mask = p_m_pos[row, col] >= np.log(args.conf_thd)
            thd_ratio = np.sum(row_mask) / max(1, len(row))
            print('thd_ratio:{}'.format(thd_ratio))
            thd_ratio_list_c.append(thd_ratio)
            row = row[row_mask]
            col = col[row_mask]


            if args.save:
                save_file = os.path.join(args.save_p, 'match_{}.pt'.format(save_idx))
                out_dict = dict()
                out_dict['ref_pt'] = pt_batch[data_batch['ref_i']]
                out_dict['ref_label'] = label[data_batch['ref_i']]
                out_dict['mov_pt'] = pt_batch[i]
                out_dict['ref_name'] = data_batch['pt_names'][data_batch['ref_i']]
                out_dict['mov_name'] = data_batch['pt_names'][i]
                cur_label = np.concatenate((label[data_batch['ref_i']], np.array([-2])))

                out_dict['mov_label'] = cur_label[col]
                out_dict['gt_label'] = np.ones(pt_batch[i].shape[0]) * -1
                out_dict['gt_label'][match_dict[i][:, 0]] = label[data_batch['ref_i']][match_dict[i][:, 1]]
                out_dict['origin_label'] = label[i]
                out_dict['col'] = col



            log_p = np.mean(p_m[row, col])
            print('temp:{}, mov:{}'.format(data_batch['pt_names'][data_batch['ref_i']], data_batch['pt_names'][i]))
            print('avg log prob:{}'.format(log_p))

            gt_match = match_dict[i]
            gt_match_dict = dict()

            for gt_m in gt_match:
                gt_match_dict[gt_m[0]] = gt_m[1]

            num_match = 0
            num_hit = 0
            for r_idx, r in enumerate(row):
                if r in gt_match_dict:
                    num_match += 1
                    if gt_match_dict[r] == col[r_idx]:
                        num_hit += 1
            num_match = len(gt_match_dict)
            acc_m = num_hit / max(1, num_match)
            num_hit_all += num_hit
            num_match_all += num_match
            print('num_hit:{}, num_match:{}, accuracy:{}'.format(num_hit, num_match, acc_m))
            acc_list.append(acc_m)

            if args.save:
                out_dict['accuracy'] = acc_m
                with open(save_file, 'wb') as f:
                    pickle.dump(out_dict, f)
                    f.close()
                save_idx += 1

            # get the top rank for gt match.
            num_pair += len(gt_match)
            for gt_m in gt_match:
                topn = np.sum(p_m[gt_m[0]] >= p_m[gt_m[0], gt_m[1]])
                for i_rank in range(10):
                    if topn <= i_rank+1:
                        num_hit_list[i_rank] += 1

            if args.method == 'hung':
                row, col = linear_sum_assignment(-p_m_pos)
            if args.method == 'max':
                col = np.argmax(p_m_pos, axis=1)
                row = np.array(range(num_neui))

            row_mask = p_m_pos[row, col] >= np.log(args.conf_thd)
            thd_ratio = np.sum(row_mask) / max(1, len(row))
            print('thd_ratio:{}'.format(thd_ratio))
            thd_ratio_list.append(thd_ratio)
            row = row[row_mask]
            col = col[row_mask]

            num_match = 0
            num_hit = 0
            for r_idx, r in enumerate(row):
                if r in gt_match_dict:
                    num_match += 1
                    if gt_match_dict[r] == col[r_idx]:
                        num_hit += 1
            acc_m = num_hit / max(1, num_match)
            num_hit_all_pos += num_hit
            num_match_all_pos += num_match
            print('num_hit:{}, num_match:{}, Pos accuracy:{}'.format(num_hit, num_match, acc_m))
            acc_list_pos.append(acc_m)

    print('Color accuracy:{}'.format(num_hit_all / num_match_all))
    print('Pos accuracy:{}'.format(num_hit_all_pos / num_match_all_pos))
    num_hit_list = np.array(num_hit_list) / num_pair
    print(num_hit_list[:10])

    #print('avg diff_gt:{}, avg diff_rand:{}'.format(color_diff_gt / color_num, color_diff_rand / color_num))
    print('color cover ratio:{}, original cover ratio:{}'.format(np.mean(thd_ratio_list_c), np.mean(thd_ratio_list)))
    out = dict()
    out['trans_w_c'] = np.array(acc_list)
    out['trans_w_p'] = np.array(acc_list_pos)
