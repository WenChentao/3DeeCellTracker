from scipy.optimize import linear_sum_assignment
import torch
import pickle
import numpy as np
from model import NIT_Registration
from scipy.special import softmax

import matplotlib.pyplot as plt


def match_color_norm(x_cs, y_cs):
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

def predict_label(temp_pos, temp_label, test_pos, temp_color=None, test_color=None, cuda=True, topn=5):
    model = NIT_Registration(input_dim=3, n_hidden=128, n_layer=6, p_rotate=0, feat_trans=0, cuda=cuda)
    device = torch.device("cuda:0" if cuda else "cpu")
    # load trained model
    model_path = "../model/model.bin"
    params = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(params['state_dict'])
    model = model.to(device)

    # put template worm data and test worm data into a batch
    pt_batch = list()
    color_batch = list()

    pt_batch.append(temp_pos[:, :3])
    pt_batch.append(test_pos[:, :3])# here we can add more test worm if provided as a list.
    if temp_color is not None and test_color is not None:
        color_batch.append(temp_color)
        color_batch.append(test_color)
    else:
        color_batch = None
    data_batch = dict()
    data_batch['pt_batch'] = pt_batch
    data_batch['color'] = color_batch
    data_batch['match_dict'] = None
    data_batch['ref_i'] = 0

    model.eval()
    pt_batch = data_batch['pt_batch']
    with torch.no_grad():
        _, output_pairs = model(pt_batch, match_dict=None, ref_idx=data_batch['ref_i'], mode='eval')
    # p_m is the match of worms to the worm0
    i = 1
    p_m = output_pairs['p_m'][i].detach().cpu().numpy()
    num_neui = len(pt_batch[i])
    p_m = p_m[:num_neui, :]

    if data_batch['color'] is None:
        color_m = 0
    else:
        color_m = match_color_norm(data_batch['color'][data_batch['ref_i']], data_batch['color'][i]) * 60

    p_m = p_m[:, :-1] + color_m * 1
    row, col = linear_sum_assignment(-p_m)

    prob_m = softmax(p_m, axis=1)

    # most probable label
    test_label = [('', 0)] * num_neui
    for row_i in range(len(row)):
        test_label[row[row_i]] = (temp_label[col[row_i]], prob_m[row[row_i], col[row_i]])
    #candidates list
    p_m_sortidx = np.argsort(-p_m, axis=1)
    candidate_list = []
    for row_i, rank_idx in enumerate(p_m_sortidx[:, :topn]):
        cur_list = [(temp_label[idx], prob_m[row_i, idx])  for idx in rank_idx]
        candidate_list.append(cur_list)
    return test_label, candidate_list



def predict(temp_f, test_f):
    temp = pre_matt(temp_f)
    test = pre_matt(test_f)

    temp_pos = temp['pts']
    temp_label = temp['name']
    temp_color = temp['color']

    test_pos = test['pts']
    test_color = test['color']
    test_label, candidate_list = predict_label(temp_pos, temp_label, test_pos, temp_color, test_color)
    return test_label, candidate_list

def pre_matt(file, scale=200):
    with open(file, 'rb') as fp:
        data = pickle.load(fp)
        fp.close()

    if not 'mask_keep' in data:
        mask_keep = np.arange(len(data['pts']))
    else:
        mask_keep = data['mask_keep']

    label = np.array(data['label'])[mask_keep] if 'label' in data else None
    name = np.array(data['name'])[mask_keep] if 'name' in data else None

    pts = data['pts'][mask_keep, :] / 0.42

    # transform the pts so that it match my results.
    pts -= np.median(pts, axis=0)
    pts /= scale
    if 'side' in data and data['side'] == 0:
        pts[:, [0, 2]] *= -1
    if label is not None:
        label = label[:, np.newaxis]
        pts_out = np.hstack((pts, label))
    else:
        pts_out = pts
    output = dict()
    output['pts'] = pts_out
    output['color'] = data['fluo'][mask_keep, :] if 'fluo' in data else None
    output['label'] = label
    output['name'] = name
    output['f_name'] = file

    return output

if __name__ == "__main__":
    temp_f = '/projects/LEIFER/Xinwei/github/NeuronNet/pts_id/neuropal_eval/20191213_142710.data'
    test_f = '/projects/LEIFER/Xinwei/github/NeuronNet/pts_id/neuropal_eval/mov/20200113_154215.data'
    test_label, candidate_list = predict(temp_f, test_f)
    test_label, candidate_list

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--batch_size", default=64, type=int)
    # parser.add_argument("--model_path", default="/scratch/gpfs2/xinweiy/github/NeuronNet/model/reg_nh128_nl6_ft0_dataall_elam_0.1_627.bin", type=str)
    # parser.add_argument("--eval_path", default="/projects/LEIFER/Xinwei/github/NeuronNet/pts_id/test_neuropal_xinwei_0930", type=str)
    # parser.add_argument("--save", default=0, type=int)
    # parser.add_argument("--save_p", default="/projects/LEIFER/Xinwei/github/NeuronNet/match_results", type=str)
    # parser.add_argument("--cuda", default=1, type=int)
    # parser.add_argument("--n_hidden", default=128, type=int)
    # parser.add_argument("--n_layer", default=6, type=int)
    # parser.add_argument("--method", default='hung', type=str)
    # parser.add_argument("--show_name", default=0, type=int)
    # parser.add_argument("--ref_idx", default=0, type=int)
    # parser.add_argument("--temp_p", default='/projects/LEIFER/Xinwei/github/NeuronNet/pts_id/neuropal_eval/20191213_142710.data', type=str)
    # parser.add_argument("--mov_p", default='/projects/LEIFER/Xinwei/github/NeuronNet/pts_id/neuropal_eval/mov', type=str)
    # args = parser.parse_args()
    #
    #
    # mov_fs = glob.glob(os.path.join(args.mov_p, "*.data"))
    # with open(args.temp_p, 'rb') as fp:
    #     data = pickle.load(fp)
    #     fp.close()
    #
    # # load model
    # model = NIT_Registration(input_dim=3, n_hidden=args.n_hidden, n_layer=args.n_layer, p_rotate=0, feat_trans=0)
    # device = torch.device("cuda:0" if args.cuda else "cpu")
    #
    # params = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    # model.load_state_dict(params['state_dict'])
    # model = model.to(device)
    #
    # num_hit_all = num_hit_all_pos = 0
    # num_match_all = num_match_all_pos = 0
    #
    # num_hit_list = [0] * 100
    # num_pair = 0
    # save_idx = 0
    #
    # color_diff_gt = 0
    # color_diff_rand = 0
    # color_num = 0
    #
    # batch_gen = get_batch(mov_fs, args.temp_p, args.batch_size)
    #
    # acc_list = list()
    # acc_list_pos = list()
    #
    # for data_batch in batch_gen:
    #     model.eval()
    #     #print('batch:{}'.format(batch_idx))
    #
    #     pt_batch = data_batch['pt_batch']
    #     match_dict = data_batch['match_dict']
    #     label = data_batch['pt_label']
    # #for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
    #     #loss = -model(src_sents, tgt_sents).sum()
    #     with torch.no_grad():
    #         _, output_pairs = model(pt_batch, match_dict=match_dict, ref_idx=data_batch['ref_i'], mode='eval')
    #     num_worm = output_pairs['p_m'].size(0)
    #     # p_m is the match of worms to the worm0
    #     for i in range(0, num_worm):
    #         if i == data_batch['ref_i']:
    #             continue
    #
    #         p_m = output_pairs['p_m'][i].detach().cpu().numpy()
    #         num_neui = len(pt_batch[i])
    #         p_m = p_m[:num_neui, :]
    #         #color_m = match_color_multiple(data_batch['color'][data_batch['ref_i']], data_batch['color'][i])
    #         color_m = match_color_norm(data_batch['color'][data_batch['ref_i']], data_batch['color'][i]) * 60
    #         #color_m = np.hstack((color_m.T, np.ones((len(p_m), 1)) * -1.5))
    #         #p_m = p_m + color_m * 0.5
    #         p_m_pos = np.copy(p_m)
    #         p_m = p_m[:, :-1] + color_m * 1
    #
    #         # gt_match = match_dict[i]
    #         # def plot_match(match, mov_i=0):
    #         #     num = p_m.shape[1]
    #         #     x = np.arange(num)
    #         #     plt.scatter(x, p_m[match[mov_i][0]], c='red')
    #         #     plt.scatter(x[:-1], color_m[match[mov_i][0]], c='green')
    #         #     print('ground truth match:{}, p_m:{}, color_m:{}'.format(match[mov_i], p_m[match[mov_i][0], match[mov_i][1]], color_m[match[mov_i][0], match[mov_i][1]]))
    #         #     plt.show()
    #         #
    #         # print(color_m[gt_match[:, 1], gt_match[:, 0]])
    #         # plot_match(gt_match, mov_i=0)
    #         #gt_match
    #
    #         if args.method == 'hung':
    #             row, col = linear_sum_assignment(-p_m)
    #         if args.method == 'max':
    #             col = np.argmax(p_m, axis=1)
    #             row = np.array(range(num_neui))
    #
    #         if args.save:
    #             save_file = os.path.join(args.save_p, 'match_{}.pt'.format(save_idx))
    #             out_dict = dict()
    #             out_dict['ref_pt'] = pt_batch[data_batch['ref_i']]
    #             out_dict['ref_label'] = label[data_batch['ref_i']]
    #             out_dict['mov_pt'] = pt_batch[i]
    #             out_dict['ref_name'] = data_batch['pt_names'][data_batch['ref_i']]
    #             out_dict['mov_name'] = data_batch['pt_names'][i]
    #             cur_label = np.concatenate((label[data_batch['ref_i']], np.array([-2])))
    #
    #             out_dict['mov_label'] = cur_label[col]
    #             out_dict['gt_label'] = np.ones(pt_batch[i].shape[0]) * -1
    #             out_dict['gt_label'][match_dict[i][:, 0]] = label[data_batch['ref_i']][match_dict[i][:, 1]]
    #             out_dict['origin_label'] = label[i]
    #             out_dict['col'] = col
    #
    #
    #
    #         log_p = np.mean(p_m[row, col])
    #         print('temp:{}, mov:{}'.format(data_batch['pt_names'][data_batch['ref_i']], data_batch['pt_names'][i]))
    #         print('avg log prob:{}'.format(log_p))
    #
    #         gt_match = match_dict[i]
    #         gt_match_dict = dict()
    #
    #         # color_m = match_color(data_batch['color'][data_batch['ref_i']][:, 2], data_batch['color'][i][:, 2])
    #         # color_diff_gt += np.sum(color_m[gt_match[:, 1], gt_match[:, 0]])
    #         # color_diff_rand += np.sum(color_m[gt_match[:, 1], 1])
    #         # color_num += len(gt_match)
    #
    #         for gt_m in gt_match:
    #             gt_match_dict[gt_m[0]] = gt_m[1]
    #
    #         num_match = 0
    #         num_hit = 0
    #         for r_idx, r in enumerate(row):
    #             if r in gt_match_dict:
    #                 num_match += 1
    #                 if gt_match_dict[r] == col[r_idx]:
    #                     num_hit += 1
    #         acc_m = num_hit / (num_match + 1e-4)
    #         num_hit_all += num_hit
    #         num_match_all += num_match
    #         print('num_hit:{}, num_match:{}, accuracy:{}'.format(num_hit, num_match, acc_m))
    #         acc_list.append(acc_m)
    #
    #         if args.save:
    #             out_dict['accuracy'] = acc_m
    #             with open(save_file, 'wb') as f:
    #                 pickle.dump(out_dict, f)
    #                 f.close()
    #             save_idx += 1
    #
    #         # get the top rank for gt match.
    #         num_pair += len(gt_match)
    #         for gt_m in gt_match:
    #             topn = np.sum(p_m[gt_m[0]] >= p_m[gt_m[0], gt_m[1]])
    #             for i_rank in range(10):
    #                 if topn <= i_rank+1:
    #                     num_hit_list[i_rank] += 1
    #
    #         if args.method == 'hung':
    #             row, col = linear_sum_assignment(-p_m_pos)
    #         if args.method == 'max':
    #             col = np.argmax(p_m_pos, axis=1)
    #             row = np.array(range(num_neui))
    #         num_match = 0
    #         num_hit = 0
    #         for r_idx, r in enumerate(row):
    #             if r in gt_match_dict:
    #                 num_match += 1
    #                 if gt_match_dict[r] == col[r_idx]:
    #                     num_hit += 1
    #         acc_m = num_hit / (num_match + 1e-4)
    #         num_hit_all_pos += num_hit
    #         num_match_all_pos += num_match
    #         print('num_hit:{}, num_match:{}, Pos accuracy:{}'.format(num_hit, num_match, acc_m))
    #         acc_list_pos.append(acc_m)
    #
    # print('Color accuracy:{}'.format(num_hit_all / num_match_all))
    # print('Pos accuracy:{}'.format(num_hit_all_pos / num_match_all_pos))
    # num_hit_list = np.array(num_hit_list) / num_pair
    # print(num_hit_list[:10])
    #
    # #print('avg diff_gt:{}, avg diff_rand:{}'.format(color_diff_gt / color_num, color_diff_rand / color_num))
    #
    # out = dict()
    # out['trans_w_c'] = np.array(acc_list)
    # out['trans_w_p'] = np.array(acc_list_pos)
    # with open(os.path.join('/projects/LEIFER/Xinwei/github/NeuronNet/pts_id/plot', 'w_color.pkl'), 'wb') as fp:
    #     pickle.dump(out, fp)
    #     fp.close()