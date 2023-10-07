"""
This file is used to synthesize worm neurons so that we can get more samples and potentially do registration on
id that is not labelled by old pipeline or arcoss worm.
"""
import matplotlib.pyplot as plt
import argparse
import os
import glob
import numpy as np
import torch
import pickle
from Himag_cline import worm_cline, standard_neurons, find_min_match
from cpd_nonrigid_sep import register_nonrigid
#from cpd_plot import cpd_plot



def syn_head(neurons_s, neurons_t=None, noise_var=0, rotate_yz=True, scale=200, label_mode='old', show=False,
             rotate_xy=True, affine=True, straighten=False):
    # synthesize head.
    # straighten the head
    cline_s = neurons_s['cline']
    pts_s = neurons_s['pts']
    cline_f_s = worm_cline(cline_s)
    pts_s_s = cline_f_s.straighten(pts_s)
    # put z to 0
    pts_s_s[:, 2] -= np.median(pts_s_s[:, 2])
    if neurons_s['side'] == 0:
        pts_s_s[:, [1, 2]] *= -1
    cline_f_out = cline_f_s
    pts_s_out = pts_s_s

    affine_lim = 0.2
    if neurons_t is not None:
        w = 0.1
        lamb = 4e3
        beta = 0.25
        cline_t = neurons_t['cline']
        pts_t = neurons_t['pts']
        cline_f_t = worm_cline(cline_t)
        pts_t_s = cline_f_t.straighten(pts_t)
        pts_t_s[:, 2] -= np.median(pts_t_s[:, 2])
        if neurons_t['side'] == 0:
            pts_t_s[:, [1, 2]] *= -1
        # use cpd to deform.
        pt_trans = standard_neurons(pts_t_s, scale)
        pts_t_s_cpd = pt_trans.transform(pts_t_s)
        pts_s_s_cpd = pt_trans.transform(pts_s_s)
        s_trans = register_nonrigid(pts_t_s_cpd, pts_s_s_cpd, w=w, lamb=lamb,
                                     beta=beta)

        pts_s_out = pt_trans.transform_inv(s_trans)
        cline_f_out = cline_f_t


    if rotate_yz:
        # rotate the cline in yoz plane.
        theta = (np.random.rand(1)[0] - 0.5) * 0.9 * np.pi
        r_m = [[1, 0, 0], [0, np.cos(theta), np.sin(theta)], [0, -np.sin(theta), np.cos(theta)]]

        if affine:
            # Affine transform in x0y plane
            affine_m = np.array([[1, 0, 0], [0, 1, (np.random.rand(1)[0] - 0.5) * affine_lim],
                                 [0, (np.random.rand(1)[0] - 0.5) * affine_lim, 1]])
            r_m = np.matmul(r_m, affine_m)

        if show:
            plt.scatter(pts_s_out[:, 1], pts_s_out[:, 2], color='red', label='before')
            for i in range(pts_s_out.shape[0]):
                plt.text(pts_s_out[i, 1], pts_s_out[i, 2],
                        str(i), fontsize=8, color='black')

        pts_s_out = np.matmul(pts_s_out, r_m)
        if show:
            plt.scatter(pts_s_out[:, 1], pts_s_out[:, 2], color='green', label='after')
            for i in range(pts_s_out.shape[0]):
                plt.text(pts_s_out[i, 1], pts_s_out[i, 2],
                        str(i), fontsize=8, color='black')
            plt.legend()
            plt.show()


    # add and miss some neurons.
    missing_prop = np.random.rand(1) * 0.2 + 0.05
    #missing_prop = 0.1
    num_neuron = pts_s_out.shape[0]
    add_lim = min(20, num_neuron * missing_prop)
    num_add = np.random.randint(add_lim, size=1)[0]
    num_add = max(1, num_add)
    pts_add = np.random.rand(num_add, 3) * np.array([[300, 120, 120]]) - np.array([[50, 60, 60]])
    _, dis_min = find_min_match(pts_add, pts_s_out)
    add_mask = np.where(dis_min > 5)[0]
    num_add = len(add_mask)
    pts_add = pts_add[add_mask, :]

    # add pts and label together.
    pts_s_out = np.vstack((pts_s_out, pts_add))

    if label_mode == 'old':
        label_ori = neurons_s['trackIdx'][:, np.newaxis]
    else:
        label_ori = np.arange(num_neuron)[:, np.newaxis]

    label = np.vstack((label_ori, np.ones((num_add, 1)) * -2))

    missing_rand = np.random.rand(len(pts_s_out))
    if straighten:
        # for straighten(prediction purpose) keep labelled neurons
        labelled_idx = np.where(label >= 0)[0]
        missing_rand[labelled_idx] = 1
    remain_idx = np.where(missing_rand > missing_prop)[0]
    pts_s_out = pts_s_out[remain_idx, :]
    label = label[remain_idx, :]

    pts_out = np.copy(pts_s_out) if straighten else cline_f_out.project(pts_s_out)

    if show:
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        ax.scatter(pts_s[:, 0], pts_s[:, 1], color='green', label='worm')
        for i in range(pts_s.shape[0]):
            ax.text(pts_s[i, 0], pts_s[i, 1],
                      str(label_ori[i, 0]), fontsize=8, color='black')

        ax.scatter(pts_out[:, 0], pts_out[:, 1], color='red', label='syn')
        for i in range(pts_out.shape[0]):
            ax.text(pts_out[i, 0], pts_out[i, 1],
                    str(label[i, 0]), fontsize=8, color='black')

        ax.legend()
        ax.set_aspect('equal')
        # ax.set_xlim([0, 511])
        # ax.set_ylim([0, 511])
        plt.show()

    if show:
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        ax.scatter(pts_out[:, 0], pts_out[:, 1], color='red', label='before')
        for i in range(pts_out.shape[0]):
            ax.text(pts_out[i, 0], pts_out[i, 1],
                    str(label[i, 0]), fontsize=8, color='black')

    if affine:
        if rotate_xy:
            theta = np.random.rand(1)[0] * 2 * np.pi
            r_m = np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0],\
                   [0, 0, 1]])
        else:
            r_m = np.eye(3)

        scale_m = np.diag(np.random.rand(3) * 0.1 - 0.05) + np.eye(3)
        t_m = np.matmul(r_m, scale_m)
        #t_m = np.eye(3)
        affine_m = np.array([[1, (np.random.rand(1)[0] - 0.5) * affine_lim, 0], [(np.random.rand(1)[0] - 0.5) * affine_lim, 1, 0],
                             [0, 0, 1]])
        t_m = np.matmul(t_m, affine_m)

        pts_out = np.matmul(pts_out, t_m)

    if show:
        ax.scatter(pts_out[:, 0], pts_out[:, 1], color='green', label='after')
        for i in range(pts_out.shape[0]):
            ax.text(pts_out[i, 0], pts_out[i, 1],
                    str(label[i, 0]), fontsize=8, color='black')
        plt.show()

    if noise_var > 0:
        pts_out += np.random.randn(pts_out.shape[0], pts_out.shape[1]) * noise_var
    # put the point to the center
    pts_out -= np.median(pts_out, axis=0)
    out = np.hstack((pts_out / scale, label))
    return out



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Synthesize Neuron Data')
    parser.add_argument("--mode", type=str, default='multiple', help='synthesize same worm or across worm: copy, single or multiple')
    parser.add_argument("--index_mode", type=str, default='old', help='save old/new(0-num) track index')
    parser.add_argument("--path", type=str, default='/projects/LEIFER/Xinwei/github/NeuronNet/pts_id/real',
                        help='the path of real data(use your own real data(1 unit=0.42um)')
    parser.add_argument("--save_p", type=str, default='../results/train',
                        help='the path to save synthesized data')
    parser.add_argument("--source_num", type=int, default=100, help='number of source worm selected randomly')
    parser.add_argument("--template_num", type=int, default=64, help='number of template worm selected randomly')
    parser.add_argument("--shuffle", type=int, default=1, help="whether shuffle the dataset")
    parser.add_argument("--scale", type=float, default=200, help="the scale applied to original coordinates(1 unit=0.42um).")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--syn_mode", default="uns", type=str)
    parser.add_argument("--save_type", default='npy', type=str)
    parser.add_argument("--var", type=float, default=0,
                        help='variance add to points')
    #parser.add_argument()
    args = parser.parse_args()

    # load data in path
    worm_folders = glob.glob(os.path.join(args.path, '*/'))
    if args.shuffle:
        np.random.shuffle(worm_folders)
    real_dict = dict()
    real_dict['folders'] = worm_folders
    real_dict['folders_num'] = len(worm_folders)
    real_dict['volumes_list'] = list()
    real_dict['volumes_num'] = list()

    for folder in real_dict['folders']:
        volume_files = glob.glob1(folder, 'pt_*.npy')
        if args.shuffle:
            np.random.shuffle(volume_files)
        real_dict['volumes_list'].append(volume_files)
        real_dict['volumes_num'].append(len(volume_files))

        if args.mode == 'copy':
            folder_last = folder.split('/')[-2]
            save_dir = os.path.join(args.save_p, 'real_' + folder_last)
            print(save_dir)

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            for volume_f in volume_files:
                f_name = os.path.join(folder, volume_f)
                with open(f_name, 'rb') as f_pt:
                    pt_dict = pickle.load(f_pt)
                    f_pt.close()

                pts = pt_dict['pts']
                if args.syn_mode != 'uns':
                    cline_s = pt_dict['cline']
                    cline_f_s = worm_cline(cline_s)
                    pts = cline_f_s.straighten(pts)

                pts -= np.median(pts, axis=0)
                pts /= args.scale
                if pt_dict['side'] == 0:
                    pts[:, [1, 2]] *= -1

                pts_out = np.hstack((pts, pt_dict['trackIdx'][:, np.newaxis]))
                # save the file
                file_name = os.path.join(save_dir, 'real_' + volume_f)
                if args.save_type == 'npy':
                    np.save(file_name, pts_out)
                else:
                    new_name = file_name.split('.')[0] + '.pt'
                    pt_t = torch.tensor(pts_out, dtype=torch.float)
                    torch.save(pt_t, new_name)

    if args.mode == 'copy':
        exit()

    # synthesize data with one source worm and one template worm.
    for source_j in range(args.source_num):
        source_i = source_j + args.start_idx
        # get source volume
        folder_idx = source_i % real_dict['folders_num']
        volume_idx = source_i // real_dict['folders_num']
        volume_idx = volume_idx % real_dict['volumes_num'][folder_idx]

        source_f = os.path.join(real_dict['folders'][folder_idx], real_dict['volumes_list'][folder_idx][volume_idx])
        with open(source_f, 'rb') as f:
            pts_s_dict = pickle.load(f)
            f.close()
        #pts_s = np.load(source_f)

        for template_i in range(args.template_num):
            if args.mode == 'single':
                folder_idx_t = folder_idx
            elif args.mode == 'multiple':
                folder_idx_t = template_i % real_dict['folders_num']

            volume_idx_t = template_i // real_dict['folders_num']
            volume_idx_t = volume_idx_t % real_dict['volumes_num'][folder_idx_t]


            template_f = os.path.join(real_dict['folders'][folder_idx_t],
                                      real_dict['volumes_list'][folder_idx_t][volume_idx_t])
            if source_f == template_f:
                pts_t_dict = None
            else:
                with open(template_f, 'rb') as f:
                    pts_t_dict = pickle.load(f)
                    f.close()

            # check if we can get cline from neurons
            if args.syn_mode == 'uns':
                pts_s_syn = syn_head(pts_s_dict, pts_t_dict, label_mode=args.index_mode, scale=args.scale, noise_var=args.var)
            else:
                pts_s_syn = syn_head(pts_s_dict, pts_t_dict, label_mode=args.index_mode, scale=args.scale,
                                     rotate_xy=False, straighten=True)
            # save the synthesized data
            if args.index_mode == 'old':
                # use the track index from old pipeline, and different source volume can be put together
                save_dir = 'real_' + real_dict['folders'][folder_idx].split('/')[-2]
                save_dir = os.path.join(args.save_p, save_dir)

            elif args.index_mode == 'new':
                # use new track index, range(num_neurons)
                save_dir = os.path.join(args.save_p, 'syn_{}_{}'.format(args.syn_mode, source_i))

            if not os.path.exists(args.save_p):
                os.mkdir(args.save_p)

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            #np.save(os.path.join(save_dir, 'syn_{}_{}.npy'.format(source_i, template_i)), pts_s_syn)
            if args.save_type == 'npy':
                file_name = os.path.join(save_dir, 'syn_{}_{}.npy'.format(source_i, template_i))
                np.save(file_name, pts_s_syn)
            else:
                new_name = os.path.join(save_dir, 'syn_{}_{}.pt'.format(source_i, template_i))
                pt_t = torch.tensor(pts_s_syn, dtype=torch.float)
                torch.save(pt_t, new_name)










