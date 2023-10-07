import numpy as np
import shapely.geometry as geom
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d
import skimage.morphology as skmorp

def extract_center(y, num_ext):
    num_pt = len(y)
    return y[num_ext:len(y)-num_ext]

def find_min_match(neurons, tmp, dim=2):
    if dim is None:
        dim = neurons.shape[1]
    dis = neurons[:, np.newaxis, :dim] - tmp[:, :dim]
    dis = np.sqrt(np.sum(dis ** 2, axis=2))
    idx = np.argmin(dis, axis=1)
    x_idx = np.arange(neurons.shape[0])
    dis_min = dis[x_idx, idx]
    return idx, dis_min

def extend_ends(y, num_ext):
    # y is array
    if num_ext > 0:
        dir_head = (y[num_ext] - y[0]) / (num_ext + 1e-3)
        first_list = [i * dir_head + y[0] for i in range(-num_ext, 0)]
        dir_tail = (y[-1] - y[-num_ext-1]) / (num_ext + 1e-3)
        last_list = [y[-1] + i * dir_tail for i in range(1, num_ext + 1)]

        y_out = first_list + list(y) + last_list
    else:
        y_out = y
    return np.array(y_out)

def smooth_2d_pts(pts_ori, degree=2, win_length=20, num_point_out=100):

    # use cum_dis as x,
    dis_list = [0]
    pts_diff = np.diff(pts_ori, axis=0)
    pts_diff_dis = np.sqrt(np.sum(pts_diff ** 2, axis=1))
    dis_list += list(np.cumsum(pts_diff_dis))

    x = np.array(dis_list)
    line_length = dis_list[-1]
    line_length_step = line_length / (num_point_out - 1)

    x_new = np.arange(num_point_out) / (num_point_out - 1) * line_length
    f1 = interp1d(x, pts_ori[:, 0], fill_value='extrapolate')
    f2 = interp1d(x, pts_ori[:, 1], fill_value='extrapolate')
    pts = np.zeros((num_point_out, 2))
    pts[:, 0] = f1(x_new)
    pts[:, 1] = f2(x_new)

    pts_out = np.copy(pts)

    win = max(3, int(win_length / line_length_step))
    win = min(win, num_point_out // 3)
    win = win + 1 if not (win // 2 == 0) else win
    num_ext = 2 * win
    num_ext = min(num_ext, int(pts.shape[0] // 3))
    y1 = extend_ends(pts[:, 0], num_ext)
    y2 = extend_ends(pts[:, 1], num_ext)
    y1_s = gaussian_filter1d(y1, sigma=win, mode='nearest')
    pts_out[:, 0] = extract_center(y1_s, num_ext)
    y2_s = gaussian_filter1d(y2, sigma=win, mode='nearest')
    pts_out[:, 1] = extract_center(y2_s, num_ext)

    return pts_out


class standard_neurons(object):
    def __init__(self, neurons, scale=200):
        self.neurons_median = np.median(neurons, axis=0)
        self.scale = scale

    def transform(self, pts):
        pts = np.copy(pts)
        dim = pts.shape[1]
        pts = pts - self.neurons_median[np.newaxis, :dim]
        pts = pts / self.scale
        return pts

    def transform_inv(self, pts):
        pts = np.copy(pts)
        dim = pts.shape[1]
        pts = pts * self.scale
        pts = pts + self.neurons_median[np.newaxis, :dim]
        return pts

class worm_cline(object):
    #This is a class for storing the centerline of worm
    def __init__(self, Centerline, degree=3, num_pt=100, length_lim=300):
        self.num_pt = num_pt
        self.update_cline(Centerline)
        self.degree = degree
        self.length_lim = length_lim


    def update_cline(self, Centerline, degree=3):
        # extend the Centerline on head and tail.
        Centerline = Centerline[:, :2]
        self.Centerline = smooth_2d_pts(Centerline, degree=degree, num_point_out=self.num_pt)

        self.shCenterline = geom.LineString(self.Centerline)

        self.length = self.shCenterline.length

        head = self.shCenterline.interpolate(0)
        self.head_pt = np.array([head.x, head.y])
        head_next = self.shCenterline.interpolate(20)
        self.head_dir = self.norm_dir(np.array([head_next.x - head.x, head_next.y - head.y]))


        tail = self.shCenterline.interpolate(self.length)
        self.tail_pt = np.array([tail.x, tail.y])
        tail_prev = self.shCenterline.interpolate(self.length - 20)
        self.tail_dir = self.norm_dir(np.array([tail.x - tail_prev.x, tail.y - tail_prev.y]))
        self.straight_origin = 0

    def norm_dir(self, cur_dir):
        # normalize direction.
        return cur_dir / (np.sqrt(np.sum(cur_dir ** 2)) + 1e-6)

    def autofluorescence_mask(self, pts_s, inten, keep_idx=None, plot=False):
        # try not to include those autofluorescence, close to body and dim
        # find x > ? and inten < ?
        # overlap with keep_idx
        inten = inten - np.min(inten)
        num_lim = 150
        if keep_idx is None:
            keep_idx = np.arange(len(pts_s))
        if len(pts_s) <= num_lim:
            return keep_idx
        # only handle situation of >150

        x = np.copy(pts_s[:, 0])
        x.sort()
        x_threshold = max(150, x[num_lim-1])
        # take 20 percentile as color threshold
        c_threshold = np.percentile(inten, 10) + 50
        mask = (pts_s[:, 0] > x_threshold) * (inten < c_threshold)

        new_keep = np.array([idx for idx in keep_idx if not mask[idx]])
        return new_keep


    def head_orient_cline(self, method='straight'):
        head_pt = self.shCenterline.interpolate(0)
        head_pt = np.array([head_pt.x, head_pt.y])
        body_pt = self.shCenterline.interpolate(120)
        body_pt = np.array([body_pt.x, body_pt.y])
        orient_dir = body_pt - head_pt

        orient_dir /= np.sqrt(np.sum(orient_dir ** 2)) + 1e-5
        num_pt = self.num_pt
        #orient_cl = np.arange(num_pt) / (num_pt - 1) * self.length * orient_dir + head_pt
        orient_cl = np.repeat(orient_dir[np.newaxis, :], num_pt, axis=0) * np.arange(num_pt)[:, np.newaxis] / (num_pt - 1) * self.length + head_pt

        if method == 'straight':
            out = worm_cline(orient_cl)
        elif method == 'combine':
            cl = (orient_cl + self.Centerline) * 0.5
            out = worm_cline(cl)
        return out

    def cut_head(self, neurons):
        # this function limit the cline to where the points are.
        # cut the cline to fit the neurons,
        # first one has s 0, length is last s
        neurons_s = self.straighten(neurons)
        num_pt = self.num_pt
        tail_s = self.straighten(np.array([self.tail_pt]))
        max_s = min(tail_s[0, 0], self.length_lim)
        max_s = max(50, max_s)
        min_s = max(0, neurons_s[:, 0].min())

        close_idx = np.where(np.abs(neurons_s[:, 1]) < 25)[0]
        if len(close_idx) > 1:
            sort_s = np.sort(neurons_s[close_idx, 0])
            diff_s = np.diff(sort_s)
            s_next = sort_s[1:]

            s_can = np.where((diff_s > 30) * (s_next < 100))[0]
            if len(s_can) > 0:
                print('get rid of tip piece')
                min_s = s_next[s_can.min()]


            step = (max_s - min_s) / (num_pt - 1)
            output = np.arange(min_s, max_s + step / 2, step)
            #output = self.length * np.arange((num_pt)) / (num_pt - 1) - ref_s + ref_s_new
            output = np.stack((output, np.zeros(num_pt)), axis=0).T
            new_cline = self.project(output)
        else:
            new_cline = self.Centerline
        return new_cline


    def cpd_cline_update(self, neurons1, neurons2, degree=3, num_pt=None):
        # transform the current centerline with respect to the pair of neurons
        # with cpd, neuron2 is cpd transform of neuron1, result cline is the same
        # cpd result of current cline.
        #num_neuron = neurons1.shape[0]
        if num_pt is None:
            num_pt = self.num_pt

        if neurons1.shape[0] < 10:
            return self.Centerline

        neurons1_s = self.straighten(neurons1)
        neurons2_s = self.straighten(neurons2)

        ref_s = np.min(neurons1_s[:, 0])

        neurons_diff = neurons2_s - neurons1_s

        # build a diff moving average.
        s_min, s_max = neurons1_s[:, 0].min(), neurons1_s[:, 0].max()

        num_new_s = 30
        s_step = (s_max - s_min) / (num_new_s - 1)
        s_step = 10
        s_array = np.arange(s_min, s_max + s_step / 2, s_step)

        s_list = list()
        diff_list = list()

        num_eval = 2
        for s in s_array:
            pos_idx = np.where((np.abs(neurons1_s[:, 0] - s) < s_step) * (neurons1_s[:, 1] > 0))[0]
            neg_idx = np.where((np.abs(neurons1_s[:, 0] - s) < s_step) * (neurons1_s[:, 1] <= 0))[0]
            if len(pos_idx) > num_eval:
                s_list.append(s)
                diff_list.append(np.mean(neurons_diff[pos_idx, :], axis=0))

            if len(neg_idx) > num_eval:
                if len(pos_idx) > num_eval:
                    diff_list[-1] = 0.5 * (diff_list[-1] + np.mean(neurons_diff[neg_idx, :], axis=0))
                else:
                    s_list.append(s)
                    diff_list.append(np.mean(neurons_diff[neg_idx, :], axis=0))

        if len(s_list) < 5:
            return self.Centerline

        diff_list = np.array(diff_list)
        s_list = np.array(s_list)

        x = s_list
        s_list_min, s_list_max = s_list.min(), s_list.max()
        step = (s_list_max - s_list_min) / (num_pt - 1)
        x_new = np.arange(s_list_min, s_list_max + step / 2, step)

        f1 = interp1d(x, diff_list[:, 0], fill_value='extrapolate')
        f2 = interp1d(x, diff_list[:, 1], fill_value='extrapolate')

        new_cline_s = np.zeros((num_pt, 2))
        diff_new = np.zeros((num_pt, 2))
        diff_new_s = np.zeros((num_pt, 2))
        new_cline_s_o = np.zeros((num_pt, 2))

        diff_new[:, 0] = f1(x_new)
        diff_new[:, 1] = f2(x_new)

        win_length = 20
        win = max(3, int(np.ceil(win_length / step)))
        win = win + 1 if not (win // 2 == 0) else win

        diff_new_s[:, 0] = gaussian_filter1d(diff_new[:, 0], sigma=win, mode='reflect')
        diff_new_s[:, 1] = gaussian_filter1d(diff_new[:, 1], sigma=win, mode='reflect')

        new_cline_s[:, 0] = diff_new_s[:, 0] + x_new
        new_cline_s[:, 1] = diff_new_s[:, 1]


        cline_tmp = worm_cline(new_cline_s)
        neurons2_s_new = cline_tmp.straighten(neurons2_s)

        ref_s_new = np.min(neurons2_s_new[:, 0])

        new_s_min = ref_s_new - ref_s
        new_s_max = max(new_s_min + 50, np.max(neurons2_s_new[:, 0]))
        new_s_max = min(self.length_lim, new_s_max)
        step = (new_s_max - new_s_min) / (num_pt - 1)
        output = np.arange(new_s_min, new_s_max + step / 2, step)
        #output = self.length * np.arange((num_pt)) / (num_pt - 1) - ref_s + ref_s_new
        output = np.stack((output, np.zeros(num_pt)), axis=0).T
        new_cline_s_reg = cline_tmp.project(output)

        new_cline = self.project(new_cline_s_reg)

        return new_cline


    def get_dir(self, s):
        # This function get the direction of worm based on the length(straightened coordinate x)
        if s <= 0:
            cur_dir = self.head_dir
        elif s <= self.length / 2:
            point_1 = self.shCenterline.interpolate(s)
            point_2 = self.shCenterline.interpolate(s + 1)
            cur_dir = np.array([point_2.x - point_1.x, point_2.y - point_1.y])
        elif s <= self.length:
            point_1 = self.shCenterline.interpolate(s - 1)
            point_2 = self.shCenterline.interpolate(s)
            cur_dir = np.array([point_2.x - point_1.x, point_2.y - point_1.y])
        else:
            cur_dir = self.tail_dir

        cur_dir = self.norm_dir(cur_dir)
        return cur_dir

    def straighten(self, Neurons):
        assert len(Neurons.shape) == 2
        #N = Neurons.shape[0]

        sNeurons = list()
        # Straighten the brain. The z coordinate does not change.
        for neuron in Neurons:
            point = geom.Point(neuron[0], neuron[1])

            # Calculate distance from the centerline and longitudinal position
            # along the centerline. Apart from the sign of y, these are the
            # coordinates in the straightened frame of reference.
            x = self.shCenterline.project(point)
            if x <= 0:
                point_coord = np.array([point.x, point.y])
                point_head = point_coord - self.head_pt

                x_out = np.dot(self.head_dir, point_head)
                y_out = self.head_dir[0] * point_head[1] - self.head_dir[1] * point_head[0]
            elif x < self.length:

                y = self.shCenterline.distance(point)

                # Find the coordinates of the projection of the neuron on the
                # centerline, in the original frame of reference.
                a = self.shCenterline.interpolate(x)
                # Find the vector going from the projection of the neuron on the
                # centerline to the neuron itself.
                vpx = point.x - a.x
                vpy = point.y - a.y

                # Move along the line in the positive direction.
                cur_dir = self.get_dir(x)
                vx, vy = cur_dir[0], cur_dir[1]

                # Calculate the cross product v x vp. Its sign is the sign of y.
                s = np.sign(vx * vpy - vy * vpx)
                x_out = x
                y_out = s * y
            else:
                point_coord = np.array([point.x, point.y])
                point_tail = point_coord - self.tail_pt

                x_out = np.dot(self.tail_dir, point_tail) + self.length
                y_out = self.tail_dir[0] * point_tail[1] - self.tail_dir[1] * point_tail[0]


            sNeurons.append([x_out, y_out])

        if Neurons.shape[1] > 2:
            sNeurons = np.hstack((np.array(sNeurons), Neurons[:, 2:]))
        else:
            sNeurons = np.array(sNeurons)
        # straight origin is the coordinate of start point in straightened coordinate system.
        sNeurons[:, 0] += self.straight_origin
        return sNeurons

    def dir_ortho(self, cur_dir):
        ortho = np.copy(cur_dir)
        ortho[0] = -cur_dir[1]
        ortho[1] = cur_dir[0]
        return ortho

    def update_straight_origin(self, straight_origin):
        self.straight_origin = straight_origin

    def project(self, sNeurons):
        assert len(sNeurons.shape) == 2
        # straight_origin is the start point of line in straight coordinate
        sNeurons = np.copy(sNeurons)
        sNeurons[:, 0] -= self.straight_origin
        Neurons = list()
        for neuron in sNeurons:

            if neuron[0] < 0:
                head_dir_ortho = self.dir_ortho(self.head_dir)
                point_out = self.head_pt + neuron[0] * self.head_dir + neuron[1] * head_dir_ortho
            elif neuron[0] < self.length:
                a = self.shCenterline.interpolate(neuron[0])
                point_out = np.array([a.x, a.y])
                cur_dir = self.get_dir(neuron[0])
                cur_dir_ortho = self.dir_ortho(cur_dir)
                point_out = point_out + cur_dir_ortho * neuron[1]
            else:
                tail_dir_ortho = self.dir_ortho(self.tail_dir)
                point_out = self.tail_pt + (neuron[0] - self.length) * self.tail_dir + neuron[1] * tail_dir_ortho
            Neurons.append(point_out)
        if sNeurons.shape[1] > 2:
            Neurons = np.hstack((np.array(Neurons), sNeurons[:, 2:]))
        else:
            Neurons = np.array(Neurons)

        return Neurons


    def find_all_neighbor(self, Neurons_s, mask, max_iter=5, dis_neigh=10):
        increase_sz = 1
        keep_idx = np.where(mask > 0)[0]
        bad_idx = np.where(mask == 0)[0]
        mask = np.copy(mask)
        i = 0
        if len(keep_idx) < Neurons_s.shape[0] and len(keep_idx) > 0:
            while increase_sz > 0 and i < max_iter and len(bad_idx) > 0:

                bad_pts = Neurons_s[bad_idx, :2]
                good_pts = Neurons_s[keep_idx, :2]

                dis = bad_pts[:, np.newaxis, :2] - good_pts[:, :2]
                dis = np.sqrt(np.sum(dis ** 2, axis=2))
                dis_min = np.min(dis, axis=1)

                new_good_idx = bad_idx[np.where(dis_min < dis_neigh)[0]]
                mask[new_good_idx] = True
                increase_sz = len(new_good_idx)
                keep_idx = new_good_idx
                bad_idx = np.where(mask == 0)[0]
                i += 1
        keep_idx = np.where(mask)[0]
        return keep_idx

    # def mask_with_threshold(self, s_array, y_array, threshold_x, threshold_y):
    #     mask_x = (s_array > threshold_x[0]) * (s_array < threshold_x[1])
    #     mask_y = (y_array > threshold_y[0]) * (y_array < threshold_y[1])
    #     return mask_x * mask_y



    # def mask_neurons(self, Neurons, straight=True):
    #     # mask out some neurons that are far from the centerline.
    #     if straight:
    #         # need to do straighting.
    #         Neurons = self.straighten(Neurons)
    #
    #     mask_out = Neurons[:, 0] < -1e6
    #
    #     x_threshold_head_ext = [-25, -5]
    #     y_threshold_head_ext = [-20, 20]
    #
    #     mask_head_ext = self.mask_with_threshold(Neurons[:, 0], Neurons[:, 1], x_threshold_head_ext, y_threshold_head_ext)
    #
    #     x_threshold_head = [-5, 100]
    #     y_threshold_head = [-50, 50]
    #
    #     mask_head = self.mask_with_threshold(Neurons[:, 0], Neurons[:, 1], x_threshold_head, y_threshold_head)
    #
    #     x_threshold_body = [100, self.length + 50]
    #     y_threshold_body = [-100, 100]
    #     mask_body = self.mask_with_threshold(Neurons[:, 0], Neurons[:, 1], x_threshold_body, y_threshold_body)
    #
    #
    #     # also keep the neurons that are close to current neurons.
    #     cur_mask = mask_body + mask_head_ext + mask_head
    #     keep_idx = self.find_all_neighbor(Neurons, cur_mask, dis_neigh=10)
    #     return keep_idx




