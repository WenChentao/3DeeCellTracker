from skimage.morphology import watershed, remove_small_objects
import skimage.measure as skmea
from skimage.measure import regionprops
from skimage.feature import hessian_matrix, hessian_matrix_eigvals, peak_local_max
from scipy import ndimage as ndi
import numpy as np

def update_bbox(box_large, box_sub):
    new_box = np.array(box_sub)
    new_box[0], new_box[3] = box_large[0] + box_sub[0], box_large[0] + box_sub[3]
    new_box[1], new_box[4] = box_large[1] + box_sub[1], box_large[1] + box_sub[4]
    new_box[2], new_box[5] = box_large[2] + box_sub[2], box_large[2] + box_sub[5]
    return new_box


def kernel_radius(kernel_size, r, dim1_scale=1, normalize=False):
    # the first dimension can be treated differently
    r_x, r_y, r_z = (kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2, (kernel_size[2] - 1) // 2
    xv, yv, zv = np.meshgrid(np.arange(-r_x, r_x+1), np.arange(-r_y, r_y+1), np.arange(-r_z, r_z+1), indexing='ij')
    # yv is dim0 xv dim1 zv dim2
    dis2 = (xv * dim1_scale) ** 2 + yv ** 2 + zv ** 2
    kernel = (dis2 <= r ** 2) * 1.
    if normalize:
        return kernel / np.sum(kernel)
    else:
        return kernel

def pt_neighbor(size_im, pt, r, dim1_scale):
    r_x, r_y, r_z = size_im[0], size_im[1], size_im[2]
    xv, yv, zv = np.meshgrid(np.arange(0, r_x), np.arange(0, r_y), np.arange(0, r_z),
                             indexing='ij')
    # yv is dim0 xv dim1 zv dim2
    dis2 = ((xv - pt[0]) * dim1_scale) ** 2 + (yv - pt[1]) ** 2 + (zv - pt[2]) ** 2
    mask = (dis2 <= r ** 2)
    return mask


class Detect_From_Deconv(object):
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu

    def check_single_neuron(self, neuron):
        # make sure that only a single neuron is here.
        image = neuron['mask']
        num_z = image.shape[0]

        conn = np.ones((3, 3, 3))
        watershed_label = np.zeros(image.shape)
        save_neuron = True
        for z_idx in range(num_z):
            image_z = image[z_idx]
            image_label = skmea.label(image_z, connectivity=2)
            num_neuron = image_label.max()
            if num_neuron > 1:
                save_neuron = False
                watershed_label[z_idx] = image_label
                label_sub = watershed(-neuron['image'], markers=watershed_label, connectivity=conn, mask=image)
                props_sub = regionprops(label_sub, intensity_image=neuron['image'])
                for prop in props_sub:
                    neu_pos = np.unravel_index(np.argmax(prop.intensity_image, axis=None),
                                               prop.intensity_image.shape)
                    new_bbox = update_bbox(neuron['bbox'], prop.bbox)
                    neu_pos = np.array(neu_pos).T + np.array(new_bbox[:3])
                    neu_pos_c = np.array(prop.centroid) + np.array(new_bbox[:3])

                    neuron_new = dict()
                    neuron_new['neu_pos'] = neu_pos
                    neuron_new['neu_pos_c'] = neu_pos_c
                    neuron_new['bbox'] = new_bbox
                    neuron_new['mask'] = prop.image
                    neuron_new['image'] = prop.intensity_image
                    neuron_new['label_o'] = neuron['label_o']
                    neuron_new['max_inten'] = prop.max_intensity
                    neuron_new['mean_inten'] = prop.mean_intensity
                    neuron_new['area'] = prop.area

                    self.check_single_neuron(neuron_new)
                break

        if save_neuron:
            self.neurons['area'].append(neuron['area'])
            self.neurons['max_inten'].append(neuron['max_inten'])
            self.neurons['mean_inten'].append(neuron['mean_inten'])
            self.neurons['pts_max'].append(neuron['neu_pos'])
            self.neurons['pts'].append(neuron['neu_pos_c'])
            self.neurons['label_o'].append(neuron['label_o'])
            self.neurons['bbox'].append(neuron['bbox'])
            self.neurons['mask'].append(neuron['mask'])

    def get_gery_mask(self, worm, fac=5):
        std_array = np.std(worm, axis=(1, 2))

        # need to handle the situation where image is not clean at
        # border.
        worm_diff = np.copy(worm)
        worm_diff[:, :5, :] = 0
        worm_diff[:, -5:, :] = 0
        worm_diff[:, :, :5] = 0
        worm_diff[:, :, -5:] = 0

        mean_sz = 3
        max_sz = 7
        worm_diff = ndi.uniform_filter(worm_diff, size=[1, mean_sz, mean_sz])
        # contrast to local minima
        worm_diff = worm_diff + ndi.maximum_filter(-worm_diff, size=[1, max_sz, max_sz])
        # Multi_Slice_Viewer(np.concatenate((worm_fil, worm_diff), axis=2))
        std_array = std_array[std_array > 5]

        if len(std_array):
            std_min_idx = np.argmin(std_array)
            grey_thd = fac * std_array[std_min_idx]
        else:
            grey_thd = 5 * fac

        print('grey threshold:{}'.format(grey_thd))
        grey_mask = worm_diff > grey_thd
        return grey_mask

    def detect_neuron_hessian(self, worm, z_diff, show=1, worm_green=None):
        # worm is the numpy array,[z,x,y]
        # calculate hessian of the worm image.
        z_scale = z_diff * 47.619
        print('z_scale:{}'.format(z_scale))

        grey_mask = self.get_gery_mask(worm)
        if worm_green is not None:
            grey_mask += self.get_gery_mask(worm_green, fac=12) * self.get_gery_mask(worm, fac=2)

        H_matrix = hessian_matrix(worm, sigma=1, order='rc')

        H_matrix_sub = list()
        for h_m in H_matrix:
            H_matrix_sub.append(h_m * grey_mask)

        H_eig = - hessian_matrix_eigvals(H_matrix_sub)[0]

        mask_n = H_eig > 1

        label_n = ndi.label(mask_n)[0]

        if label_n.max() > 600:
            return None
        props = regionprops(label_n, intensity_image=worm)

        max_kernel3 = kernel_radius([3, 5, 5], r=3, dim1_scale=2.5, normalize=False)
        conn = np.ones((3, 3, 3))

        area_thd = 6
        self.neurons = dict()
        self.neurons['pts'] = list()
        self.neurons['pts_max'] = list()
        self.neurons['area'] = list()
        self.neurons['max_inten'] = list()
        self.neurons['mean_inten'] = list()
        self.neurons['label_o'] = list()
        self.neurons['bbox'] = list()
        self.neurons['mask'] = list()
        for prop in props:
            save_prop = True
            if prop.area > area_thd:
                worm_max = peak_local_max(prop.intensity_image, threshold_abs=None, exclude_border=False,
                                          indices=False, footprint=max_kernel3)

                worm_max_image = prop.intensity_image * worm_max
                candidate = list()
                worm_max_new = np.zeros(worm_max.shape) > 1
                while worm_max_image.sum() > 0:
                    pt = np.unravel_index(np.argmax(worm_max_image, axis=None), worm_max_image.shape)
                    max_mask = pt_neighbor(size_im=worm_max.shape, pt=pt, r=4.2, dim1_scale=1.5)  # 4.2, 2
                    candidate.append(pt)
                    worm_max_new[pt] = True
                    worm_max_image[max_mask] = 0

                if len(candidate) >= 2:

                    save_prop = False
                    markers = ndi.label(worm_max_new)[0]
                    label_sub = watershed(-prop.intensity_image, markers=markers, connectivity=conn, mask=prop.image)
                    props_sub = regionprops(label_sub, intensity_image=prop.intensity_image)

                    for prop_sub in props_sub:
                        if prop_sub.area > area_thd:
                            # neu_pos = np.unravel_index(np.argmax(prop_sub.intensity_image, axis=None), prop_sub.intensity_image.shape)
                            # marker is of size prop
                            neu_pos = np.where(markers == prop_sub.label)
                            new_bbox = update_bbox(prop.bbox, prop_sub.bbox)
                            neu_pos = np.array(neu_pos).T[0] + np.array(prop.bbox[:3])
                            neu_pos_c = np.array(prop_sub.centroid) + np.array(new_bbox[:3])
                            neuron = dict()
                            neuron['neu_pos'] = neu_pos
                            neuron['neu_pos_c'] = neu_pos_c
                            neuron['bbox'] = new_bbox
                            neuron['mask'] = prop_sub.image
                            neuron['image'] = prop_sub.intensity_image
                            neuron['label_o'] = prop.label
                            neuron['max_inten'] = prop_sub.max_intensity
                            neuron['mean_inten'] = prop_sub.mean_intensity
                            neuron['area'] = prop_sub.area

                            self.check_single_neuron(neuron)

                if save_prop:
                    neu_pos = np.unravel_index(np.argmax(prop.intensity_image, axis=None),
                                               prop.intensity_image.shape)
                    neu_pos = np.array(neu_pos) + np.array(prop.bbox[:3])
                    neu_pos_c = np.array(prop.centroid)
                    neuron = dict()
                    neuron['neu_pos'] = neu_pos
                    neuron['neu_pos_c'] = neu_pos_c
                    neuron['bbox'] = prop.bbox
                    neuron['mask'] = prop.image
                    neuron['image'] = prop.intensity_image
                    neuron['label_o'] = prop.label
                    neuron['max_inten'] = prop.max_intensity
                    neuron['mean_inten'] = prop.mean_intensity
                    neuron['area'] = prop.area

                    self.check_single_neuron(neuron)

        neurons_pos = np.array(self.neurons['pts_max'])
        self.neurons['num_neuron'] = len(neurons_pos)
        print('find {} neurons'.format(self.neurons['num_neuron']))
        print('\n')

        return self.neurons
