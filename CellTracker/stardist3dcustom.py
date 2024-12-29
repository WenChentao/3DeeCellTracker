from __future__ import division, absolute_import, print_function, unicode_literals
from __future__ import print_function, unicode_literals, absolute_import, division

import datetime
import functools
import numbers
import sys
import warnings

import numpy as np
import scipy.ndimage as ndi
from csbdeep.utils import _raise, save_json
from csbdeep.utils.tf import keras_import
from scipy.optimize import minimize_scalar
from stardist.matching import matching_dataset, relabel_sequential
from stardist.models import StarDist3D
from stardist.nms import _ind_prob_thresh, non_maximum_suppression_3d_sparse
from stardist.rays3d import rays_from_json
from stardist.geometry import polyhedron_to_label
from tqdm import tqdm

K = keras_import('backend')
Sequence = keras_import('utils', 'Sequence')
Adam = keras_import('optimizers', 'Adam')
ReduceLROnPlateau, TensorBoard = keras_import('callbacks', 'ReduceLROnPlateau', 'TensorBoard')


class StarDist3DCustom(StarDist3D):

    def _predict_instances_generator(self, img, axes=None, normalizer=None,
                                     sparse=True,
                                     prob_thresh=None, nms_thresh=None,
                                     scale=None,
                                     n_tiles=None, show_tile_progress=True,
                                     verbose=False,
                                     return_labels=True,
                                     predict_kwargs=None, nms_kwargs=None,
                                     overlap_label=None, return_predict=False):
        """Predict instance segmentation from input image.

        Parameters
        ----------
        img : :class:`numpy.ndarray`
            Input image
        axes : str or None
            Axes of the input ``img``.
            ``None`` denotes that axes of img are the same as denoted in the config.
        normalizer : :class:`csbdeep.data.Normalizer` or None
            (Optional) normalization of input image before prediction.
            Note that the default (``None``) assumes ``img`` to be already normalized.
        sparse: bool
            If true, aggregate probabilities/distances sparsely during tiled
            prediction to save memory (recommended).
        prob_thresh : float or None
            Consider only object candidates from pixels with predicted object probability
            above this threshold (also see `optimize_thresholds`).
        nms_thresh : float or None
            Perform non-maximum suppression that considers two objects to be the same
            when their area/surface overlap exceeds this threshold (also see `optimize_thresholds`).
        scale: None or float or iterable
            Scale the input image internally by this factor and rescale the output accordingly.
            All spatial axes (X,Y,Z) will be scaled if a scalar value is provided.
            Alternatively, multiple scale values (compatible with input `axes`) can be used
            for more fine-grained control (scale values for non-spatial axes must be 1).
        n_tiles : iterable or None
            Out of memory (OOM) errors can occur if the input image is too large.
            To avoid this problem, the input image is broken up into (overlapping) tiles
            that are processed independently and re-assembled.
            This parameter denotes a tuple of the number of tiles for every image axis (see ``axes``).
            ``None`` denotes that no tiling should be used.
        show_tile_progress: bool
            Whether to show progress during tiled prediction.
        verbose: bool
            Whether to print some info messages.
        return_labels: bool
            Whether to create a label image, otherwise return None in its place.
        predict_kwargs: dict
            Keyword arguments for ``predict`` function of Keras model.
        nms_kwargs: dict
            Keyword arguments for non-maximum suppression.
        overlap_label: scalar or None
            if not None, label the regions where polygons overlap with that value
        return_predict: bool
            Also return the outputs of :func:`predict` (in a separate tuple)
            If True, implies sparse = False

        Returns
        -------
        (:class:`numpy.ndarray`, dict), (optional: return tuple of :func:`predict`)
            Returns a tuple of the label instances image and also
            a dictionary with the details (coordinates, etc.) of all remaining polygons/polyhedra.

        """
        if predict_kwargs is None:
            predict_kwargs = {}
        if nms_kwargs is None:
            nms_kwargs = {}

        if return_predict and sparse:
            sparse = False
            warnings.warn("Setting sparse to False because return_predict is True")

        nms_kwargs.setdefault("verbose", verbose)

        _axes         = self._normalize_axes(img, axes)
        _axes_net     = self.config.axes
        _permute_axes = self._make_permute_axes(_axes, _axes_net)
        _shape_inst   = tuple(s for s,a in zip(_permute_axes(img).shape, _axes_net) if a != 'C')

        if scale is not None:
            if isinstance(scale, numbers.Number):
                scale = tuple(scale if a in 'XYZ' else 1 for a in _axes)
            scale = tuple(scale)
            len(scale) == len(_axes) or _raise(ValueError(f"scale {scale} must be of length {len(_axes)}, i.e. one value for each of the axes {_axes}"))
            for s,a in zip(scale,_axes):
                s > 0 or _raise(ValueError("scale values must be greater than 0"))
                (s in (1,None) or a in 'XYZ') or warnings.warn(f"replacing scale value {s} for non-spatial axis {a} with 1")
            scale = tuple(s if a in 'XYZ' else 1 for s,a in zip(scale,_axes))
            verbose and print(f"scaling image by factors {scale} for axes {_axes}")
            img = ndi.zoom(img, scale, order=1)

        yield 'predict'  # indicate that prediction is starting
        res = None
        if sparse:
            for res in self._predict_sparse_generator(img, axes=axes, normalizer=normalizer, n_tiles=n_tiles,
                                                      prob_thresh=prob_thresh, show_tile_progress=show_tile_progress, **predict_kwargs):
                if res is None:
                    yield 'tile'  # yield 'tile' each time a tile has been processed
        else:
            for res in self._predict_generator(img, axes=axes, normalizer=normalizer, n_tiles=n_tiles,
                                               show_tile_progress=show_tile_progress, **predict_kwargs):
                if res is None:
                    yield 'tile'  # yield 'tile' each time a tile has been processed
            res = tuple(res) + (None,)

        if self._is_multiclass():
            prob_n, dist_nxray, prob_class, points_coords_nx3, prob_map_3d_zyx = res
        else:
            prob_n, dist_nxray, points_coords_nx3, prob_map_3d_zyx = res
            print(prob_n.shape, dist_nxray.shape, points_coords_nx3.shape, prob_map_3d_zyx.shape)
            prob_class = None

        yield 'nms'  # indicate that non-maximum suppression is starting
        res_instances = self._instances_from_prediction(_shape_inst, prob_n, dist_nxray,
                                                        points=points_coords_nx3,
                                                        prob_class=prob_class,
                                                        prob_thresh=prob_thresh,
                                                        nms_thresh=nms_thresh,
                                                        scale=(None if scale is None else dict(zip(_axes,scale))),
                                                        return_labels=return_labels,
                                                        overlap_label=overlap_label,
                                                        **nms_kwargs)

        # last "yield" is the actual output that would have been "return"ed if this was a regular function
        # Note: the size of prob_map is compressed according to the grid parameter. It will be recovered during tracking
        if return_predict:
            yield res_instances, tuple(res[:-1]), prob_map_3d_zyx
        else:
            yield res_instances, prob_map_3d_zyx

    @functools.wraps(_predict_instances_generator)
    def predict_instances(self, *args, **kwargs):
        # the reason why the actual computation happens as a generator function
        # (in '_predict_instances_generator') is that the generator is called
        # from the stardist napari plugin, which has its benefits regarding
        # control flow and progress display. however, typical use cases should
        # almost always use this function ('predict_instances'), and shouldn't
        # even notice (thanks to @functools.wraps) that it wraps the generator
        # function. note that similar reasoning applies to 'predict' and
        # 'predict_sparse'.

        # return last "yield"ed value of generator
        warnings.warn(
            "predict_instances is deprecated and will be removed in future versions.",
            DeprecationWarning,
            stacklevel=2
        )
        r = None
        for r in self._predict_instances_generator(*args, **kwargs):
            pass
        return r

    def _predict_instances_simple(self, img, prob_thresh=None, nms_thresh=None,
                                  n_tiles=None, show_tile_progress=True, force_include_bright: bool = False):
        """Predict instance segmentation from input image.

        Parameters
        ----------
        img : :class:`numpy.ndarray`
            Input image
        prob_thresh : float or None
            Consider only object candidates from pixels with predicted object probability
            above this threshold (also see `optimize_thresholds`).
        nms_thresh : float or None
            Perform non-maximum suppression that considers two objects to be the same
            when their area/surface overlap exceeds this threshold (also see `optimize_thresholds`).
        n_tiles : iterable or None
            Out of memory (OOM) errors can occur if the input image is too large.
            To avoid this problem, the input image is broken up into (overlapping) tiles
            that are processed independently and re-assembled.
            This parameter denotes a tuple of the number of tiles for every image axis (see ``axes``).
            ``None`` denotes that no tiling should be used.
        show_tile_progress: bool
            Whether to show progress during tiled prediction.

        Returns
        -------
        (:class:`numpy.ndarray`, dict), (optional: return tuple of :func:`predict`)
            Returns a tuple of the label instances image and also
            a dictionary with the details (coordinates, etc.) of all remaining polygons/polyhedra.

        """
        _shape_inst = self._get_image_shape(img)

        prob_n, dist_nxray, points_coords_nx3, prob_map_reduced_zyx, dist_map_reduced_zyxr = self._predict_sparse_custom(
            img, n_tiles=n_tiles, prob_thresh=prob_thresh, show_tile_progress=show_tile_progress)

        labels, res_dict = self._instances_from_prediction_simple(
            _shape_inst,
            prob_n,
            dist_nxray,
            points=points_coords_nx3,
            nms_thresh=nms_thresh
        )
        res_dict["img_shape"] = _shape_inst

        # Note: the size of prob_map is compressed according to the grid parameter. It will be recovered during tracking
        return labels, res_dict, prob_map_reduced_zyx, dist_map_reduced_zyxr


    def _get_image_shape(self, img):
        _axes = self._normalize_axes(img, None)
        _axes_net = self.config.axes
        _permute_axes = self._make_permute_axes(_axes, _axes_net)
        _shape_inst = tuple(s for s, a in zip(_permute_axes(img).shape, _axes_net) if a != 'C')
        return _shape_inst

    def _instances_from_prediction_simple(self, img_shape, prob, dist, points, nms_thresh=None):
        if nms_thresh  is None:
            nms_thresh  = self.thresholds.nms

        rays = rays_from_json(self.config.rays_json)

        points, probi, disti, indsi = non_maximum_suppression_3d_sparse(
            dist, prob, points, rays, nms_thresh=nms_thresh, verbose=False)

        labels = polyhedron_to_label(disti, points, rays=rays, prob=probi,
                                     shape=img_shape, overlap_label=None, verbose=False)
        labels, _,_ = relabel_sequential(labels)

        res_dict = dict(dist=disti, points=points, prob=probi, rays=rays,
                        rays_vertices=rays.vertices, rays_faces=rays.faces)
        return labels, res_dict

    def _predict_sparse_custom(self, img, prob_thresh=None,
                               n_tiles=None, show_tile_progress=True, **predict_kwargs):
        """ Sparse version of model.predict()
        Returns
        -------
        (prob, dist, [prob_class], points)   flat list of probs, dists, (optional prob_class) and points
        """
        if prob_thresh is None: prob_thresh = self.thresholds.prob
        predict_kwargs["verbose"] = False
        (cell_image_zyx, axes, axes_net, axes_net_div_by, _permute_axes, resizer, n_tiles,
         grid, grid_dict, channel, predict_direct, tiling_setup) = \
            self._predict_setup(img, None, None, n_tiles,
                                show_tile_progress, predict_kwargs)

        def _prep(prob, dist):
            prob = np.take(prob,0,axis=channel)
            dist = np.moveaxis(dist,channel,-1)
            dist = np.maximum(1e-3, dist)
            return prob, dist

        if np.prod(n_tiles) > 1:
            tile_generator, output_shape, create_empty_output = tiling_setup()

            sh = list(output_shape)
            sh[channel] = 1

            prob = np.zeros(output_shape[:3])
            dist = np.zeros(output_shape[:3] + (self.config.n_rays,))
            proba, dista, pointsa, prob_classa = [], [], [], []

            for tile, s_src, s_dst in tile_generator:

                results_tile = predict_direct(tile)

                # account for grid
                s_src = [slice(s.start // grid_dict.get(a, 1), s.stop // grid_dict.get(a, 1)) for s, a in
                         zip(s_src, axes_net)]
                s_dst = [slice(s.start // grid_dict.get(a, 1), s.stop // grid_dict.get(a, 1)) for s, a in
                         zip(s_dst, axes_net)]
                s_src[channel] = slice(None)
                s_dst[channel] = slice(None)
                s_src, s_dst = tuple(s_src), tuple(s_dst)

                prob_tile, dist_tile = results_tile[:2]
                prob_tile, dist_tile = _prep(prob_tile[s_src], dist_tile[s_src])
                prob[s_dst[:3]] = prob_tile.copy()
                dist[s_dst[:3]] = dist_tile.copy() #TODO: many may incorrect here when use tiles

                inds = self._ind_prob_raw_thresh_(prob_tile, prob_thresh)
                proba.extend(prob_tile[inds].copy())
                dista.extend(dist_tile[inds].copy())
                _points = np.stack(np.where(inds), axis=1)
                offset = list(s.start for i, s in enumerate(s_dst))
                offset.pop(channel)
                _points = _points + np.array(offset).reshape((1, len(offset)))
                _points = _points * np.array(self.config.grid).reshape((1, len(self.config.grid)))
                pointsa.extend(_points)
        else:
            # predict_direct -> prob, dist, [prob_class if multi_class]
            results = predict_direct(cell_image_zyx)
            prob, dist = results[:2]
            prob, dist = _prep(prob, dist)
            # grid_zyx = self.config.grid
            # x_resized = cell_image_zyx[::grid_zyx[0], ::grid_zyx[1], ::grid_zyx[2], 0]
            inds   = self._ind_prob_raw_thresh_(prob, prob_thresh)
            proba = prob[inds].copy()
            dista = dist[inds].copy()
            _points = np.stack(np.where(inds), axis=1)
            pointsa = (_points * np.array(self.config.grid).reshape((1,len(self.config.grid))))

        proba = np.asarray(proba)
        dista = np.asarray(dista).reshape((-1,self.config.n_rays))
        pointsa = np.asarray(pointsa).reshape((-1,self.config.n_dim))

        prob_map_reduced = resizer.after(prob[:, :, :, None], self.config.axes)[..., 0]
        dist_map_reduced = resizer.after(dist, self.config.axes)

        idx = resizer.filter_points(cell_image_zyx.ndim, pointsa, axes_net)
        proba = proba[idx]
        dista = dista[idx]
        pointsa = pointsa[idx]

        return proba, dista, pointsa, prob_map_reduced, dist_map_reduced.astype(np.float16)

    @staticmethod
    def _ind_prob_raw_thresh_(prob, prob_thresh, raw=None):
        ind_thresh = prob > prob_thresh
        if raw is not None:
            median_intensity = np.percentile(raw[ind_thresh], 95)
            ind_thresh = np.logical_or(ind_thresh, raw > median_intensity)
        return ind_thresh

    def _predict_sparse_generator(self, img,
                                  prob_thresh=None, axes=None, normalizer=None, n_tiles=None,
                                  show_tile_progress=True, b=2, **predict_kwargs):
        """ Sparse version of model.predict()
        Returns
        -------
        (prob, dist, [prob_class], points)   flat list of probs, dists, (optional prob_class) and points
        """
        if prob_thresh is None: prob_thresh = self.thresholds.prob

        x, axes, axes_net, axes_net_div_by, _permute_axes, resizer, n_tiles, grid, grid_dict, channel, predict_direct, tiling_setup = \
            self._predict_setup(img, axes, normalizer, n_tiles, show_tile_progress, predict_kwargs)

        def _prep(prob, dist):
            prob = np.take(prob,0,axis=channel)
            dist = np.moveaxis(dist,channel,-1)
            dist = np.maximum(1e-3, dist)
            return prob, dist

        if np.prod(n_tiles) > 1:
            tile_generator, output_shape, create_empty_output = tiling_setup()

            sh = list(output_shape)
            sh[channel] = 1

            prob = np.zeros(output_shape[:3])
            proba, dista, pointsa, prob_classa = [], [], [], []

            for tile, s_src, s_dst in tile_generator:

                results_tile = predict_direct(tile)

                # account for grid
                s_src = [slice(s.start // grid_dict.get(a, 1), s.stop // grid_dict.get(a, 1)) for s, a in
                         zip(s_src, axes_net)]
                s_dst = [slice(s.start // grid_dict.get(a, 1), s.stop // grid_dict.get(a, 1)) for s, a in
                         zip(s_dst, axes_net)]
                s_src[channel] = slice(None)
                s_dst[channel] = slice(None)
                s_src, s_dst = tuple(s_src), tuple(s_dst)

                prob_tile, dist_tile = results_tile[:2]
                prob_tile, dist_tile = _prep(prob_tile[s_src], dist_tile[s_src])
                prob[s_dst[:3]] = prob_tile.copy()

                bs = list((b if s.start == 0 else -1, b if s.stop == _sh else -1) for s, _sh in zip(s_dst, sh))
                bs.pop(channel)
                inds = _ind_prob_thresh(prob_tile, prob_thresh, b=bs)
                proba.extend(prob_tile[inds].copy())
                dista.extend(dist_tile[inds].copy())
                _points = np.stack(np.where(inds), axis=1)
                offset = list(s.start for i, s in enumerate(s_dst))
                offset.pop(channel)
                _points = _points + np.array(offset).reshape((1, len(offset)))
                _points = _points * np.array(self.config.grid).reshape((1, len(self.config.grid)))
                pointsa.extend(_points)

                if self._is_multiclass():
                    p = results_tile[2][s_src].copy()
                    p = np.moveaxis(p, channel, -1)
                    prob_classa.extend(p[inds])
                yield  # yield None after each processed tile

        else:
            # predict_direct -> prob, dist, [prob_class if multi_class]
            results = predict_direct(x)
            prob, dist = results[:2]
            prob, dist = _prep(prob, dist)
            inds   = _ind_prob_thresh(prob, prob_thresh, b=b)
            proba = prob[inds].copy()
            dista = dist[inds].copy()
            _points = np.stack(np.where(inds), axis=1)
            pointsa = (_points * np.array(self.config.grid).reshape((1,len(self.config.grid))))

            if self._is_multiclass():
                p = np.moveaxis(results[2],channel,-1)
                prob_classa = p[inds].copy()

        proba = np.asarray(proba)
        dista = np.asarray(dista).reshape((-1,self.config.n_rays))
        pointsa = np.asarray(pointsa).reshape((-1,self.config.n_dim))

        prob_map = resizer.after(prob[:, :, :, None], self.config.axes)[..., 0]

        idx = resizer.filter_points(x.ndim, pointsa, axes_net)
        proba = proba[idx]
        dista = dista[idx]
        pointsa = pointsa[idx]

        # last "yield" is the actual output that would have been "return"ed if this was a regular function
        if self._is_multiclass():
            prob_classa = np.asarray(prob_classa).reshape((-1,self.config.n_classes+1))
            prob_classa = prob_classa[idx]
            yield proba, dista, prob_classa, pointsa, prob_map
        else:
            prob_classa = None
            yield proba, dista, pointsa, prob_map

    def optimize_thresholds(self, X_val, Y_val, nms_threshs=[0.2, 0.3,0.4], iou_threshs=[0.3,0.5,0.7], predict_kwargs=None, optimize_kwargs=None, save_to_json=True):
        """Optimize two thresholds (probability, NMS overlap) necessary for predicting object instances.

        Note that the default thresholds yield good results in many cases, but optimizing
        the thresholds for a particular dataset can further improve performance.

        The optimized thresholds are automatically used for all further predictions
        and also written to the model directory.

        See ``utils.optimize_threshold`` for details and possible choices for ``optimize_kwargs``.

        Parameters
        ----------
        X_val : list of ndarray
            (Validation) input images (must be normalized) to use for threshold tuning.
        Y_val : list of ndarray
            (Validation) label images to use for threshold tuning.
        nms_threshs : list of float
            List of overlap thresholds to be considered for NMS.
            For each value in this list, optimization is run to find a corresponding prob_thresh value.
        iou_threshs : list of float
            List of intersection over union (IOU) thresholds for which
            the (average) matching performance is considered to tune the thresholds.
        predict_kwargs: dict
            Keyword arguments for ``predict`` function of this class.
            (If not provided, will guess value for `n_tiles` to prevent out of memory errors.)
        optimize_kwargs: dict
            Keyword arguments for ``utils.optimize_threshold`` function.

        """
        if predict_kwargs is None:
            predict_kwargs = {}
        if optimize_kwargs is None:
            optimize_kwargs = {}

        def _predict_kwargs(x):
            if 'n_tiles' in predict_kwargs:
                return predict_kwargs
            else:
                return {**predict_kwargs, 'n_tiles': self._guess_n_tiles(x), 'show_tile_progress': False}

        # only take first two elements of predict in case multi class is activated
        Yhat_val = [self.predict(x, **_predict_kwargs(x))[:2] for x in X_val]

        opt_prob_thresh, opt_measure, opt_nms_thresh = None, -np.inf, None
        for _opt_nms_thresh in nms_threshs:
            _opt_prob_thresh, _opt_measure = optimize_threshold(Y_val, Yhat_val, model=self, nms_thresh=_opt_nms_thresh, iou_threshs=iou_threshs, **optimize_kwargs)
            if _opt_measure > opt_measure:
                opt_prob_thresh, opt_measure, opt_nms_thresh = _opt_prob_thresh, _opt_measure, _opt_nms_thresh
        opt_threshs = dict(prob=opt_prob_thresh, nms=opt_nms_thresh)

        self.thresholds = opt_threshs
        print(end='', file=sys.stderr, flush=True)
        print("Using optimized values: prob_thresh={prob:g}, nms_thresh={nms:g}.".format(prob=self.thresholds.prob, nms=self.thresholds.nms))
        if save_to_json and self.basedir is not None:
            print("Saving to 'thresholds.json'.")
            save_json(opt_threshs, str(self.logdir / 'thresholds.json'))
        return opt_threshs


def optimize_threshold(Y, Yhat, model, nms_thresh, measure='accuracy', iou_threshs=[0.3,0.5,0.7], bracket=None, tol=1e-2, maxiter=20, verbose=1):
    """ Tune prob_thresh for provided (fixed) nms_thresh to maximize matching score (for given measure and averaged over iou_threshs). """
    np.isscalar(nms_thresh) or _raise(ValueError("nms_thresh must be a scalar"))
    iou_threshs = [iou_threshs] if np.isscalar(iou_threshs) else iou_threshs
    values = dict()

    if bracket is None:
        max_prob = max([np.max(prob) for prob, dist in Yhat])
        bracket = max_prob / 3, max_prob
    # print("bracket =", bracket)

    with tqdm(total=maxiter, disable=(verbose!=1), desc="NMS threshold = %g" % nms_thresh) as progress:

        def fn(thr):
            prob_thresh = np.clip(thr, *bracket)
            value = values.get(prob_thresh)
            if value is None:
                Y_instances = [model._instances_from_prediction(y.shape, *prob_dist, prob_thresh=prob_thresh, nms_thresh=nms_thresh)[0] for y,prob_dist in zip(Y,Yhat)]
                stats = matching_dataset(Y, Y_instances, thresh=iou_threshs, show_progress=False, parallel=True)
                values[prob_thresh] = value = np.mean([s._asdict()[measure] for s in stats])
            if verbose > 1:
                print("{now}   thresh: {prob_thresh:f}   {measure}: {value:f}".format(
                    now = datetime.datetime.now().strftime('%H:%M:%S'),
                    prob_thresh = prob_thresh,
                    measure = measure,
                    value = value,
                ), flush=True)
            else:
                progress.update()
                progress.set_postfix_str("{prob_thresh:.3f} -> {value:.3f}".format(prob_thresh=prob_thresh, value=value))
                progress.refresh()
            return -value

        opt = minimize_scalar(fn, method='golden', bracket=bracket, tol=tol, options={'maxiter': maxiter})

    verbose > 1 and print('\n',opt, flush=True)
    return opt.x, -opt.fun