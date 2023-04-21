from __future__ import division, absolute_import, print_function, unicode_literals
from __future__ import print_function, unicode_literals, absolute_import, division

import functools
import numbers
import warnings

import numpy as np
import scipy.ndimage as ndi
from csbdeep.utils import _raise
from csbdeep.utils.tf import keras_import
from stardist.models import StarDist3D
from stardist.nms import _ind_prob_thresh

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
            prob, dist, prob_class, points, prob_map = res
        else:
            prob, dist, points, prob_map = res
            prob_class = None

        yield 'nms'  # indicate that non-maximum suppression is starting
        res_instances = self._instances_from_prediction(_shape_inst, prob, dist,
                                                        points=points,
                                                        prob_class=prob_class,
                                                        prob_thresh=prob_thresh,
                                                        nms_thresh=nms_thresh,
                                                        scale=(None if scale is None else dict(zip(_axes,scale))),
                                                        return_labels=return_labels,
                                                        overlap_label=overlap_label,
                                                        **nms_kwargs)

        # last "yield" is the actual output that would have been "return"ed if this was a regular function
        if return_predict:
            yield res_instances, tuple(res[:-1]), prob_map
        else:
            yield res_instances, prob_map

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
        r = None
        for r in self._predict_instances_generator(*args, **kwargs):
            pass
        return r

    def _predict_sparse_generator(self, img, prob_thresh=None, axes=None, normalizer=None, n_tiles=None, show_tile_progress=True, b=2, **predict_kwargs):
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

        proba, dista, pointsa, prob_class = [],[],[], []

        if np.prod(n_tiles) > 1:
            raise NotImplementedError("The prediction function has not been realized when np.prod(n_tiles)>1")
            tile_generator, output_shape, create_empty_output = tiling_setup()

            sh = list(output_shape)
            sh[channel] = 1;

            proba, dista, pointsa, prob_classa = [], [], [], []

            for tile, s_src, s_dst in tile_generator:

                results_tile = predict_direct(tile)

                # account for grid
                s_src = [slice(s.start//grid_dict.get(a,1),s.stop//grid_dict.get(a,1)) for s,a in zip(s_src,axes_net)]
                s_dst = [slice(s.start//grid_dict.get(a,1),s.stop//grid_dict.get(a,1)) for s,a in zip(s_dst,axes_net)]
                s_src[channel] = slice(None)
                s_dst[channel] = slice(None)
                s_src, s_dst = tuple(s_src), tuple(s_dst)

                prob_tile, dist_tile = results_tile[:2]
                prob_tile, dist_tile = _prep(prob_tile[s_src], dist_tile[s_src])

                bs = list((b if s.start==0 else -1, b if s.stop==_sh else -1) for s,_sh in zip(s_dst, sh))
                bs.pop(channel)
                inds   = _ind_prob_thresh(prob_tile, prob_thresh, b=bs)
                proba.extend(prob_tile[inds].copy())
                dista.extend(dist_tile[inds].copy())
                _points = np.stack(np.where(inds), axis=1)
                offset = list(s.start for i,s in enumerate(s_dst))
                offset.pop(channel)
                _points = _points + np.array(offset).reshape((1,len(offset)))
                _points = _points * np.array(self.config.grid).reshape((1,len(self.config.grid)))
                pointsa.extend(_points)

                if self._is_multiclass():
                    p = results_tile[2][s_src].copy()
                    p = np.moveaxis(p,channel,-1)
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
