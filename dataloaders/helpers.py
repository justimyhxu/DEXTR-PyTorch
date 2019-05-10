import os

import torch, cv2
import random
import numpy as np
from PIL import Image


def tens2image(im):
    if im.size()[0] == 1:
        tmp = np.squeeze(im.numpy(), axis=0)
    else:
        tmp = im.numpy()
    if tmp.ndim == 2:
        return tmp
    else:
        return tmp.transpose((1, 2, 0))


def crop2fullmask(crop_mask, bbox, im=None, im_size=None, zero_pad=False, relax=0, mask_relax=True,
                  interpolation=cv2.INTER_CUBIC, scikit=False):
    if scikit:
        from skimage.transform import resize as sk_resize
    assert(not(im is None and im_size is None)), 'You have to provide an image or the image size'
    if im is None:
        im_si = im_size
    else:
        im_si = im.shape
    # Borers of image
    bounds = (0, 0, im_si[1] - 1, im_si[0] - 1)

    # Valid bounding box locations as (x_min, y_min, x_max, y_max)
    bbox_valid = (max(bbox[0], bounds[0]),
                  max(bbox[1], bounds[1]),
                  min(bbox[2], bounds[2]),
                  min(bbox[3], bounds[3]))

    # Bounding box of initial mask
    bbox_init = (bbox[0] + relax,
                 bbox[1] + relax,
                 bbox[2] - relax,
                 bbox[3] - relax)

    if zero_pad:
        # Offsets for x and y
        offsets = (-bbox[0], -bbox[1])
    else:
        assert((bbox == bbox_valid).all())
        offsets = (-bbox_valid[0], -bbox_valid[1])

    # Simple per element addition in the tuple
    inds = tuple(map(sum, zip(bbox_valid, offsets + offsets)))

    if scikit:
        crop_mask = sk_resize(crop_mask, (bbox[3] - bbox[1] + 1, bbox[2] - bbox[0] + 1), order=0, mode='constant').astype(crop_mask.dtype)
    else:
        crop_mask = cv2.resize(crop_mask, (bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1), interpolation=interpolation)
    result_ = np.zeros(im_si)
    result_[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1] = \
        crop_mask[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1]

    result = np.zeros(im_si)
    if mask_relax:
        result[bbox_init[1]:bbox_init[3]+1, bbox_init[0]:bbox_init[2]+1] = \
            result_[bbox_init[1]:bbox_init[3]+1, bbox_init[0]:bbox_init[2]+1]
    else:
        result = result_

    return result


def overlay_mask(im, ma, colors=None, alpha=0.5):
    assert np.max(im) <= 1.0
    if colors is None:
        colors = np.load(os.path.join(os.path.dirname(__file__), 'pascal_map.npy'))/255.
    else:
        colors = np.append([[0.,0.,0.]], colors, axis=0);

    if ma.ndim == 3:
        assert len(colors) >= ma.shape[0], 'Not enough colors'
    ma = ma.astype(np.bool)
    im = im.astype(np.float32)

    if ma.ndim == 2:
        fg = im * alpha+np.ones(im.shape) * (1 - alpha) * colors[1, :3]   # np.array([0,0,255])/255.0
    else:
        fg = []
        for n in range(ma.ndim):
            fg.append(im * alpha + np.ones(im.shape) * (1 - alpha) * colors[1+n, :3])
    # Whiten background
    bg = im.copy()
    if ma.ndim == 2:
        bg[ma == 0] = im[ma == 0]
        bg[ma == 1] = fg[ma == 1]
        total_ma = ma
    else:
        total_ma = np.zeros([ma.shape[1], ma.shape[2]])
        for n in range(ma.shape[0]):
            tmp_ma = ma[n, :, :]
            total_ma = np.logical_or(tmp_ma, total_ma)
            tmp_fg = fg[n]
            bg[tmp_ma == 1] = tmp_fg[tmp_ma == 1]
        bg[total_ma == 0] = im[total_ma == 0]

    # [-2:] is s trick to be compatible both with opencv 2 and 3
    contours = cv2.findContours(total_ma.copy().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cv2.drawContours(bg, contours[0], -1, (0.0, 0.0, 0.0), 1)

    return bg

def overlay_masks(im, masks, alpha=0.5):
    colors = np.load(os.path.join(os.path.dirname(__file__), 'pascal_map.npy'))/255.
    
    if isinstance(masks, np.ndarray):
        masks = [masks]

    assert len(colors) >= len(masks), 'Not enough colors'

    ov = im.copy()
    im = im.astype(np.float32)
    total_ma = np.zeros([im.shape[0], im.shape[1]])
    i = 1
    for ma in masks:
        ma = ma.astype(np.bool)
        fg = im * alpha+np.ones(im.shape) * (1 - alpha) * colors[i, :3]   # np.array([0,0,255])/255.0
        i = i + 1
        ov[ma == 1] = fg[ma == 1]
        total_ma += ma

        # [-2:] is s trick to be compatible both with opencv 2 and 3
        contours = cv2.findContours(ma.copy().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        cv2.drawContours(ov, contours[0], -1, (0.0, 0.0, 0.0), 1)
    ov[total_ma == 0] = im[total_ma == 0]

    return ov


def extreme_points(mask, pert):
    def find_point(id_x, id_y, ids):
        sel_id = ids[0][random.randint(0, len(ids[0]) - 1)]
        return [id_x[sel_id], id_y[sel_id]]

    # List of coordinates of the mask
    inds_y, inds_x = np.where(mask > 0.5)

    # Find extreme points
    return np.array([find_point(inds_x, inds_y, np.where(inds_x <= np.min(inds_x)+pert)), # left
                     find_point(inds_x, inds_y, np.where(inds_x >= np.max(inds_x)-pert)), # right
                     find_point(inds_x, inds_y, np.where(inds_y <= np.min(inds_y)+pert)), # top
                     find_point(inds_x, inds_y, np.where(inds_y >= np.max(inds_y)-pert)) # bottom
                     ])
def mask_to_poly(mask, visualize=False):
    # import ipdb
    # ipdb.set_trace()
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) > 4:
            polygons.append(contour)
    import matplotlib.pyplot as plt
    if visualize:
        plt.imshow(mask)
        for polygon in polygons:
            # self.plot_polygon(polygon)
            print(polygon)
            y = polygon[1::2]
            y.append(y[0])
            x = polygon[0::2]
            x.append(x[0])
            plt.plot(x, y)
        plt.show()
        # plt.savefig('vis.png')
        # # plt.clf()
        # import pdb
        # pdb.set_trace()
    return polygons

def sample_on_polygon(polygon, n_points=50):
    y = polygon[0::2]
    x = polygon[1::2]
    x_t, y_t = x[1:], y[1:]
    x_t.append(x[0])
    y_t.append(y[0])
    x_t, y_t = np.asarray(x_t), np.asarray(y_t)
    dist = np.sqrt((x_t - x) * (x_t - x) + (y_t - y) * (y_t - y))
    dist_sum = np.cumsum(dist)
    stride = dist_sum[-1] / n_points

    x_set, y_set = [], []
    for i in range(n_points):
        length = i * stride
        idx = np.where(length <= dist_sum)[0][0]
        length_remained = dist_sum[idx] - length
        eps = 1e-12
        alpha = 1 - length_remained / (dist[idx] + eps)
        if idx < np.shape(dist)[0] - 1:
            x_i = (1 - alpha) * x[idx] + alpha * x[idx + 1]
            y_i = (1 - alpha) * y[idx] + alpha * y[idx + 1]
        else:
            x_i = (1 - alpha) * x[idx] + alpha * x[0]
            y_i = (1 - alpha) * y[idx] + alpha * y[0]
        x_set.append(x_i)
        y_set.append(y_i)
    x_set, y_set = np.array(x_set), np.array(y_set)
    point_set = np.concatenate((y_set[:,None], x_set[:,None]), 1)
    point_set = point_set.reshape(-1)
    return point_set

def polygon_len(polygon):
    y = polygon[0::2]
    x = polygon[1::2]
    x_t, y_t = x[1:], y[1:]
    x_t.append(x[0])
    y_t.append(y[0])
    x_t, y_t = np.asarray(x_t), np.asarray(y_t)
    dist = np.sqrt((x_t - x) * (x_t - x) + (y_t - y) * (y_t - y))
    perimeter = sum(dist)
    return perimeter

def get_polygon_points(polygons, num_pts, img_shape):

    if len(polygons) == 0:
        polygon_sample = np.zeros([2*num_pts], np.float64)
    else:
        polygons_len = np.array([polygon_len(polygon) for polygon in polygons])
        polygons_ratio = polygons_len/polygons_len.sum()
        polynums_num = (polygons_ratio*num_pts).astype(np.int)
        polynums_num[-1] = num_pts-polynums_num[:-1].sum()
        fuse_polygon = []
        for poly_numlen, polygon in zip(polynums_num, polygons):
            if poly_numlen == 0:
                continue
            fuse_polygon.append(sample_on_polygon(polygon, poly_numlen))
        polygon_sample = np.concatenate(fuse_polygon)
    polygon_scale = polygon_sample * 1
    polygon_scale[0::2] = np.clip(polygon_scale[0::2], 0, img_shape[1] - 1)
    polygon_scale[1::2] = np.clip(polygon_scale[1::2], 0, img_shape[0] - 1)

    polygon_scale = polygon_scale.reshape(-1,2)
    return polygon_scale

def get_mask_sample_points(mask, num_pts):
    index = np.nonzero(mask>0)
    index = np.vstack(index).transpose(1,0)
    index = index[:,::-1]
    _len = index.shape[0]
    if _len > num_pts:
        np.random.shuffle(index)
        index = index[:num_pts]
    else:
        sindex = [np.random.choice(_len)for i in range(num_pts)]
        index = index[sindex]
    return index

def get_mask_noise_sample_masks(mask, num_pts, ratio=0):
    index = np.nonzero(mask>0)
    y1,y2, x1, x2 = min(index[0]), max(index[0]), min(index[1]), max(index[1])

    index = np.vstack(index).transpose(1,0)
    index = index[:,::-1]


    nindex = np.nonzero(mask == 0)
    nindex = np.vstack(nindex).transpose(1, 0)
    nindex = nindex[:, ::-1]
    index_mask  = np.logical_and(np.logical_and(nindex[:,0]<=x2,nindex[:,0]>=x1), np.logical_and(nindex[:,1]<=y2, nindex[:, 1]>=y1))
    nindex = nindex[index_mask]


    _len_in = index.shape[0]
    _len_out = nindex.shape[0]
    pts_in = int(num_pts*(1-ratio))
    if pts_in == 0:
        pts_in += 1
    pts_out = num_pts-pts_in
    if _len_in > pts_in and _len_out > pts_out :
        np.random.shuffle(index)
        index = index[:pts_in]
        np.random.shuffle(nindex)
        nindex = nindex[:pts_out]
    elif _len_in > pts_in and _len_out < pts_out:
        np.random.shuffle(index)
        index = index[:pts_in]
        sindex = [np.random.choice(_len_out)for i in range(pts_out)]
        nindex = nindex[sindex]
    elif _len_in < pts_in and _len_out > pts_out:
        sindex = [np.random.choice(_len_in) for i in range(pts_in)]
        index = nindex[sindex]
        np.random.shuffle(nindex)
        nindex = nindex[:pts_out]
    else:
        sindex = [np.random.choice(_len_in) for i in range(pts_in)]
        index = nindex[sindex]
        sindex = [np.random.choice(_len_out) for i in range(pts_out)]
        nindex = nindex[sindex]

    return np.concatenate((index, nindex), axis=0)


def get_bbox_sample_points(mask, num_pts):
    index = np.nonzero(mask>0)
    x1,x2, y1, y2 = min(index[0]), max(index[0]), min(index[1]), max(index[1])
    points = []
    for i in range(num_pts):
        points.append((np.random.randint(y1,y2),np.random.randint(x1,x2)))
    return np.array(points)

def get_bbox(mask, points=None, pad=0, zero_pad=False):
    if points is not None:
        inds = np.flip(points.transpose(), axis=0)
    else:
        inds = np.where(mask > 0)

    if inds[0].shape[0] == 0:
        return None

    if zero_pad:
        x_min_bound = -np.inf
        y_min_bound = -np.inf
        x_max_bound = np.inf
        y_max_bound = np.inf
    else:
        x_min_bound = 0
        y_min_bound = 0
        x_max_bound = mask.shape[1] - 1
        y_max_bound = mask.shape[0] - 1

    x_min = max(inds[1].min() - pad, x_min_bound)
    y_min = max(inds[0].min() - pad, y_min_bound)
    x_max = min(inds[1].max() + pad, x_max_bound)
    y_max = min(inds[0].max() + pad, y_max_bound)

    return x_min, y_min, x_max, y_max


def crop_from_bbox(img, bbox, zero_pad=False):
    # Borders of image
    bounds = (0, 0, img.shape[1] - 1, img.shape[0] - 1)

    # Valid bounding box locations as (x_min, y_min, x_max, y_max)
    bbox_valid = (max(bbox[0], bounds[0]),
                  max(bbox[1], bounds[1]),
                  min(bbox[2], bounds[2]),
                  min(bbox[3], bounds[3]))

    if zero_pad:
        # Initialize crop size (first 2 dimensions)
        crop = np.zeros((bbox[3] - bbox[1] + 1, bbox[2] - bbox[0] + 1), dtype=img.dtype)

        # Offsets for x and y
        offsets = (-bbox[0], -bbox[1])

    else:
        assert(bbox == bbox_valid)
        crop = np.zeros((bbox_valid[3] - bbox_valid[1] + 1, bbox_valid[2] - bbox_valid[0] + 1), dtype=img.dtype)
        offsets = (-bbox_valid[0], -bbox_valid[1])

    # Simple per element addition in the tuple
    inds = tuple(map(sum, zip(bbox_valid, offsets + offsets)))

    img = np.squeeze(img)
    if img.ndim == 2:
        crop[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1] = \
            img[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1]
    else:
        crop = np.tile(crop[:, :, np.newaxis], [1, 1, 3])  # Add 3 RGB Channels
        crop[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1, :] = \
            img[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1, :]

    return crop


def fixed_resize(sample, resolution, flagval=None):

    if flagval is None:
        if ((sample == 0) | (sample == 1)).all():
            flagval = cv2.INTER_NEAREST
        else:
            flagval = cv2.INTER_CUBIC

    if isinstance(resolution, int):
        tmp = [resolution, resolution]
        tmp[np.argmax(sample.shape[:2])] = int(round(float(resolution)/np.min(sample.shape[:2])*np.max(sample.shape[:2])))
        resolution = tuple(tmp)

    if sample.ndim == 2 or (sample.ndim == 3 and sample.shape[2] == 3):
        sample = cv2.resize(sample, resolution[::-1], interpolation=flagval)
    else:
        tmp = sample
        sample = np.zeros(np.append(resolution, tmp.shape[2]), dtype=np.float32)
        for ii in range(sample.shape[2]):
            sample[:, :, ii] = cv2.resize(tmp[:, :, ii], resolution[::-1], interpolation=flagval)
    return sample


def crop_from_mask(img, mask, relax=0, zero_pad=False):
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, dsize=tuple(reversed(img.shape[:2])), interpolation=cv2.INTER_NEAREST)

    assert(mask.shape[:2] == img.shape[:2])

    bbox = get_bbox(mask, pad=relax, zero_pad=zero_pad)

    if bbox is None:
        return None

    crop = crop_from_bbox(img, bbox, zero_pad)

    return crop


def make_gaussian(size, sigma=10, center=None, d_type=np.float64):
    """ Make a square gaussian kernel.
    size: is the dimensions of the output gaussian
    sigma: is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size[1], 1, float)
    y = np.arange(0, size[0], 1, float)
    y = y[:, np.newaxis]

    if center is None:
        x0 = y0 = size[0] // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2).astype(d_type)


def make_gt(img, labels, sigma=10, one_mask_per_point=False):
    """ Make the ground-truth for  landmark.
    img: the original color image
    labels: label with the Gaussian center(s) [[x0, y0],[x1, y1],...]
    sigma: sigma of the Gaussian.
    one_mask_per_point: masks for each point in different channels?
    """
    h, w = img.shape[:2]
    if labels is None:
        gt = make_gaussian((h, w), center=(h//2, w//2), sigma=sigma)
    else:
        labels = np.array(labels)
        if labels.ndim == 1:
            labels = labels[np.newaxis]
        if one_mask_per_point:
            gt = np.zeros(shape=(h, w, labels.shape[0]))
            for ii in range(labels.shape[0]):
                gt[:, :, ii] = make_gaussian((h, w), center=labels[ii, :], sigma=sigma)
        else:
            gt = np.zeros(shape=(h, w), dtype=np.float64)
            for ii in range(labels.shape[0]):
                gt = np.maximum(gt, make_gaussian((h, w), center=labels[ii, :], sigma=sigma))

    gt = gt.astype(dtype=img.dtype)

    return gt


def cstm_normalize(im, max_value):
    """
    Normalize image to range 0 - max_value
    """
    imn = max_value*(im - im.min()) / max((im.max() - im.min()), 1e-8)
    return imn


def generate_param_report(logfile, param):
    log_file = open(logfile, 'w')
    for key, val in param.items():
        log_file.write(key+':'+str(val)+'\n')
    log_file.close()

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def save_mask(results, mask_path):
    mask = np.zeros(results[0].shape)
    for ii, r in enumerate(results):
        mask[r] = ii + 1
    result = Image.fromarray(mask.astype(np.uint8))
    result.putpalette(list(color_map(80).flatten()))
    result.save(mask_path)