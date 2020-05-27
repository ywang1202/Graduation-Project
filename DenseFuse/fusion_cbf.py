import numpy as np
import tensorflow as tf


def gauss_kernel(sigma, ksize):
    x = np.arange(-(ksize//2), ksize//2 + 1)
    x = np.tile(x ** 2, [ksize, 1])
    x = x + np.transpose(x)

    gk = tf.constant(np.exp(-(x/(2*sigma**2))), dtype="float32")
    gk = tf.expand_dims(gk, -1)
    gk = tf.expand_dims(gk, 0)
    gk = tf.tile(gk, [1, 1, 1, 64])

    return gk


def cross_bilateral_filter(fm1, fm2, sigmas, sigmar, ksize):
    half_size = ksize // 2

    gk = gauss_kernel(sigmas, ksize)

    fm1_padded = tf.pad(fm1, [[0, 0], [half_size, half_size], [half_size, half_size], [0, 0]], mode="REFLECT")
    fm2_padded = tf.pad(fm2, [[0, 0], [half_size, half_size], [half_size, half_size], [0, 0]], mode="REFLECT")

    _, h, w, _ = fm1_padded.shape
    # cbf_out = np.zeros(fm1.shape)
    flag = True
    for i in range(half_size, h - half_size):
        for j in range(half_size, w - half_size):
            xtemp1 = fm1_padded[:, i - half_size:i + half_size + 1, j - half_size:j + half_size + 1, :]
            xtemp2 = fm2_padded[:, i - half_size:i + half_size + 1, j - half_size:j + half_size + 1, :]
            # a = xtemp2[:, half_size:half_size+1, half_size:half_size+1, :]
            pix_diff = tf.math.abs(xtemp2 - xtemp2[:, half_size:half_size+1, half_size:half_size+1, :])
            rgk = tf.math.exp(-(pix_diff**2/(2*sigmar**2)))
            tmp = tf.reduce_sum(xtemp1 * gk * rgk, axis=[1, 2], keepdims=True) / tf.reduce_sum(gk * rgk, axis=[1, 2], keepdims=True)
            if flag:
                cbf_out = tmp
                flag = False
            else:
                cbf_out = tf.concat([cbf_out, tmp], axis=1)
            # print(cbf_out.shape)
    cbf_out = tf.reshape(cbf_out, shape=fm1.shape)

    return cbf_out


def compute_covariance(x, cov_wsize):
    _mean = tf.reduce_mean(x, axis=[1], keepdims=True)
    _mean = tf.tile(_mean, [1, cov_wsize, 1, 1])
    tr = x - _mean

    tr = tf.transpose(tr, [0, 3, 1, 2])
    cov = tf.matmul(tr, tr, transpose_a=True)
    # cov = tf.transpose(cov, [0, 2, 3, 1])

    return cov / (cov_wsize - 1)


def compute_weight_matrix(detail, cov_wsize):
    half_wsize = cov_wsize // 2

    detail_padded = tf.pad(detail, [[0, 0], [half_wsize, half_wsize], [half_wsize, half_wsize], [0, 0]], mode="REFLECT")

    _, h, w, _ = detail_padded.shape
    # wt = np.zeros(detail.shape)
    flag = True
    for i in range(half_wsize, h - half_wsize):
        for j in range(half_wsize, w - half_wsize):
            temp = detail_padded[:, i - half_wsize:i + half_wsize + 1, j - half_wsize: j + half_wsize + 1, :]

            hor_cov = compute_covariance(temp, cov_wsize)
            ver_cov = compute_covariance(tf.transpose(temp, [0, 2, 1, 3]), cov_wsize)

            hor_eigvals = tf.linalg.eigvalsh(hor_cov)
            ver_eigvals = tf.linalg.eigvalsh(ver_cov)

            hor_sum = tf.reduce_sum(hor_eigvals, axis=[-1], keepdims=True)
            ver_sum = tf.reduce_sum(ver_eigvals, axis=[-1], keepdims=True)
            eigvals_sum = hor_sum + ver_sum

            eigvals_sum = tf.expand_dims(eigvals_sum, -1)
            eigvals_sum = tf.transpose(eigvals_sum, [0, 2, 3, 1])
            # wt[:, i - half_wsize, j - half_wsize, :] = eigvals_sum.eval(session=sess)
            if flag:
                wt = eigvals_sum
                flag = False
            else:
                wt = tf.concat([wt, eigvals_sum], axis=1)
            # print(wt.shape)
    wt = tf.reshape(wt, detail.shape)
    # print(wt.shape)
    # wt[wt == 0] = np.spacing(1)
    return wt


def CBF_Strategy(fm1, fm2, sigmas, sigmar, ksize, cov_wsize):
    cbf_out1 = cross_bilateral_filter(fm1, fm2, sigmas, sigmar, ksize)
    detail1 = fm1 - cbf_out1
    cbf_out2 = cross_bilateral_filter(fm2, fm1, sigmas, sigmar, ksize)
    detail2 = fm2 - cbf_out2

    wt1 = compute_weight_matrix(detail1, cov_wsize)
    wt2 = compute_weight_matrix(detail2, cov_wsize)

    fuse_image = (fm1 * wt1 + fm2 * wt2) / (wt1 + wt2)
    return fuse_image