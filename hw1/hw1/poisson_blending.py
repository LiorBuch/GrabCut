import math

import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg
import scipy.sparse
from scipy.sparse import lil_matrix, block_diag
from scipy.sparse.linalg import spsolve
import argparse
from scipy.ndimage import binary_dilation
from tqdm import tqdm


def poisson_blend(im_src, im_tgt, im_mask, center):
    # LAPLACIAN OPERATOR

    mask_height, mask_width = im_mask.shape
    num_pixels_to_paste = im_mask.shape[0] * im_mask.shape[1]
    # src_center = int(im_src.shape[0] / 2), int(im_src.shape[1] / 2)
    #
    # relative_h = center[0] - src_center[0]
    # relative_w = center[1] - src_center[1]

    relative_h = center[1] - mask_height // 2
    relative_w = center[0] - mask_width // 2

    # We build A according to the formulas in rec3
    A = [lil_matrix((num_pixels_to_paste, num_pixels_to_paste), dtype='float64') for _ in range(im_tgt.shape[2])]
    b = [np.zeros(num_pixels_to_paste, dtype='float64') for _ in range(im_tgt.shape[2])]
    f = []

    bound = get_boundary(im_mask)

    for pixel in bound:
        for color in range(im_tgt.shape[2]):
            im_src[pixel[0],pixel[1],color] = im_tgt[pixel[0]+relative_h,pixel[1]+relative_w,color]

    im_src = im_src.astype(np.int32)
    im_tgt = im_tgt.astype(np.int32)
    for color in range(im_tgt.shape[2]):
        for i in tqdm(range(mask_height)):
            for j in range(mask_width):
                idx = i * mask_width + j
                # up down left right
                src_buffer = [0, 0, 0, 0]
                tgt_buffer = [0, 0, 0, 0]
                if im_mask[i, j] >= 100:
                    A[color][idx, idx] = -4
                    # up
                    if 0 <= i - 1 < mask_height and 0 <= j <= mask_width:
                        if im_mask[i - 1, j] > 0:
                            A[color][idx, idx - mask_width] = 1
                            src_buffer[0] =(im_src[(i - 1), j, color] - im_src[i, j, color])
                            tgt_buffer[0] = (im_tgt[relative_h + (i - 1), relative_w + j, color] - im_tgt[
                                relative_h + i, relative_w + j, color])
                        else:
                            A[color][idx, idx - mask_width] = 0
                            #b[color][idx] += (im_src[(i - 1), j, color] - im_src[i, j, color])
                            b[color][idx] -= (im_tgt[relative_h + (i - 1), relative_w + j, color] - im_tgt[
                                relative_h + i, relative_w + j, color])

                    # down
                    if 0 <= i + 1 < mask_height and 0 <= j <= mask_width:
                        if im_mask[i + 1, j] > 0:
                            A[color][idx, idx + mask_width] = 1
                            src_buffer[1] = (im_src[(i + 1), j, color] - im_src[i, j, color])
                            tgt_buffer[1] = (im_tgt[relative_h + (i + 1), relative_w + j, color] - im_tgt[
                                relative_h + i, relative_w + j, color])

                        else:
                            A[color][idx, idx + mask_width] = 0
                            #b[color][idx] += (im_src[(i + 1), j, color] - im_src[i, j, color])
                            b[color][idx] -= (im_tgt[relative_h + (i + 1), relative_w + j, color] - im_tgt[
                                relative_h + i, relative_w + j, color])

                    # left
                    if 0 <= i < mask_height and 0 <= j - 1 <= mask_width:
                        if im_mask[i, j - 1] > 0:
                            A[color][idx, idx - 1] = 1
                            src_buffer[2]= (im_src[i, j - 1, color] - im_src[i, j, color])
                            tgt_buffer[2]= (im_tgt[relative_h + i, relative_w + j - 1, color] - im_tgt[
                                relative_h + i, relative_w + j, color])

                        else:
                            A[color][idx, idx - 1] = 0
                            #b[color][idx] += (im_src[i, j - 1, color] - im_src[i, j, color])
                            b[color][idx] -= (im_tgt[relative_h + i, relative_w + j - 1, color] - im_tgt[
                                relative_h + i, relative_w + j, color])

                    # right
                    if 0 <= i < mask_height and 0 <= j - 1 <= mask_width:
                        if im_mask[i, j + 1] > 0:
                            A[color][idx, idx + 1] = 1
                            src_buffer[3]= (im_src[i, j + 1, color] - im_src[i, j, color])
                            tgt_buffer[3]= (im_tgt[relative_h + i, relative_w + j + 1, color] - im_tgt[
                                relative_h + i, relative_w + j, color])
                        else:
                            A[color][idx, idx + 1] = 0
                            #b[color][idx] += (im_src[i, j + 1, color] - im_src[i, j, color])
                            b[color][idx] -= (im_tgt[relative_h + i, relative_w + j + 1, color] - im_tgt[
                                relative_h + i, relative_w + j, color])

                    b[color][idx] = sum(src_buffer) - sum(tgt_buffer)

                else:
                    A[color][idx, idx] = -1
                    b[color][idx] = im_tgt[relative_h + i, relative_w + j, color]

        # POISSON SOLVE
        f.append(spsolve(A[color].tocsc(), b[color]))

    # BLEND
    for color in range(im_tgt.shape[2]):
        for i in range(mask_height):
            for j in range(mask_width):
                if im_mask[i, j] >= 100:
                    value = np.clip(abs((f[color][i * mask_width + j]*0.95 +im_tgt[relative_h+i,relative_w+j,color])),0,255) # play with this?
                    print(f"color:{color} value -> {value}")
                    im_tgt[relative_h + i, relative_w + j, color] = value

    # SAVE
    print(len(bound))
    im_tgt = im_tgt.astype(np.uint8)
    im_blend = im_tgt
    return im_blend


def get_boundary(mask):
    structure = np.ones((3,3))
    dilated_mask = binary_dilation(mask, structure=structure)
    boundary = dilated_mask - mask
    boundary_pixels = np.argwhere(boundary == 1)
    return boundary_pixels


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/banana2.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/banana2.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/wall.jpg', help='mask file path')
    return parser.parse_args()


if __name__ == "__main__":
    # Load the source and target images
    args = parse()

    im_tgt = cv2.imread(args.tgt_path, cv2.IMREAD_COLOR)
    im_src = cv2.imread(args.src_path, cv2.IMREAD_COLOR)
    if args.mask_path == '':
        im_mask = np.full(im_src.shape, 255, dtype=np.uint8)
    else:
        im_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]

    center = (int(im_tgt.shape[1] / 2), int(im_tgt.shape[0] / 2))

    im_clone = poisson_blend(im_src, im_tgt, im_mask, center)
    #im_clone = cv2.seamlessClone(im_src, im_tgt, im_mask, center, cv2.NORMAL_CLONE)

    cv2.imshow('Cloned image', im_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
