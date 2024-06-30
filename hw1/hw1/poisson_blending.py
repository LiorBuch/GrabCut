import math

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
import argparse
from scipy.ndimage import binary_dilation


def poisson_blend(im_src, im_tgt, im_mask, center):
    # LAPLACIAN OPERATOR

    mask_height,mask_width = im_mask.shape
    num_pixels_to_paste = im_mask.shape[0]*im_mask.shape[1]
    src_center = int(im_src.shape[0] / 2) , int(im_src.shape[1] / 2)

    relative_h = center[0] - src_center[0]
    relative_w = center[1] - src_center[1]

    # We build A according to the formulas in rec3
    A = [lil_matrix((num_pixels_to_paste, num_pixels_to_paste), dtype='float64') for i in range(im_tgt.shape[2])]
    b = [np.zeros(num_pixels_to_paste) for i in range(im_tgt.shape[2])]
    f = []
    for color in range(im_tgt.shape[2]):
        print("color")
        for i in range(mask_height):
            print(f"row number {i} out of {mask_height}")
            for j in range(mask_width):
                idx = i * mask_width + j
                if im_mask[i, j] >= 100:
                    A[color][i*mask_width+j,i*mask_width+j] = 4
                    for k, m in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                        ni, nj = i + k, j + m
                        if 0 <= ni < mask_height and 0 <= nj < mask_width:
                            if im_mask[i + k, j + m] >= 100:
                                n_idx = ni * mask_width + nj
                                A[color][n_idx,n_idx] = -1
                                b[color][idx] += im_src[i,j, color]/3
                            else:  # If on the edge of the mask
                                # TODO: Check that not out of bound in im_tgt
                                b[color][idx] += im_tgt[relative_h + ni, relative_w + nj, color]
                else:
                    A[color][i * mask_width + j, i * mask_width + j] = 1
                    try:
                        b[color][idx] = im_tgt[relative_h+i,relative_w + j,color]
                    except:
                        b[color][idx] = 10

        # POISSON SOLVE
        f.append(spsolve(A[color].tocsc(), b[color]))

    # BLEND
    # boundary = get_boundary(im_mask)
    for color in range(im_tgt.shape[2]):
        for i in range(im_src.shape[0]):
            for j in range(im_src.shape[1]):
                if im_mask[i, j] >= 100:
                    print(f"Pasting {i}, {j} -> {f[color][i * mask_width + j]}")
                    im_tgt[relative_h+i,relative_w+j,color] = np.abs(f[color][i * mask_width + j])

    # SAVE

    im_blend = im_tgt
    return im_blend

def get_boundary(mask):
    structure = np.ones((3, 3))
    dilated_mask = binary_dilation(mask, structure=structure)
    boundary = dilated_mask - mask
    boundary_pixels = np.argwhere(boundary == 1)
    return boundary_pixels



def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/banana2.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/banana1.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/table.jpg', help='mask file path')
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

    cv2.imshow('Cloned image', im_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
