import time

import numpy as np
import cv2
import igraph as ig
import argparse
from sklearn.mixture import GaussianMixture

GC_BGD = 0  # Hard bg pixel
GC_FGD = 1  # Hard fg pixel, will not be used
GC_PR_BGD = 2  # Soft bg pixel
GC_PR_FGD = 3  # Soft fg pixel


# Define the GrabCut algorithm function
def grabcut(img, rect, n_iter=5):
    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect

    # Convert from absolute cordinates
    w -= x
    h -= y

    # Initalize the inner square to Foreground
    mask[y:y + h, x:x + w] = GC_PR_FGD
    mask[rect[1] + rect[3] // 2, rect[0] + rect[2] // 2] = GC_FGD

    bgGMM, fgGMM = initalize_GMMs(img, mask)

    num_iters = 1000
    for i in range(num_iters):
        # Update GMM
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)

        mask = update_mask(mincut_sets, mask)

        if check_convergence(energy):
            break

    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM


def initalize_GMMs(img, mask):
    # TODO: implement initalize_GMMs
    bgGMM = GaussianMixture(n_components=5, random_state=0)
    fgGMM = GaussianMixture(n_components=5, random_state=0)
    bgGMM.fit(img.reshape(-1, 3))
    fgGMM.fit(mask.reshape(-1, 3))

    return bgGMM, fgGMM


# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):
    # TODO: implement GMM component assignment step
    bgGMM.fit(img.reshape(-1, 3))
    fgGMM.fit(mask.reshape(-1, 3))
    return bgGMM, fgGMM


def calculate_mincut(img, mask, bgGMM, fgGMM):
    t1 = time.time()
    # TODO: implement energy (cost) calculation step and mincut
    height, width = img.shape[:2]
    g = ig.Graph(directed=False)
    g.add_vertices(height * width + 2)
    # Last 2 vertices are the source and the sink.
    source_index = height * width  # bg
    sink_index = source_index + 1  # fg
    beta = 1 / (2 * np.mean((img[:-1, :-1] - img[1:, 1:]) ** 2))
    gamma = 50
    bg_img_prob = bgGMM.predict_proba(img.reshape(-1, 3))
    fg_img_prob = fgGMM.predict_proba(img.reshape(-1, 3))
    src_weights = bg_img_prob.max(axis=1)
    sink_weights = fg_img_prob.max(axis=1)
    src_edges = [(i, source_index) for i in range(height * width)]
    sink_edges = [(i, sink_index) for i in range(height * width)]

    tween_edges = []
    tween_weights = []
    for row_index in range(0, height):
        for col_index in range(0, width):
            nei_list = neighborhood(row_index, col_index, width, height)
            pixel_index = col_index + width * row_index
            for nei in nei_list:
                w = gamma * np.exp(-1 * beta * np.linalg.norm(img[row_index, col_index] - img[nei[0], nei[1]]) ** 2) * (
                        (
                                (row_index - nei[0]) ** 2 + (col_index - nei[1]) ** 2) ** -0.5)
                tween_edges.append((pixel_index, nei[0] * width + nei[1]))
                tween_weights.append(w)

    g.add_edges(src_edges)
    g.add_edges(sink_edges)
    g.add_edges(tween_edges)
    g.es['weight'] = np.concatenate([src_weights, sink_weights, tween_weights])
    min_cut = g.mincut(source=source_index, target=sink_index).partition  # [[], []]
    energy = g.mincut_value(source=source_index, target=sink_index)
    print(time.time() - t1)  # time = 13.0004
    return min_cut, energy


def neighborhood(row_index, col_index, max_width, max_height) -> np.ndarray:
    neighborhood_list = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i != 0 and j != 0):
                if 0 <= row_index + i < max_height and 0 <= col_index + j < max_width:
                    neighborhood_list.append((row_index + i, col_index + j))
    return np.array(neighborhood_list)


def update_mask(mincut_sets, mask):
    # TODO findout how the cutsets looks like
    height, width = mask.shape[:2]
    for row in height:
        for col in width:
            pass
            mask[row][col] = 0 if mincut_sets[0] else 3
    return mask


def check_convergence(energy):
    # TODO: implement convergence check
    convergence = False
    return convergence


def cal_metric(predicted_mask, gt_mask):
    # TODO: implement metric calculation

    return 100, 100


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='banana1', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()


if __name__ == '__main__':
    # Load an example image and define a bounding box around the object of interest
    args = parse()

    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int, args.rect.split(',')))

    img = cv2.imread(input_path)

    # Run the GrabCut algorithm on the image and bounding box
    mask, bgGMM, fgGMM = grabcut(img, rect)
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = cal_metric(mask, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])
    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCut Mask', 255 * mask)
    cv2.imshow('GrabCut Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
