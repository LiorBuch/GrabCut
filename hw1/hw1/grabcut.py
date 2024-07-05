import time

import numpy as np
import cv2
import igraph as ig
import argparse

from sklearn.cluster import KMeans

from mix import GaussianMixture

GC_BGD = 0  # Hard bg pixel
GC_FGD = 1  # Hard fg pixel, will not be used
GC_PR_BGD = 2  # Soft bg pixel
GC_PR_FGD = 3  # Soft fg pixel

LOOP_TRACK = 0  # user metric
PREV_ENERGY = -1
BETA = -1
K = -1
BETWEEN_EDGES = []
BETWEEN_WEIGHTS = []
NEIGHBORHOOD_LIST = []

SRC_EDGES = []
SINK_EDGES = []


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
            height, width = mask.shape
            flat_mask = mask.flatten()
            flat_mask[(flat_mask == GC_PR_FGD)] = 1
            flat_mask[(flat_mask == GC_PR_BGD)] = 0

            mask = flat_mask.reshape(height, width)
            break

    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM


def initalize_GMMs(img, mask):
    bgd_mask = (mask == GC_BGD) | (mask == GC_PR_BGD)
    background_pixels = img[bgd_mask]
    fg_mask = (mask == GC_FGD) | (mask == GC_PR_FGD)
    foreground_pixels = img[fg_mask]
    bgGMM = GaussianMixture(n_components=5, X=background_pixels.reshape((-1, img.shape[-1])))
    fgGMM = GaussianMixture(n_components=5, X=foreground_pixels.reshape((-1, img.shape[-1])))
    return bgGMM, fgGMM


# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):
    bgd_mask = (mask == GC_BGD) | (mask == GC_PR_BGD)
    background_pixels = img[bgd_mask]
    fg_mask = (mask == GC_FGD) | (mask == GC_PR_FGD)
    foreground_pixels = img[fg_mask]

    bg_label = KMeans(n_clusters=bgGMM.n_components, n_init=1).fit(background_pixels.reshape((-1, img.shape[-1]))).labels_
    fg_label = KMeans(n_clusters=fgGMM.n_components, n_init=1).fit(foreground_pixels.reshape((-1, img.shape[-1]))).labels_

    bgGMM.fit(background_pixels.reshape((-1, img.shape[-1])),labels=bg_label)
    fgGMM.fit(foreground_pixels.reshape((-1, img.shape[-1])),labels=fg_label)
    return bgGMM, fgGMM


def calculate_mincut(img, mask, bgGMM: GaussianMixture, fgGMM: GaussianMixture):
    global BETA, BETWEEN_EDGES, BETWEEN_WEIGHTS, K, SRC_EDGES, SINK_EDGES

    t1 = time.time()
    height, width = img.shape[:2]
    g = ig.Graph(directed=False)
    num_of_pixels = height * width

    g.add_vertices(num_of_pixels + 2)
    # Last 2 vertices are the source and the sink.
    source_index = height * width  # bg
    sink_index = source_index + 1  # fg

    if BETA == -1:
        SRC_EDGES = [(source_index, i) for i in range(num_of_pixels)]
        SINK_EDGES = [(i, sink_index) for i in range(num_of_pixels)]
        print("calc beta")
        BETA = 0
        for row in range(height):
            for col in range(width):
                nei_list = neighborhood(row, col, width, height)
                sum_dist = 0
                for nei in nei_list:
                    sum_dist += np.linalg.norm(img[row, col] - img[nei[0], nei[1]])
                BETA += sum_dist / len(nei_list)
        BETA = BETA / num_of_pixels
        print(f"beta value -> {BETA}")
    if K == -1:
        for row_index in range(0, height):
            for col_index in range(0, width):
                nei_list = neighborhood(row_index, col_index, width, height)
                pixel_index = col_index + width * row_index
                sum_n = 0
                for nei in nei_list:
                    N = (50 * np.exp(-1 * BETA * np.linalg.norm(img[row_index, col_index] - img[nei[0], nei[1]]) ** 2) *
                         (((row_index - nei[0]) ** 2 + (col_index - nei[1]) ** 2) ** -0.5))
                    sum_n += N
                    BETWEEN_EDGES.append((pixel_index, nei[0] * width + nei[1]))
                    BETWEEN_WEIGHTS.append(N)
                K = max(K, sum_n)
        print(f"K={K}")

    # Reshape image and calculate probabilities
    img_reshaped = img.reshape((-1, img.shape[-1]))
    fg_img_prob = fgGMM.calc_prob(img_reshaped).reshape(img.shape[:-1])
    bg_img_prob = bgGMM.calc_prob(img_reshaped).reshape(img.shape[:-1])

    ts = time.time()
    # Initialize weights with zeros
    src_weights = np.zeros((height, width))
    sink_weights = np.zeros((height, width))

    # Use boolean indexing to set weights based on the mask
    src_weights[mask == GC_BGD] = K
    sink_weights[mask == GC_FGD] = K

    # For the rest of the pixels, use the probabilities
    mask_other = (mask != GC_BGD) & (mask != GC_FGD)
    src_weights[mask_other] = bg_img_prob[mask_other]
    sink_weights[mask_other] = fg_img_prob[mask_other]

    # Flatten the weights to match the original list structure
    src_weights = src_weights.flatten().tolist()
    sink_weights = sink_weights.flatten().tolist()
    g.add_edges(SRC_EDGES)
    g.add_edges(SINK_EDGES)
    g.add_edges(BETWEEN_EDGES)
    g.es['weight'] = np.concatenate([src_weights, sink_weights, BETWEEN_WEIGHTS])
    min_cut = g.mincut(source=source_index, target=sink_index, capacity='weight')
    min_cut_part = min_cut.partition  # [[], []]
    energy = min_cut.value
    print(f"min cut runtime -> {time.time() - t1}")
    print(f"energy value -> {energy}")
    return min_cut_part, energy


def neighborhood(row_index, col_index, max_width, max_height):
    if len(NEIGHBORHOOD_LIST) == 0:
        for row_index in range(max_height):
            for col_index in range(max_width):
                nei_list = []
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if (i != 0 or j != 0):
                            if 0 <= row_index + i < max_height and 0 <= col_index + j < max_width:
                                nei_list.append((row_index + i, col_index + j))
                NEIGHBORHOOD_LIST.append(nei_list)

    return NEIGHBORHOOD_LIST[row_index * max_width + col_index]


def update_mask(mincut_sets, mask):
    height, width = mask.shape
    flat_mask = mask.flatten()
    # 0 stays 0, 1 stays 1, 3 can be 3 or 2, 2 can be 2 or 3
    # cut[0] source BG , cut[1] sink FG.

    # Create a boolean mask for the indexes in mincut_sets[0]
    source_mask = np.isin(np.arange(flat_mask.size), mincut_sets[0])

    # Update the mask based on the conditions
    flat_mask[(source_mask) & (flat_mask == GC_PR_FGD)] = GC_PR_BGD
    flat_mask[(~source_mask) & (flat_mask == GC_PR_BGD)] = GC_PR_FGD

    # Reshape the flattened mask back to the original shape
    updated_mask = flat_mask.reshape(height, width)
    return updated_mask


def check_convergence(energy):
    # TODO: implement convergence check
    global LOOP_TRACK, PREV_ENERGY
    LOOP_TRACK = LOOP_TRACK + 1
    print(f"loop track -> {LOOP_TRACK}")
    if -1 == PREV_ENERGY:
        PREV_ENERGY = energy
        print("")
        return False
    print(f"energy conver -> {np.abs(energy - PREV_ENERGY) / PREV_ENERGY} \n")
    result = (np.abs(energy - PREV_ENERGY) / PREV_ENERGY) <= 0.01 or LOOP_TRACK == 10  # TODO: Update this value
    PREV_ENERGY = energy
    return result


def cal_metric(predicted_mask, gt_mask):
    number_of_correct_pixels = np.sum(predicted_mask == gt_mask)
    matrix_size = img.shape[0] * img.shape[1]
    accuracy = number_of_correct_pixels / matrix_size

    intersection = np.sum((predicted_mask == 1) & (gt_mask == 1))
    union = np.sum((predicted_mask == 1) | (gt_mask == 1))
    jaccard = intersection / union
    return accuracy, jaccard


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='teddy', help='name of image from the course files')
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
