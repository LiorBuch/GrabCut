import time

import numpy as np
import cv2
import igraph as ig
import argparse

import scipy
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

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


class Gaussian:
    def __init__(self, comp_num: int, pixels):
        self.comp_num = comp_num
        kmeans_clusters = KMeans(n_clusters=self.comp_num, n_init=1).fit(pixels)
        self.means = kmeans_clusters.cluster_centers_
        self.weights = np.array([np.sum(kmeans_clusters.labels_ == i) / len(pixels) for i in range(comp_num)])
        self.covariance = np.array([np.cov(pixels[kmeans_clusters.labels_ == i].T) for i in range(comp_num)])
        for i in range(comp_num):
            if np.linalg.det(self.covariance[i]) == 0:
                reg_cov = 1e-6 * np.eye(len(self.covariance[i]))
                self.covariance[i] += reg_cov
        self.covariance_inverse = np.array([np.linalg.inv(self.covariance[i]) for i in range(comp_num)])
        self.covariance_det = np.array([np.linalg.det(self.covariance[i]) for i in range(comp_num)])

    def calc_N(self, xi, mu, covariance_inverse, covariance_determinant):
        xi_minus_mu = np.array(xi) - np.array(mu)
        xi_dimension = len(xi)
        numerator = np.e ** (-0.5 * np.dot(np.dot(xi_minus_mu.T, covariance_inverse), xi_minus_mu))
        denominator = ((2 * np.pi) ** (xi_dimension / 2)) * (covariance_determinant ** 0.5)
        return numerator / denominator

    # ric
    def evaluate_responsibility_for_pixel(self, pixel: list):
        probability = np.zeros(self.comp_num)
        for cluster_index in range(self.comp_num):
            covariance_inverse = self.covariance_inverse[cluster_index]
            covariance_determinant = self.covariance_det[cluster_index]
            N = self.calc_N(pixel, self.means[cluster_index], covariance_inverse, covariance_determinant)
            weight = self.weights[cluster_index]
            numerator = weight * N
            probability[cluster_index] = numerator
        # Normalize
        return np.array(probability) / sum(probability)

    def re_estimate_gmms_parameters(self, responsibility, pixels, cluster_index):
        sum_ric = np.sum(responsibility[:, cluster_index])
        mu = np.sum(pixels * responsibility[:, cluster_index].reshape(-1, 1), axis=0) / sum_ric

        weight = sum_ric / len(pixels)

        diff = pixels - mu
        sigma = np.dot((responsibility[:, cluster_index].reshape(-1, 1) * diff).T, diff) / sum_ric

        return weight, mu, sigma

    def fit(self, pixels):
        responsibilities = np.zeros((pixels.shape[0],5))
        for pixel_index in range(pixels.shape[0]):
                responsibilities[pixel_index] = self.evaluate_responsibility_for_pixel(pixels[pixel_index])

        responsibilities = np.array(responsibilities)
        self.weights = np.zeros(self.comp_num)
        self.means = np.zeros((self.comp_num, 3))
        self.covariance = np.zeros((self.comp_num, 3, 3))
        for cluster_index in range(self.comp_num):
            weight, mu, sigma = self.re_estimate_gmms_parameters(responsibilities, pixels, cluster_index)
            self.weights[cluster_index] = weight
            self.means[cluster_index] = mu
            self.covariance[cluster_index] = sigma

        self.weights = np.array(self.weights)
        self.means = np.array(self.means)
        self.covariance = np.array(self.covariance)
        for i in range(self.comp_num):
            if np.linalg.det(self.covariance[i]) == 0:
                reg_cov = 1e-6 * np.eye(len(self.covariance[i]))
                self.covariance[i] += reg_cov
        self.covariance_inverse = np.array([np.linalg.inv(self.covariance[i]) for i in range(self.comp_num)])
        self.covariance_det = np.array([np.linalg.det(self.covariance[i]) for i in range(self.comp_num)])


    def score_samples(self, pixels):
        likelihoods = []
        for row in pixels:
            current_row = []
            for pixel in row:
                pixel_probability = 0
                for cluster_index in range(self.comp_num):
                    if self.covariance_det[cluster_index] <= 0:
                        continue  # Skip this component if covariance is singular

                    x_minus_u = pixel - self.means[cluster_index]

                    # Calculate exponent part more stably
                    exponent = -0.5 * np.dot(np.dot(x_minus_u, self.covariance_inverse[cluster_index]), x_minus_u)

                    # Calculate probability density for the current component
                    prob_density = (self.weights[cluster_index] / ((2 * np.pi) ** (len(pixel) / 2) * np.sqrt(
                        self.covariance_det[cluster_index]))) * np.exp(exponent)

                    pixel_probability += prob_density

                # Avoid log(0) by ensuring pixel_probability is positive
                if pixel_probability <= 0:
                    pixel_probability = 1e-10

                current_row.append(-np.log(pixel_probability))
            likelihoods.append(current_row)
        return np.array(likelihoods)


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

    bgGMM, fgGMM = initalize_GMMs(img, mask, n_iter)

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


def initalize_GMMs(img, mask, n_components):
    bgd_mask = (mask == GC_BGD) | (mask == GC_PR_BGD)
    background_pixels = img[bgd_mask]
    fg_mask = (mask == GC_FGD) | (mask == GC_PR_FGD)
    foreground_pixels = img[fg_mask]

    # Create the Gaussian mixture objects
    # bgGMM = GaussianMixture(n_components=n_components, random_state=0, warm_start=True)
    # fgGMM = GaussianMixture(n_components=n_components, random_state=0, warm_start=True)

    # bgGMM.fit(background_pixels)
    # fgGMM.fit(foreground_pixels)

    # return bgGMM, fgGMM
    return Gaussian(5, background_pixels), Gaussian(5, foreground_pixels)


# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):
    bgd_mask = (mask == GC_BGD) | (mask == GC_PR_BGD)
    background_pixels = img[bgd_mask]
    fg_mask = (mask == GC_FGD) | (mask == GC_PR_FGD)
    foreground_pixels = img[fg_mask]

    bgGMM.fit(background_pixels)
    fgGMM.fit(foreground_pixels)

    return bgGMM, fgGMM


def calculate_mincut(img, mask, bgGMM: Gaussian, fgGMM: Gaussian):
    global BETA, BETWEEN_EDGES, BETWEEN_WEIGHTS, K, SRC_EDGES, SINK_EDGES

    t1 = time.time()
    height, width = img.shape[:2]
    g = ig.Graph(directed=False)
    num_of_pixels = height * width

    g.add_vertices(num_of_pixels + 2)
    # Last 2 vertices are the source and the sink.
    bg_index = height * width  # bg
    fg_index = bg_index + 1  # fg

    if BETA == -1:
        SRC_EDGES = [(bg_index, i) for i in range(num_of_pixels)]
        SINK_EDGES = [(i, fg_index) for i in range(num_of_pixels)]
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

    fg_img_prob = fgGMM.score_samples(img)#.reshape((-1, img.shape[-1]))).reshape(img.shape[:-1])
    bg_img_prob = bgGMM.score_samples(img)#.reshape((-1, img.shape[-1]))).reshape(img.shape[:-1])

    bg_weights = []  # bg
    fg_weights = []  # fg

    for row_index in range(0, height):
        for col_index in range(0, width):
            if mask[row_index][col_index] == GC_BGD:
                bg_weights.append(K)  # if we know its bg, the weight should be K
                fg_weights.append(0)
            elif mask[row_index][col_index] == GC_FGD:
                bg_weights.append(0)
                fg_weights.append(K)  # if we know its fg, the weight should be K
            else:  # If the mask is 2 or 3, use the prob
                # TODO: Check if should be bg-fg instead
                bg_weights.append(fg_img_prob[row_index][col_index])
                fg_weights.append(bg_img_prob[row_index][col_index])

    g.add_edges(SRC_EDGES)
    g.add_edges(SINK_EDGES)
    g.add_edges(BETWEEN_EDGES)
    g.es['weight'] = np.concatenate([bg_weights, fg_weights, BETWEEN_WEIGHTS])
    min_cut = g.mincut(source=bg_index, target=fg_index, capacity='weight')
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
    result = (np.abs(energy - PREV_ENERGY) / PREV_ENERGY) <= 0.01 or LOOP_TRACK == 20  # TODO: Update this value
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
