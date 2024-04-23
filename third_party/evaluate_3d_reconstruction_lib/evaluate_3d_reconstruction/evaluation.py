# This script is modified from the original source by Erik Sandstroem

# ----------------------------------------------------------------------------
# -                   TanksAndTemples Website Toolbox                        -
# -                    http://www.tanksandtemples.org                        -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2017
# Arno Knapitsch <arno.knapitsch@gmail.com >
# Jaesik Park <syncle@gmail.com>
# Qian-Yi Zhou <Qianyi.Zhou@gmail.com>
# Vladlen Koltun <vkoltun@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ----------------------------------------------------------------------------

import copy
import json
import shutil

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib import rc

matplotlib.use('Agg')

rc('font', **{'family': 'serif', 'sans-serif': ['Times New Roman']})
rc('text', usetex=True if shutil.which('latex') else
   False)  # if Latex is installed and executable on PATH


def write_color_distances_mesh(path, mesh, distances, max_distance):
    cmap = plt.get_cmap('hsv')
    distances = np.array(distances)

    max_dist = max_distance
    c = distances / max_dist
    c[c > 0.85] = 0.85
    c += 0.33
    c[c > 1] = c[c > 1] - 1

    colors = cmap(c)[:, :3]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_triangle_mesh(path, mesh)


def EvaluateHisto(
    geometric_dict,
    threshold,
    filename_mvs,
    plot_stretch,
    scene_name,
):
    print('[Evaluate]')
    s = copy.deepcopy(geometric_dict['pcd'])

    t = copy.deepcopy(geometric_dict['gt_pcd'])
    print('[compute distance from source to target]')
    distance1 = s.compute_point_cloud_distance(
        t)  # distance from source to target
    print('[compute distance from target to source]')
    distance2 = t.compute_point_cloud_distance(
        s)  # distance from target to source

    # plot histograms of the distances
    cm = plt.get_cmap('hsv')
    _, bins, patches = plt.hist(distance1, bins=1000)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # scale values to interval [0,1]
    max_col = 3 * threshold
    col = bin_centers - min(bin_centers)

    col /= max_col
    for c, p in zip(col, patches):
        if c > 0.85:
            c = 0.85
        c += 0.33
        if c > 1:
            c -= 1
        plt.setp(p, 'facecolor', cm(c))

    plt.ylabel('$\#$ of points', fontsize=18)
    plt.xlabel('Meters', fontsize=18)
    plt.title('Precision Histogram', fontsize=18)
    plt.grid(True)
    plt.savefig(filename_mvs + '/' + 'histogram_rec_to_gt')
    plt.clf()
    _, bins, patches = plt.hist(distance2, bins=1000)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max_col
    for c, p in zip(col, patches):
        if c > 0.85:
            c = 0.85
        c += 0.33
        if c > 1:
            c -= 1
        plt.setp(p, 'facecolor', cm(c))

    plt.ylabel('$\#$ of points', fontsize=18)
    plt.xlabel('Meters', fontsize=18)
    plt.title('Recall Histogram', fontsize=18)
    plt.grid(True)
    plt.savefig(filename_mvs + '/' + 'histogram_gt_to_rec')

    source_n_fn = filename_mvs + '/' + scene_name + '.precision.ply'
    target_n_fn = filename_mvs + '/' + scene_name + '.recall.ply'

    print('[Add color coding to predicted mesh to visualize error]')
    if 'pcd_color' in geometric_dict:
        distance1 = geometric_dict['pcd_color'].compute_point_cloud_distance(
            t)  # distance from source to target
    write_color_distances_mesh(source_n_fn, geometric_dict['mesh'], distance1,
                               3 * threshold)
    print('[Written to ', source_n_fn, ']')

    print('[Add color coding to target mesh to visualize error]')
    if 'gt_pcd_color' in geometric_dict:
        distance2 = geometric_dict[
            'gt_pcd_color'].compute_point_cloud_distance(
                s)  # distance from target to source
    write_color_distances_mesh(target_n_fn, geometric_dict['gt_mesh'],
                               distance2, 3 * threshold)
    print('[Written to ', target_n_fn, ']')

    # get histogram and f-score
    [
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
    ] = get_f1_score_histo(threshold, filename_mvs, plot_stretch, distance1,
                           distance2)
    np.savetxt(filename_mvs + '/' + scene_name + '.recall.txt', cum_target)
    np.savetxt(filename_mvs + '/' + scene_name + '.precision.txt', cum_source)
    np.savetxt(
        filename_mvs + '/' + scene_name + '.prf_tau_plotstr.txt',
        np.array([precision, recall, fscore, threshold, plot_stretch]),
    )
    # calculate mean, median, min, max, std of distance1 and distance2
    min1 = np.amin(distance1)
    min2 = np.amin(distance2)
    max1 = np.amax(distance1)
    max2 = np.amax(distance2)
    mean1 = np.mean(distance1)
    mean2 = np.mean(distance2)
    median1 = np.median(distance1)
    median2 = np.median(distance2)
    std1 = np.std(distance1)
    std2 = np.std(distance2)
    np.savetxt(
        filename_mvs + '/' + scene_name +
        '.min12_max12_mean12_median12_std12.txt',
        np.array([
            min1, min2, max1, max2, mean1, mean2, median1, median2, std1, std2
        ]),
    )

    return [
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
        min1,
        min2,
        max1,
        max2,
        mean1,
        mean2,
        median1,
        median2,
        std1,
        std2,
    ]


def get_f1_score_histo(threshold, filename_mvs, plot_stretch, distance1,
                       distance2):
    print('[get_f1_score_histo]')
    dist_threshold = threshold
    if len(distance1) and len(distance2):

        recall = float(sum(d < threshold
                           for d in distance2)) / float(len(distance2))
        precision = float(sum(d < threshold
                              for d in distance1)) / float(len(distance1))
        fscore = 2 * recall * precision / (recall + precision)
        num = len(distance1)
        bins = np.arange(0, dist_threshold * plot_stretch,
                         dist_threshold / 100)
        hist, edges_source = np.histogram(distance1, bins)
        cum_source = np.cumsum(hist).astype(float) / num

        num = len(distance2)
        bins = np.arange(0, dist_threshold * plot_stretch,
                         dist_threshold / 100)
        hist, edges_target = np.histogram(distance2, bins)
        cum_target = np.cumsum(hist).astype(float) / num

    else:
        precision = 0
        recall = 0
        fscore = 0
        edges_source = np.array([0])
        cum_source = np.array([0])
        edges_target = np.array([0])
        cum_target = np.array([0])

    return [
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
    ]
