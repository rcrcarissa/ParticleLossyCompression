#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_set>
#include <limits>
#include <ctime>
#include <omp.h>
#include <zstd.h>
#include "findNearestPoint.cpp"
#include "huffman.cpp"

#pragma omp parallel for

void printUsage() {
    std::cout << "Usage: <cpp_filename> -ds <dataset_name> -dim <dimension> -f" << std::endl;
    std::cout << "  -ds  Specify the dataset name" << std::endl;
    std::cout << "  -dim   Specify the dimension" << std::endl;
    std::cout << "  -f   Specify the precision. -f for float and -d for double" << std::endl;
    std::cout << "  -reb   Specify the relative error bound." << std::endl;
    std::cout << "  -r   Specify the max number of particles allowed in a leaf node." << std::endl;
}

std::tuple<char *, size_t, bool, double, double> Parsing(int argc, char *argv[]) {
    std::tuple<char *, size_t, bool, double, double> error_return_value = std::make_tuple(nullptr, 0, false, 0, 0);
    if (argc < 10) {
        printUsage();
        return error_return_value;
    }

    char *dataset;
    size_t dim;
    bool isSinglePrecision;
    double reb;
    double r;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-ds") {
            if (i + 1 < argc) {
                dataset = argv[++i];
            } else {
                std::cerr << "Error: -ds requires a value." << std::endl;
                printUsage();
                return error_return_value;
            }
        } else if (arg == "-dim") {
            if (i + 1 < argc) {
                dim = std::atoi(argv[++i]);
            } else {
                std::cerr << "Error: -dim requires a value." << std::endl;
                printUsage();
                return error_return_value;
            }
        } else if (arg == "-reb") {
            if (i + 1 < argc) {
                reb = std::atof(argv[++i]);
            } else {
                std::cerr << "Error: -reb requires a value." << std::endl;
                printUsage();
                return error_return_value;
            }
        } else if (arg == "-r") {
            if (i + 1 < argc) {
                r = std::atof(argv[++i]);
            } else {
                std::cerr << "Error: -r requires a value." << std::endl;
                printUsage();
                return error_return_value;
            }
        } else if (arg == "-f") {
            isSinglePrecision = true;
        } else if (arg == "-d") {
            isSinglePrecision = false;
        }
    }
    return std::make_tuple(dataset, dim, isSinglePrecision, reb, r);
}

typedef struct KDNode {
    int depth;
    int start_idx;
    int end_idx;
    int center_idx;
    std::vector<double> attrs;  // box_center, box_half_len
    KDNode *parent;
    KDNode *left;
    KDNode *right;

    KDNode(int depth, int start_id, int end_id, KDNode *parent) : depth(depth), start_idx(start_id), end_idx(end_id),
                                                                  center_idx(-1), parent(parent), left(nullptr),
                                                                  right(nullptr) {}
} KDNode;

typedef struct KDTree {
    std::vector<std::vector<double>> pts;
    size_t dim;
    int r;  // maximum number of particles in a KDNode
    KDNode root;

    KDTree(const std::vector<std::vector<double>> &points, int r) : pts(points), dim(pts[0].size()), r(r),
                                                                    root(0, 0, points.size(), nullptr) {
        build_tree(&root);
    }

    void build_tree(KDNode *node) {
        size_t num_pts = node->end_idx - node->start_idx;
        if (num_pts > r) {
            int median_index = num_pts / 2;
            int split_dim = node->depth % dim;
            std::nth_element(pts.begin() + node->start_idx,
                             pts.begin() + node->start_idx + median_index,
                             pts.begin() + node->start_idx + num_pts,
                             [split_dim](const std::vector<double> &a, const std::vector<double> &b) {
                                 return a[split_dim] < b[split_dim];
                             });
            KDNode *left_child = new KDNode(node->depth + 1, node->start_idx, node->start_idx + median_index, node);
            KDNode *right_child = new KDNode(node->depth + 1, node->start_idx + median_index, node->end_idx, node);
            node->left = left_child;
            node->right = right_child;
            build_tree(left_child);
            build_tree(right_child);
        }
    }

    void get_leaf_nodes(KDNode *root_node, std::vector<KDNode *> &leaves_ptr) {
        if (root_node == nullptr) return;
        if (root_node->left == nullptr && root_node->right == nullptr) leaves_ptr.push_back(root_node);
        get_leaf_nodes(root_node->left, leaves_ptr);
        get_leaf_nodes(root_node->right, leaves_ptr);
    }

    void initialize(KDNode *node, std::vector<double> (*func)(std::vector<double>, std::vector<double>)) {
        if (node->left->left != nullptr) {
            initialize(node->left, func);
            initialize(node->right, func);
        }
        node->attrs = func(node->left->attrs, node->right->attrs);
    }

    void bottom_up_update(KDNode *node, std::vector<double> (*func)(std::vector<double>, std::vector<double>)) {
        KDNode *curr_node = node;
        while (curr_node->parent != nullptr) {
            curr_node = curr_node->parent;
            curr_node->attrs = func(curr_node->left->attrs, curr_node->right->attrs);
        }
    }

    std::vector<KDNode *>
    top_down_search(KDNode *node, bool (*func)(std::vector<double>, std::vector<double>)) {
        std::vector<KDNode *> target_nodes;
        std::vector<KDNode *> nodes_to_visit;
        nodes_to_visit.push_back(&root);
        while (nodes_to_visit.size() > 0) {
            KDNode *curr_node = nodes_to_visit.back();
            nodes_to_visit.pop_back();
            if (curr_node != node) {
                if (func(curr_node->attrs, node->attrs)) {
                    if (curr_node->left == nullptr) target_nodes.push_back(curr_node);
                    else if (curr_node->depth < node->depth) {
                        nodes_to_visit.push_back(curr_node->left);
                        nodes_to_visit.push_back(curr_node->right);
                    }
                }
            }
        }
        return target_nodes;
    };


} KDTree;

std::vector<std::vector<double>> readBinaryFile(const std::string &fileName, size_t dim) {
    std::ifstream inputFile(fileName, std::ios::binary | std::ios::ate);
    if (!inputFile) {
        throw std::runtime_error("Error opening file");
    }

    std::streamsize fileSize = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);
    if (fileSize % sizeof(double) != 0) {
        throw std::runtime_error("File size is not a multiple of double size");
    }
    std::size_t numDoubles = fileSize / sizeof(double);

    std::vector<double> buffer(numDoubles);

    if (!inputFile.read(reinterpret_cast<char *>(buffer.data()), fileSize)) {
        throw std::runtime_error("Error reading file");
    }

    inputFile.close();

    size_t n_pts = buffer.size() / dim;
    std::vector<std::vector<double> > vec(n_pts, std::vector<double>(dim));
    for (int i = 0; i < n_pts; i++)
        for (int j = 0; j < dim; j++) {
            vec[i][j] = buffer[i * dim + j];
        }

    return vec;
}

void exportBinaryFile(const std::vector<std::vector<double>> &data, const std::string &filename) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile.is_open()) {
        std::cerr << "Error opening file for writing!" << std::endl;
        return;
    }
    for (const auto &row: data) {
        outfile.write(reinterpret_cast<const char *>(row.data()), row.size() * sizeof(double));
    }
    outfile.close();
}

int half_length2bits(double half_len, double xi) {
    return (int) std::ceil(std::log(half_len / (2 * xi) + 0.5) / std::log(2) + 1);
}

double bits2half_length(int m, double xi) {
    return (std::pow(2, m - 1) - 0.5) * 2 * xi;
}


std::tuple<std::vector<double>, std::vector<double>, int, std::vector<int>>
get_bit_box(std::vector<std::vector<double>> &pts, double xi) {
    size_t dim = pts[0].size();
    std::vector<double> pos_max;
    std::vector<double> pos_min;
    std::vector<double> center;
    for (int d = 0; d < dim; d++) {
        auto maxInCol = *std::max_element(pts.begin(), pts.end(),
                                          [d](const std::vector<double> &a, const std::vector<double> &b) {
                                              return a[d] < b[d];
                                          });
        auto minInCol = *std::min_element(pts.begin(), pts.end(),
                                          [d](const std::vector<double> &a, const std::vector<double> &b) {
                                              return a[d] < b[d];
                                          });
        pos_max.push_back(maxInCol[d]);
        pos_min.push_back(minInCol[d]);
        center.push_back((maxInCol[d] + minInCol[d]) / 2);
    }
    int nearest_pt_id = findNearestPointIndex(pts, center);
    center = pts[nearest_pt_id];
    std::vector<int> num_bits;
    std::vector<double> half_length;
    for (int d = 0; d < dim; d++) {
        num_bits.push_back(half_length2bits(std::max(center[d] - pos_min[d], pos_max[d] - center[d]), xi));
        half_length.push_back(bits2half_length(num_bits[d], xi));
    }
    return std::make_tuple(center, half_length, nearest_pt_id, num_bits);
}

std::vector<double>
get_union_of_two_bounding_boxes(const std::vector<double> &box0_attrs, const std::vector<double> &box1_attrs) {
    std::vector<double> empty_box;
    if (box0_attrs.size() == 0) {
        if (box1_attrs.size() == 0) return empty_box;
        else return box1_attrs;
    } else {
        if (box1_attrs.size() == 0) return box0_attrs;
        else {
            size_t dim = box0_attrs.size() / 2;
            std::vector<double> center0(box0_attrs.begin(), box0_attrs.begin() + dim);
            std::vector<double> half_length0(box0_attrs.begin() + dim, box0_attrs.end());
            std::vector<double> center1(box1_attrs.begin(), box1_attrs.begin() + dim);
            std::vector<double> half_length1(box1_attrs.begin() + dim, box1_attrs.end());

            std::vector<double> min_corner(dim);
            std::vector<double> max_corner(dim);
            for (int d = 0; d < dim; d++) {
                min_corner[d] = std::min(center0[d] - half_length0[d], center1[d] - half_length1[d]);
                max_corner[d] = std::max(center0[d] + half_length0[d], center1[d] + half_length1[d]);
            }
            std::vector<double> box_attrs(dim * 2);
            for (int d = 0; d < dim; d++) {
                box_attrs[d] = (min_corner[d] + max_corner[d]) / 2;
            }
            for (int d = 0; d < dim; d++) {
                box_attrs[d + dim] = (max_corner[d] - min_corner[d]) / 2;
            }
            return box_attrs;
        }
    }
}

std::vector<std::vector<double>> get_box_range(const std::vector<double> &box_attrs) {
    size_t dim = box_attrs.size() / 2;
    std::vector<std::vector<double>> box_range;
    for (int d = 0; d < dim; d++) {
        std::vector<double> range(2);
        range[0] = box_attrs[d] - box_attrs[d + dim];
        range[1] = box_attrs[d] + box_attrs[d + dim];
        box_range.push_back(range);
    }
    return box_range;
}

bool is_overlapping1D(const std::vector<double> &interval0, const std::vector<double> &interval1) {
    return interval0[1] >= interval1[0] && interval1[1] >= interval0[0];
}

bool is_overlappingND(const std::vector<std::vector<double>> &box0_range,
                      const std::vector<std::vector<double>> &box1_range) {
    size_t dim = box0_range.size();
    for (int d = 0; d < dim; d++) {
        if (!is_overlapping1D(box0_range[d], box1_range[d]))
            return false;
    }
    return true;
}

bool is_two_boxes_overlap(const std::vector<double> &box0_attrs, const std::vector<double> &box1_attrs) {
    if (box0_attrs.size() == 0 or box1_attrs.size() == 0) return false;
    std::vector<std::vector<double>> box0_range = get_box_range(box0_attrs);
    std::vector<std::vector<double>> box1_range = get_box_range(box1_attrs);
    return is_overlappingND(box0_range, box1_range);
}

bool is_point_in_box(const std::vector<double> &pt, const std::vector<std::vector<double>> &box_range) {
    size_t dim = pt.size();
    for (int d = 0; d < dim; d++) {
        if (!(box_range[d][0] < pt[d] && pt[d] < box_range[d][1]))
            return false;
    }
    return true;
}

void
one_box_compression(const std::vector<std::vector<double>> &pts, const std::vector<int> &pts_in_box_id,
                    const std::vector<int> &num_bits, const int center_id, const double xi,
                    std::vector<int> &quantization_code, std::vector<double> &lossless_data,
                    std::vector<double> &decompressed_data) {
    std::vector<double> center = pts[center_id];
    lossless_data.insert(lossless_data.end(), center.begin(), center.end());
    quantization_code.insert(quantization_code.end(), num_bits.begin(), num_bits.end());
    decompressed_data.insert(decompressed_data.end(), center.begin(), center.end());
    for (int i = 0; i < pts_in_box_id.size(); i++) {
        std::vector<double> pt = pts[pts_in_box_id[i]];
        std::vector<double> code_distance_tmp(pt.size());
        std::transform(pt.begin(), pt.end(), center.begin(), code_distance_tmp.begin(), std::minus<double>());
        std::transform(code_distance_tmp.begin(), code_distance_tmp.end(), code_distance_tmp.begin(),
                       [xi](double element) { return element / xi; });
        std::vector<int> code_distance(code_distance_tmp.size());
        std::transform(code_distance_tmp.begin(), code_distance_tmp.end(), code_distance.begin(),
                       [](double value) { return static_cast<int>(value); });
        for (int d = 0; d < pt.size(); d++) {
            int code = (int) std::pow(2, num_bits[d] - 1);
            if (code_distance[d] < 0)
                code += (int) std::floor(code_distance[d] / 2);
            else
                code += (int) std::ceil(code_distance[d] / 2);
            if (code < 1 or code > std::pow(2, num_bits[d]) - 1)
                std::cout << "Error! Points inside box is unpredictable." << std::endl;
            else {
                decompressed_data.push_back(center[d] + (code - std::pow(2, num_bits[d] - 1)) * 2 * xi);
                quantization_code.push_back(code);
            }
        }
    }
    quantization_code.push_back(std::numeric_limits<double>::infinity());
}

std::vector<char> zstdCompress(const std::string &input) {
    size_t compressedSize = ZSTD_compressBound(input.size());  // Get maximum compressed size
    std::vector<char> compressedData(compressedSize);

    size_t actualCompressedSize = ZSTD_compress(compressedData.data(), compressedSize, input.data(), input.size(),
                                                1);  // Compression level 1

    if (ZSTD_isError(actualCompressedSize)) {
        std::cerr << "ZSTD compression error: " << ZSTD_getErrorName(actualCompressedSize) << std::endl;
        return {};
    }

    compressedData.resize(actualCompressedSize);
    return compressedData;
}

std::string zstdDecompress(const std::vector<char> &compressedData, size_t originalSize) {
    std::vector<char> decompressedData(originalSize);

    size_t decompressedSize = ZSTD_decompress(decompressedData.data(), originalSize, compressedData.data(),
                                              compressedData.size());

    if (ZSTD_isError(decompressedSize)) {
        std::cerr << "ZSTD decompression error: " << ZSTD_getErrorName(decompressedSize) << std::endl;
        return {};
    }

    return std::string(decompressedData.begin(), decompressedData.begin() + decompressedSize);
}


double getMaxColumnRange(const std::vector<std::vector<double>> &matrix) {
    size_t numCols = matrix[0].size();
    double maxRange = 0;

    for (size_t col = 0; col < numCols; ++col) {
        double minVal = std::numeric_limits<double>::max();
        double maxVal = std::numeric_limits<double>::min();

        for (const auto &row: matrix) {
            if (row.size() > col) {
                double value = row[col];
                if (value < minVal) minVal = value;
                if (value > maxVal) maxVal = value;
            }
        }

        double range = maxVal - minVal;
        if (range > maxRange) maxRange = range;
    }

    return maxRange;
}

void box_compression(const std::vector<std::vector<double>> &pts, double xi, int r) {
    KDTree kd_tree(pts, r);
    std::vector<KDNode *> leaves_ptr;
    kd_tree.get_leaf_nodes(&kd_tree.root, leaves_ptr);
    int n_remaining_pts = pts.size();
    std::unordered_set<int> removed_pts_idx;
    std::vector<int> total_n_bits(leaves_ptr.size());
    for (int i = 0; i < leaves_ptr.size(); i++) {
        // build bit boxes
        std::vector<std::vector<double>> leaf_pts(kd_tree.pts.begin() + leaves_ptr[i]->start_idx,
                                                  kd_tree.pts.begin() + leaves_ptr[i]->end_idx);
        auto [center, half_length, center_id, num_bits] = get_bit_box(leaf_pts, xi);
        leaves_ptr[i]->attrs.insert(leaves_ptr[i]->attrs.end(), center.begin(), center.end());
        leaves_ptr[i]->attrs.insert(leaves_ptr[i]->attrs.end(), half_length.begin(), half_length.end());
        leaves_ptr[i]->center_idx = leaves_ptr[i]->start_idx + center_id;
        total_n_bits[i] = std::accumulate(num_bits.begin(), num_bits.end(), 0);
    }
    kd_tree.initialize(&kd_tree.root, reinterpret_cast<std::vector<double> (*)(std::vector<double>,
                                                                               std::vector<double>)>(get_union_of_two_bounding_boxes));
    std::vector<std::vector<int>> nums_bits;
    std::vector<int> quantization_codes;
    std::vector<double> lossless_data;
    std::vector<double> decompressed_data;
    while (n_remaining_pts > 0) {
        int box_idx = std::distance(total_n_bits.begin(), std::min_element(total_n_bits.begin(), total_n_bits.end()));
        KDNode *leaf_to_remove = leaves_ptr[box_idx];
        total_n_bits.erase(total_n_bits.begin() + box_idx);
        leaves_ptr.erase(leaves_ptr.begin() + box_idx);
        std::vector<int> num_bits;
        for (int i = 0; i < kd_tree.dim; i++) {
            num_bits.push_back(half_length2bits(leaf_to_remove->attrs[i + kd_tree.dim], xi));
        }
        nums_bits.push_back(num_bits);
        std::vector<std::vector<double>> box_range = get_box_range(leaf_to_remove->attrs);
        std::vector<KDNode *> intersected_leaves = kd_tree.top_down_search(leaf_to_remove,
                                                                           reinterpret_cast<bool (*)(
                                                                                   std::vector<double>,
                                                                                   std::vector<double>)>(is_two_boxes_overlap));
        // find bit boxes overlapping the bit box to be removed
        // if the bit box to be removed containing points in other bit boxes, remove these points & update bit boxes
        std::vector<int> pts_to_remove_idx;
        for (int i = 0; i < intersected_leaves.size(); i++) {
            int idx = std::distance(leaves_ptr.begin(),
                                    std::find(leaves_ptr.begin(), leaves_ptr.end(), intersected_leaves[i]));
            std::unordered_set<int> overlap_pts_idx;
            for (int j = intersected_leaves[i]->start_idx; j < intersected_leaves[i]->end_idx; j++) {
                if (!removed_pts_idx.count(j)) {
                    std::vector<double> pt = kd_tree.pts[j];
                    if (is_point_in_box(pt, box_range)) {
                        pts_to_remove_idx.push_back(j);
                        overlap_pts_idx.insert(j);
                    }
                }
            }
            if (overlap_pts_idx.size() > 0) {
                std::vector<std::vector<double>> pts_in_box;
                std::vector<int> pts_in_box_idx;
                for (int j = intersected_leaves[i]->start_idx; j < intersected_leaves[i]->end_idx; j++) {
                    if (!overlap_pts_idx.count(j) && !removed_pts_idx.count(j)) {
                        pts_in_box.push_back(kd_tree.pts[j]);
                        pts_in_box_idx.push_back(j);
                    }
                }
                if (pts_in_box.size() > 0) {
                    auto [i_center, i_half_length, i_center_id, i_num_bits] = get_bit_box(pts_in_box, xi);
                    std::copy(i_center.begin(), i_center.end(), intersected_leaves[i]->attrs.begin());
                    std::copy(i_half_length.begin(), i_half_length.end(),
                              intersected_leaves[i]->attrs.begin() + kd_tree.dim);
                    intersected_leaves[i]->center_idx = pts_in_box_idx[i_center_id];
                    total_n_bits[idx] = std::accumulate(i_num_bits.begin(), i_num_bits.end(), 0);
                } else {
                    leaves_ptr.erase(leaves_ptr.begin() + idx);
                    total_n_bits.erase(total_n_bits.begin() + idx);
                    intersected_leaves[i]->attrs.clear();
                }
                kd_tree.bottom_up_update(intersected_leaves[i],
                                         reinterpret_cast<std::vector<double> (*)(std::vector<double>,
                                                                                  std::vector<double>)>(get_union_of_two_bounding_boxes));
            }
        }
        // remove the points in the bit box
        for (int i = leaf_to_remove->start_idx; i < leaf_to_remove->end_idx; i++) {
            if (!removed_pts_idx.count(i) && i != leaf_to_remove->center_idx) {
                pts_to_remove_idx.push_back(i);
            }
        }
        leaf_to_remove->attrs.clear();
        one_box_compression(kd_tree.pts, pts_to_remove_idx, num_bits, leaf_to_remove->center_idx, xi,
                            quantization_codes, lossless_data, decompressed_data);
        n_remaining_pts -= pts_to_remove_idx.size() + 1;
        removed_pts_idx.insert(pts_to_remove_idx.begin(), pts_to_remove_idx.end());
        removed_pts_idx.insert(leaf_to_remove->center_idx);
    }
    std::string huffman = huffmanCompress(quantization_codes);
    std::vector<char> compressedData = zstdCompress(huffman);

    // print results
    int original_size = pts.size() * 3 * sizeof(double);
    int compressed_size = compressedData.size() + lossless_data.size() * sizeof(double);
    std::cout << "Compression ratio = " << original_size / compressed_size << std::endl;
    std::vector<double> original_data;
    for (const auto &row: kd_tree.pts) {
        original_data.insert(original_data.end(), row.begin(), row.end());
    }

    std::vector<double> tmp1(original_data.size());
    std::transform(original_data.begin(), original_data.end(), decompressed_data.begin(), tmp1.begin(),
                   [](double a, double b) { return a - b; });
    std::vector<double> tmp2(original_data.size());
    std::transform(tmp1.begin(), tmp1.end(), tmp2.begin(),
                   [](double val) { return std::pow(val, 2); });
    double mse = std::accumulate(tmp2.begin(), tmp2.end(), 0) / original_data.size();
    double rmse = std::sqrt(mse);
    double range = getMaxColumnRange(kd_tree.pts);
    double nrmse = rmse / range;
    double psnr = 20 * std::log10(range) - 10 * std::log10(mse);
    std::cout << "MSE = " << mse << std::endl;
    std::cout << "RMSE = " << rmse << std::endl;
    std::cout << "NRMSE = " << nrmse << std::endl;
    std::cout << "PSNR = " << psnr << std::endl;
}


int main(int argc, char *argv[]) {
    auto [dataset, dim, isSinglePrecision, xi_pct, r] = Parsing(argc, argv);
    std::vector<std::vector<double> > points = readBinaryFile(strcat(dataset, ".dat"), dim);
    double range = getMaxColumnRange(points);
    clock_t start_time = clock();
    box_compression(points, xi_pct * range, r);
    clock_t end_time = clock();
    double duration = double(end_time - start_time) / CLOCKS_PER_SEC;
    std::cout << duration << std::endl;
}
