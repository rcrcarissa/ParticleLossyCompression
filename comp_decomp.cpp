#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <algorithm>
#include <cmath>
#include <numeric>
#include "findNearestPoint.cpp"

void printUsage() {
    std::cout << "Usage: <cpp_filename> -ds <dataset_name> -dim <dimension> -f" << std::endl;
    std::cout << "  -ds  Specify the dataset name" << std::endl;
    std::cout << "  -dim   Specify the dimension" << std::endl;
    std::cout << "  -f   Specify the precision. -f for float and -d for double" << std::endl;
    std::cout << "  -reb   Specify the relative error bound." << std::endl;
}

std::tuple<char *, size_t, bool, double> Parsing(int argc, char *argv[]) {
    std::tuple<char *, size_t, bool, double> error_return_value = std::make_tuple(nullptr, 0, false, 0);
    if (argc < 8) {
        printUsage();
        return error_return_value;
    }

    char *dataset;
    size_t dim;
    bool isSinglePrecision;
    double reb;

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
        } else if (arg == "-f") {
            isSinglePrecision = true;
        } else if (arg == "-d") {
            isSinglePrecision = false;
        }
    }
    return std::make_tuple(dataset, dim, isSinglePrecision, reb);
}

typedef struct KDNode {
    int depth;
    int start_idx;
    int end_idx;
    int split_pt_idx;
    std::vector<double> attrs;  // box_center, box_half_len
    KDNode *parent;
    KDNode *left;
    KDNode *right;

    KDNode(int depth, int start_id, int end_id, KDNode *parent) : depth(depth), start_idx(start_id), end_idx(end_id),
                                                                  parent(parent), left(nullptr), right(nullptr) {}
} KDNode;

typedef struct KDTree {
    std::vector<std::vector<double>> pts;
    size_t dim;
    int r;  // maximum number of particles in a KDNode
    KDNode root;

    KDTree(std::vector<std::vector<double>> points, int r) : pts(points), dim(pts[0].size()), r(r),
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

    void get_leaf_nodes(KDNode *root_node, std::vector<KDNode *> leaves_ptr) {
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

    void top_down_search(KDNode *node, std::vector<double> (*func)(std::vector<double>, std::vector<double>));


} KDTree;

std::vector<std::vector<double>> readFile(const std::string &fileName, size_t dim) {
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

int half_length2bits(double half_len, double xi) {
    return (int) std::ceil(std::log(half_len / xi + 1) / std::log(2) + 1);
}

double bits2half_length(int m, double xi) {
    return (std::pow(2, m - 1) - 1) * xi;
}


std::tuple<std::vector<double>, std::vector<double>, int, std::vector<int>>
get_bit_box(std::vector<std::vector<double>> pts, double xi) {
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

std::vector<double> get_union_of_two_bounding_boxes(std::vector<double> box0_attrs, std::vector<double> box1_attrs) {
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

std::vector<std::vector<double>> get_box_range(std::vector<double> box_attrs) {
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

bool is_overlapping1D(std::vector<double> interval0, std::vector<double> interval1) {
    return interval0[1] >= interval1[0] && interval1[1] >= interval0[0];
}

bool is_overlappingND(std::vector<std::vector<double>> box0_range, std::vector<std::vector<double>> box1_range) {
    size_t dim = box0.size();
    bool is_overlap = true;
    for (int d = 0; d < dim; d++) {
        is_overlap = is_overlap && is_overlapping1D(box0_range[d], box1_range[d]);
    }
    return is_overlap;
}

bool is_two_boxes_overlap(std::vector<double> box0_attrs, std::vector<double> box1_attrs) {
    if (box0_attrs.size() == 0 or box1_attrs.size() == 0) return false;
    std::vector<std::vector<double>> box0_range = get_box_range(box0_attrs);
    std::vector<std::vector<double>> box1_range = get_box_range(box1_attrs);
    return is_overlappingND(box0_range, box1_range);
}

bool is_point_in_box(std::vector<double> pt, std::vector<std::vector<double>> box_range) {
    size_t dim = pt.size();
    bool inside = true;
    for (int d = 0; d < dim; d++) {
        inside = inside && box_range[d][0] < pt[d] < box_range[d][1];
    }
    return inside;
}

void box_compression(std::vector<std::vector<double>> pts, double xi, int r) {
    KDTree kd_tree(pts, r);
    std::vector<KDNode *> leaves_ptr;
    kd_tree.get_leaf_nodes(&kd_tree.root, leaves_ptr);
    int n_remaining_pts = pts.size();
    std::vector<int> total_n_bits(leaves_ptr.size());
    for (int i = 0; i < leaves_ptr.size(); i++) {
        // build bit boxes
        std::vector<std::vector<double>> leaf_pts(pts.begin() + leaves_ptr[i]->start_idx,
                                                  pts.begin() + leaves_ptr[i]->end_idx);
        auto [center, half_length, nearest_pt_id, num_bits] = get_bit_box(leaf_pts, xi);
        std::copy(center.begin(), center.end(), leaves_ptr[i]->attrs.begin());
        std::copy(half_length.begin(), half_length.end(), leaves_ptr[i]->attrs.begin() + kd_tree.dim);
        leaves_ptr[i]->split_pt_idx = nearest_pt_id;
        total_n_bits[i] = std::accumulate(num_bits.begin(), num_bits.end(), 0);
    }
    kd_tree.initialize(&kd_tree.root, get_union_of_two_bounding_boxes);
    std::vector<std::vector<int>>nums_bits;
//    while (n_remaining_pts > 0) {
//        int box_idx = std::distance(total_n_bits.begin(), std::min_element(total_n_bits.begin(), total_n_bits.end()));
//        KDNode *leaf_to_remove = leaves_ptr[box_idx];
//        nums_bits.push_back()
//    }
}

int main(int argc, char *argv[]) {

//    auto [dataset, dim, isSinglePrecision, xi_pct] = Parsing(argc, argv);
//
//    std::cout << dataset << std::endl;
//    std::cout << dim << std::endl;
//    std::cout << isSinglePrecision << std::endl;

//    std::vector<std::vector<double> > arr = readFile("a.dat", dim);
//    KDNode node(1, nullptr);
    std::vector<std::vector<double>> points = {
            {10.0, 11.0, 12.0},
            {1.0,  2.0,  3.0},
            {4.0,  5.0,  6.0},
            {13.0, 14.0, 15.0},
            {7.0,  8.0,  9.0}
    };
    KDTree tree(points, 1);
}