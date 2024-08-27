#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

// Define a structure to represent a point in space
struct Point {
    std::vector<double> coords; // Coordinates of the point
    int index; // Index of the point

    Point(const std::vector<double> &coordinates, int idx) : coords(coordinates), index(idx) {}
};

std::vector<Point> convertToPoints(const std::vector<std::vector<double>> &vec2D) {
    std::vector<Point> points;
    for (size_t i = 0; i < vec2D.size(); ++i) {
        // Ensure the inner vector is not empty and has a consistent size
        if (!vec2D[i].empty()) {
            points.emplace_back(vec2D[i], static_cast<int>(i)); // Use i as the index
        }
    }
    return points;
}

// Define a structure to represent a node in the k-d tree
struct kdNode {
    Point point; // Point stored in the node
    kdNode *left; // Pointer to the left child
    kdNode *right; // Pointer to the right child

    kdNode(const Point &pt) : point(pt), left(nullptr), right(nullptr) {}
};

// Function to calculate the squared Euclidean distance between two points
double squaredDistance(const Point &a, const Point &b) {
    double dist = 0.0;
    for (size_t i = 0; i < a.coords.size(); ++i) {
        double diff = a.coords[i] - b.coords[i];
        dist += diff * diff;
    }
    return dist;
}

// Function to build the k-d tree
kdNode *buildKDTree(std::vector<Point> &points, int depth = 0) {
    if (points.empty()) {
        return nullptr;
    }

    size_t axis = depth % points[0].coords.size(); // Select axis based on depth
    size_t median = points.size() / 2;

    // Sort points based on the selected axis
    std::nth_element(points.begin(), points.begin() + median, points.end(),
                     [axis](const Point &a, const Point &b) {
                         return a.coords[axis] < b.coords[axis];
                     });

    // Create a new node and construct subtrees
    kdNode *node = new kdNode(points[median]);
    std::vector<Point> leftPoints(points.begin(), points.begin() + median);
    std::vector<Point> rightPoints(points.begin() + median + 1, points.end());

    node->left = buildKDTree(leftPoints, depth + 1);
    node->right = buildKDTree(rightPoints, depth + 1);

    return node;
}

// Function to find the nearest neighbor
void nearestNeighbor(kdNode *root, const Point &query, kdNode *&bestNode, double &bestDist, int depth = 0) {
    if (!root) return;

    double dist = squaredDistance(query, root->point);
    if (dist < bestDist) {
        bestDist = dist;
        bestNode = root;
    }

    size_t axis = depth % query.coords.size();
    double diff = query.coords[axis] - root->point.coords[axis];

    kdNode *nextBranch = (diff < 0) ? root->left : root->right;
    kdNode *otherBranch = (diff < 0) ? root->right : root->left;

    nearestNeighbor(nextBranch, query, bestNode, bestDist, depth + 1);

    if (diff * diff < bestDist) {
        nearestNeighbor(otherBranch, query, bestNode, bestDist, depth + 1);
    }
}

int findNearestPointIndex(const std::vector<std::vector<double>> &vec2D, const std::vector<double> point) {
    std::vector<Point> pointsCopy = convertToPoints(vec2D);
    Point query(point, -1);
    kdNode *root = buildKDTree(pointsCopy);

    kdNode *bestNode = nullptr;
    double bestDist = std::numeric_limits<double>::max();

    nearestNeighbor(root, query, bestNode, bestDist);

    return bestNode ? bestNode->point.index : -1;
}