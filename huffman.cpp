#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <string>
#include <memory>

struct HuffmanNode {
    int data;
    int freq;
    std::shared_ptr<HuffmanNode> left, right;

    HuffmanNode(int data, int freq) : data(data), freq(freq), left(nullptr), right(nullptr) {}
};

struct Compare {
    bool operator()(std::shared_ptr<HuffmanNode> a, std::shared_ptr<HuffmanNode> b) {
        return a->freq > b->freq;
    }
};

void generateCodes(const std::shared_ptr<HuffmanNode> &root, const std::string &code,
                   std::unordered_map<int, std::string> &huffmanCodes) {
    if (!root) return;
    if (!root->left && !root->right) {
        huffmanCodes[root->data] = code;
    }
    generateCodes(root->left, code + "0", huffmanCodes);
    generateCodes(root->right, code + "1", huffmanCodes);
}

std::string huffmanCompress(const std::vector<int> &data) {
    std::unordered_map<int, int> freqTable;
    for (int num: data) {
        freqTable[num]++;
    }
    std::priority_queue<std::shared_ptr<HuffmanNode>, std::vector<std::shared_ptr<HuffmanNode>>, Compare> pq;
    for (const auto &pair: freqTable) {
        pq.push(std::make_shared<HuffmanNode>(pair.first, pair.second));
    }

    while (pq.size() > 1) {
        auto left = pq.top();
        pq.pop();
        auto right = pq.top();
        pq.pop();
        auto parent = std::make_shared<HuffmanNode>(-1, left->freq + right->freq);
        parent->left = left;
        parent->right = right;
        pq.push(parent);
    }

    auto root = pq.top();

    std::unordered_map<int, std::string> huffmanCodes;
    generateCodes(root, "", huffmanCodes);

    std::string encodedData;
    for (int num: data) {
        encodedData += huffmanCodes[num];
    }

    return encodedData;
}