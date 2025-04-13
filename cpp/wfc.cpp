// Simple WFC implementation based on: https://github.com/math-fehr/fast-wfc
#include <vector>
#include <array>
#include <random>
#include <optional>
#include <cmath>
#include <algorithm>
#include <tuple>
#include <cassert>
#include <iostream>
#include <limits>
#include <string>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

// direction utils: up, down, left, right
constexpr int directions_x[4] = {0, -1, 1, 0};
constexpr int directions_y[4] = {-1, 0, 0, 1};

constexpr unsigned get_opposite_direction(unsigned direction) noexcept {
    return 3 - direction;
}

// 2D array implementation
template <typename T>
class Array2D {
    public:
        std::size_t height;
        std::size_t width;
        std::vector<T> data;

        Array2D(std::size_t height, std::size_t width) noexcept
            : height(height), width(width), data(width * height) {}
        
        Array2D(std::size_t height, std::size_t width, T value) noexcept
            : height(height), width(width), data(width * height, value) {}

        T get(std::size_t i, std::size_t j) const noexcept {
            assert(i < height && j < width);
            return data[i * width + j];
        }

        void set(std::size_t i, std::size_t j, T value) noexcept {
            assert(i < height && j < width);
            data[i * width + j] = value;
        }

        bool operator==(const Array2D<T> &other) const noexcept {
            if (height != other.height || width != other.width) {
                return false;
            }

            for (std::size_t i = 0; i < data.size(); i++) {
                if (other.data[i] != data[i]) {
                    return false;
                }
            }

            return true;
        }
};

// 3D array implementation
template <typename T>
class Array3D {
    public:
        std::size_t height;
        std::size_t width;
        std::size_t depth;
        std::vector<T> data;

        Array3D(std::size_t height, std::size_t width, std::size_t depth) noexcept
            : height(height), width(width), depth(depth), data(height * width * depth) {}
        Array3D(std::size_t height, std::size_t width, std::size_t depth, T value) noexcept
            : height(height), width(width), depth(depth), data(height * width * depth, value) {}
        
        T get(std::size_t i, std::size_t j, std::size_t k) const noexcept {
            assert(i < height && j < width && k < depth);
            return data[i * width * depth + j * depth + k];
        }

        void set(std::size_t i, std::size_t j, std::size_t k, T value) noexcept {
            assert(i < height && j < width && k < depth);
            data[i * width * depth + j * depth + k] = value;
        }

        bool operator==(const Array3D<T> &other) const noexcept {
            if (height != other.height || width != other.width || depth != other.depth) {
                return false;
            }

            for (std::size_t i = 0; i < data.size(); i++) {
                if (other.data[i] != data[i]) {
                    return false;
                }
            }

            return true;
        }
};

// Entropy memoization
struct EntropyMemoization {
    std::vector<double> p_logp_sum; // Sum of p * log(p) for each cell
    std::vector<double> p_sum; // Sum of p for each cell
    std::vector<double> log_sum; // Log of sum for each cell
    std::vector<unsigned> num_patterns; // Number of patterns in each cell
    std::vector<double> entropy; // Entropy for each cell
};

// Wave class to track possible patterns for each cell
class Wave {
    private:
        const std::vector<double> pattern_frequencies; // Normalized
        const std::vector<double> p_logp_pattern_frequencies; // Precomputed p * log(p) 
        const double min_abs_half_p_logp; // Used for noise in entropy calculation
        EntropyMemoization memoization;
        bool is_impossible;
        const size_t num_patterns;
        // Wave data: data[cell][pattern] = 1 if pattern is possible
        Array2D<uint8_t> data;

        // Calculate p * log(p) for all pattern frequencies
        static std::vector<double> get_p_logp (const std::vector<double> &distribution) noexcept {
            std::vector<double> p_logp;
            for (unsigned i = 0; i < distribution.size(); i++) {
                p_logp.push_back(distribution[i] * log(distribution[i]));
            }
            return p_logp;
        }

        // Calculate min(|p * log(p)|) / 2 for noise
        static double get_min_abs_half(const std::vector<double> &v) noexcept {
            double min_abs_half = std::numeric_limits<double>::infinity();
            for (unsigned i = 0; i < v.size(); i++) {
                min_abs_half = std::min(min_abs_half, std::abs(v[i] / 2.0));
            }
            return min_abs_half;
        }

    public:
        // Grid dimensions
        const unsigned width;
        const unsigned height;
        const unsigned size;

        // Wave constructor
        Wave(unsigned height, unsigned width, const std::vector<double> &pattern_frequencies) noexcept
            : pattern_frequencies(pattern_frequencies),
              p_logp_pattern_frequencies(get_p_logp(pattern_frequencies)),
              min_abs_half_p_logp(get_min_abs_half(p_logp_pattern_frequencies)),
              is_impossible(false),
              num_patterns(pattern_frequencies.size()),
              data(height * width, num_patterns, 1),
              height(height),
              width(width),
              size(height * width) {
            // Init memoization
            double base_entropy = 0.0;
            double base_s = 0.0;
            for (unsigned i = 0; i < num_patterns; i++) {
                base_entropy += p_logp_pattern_frequencies[i];
                base_s += pattern_frequencies[i];
            }
            double log_base_s = log(base_s);
            double entropy_base = log_base_s - base_entropy / base_s;

            memoization.p_logp_sum = std::vector<double>(height * width, base_entropy);
            memoization.p_sum = std::vector<double>(height * width, base_s);
            memoization.log_sum = std::vector<double>(height * width, log_base_s);
            memoization.num_patterns = std::vector<unsigned>(height * width, static_cast<unsigned>(num_patterns));
            memoization.entropy = std::vector<double>(height * width, entropy_base);
        }

        // Check if pattern can be placed in cell at index
        bool get(unsigned index, unsigned pattern) const noexcept {
            return data.get(index, pattern);
        }

        // Check if pattern can be placed in cell at (i, j)
        bool get(unsigned i, unsigned j, unsigned pattern) const noexcept {
            return get(i * width +j, pattern);
        }

        // Set pattern possibility in cell at index
        void set(unsigned index, unsigned pattern, bool value) noexcept {
            bool old_value = data.get(index, pattern);
            if (old_value == value) {
                return;
            }

            data.set(index, pattern, value);
            
            // Update entropy memoization
            if (value == false) { // Remove pattern
                memoization.p_logp_sum[index] -= p_logp_pattern_frequencies[pattern];
                memoization.p_sum[index] -= pattern_frequencies[pattern];
                memoization.log_sum[index] = log(memoization.p_sum[index]);
                memoization.num_patterns[index]--;
                memoization.entropy[index] = memoization.log_sum[index] - memoization.p_logp_sum[index] / memoization.p_sum[index];

                // Check contradiction
                if (memoization.num_patterns[index] == 0) {
                    is_impossible = true;
                }
            }
        }

        // Set pattern possibility in cell at (i, j)
        void set(unsigned i, unsigned j, unsigned pattern, bool value) noexcept {
            set(i * width + j, pattern, value);
        }

        // Find cell with minimum entropy (-2: contradiction, -1: success, otherwise: cell index)
        int get_min_entropy(std::minstd_rand &gen) const noexcept {
            if (is_impossible) {
                return -2;
            }

            std::uniform_real_distribution<> dis(0, min_abs_half_p_logp);
            double min_entropy = std::numeric_limits<double>::infinity();
            int argmin = -1;

            for (unsigned i = 0; i < size; i++) {
                // Skip cells with one pattern (already collapsed)
                if (memoization.num_patterns[i] == 1) {
                    continue;
                }

                // Get noised entropy
                double entropy = memoization.entropy[i];
                if (entropy <= min_entropy) {
                    double noise = dis(gen);
                    if (entropy + noise < min_entropy) {
                        min_entropy = entropy + noise;
                        argmin = i;
                    }
                }
            }
            
            return argmin; 
        }
};

// Propagator class to propagate constraints
class Propagator {
    public:
        // Pattern compatibility rules
        using PropagatorState = std::vector<std::array<std::vector<unsigned>, 4>>;

    private:
        // Number of patterns
        const std::size_t patterns_size;
        // Compatibility rules: propagator_state[pattern][direction] = compatible patterns
        PropagatorState propagator_state;
        // Grid dimensions
        const unsigned wave_width;
        const unsigned wave_height;
        // True if output is toric/periodic
        const bool periodic_output;
        // Queue of (y,x,pattern) tuples to propagate
        std::vector<std::tuple<unsigned, unsigned, unsigned>> propagating;
        // compatible[y][x][pattern][direction] = number of compatible patterns
        Array3D<std::array<int, 4>> compatible;
        
        // Initialize the compatible array
        void init_compatible() noexcept {
            std::array<int, 4> value;
            for (unsigned y = 0; y < wave_height; y++) {
                for (unsigned x = 0; x < wave_width; x++) {
                    for (unsigned pattern = 0; pattern < patterns_size; pattern++) {
                        for (int direction = 0; direction < 4; direction++) {
                            value[direction] = static_cast<unsigned>(
                                propagator_state[pattern][get_opposite_direction(direction)].size());
                        }
                    compatible.get(y, x, pattern) = value;
                    }
                }
            }
        }
      
    public:
        // Constructor for the propagator
        Propagator(unsigned wave_height, unsigned wave_width, bool periodic_output,
                    PropagatorState propagator_state) noexcept
            : patterns_size(propagator_state.size()),
                propagator_state(propagator_state), 
                wave_width(wave_width),
                wave_height(wave_height), 
                periodic_output(periodic_output),
                compatible(wave_height, wave_width, patterns_size) {
            init_compatible();
        }
        
        // Add a (y,x,pattern) to the propagation queue
        void add_to_propagator(unsigned y, unsigned x, unsigned pattern) noexcept {
            std::array<int, 4> temp = {};
            compatible.get(y, x, pattern) = temp;
            propagating.emplace_back(y, x, pattern);
        }
        
        // Propagate constraints through the wave
        void propagate(Wave &wave) noexcept {
            while (propagating.size() != 0) {
                unsigned y1, x1, pattern;
                std::tie(y1, x1, pattern) = propagating.back();
                propagating.pop_back();
            
                // Propagate in all four directions
                for (unsigned direction = 0; direction < 4; direction++) {
                    int dx = directions_x[direction];
                    int dy = directions_y[direction];
                    int x2, y2;
                    
                    // Calculate the next cell coordinates, handling periodicity
                    if (periodic_output) {
                        x2 = ((int)x1 + dx + (int)wave.width) % wave.width;
                        y2 = ((int)y1 + dy + (int)wave.height) % wave.height;
                    } else {
                        x2 = x1 + dx;
                        y2 = y1 + dy;
                    // Skip if outside grid boundaries
                    if (x2 < 0 || x2 >= (int)wave.width) {
                        continue;
                    }
                    if (y2 < 0 || y2 >= (int)wave.height) {
                        continue;
                    }
                    }
            
                    unsigned i2 = x2 + y2 * wave.width;
                    const std::vector<unsigned> &patterns = propagator_state[pattern][direction];
            
                    // Update compatible counts for all affected patterns
                    for (auto it = patterns.begin(), it_end = patterns.end(); it < it_end; ++it) {
                        std::array<int, 4> value = compatible.get(y2, x2, *it);
                        value[direction]--;
                        compatible.set(y2, x2, *it, value);
                
                        // If no compatible patterns remain in a direction, remove pattern from wave
                        if (value[direction] == 0) {
                            add_to_propagator(y2, x2, *it);
                            wave.set(i2, *it, false);
                        }
                    }
                }
            }
        }
};

// Main WFC algorithm implementation
class WFC {
    private:
        // Random number generator
        std::minstd_rand gen;
        // Pattern frequencies
        const std::vector<double> pattern_frequencies;
        // The wave of possibilities
        Wave wave;
        // Number of patterns
        const size_t num_patterns;
        // Propagator for constraint propagation
        Propagator propagator;
        
        // Normalize a vector of weights to sum to 1.0
        std::vector<double>& normalize(std::vector<double>& v) {
            double sum_weights = 0.0;
            for(double weight: v) {
                sum_weights += weight;
            }
        
            double inv_sum_weights = 1.0/sum_weights;
            for(double& weight: v) {
                weight *= inv_sum_weights;
            }
        
            return v;
        }
        
        // Convert the wave to an output grid of pattern indices
        Array2D<unsigned> wave_to_output() const noexcept {
            Array2D<unsigned> output_patterns(wave.height, wave.width);
            for (unsigned i = 0; i < wave.size; i++) {
                for (unsigned k = 0; k < num_patterns; k++) {
                    if (wave.get(i, k)) {
                        output_patterns.data[i] = k;
                    }
                }
            }
            return output_patterns;
        }
    
    public:
        // Status of observation step
        enum ObserveStatus {
            success,      // WFC has finished and has succeeded
            failure,      // WFC has finished and failed
            to_continue   // WFC isn't finished
        };
        
        // Constructor for the WFC algorithm
        WFC(bool periodic_output, int seed,
            std::vector<double> pattern_frequencies,
            Propagator::PropagatorState propagator_state, 
            unsigned wave_height,
            unsigned wave_width) noexcept
            : gen(seed), 
            pattern_frequencies(normalize(pattern_frequencies)),
            wave(wave_height, wave_width, pattern_frequencies),
            num_patterns(pattern_frequencies.size()),
            propagator(wave.height, wave.width, periodic_output, propagator_state) {}
        
        // Observe the cell with minimum entropy
        ObserveStatus observe() noexcept {
            int argmin = wave.get_min_entropy(gen);
        
            if (argmin == -2) {
                return failure;  // Contradiction found
            }
        
            if (argmin == -1) {
                return success;  // All cells decided
            }
        
            // Calculate sum of weights for possible patterns
            double s = 0;
            for (unsigned k = 0; k < num_patterns; k++) {
                s += wave.get(argmin, k) ? pattern_frequencies[k] : 0;
            }
        
            // Randomly choose a pattern according to weights
            std::uniform_real_distribution<> dis(0, s);
            double random_value = dis(gen);
            size_t chosen_value = num_patterns - 1;
        
            for (unsigned k = 0; k < num_patterns; k++) {
                random_value -= wave.get(argmin, k) ? pattern_frequencies[k] : 0;
                if (random_value <= 0) {
                    chosen_value = k;
                    break;
                }
            }
        
            // Set the chosen pattern and remove others
            for (unsigned k = 0; k < num_patterns; k++) {
                if (wave.get(argmin, k) != (k == chosen_value)) {
                    propagator.add_to_propagator(argmin / wave.width, argmin % wave.width, k);
                    wave.set(argmin, k, false);
                }
            }
        
            return to_continue;
        }

        // Get full wave state (pattern probabilities for each cell)
        nb::ndarray<bool> get_wave_state() const {
            // Dimensions
            size_t height = wave.height;
            size_t width = wave.width;
            size_t depth = num_patterns;

            // Capsule for memory management
            bool* data = new bool[height * width * depth];
            auto capsule = nb::capsule(data, [](void *p) noexcept {
                delete[] static_cast<bool *>(p);
            });

            // Fill the ndarray with the wave data
            for (size_t y = 0; y < height; y++) {
                for (size_t x = 0; x < width; x++) {
                    for (size_t p = 0; p < depth; p++) {
                        data[y * wave.width * depth + x * depth + p] = wave.get(y, x, p);
                    }
                }
            }

            // Create ndarray with allocated memory
            auto result = nb::ndarray<bool>(
                data,
                {height, width, depth},
                capsule,
                {
                    width * depth * sizeof(bool),
                    depth * sizeof(bool),
                    sizeof(bool)
                }
            );
            return result;
        }

        // Get next cell to collapse with probabilities
        std::tuple<int, int, std::vector<double>> get_next_collapse_cell() {
            int argmin = wave.get_min_entropy(gen);
            if (argmin == -1 || argmin == -2) {
                return std::make_tuple(-1, -1, std::vector<double>());
            }
            
            int y = argmin / wave.width;
            int x = argmin % wave.width;
            
            std::vector<double> probabilities(num_patterns, 0.0);
            double total_weight = 0.0;
            for (unsigned p = 0; p < num_patterns; p++) {
                if (wave.get(y, x, p)) {
                    probabilities[p] = pattern_frequencies[p];
                    total_weight += probabilities[p];
                }
            }

            // Norm probabilities
            if (total_weight > 0) {
                for (auto& p: probabilities) {
                    p /= total_weight;
                }
            }
            return std::make_tuple(x, y, probabilities);
        }

        // Single collapse step with action vector
        std::tuple<bool, bool> collapse_step(const std::vector<double>& action_vec) {
            ObserveStatus status = observe();

            if (status == failure) {
                return std::make_tuple(false, true); // terminate=false, truncate=true
            } else if (status == success) {
                return std::make_tuple(true, false); // terminate=true, truncate=false
            }

            int argmin = wave.get_min_entropy(gen);
            int y = argmin / wave.width;
            int x = argmin % wave.width;

            // Use action to influence collapse
            double total_weight = 0.0;
            for (unsigned p = 0; p < num_patterns; p++) {
                if (wave.get(y, x, p)) {
                    total_weight += action_vec[p];
                }
            }

            // If total weight is zero, use uniform
            if (total_weight <= 0.0) {
                for (unsigned p = 0; p < num_patterns; p++) {
                    if (wave.get(y, x, p)) {
                        total_weight += 1.0;
                    }
                }
            }

            double rand_val = std::uniform_real_distribution<>(0, total_weight)(gen);
            int chosen = -1;
            double running_sum = 0.0;

            for (unsigned p = 0; p < num_patterns; p++) {
                if (wave.get(y, x, p)) {
                    running_sum += action_vec[p] > 0 ? action_vec[p] : 1.0;
                    if (running_sum >= rand_val) {
                        chosen = p;
                        break;
                    }
                }
            }

            // Collapse to chosen pattern
            for (unsigned p = 0; p < num_patterns; p++) {
                if (p != chosen && wave.get(y, x, p)) {
                    propagator.add_to_propagator( y, x, p);
                    wave.set(y, x, p, false);
                }
            }

            propagator.propagate(wave); // Propagate constraints
            return std::make_tuple(false, false); // continue
        }
        
        // Propagate constraints in the wave
        void propagate() noexcept { 
            propagator.propagate(wave); 
        }
        
        // Remove a pattern from a cell
        void remove_wave_pattern(unsigned i, unsigned j, unsigned pattern) noexcept {
            if (wave.get(i, j, pattern)) {
                wave.set(i, j, pattern, false);
                propagator.add_to_propagator(i, j, pattern);
            }
        }
        
        // Run the full WFC algorithm
        std::optional<Array2D<unsigned>> run() noexcept {
            while (true) {
                ObserveStatus result = observe();
            
                if (result == failure) {
                    return std::nullopt;  // Algorithm failed
                } else if (result == success) {
                    return wave_to_output();  // Algorithm succeeded
                }
            
                propagator.propagate(wave);  // Propagate constraints
            }
        }
};
    
// Helper to generate simple adjacency rules for a test
Propagator::PropagatorState generate_adjacency_rules(int num_patterns) {
    // Initialize the rules
    Propagator::PropagatorState rules(num_patterns);
    
    // Set up some simple adjacency rules
    for (int i = 0; i < num_patterns; i++) {
        for (int dir = 0; dir < 4; dir++) {
            // Add compatible patterns for each direction
            // Simple rule: patterns can be adjacent if they differ by at most 1
            for (int j = 0; j < num_patterns; j++) {
                if (std::abs(i - j) <= 1) {
                    rules[i][dir].push_back(j);
                }
            }
        }
    }
    
    return rules;
}

// Helper function to visualize the output with ASCII symbols
void visualize_output(const Array2D<unsigned>& output) {
    const std::string symbols = ".#X$@%&*";
    
    for (unsigned y = 0; y < output.height; y++) {
        for (unsigned x = 0; x < output.width; x++) {
            unsigned pattern = output.get(y, x);
            char symbol = symbols[pattern % symbols.length()];
            std::cout << symbol << ' ';
        }
        std::cout << std::endl;
    }
}
    
// Main function with a simple example
int main() {
    // Set up the parameters
    const int width = 20;
    const int height = 10;
    const int num_patterns = 3;
    const bool periodic = false;
    const int seed = 42;
    
    std::cout << "Wave Function Collapse Algorithm Example" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Grid size: " << width << "x" << height << std::endl;
    std::cout << "Number of patterns: " << num_patterns << std::endl;
    std::cout << "Periodic output: " << (periodic ? "Yes" : "No") << std::endl;
    std::cout << "Seed: " << seed << std::endl;
    std::cout << std::endl;
    
    // Initialize pattern frequencies (equal in this simple example)
    std::vector<double> frequencies(num_patterns, 1.0);
    
    // Generate simple adjacency rules
    auto rules = generate_adjacency_rules(num_patterns);
    
    // Create the WFC instance
    WFC wfc(periodic, seed, frequencies, rules, height, width);
    
    // Run the algorithm
    auto result = wfc.run();
    
    // Check if generation was successful
    if (!result) {
    std::cout << "WFC failed to generate a valid output" << std::endl;
    return 1;
    }
    
    // Print the resulting pattern
    std::cout << "Generated pattern (numeric):" << std::endl;
    for (unsigned y = 0; y < height; y++) {
    for (unsigned x = 0; x < width; x++) {
        std::cout << result->get(y, x) << " ";
    }
    std::cout << std::endl;
    }
    
    std::cout << "\nGenerated pattern (visual):" << std::endl;
    visualize_output(*result);
    
    return 0;
}