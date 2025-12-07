/**
 * 2025 Lord Kelvin Programming Competition
 */

#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <math.h>
#include <float.h>

// ==========================================
// 1. Constants & Data Structures
// ==========================================

#define MAX_T 3005
#define MAX_N 45
#define MAX_M 85
#define MAX_BELIEF_STATES 1000  // For discretization
#define EPSILON 1e-12

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    int T;      // Total duration
    int N;      // Number of states
    int M;      // Number of sound effects
    int L;      // Wandering cycle period
    int Type;   // 0 = No Penalty, 1 = Matrix Penalty
} Config;

typedef struct {
    Config cfg;
    int initial_rewards[MAX_N];
    double mu;
    int penalty_matrix[MAX_N][MAX_N];
    double transition_matrices[MAX_M][MAX_N][MAX_N];
} ProblemData;

typedef struct {
    double prob[MAX_N];
} BeliefState;

// ==========================================
// 2. Mathematical Functions
// ==========================================

double get_reward(const ProblemData* data, int state, int t) {
    if (data->cfg.L == 0) {
        return (double)data->initial_rewards[state];
    }
    double r_init = (double)data->initial_rewards[state];
    double cosine_term = cos((2.0 * M_PI * t) / (double)data->cfg.L);
    return data->mu + (r_init - data->mu) * cosine_term;
}

int get_penalty(const ProblemData* data, int prev_state, int curr_state, int prev_action, int curr_action) {
    if (prev_action == curr_action) return 0;
    if (data->cfg.Type == 0) return 0;
    return data->penalty_matrix[prev_state][curr_state];
}

// ==========================================
// 3. Input Parsing
// ==========================================

void read_input(ProblemData* data) {
    scanf("%d %d %d %d %d", &data->cfg.T, &data->cfg.N, &data->cfg.M, &data->cfg.L, &data->cfg.Type);

    double sum = 0.0;
    for (int i = 0; i < data->cfg.N; i++) {
        scanf("%d", &data->initial_rewards[i]);
        sum += data->initial_rewards[i];
    }
    data->mu = sum / data->cfg.N;

    for (int i = 0; i < data->cfg.N; i++) {
        for (int j = 0; j < data->cfg.N; j++) {
            scanf("%d", &data->penalty_matrix[i][j]);
        }
    }

    for (int m = 0; m < data->cfg.M; m++) {
        for (int i = 0; i < data->cfg.N; i++) {
            for (int j = 0; j < data->cfg.N; j++) {
                scanf("%lf", &data->transition_matrices[m][i][j]);
            }
        }
    }
}

// ==========================================
// 4. Belief State Operations
// ==========================================

void update_belief(const ProblemData* data, const BeliefState* old_belief,
                   int action, BeliefState* new_belief) {
    for (int j = 0; j < data->cfg.N; j++) {
        new_belief->prob[j] = 0.0;
    }
    
    for (int i = 0; i < data->cfg.N; i++) {
        if (old_belief->prob[i] < EPSILON) continue;
        for (int j = 0; j < data->cfg.N; j++) {
            new_belief->prob[j] += old_belief->prob[i] * 
                                 data->transition_matrices[action][i][j];
        }
    }
    
    // Normalize
    double sum = 0.0;
    for (int j = 0; j < data->cfg.N; j++) {
        sum += new_belief->prob[j];
    }
    if (sum > EPSILON) {
        for (int j = 0; j < data->cfg.N; j++) {
            new_belief->prob[j] /= sum;
        }
    }
}

// ==========================================
// 5. Exact Solution for Small Problems (T ≤ 100)
// ==========================================

// DP table for exact solution of small problems
double dp_small[MAX_T][MAX_N][MAX_M];  // [time][state][prev_action]
int policy_small[MAX_T][MAX_N][MAX_M];

void solve_small_exact(const ProblemData* data, int* result_actions) {
    int T = data->cfg.T;
    int N = data->cfg.N;
    int M = data->cfg.M;
    
    // Initialize DP table
    for (int i = 0; i < T; i++) {
        for (int s = 0; s < N; s++) {
            for (int a = 0; a < M; a++) {
                dp_small[i][s][a] = -DBL_MAX;
            }
        }
    }
    
    // Fill DP table backwards
    for (int t = T - 1; t >= 0; t--) {
        for (int s = 0; s < N; s++) {
            for (int prev_a = 0; prev_a < M; prev_a++) {
                double best_value = -DBL_MAX;
                int best_action = 0;
                
                for (int a = 0; a < M; a++) {
                    double value = 0.0;
                    
                    // Immediate reward and penalty
                    double imm_reward = get_reward(data, s, t);
                    int penalty = (t > 0) ? get_penalty(data, s, s, prev_a, a) : 0;
                    
                    value = imm_reward - penalty;
                    
                    // Future value
                    if (t < T - 1) {
                        double future_value = 0.0;
                        for (int next_s = 0; next_s < N; next_s++) {
                            double trans = data->transition_matrices[a][s][next_s];
                            if (trans > EPSILON) {
                                future_value += trans * dp_small[t + 1][next_s][a];
                            }
                        }
                        value += future_value;
                    }
                    
                    if (value > best_value) {
                        best_value = value;
                        best_action = a;
                    }
                }
                
                dp_small[t][s][prev_a] = best_value;
                policy_small[t][s][prev_a] = best_action;
            }
        }
    }
    
    // Reconstruct policy starting from state 0, no previous action
    int current_state = 0;
    int prev_action = 0;
    
    for (int t = 0; t < T; t++) {
        result_actions[t] = policy_small[t][current_state][prev_action];
        prev_action = result_actions[t];
        
        // Sample next state according to transition probabilities
        double rand_val = (double)(t * 1234567 % 10000) / 10000.0;
        double cumsum = 0.0;
        int next_state = 0;
        for (int s = 0; s < N; s++) {
            cumsum += data->transition_matrices[prev_action][current_state][s];
            if (rand_val <= cumsum) {
                next_state = s;
                break;
            }
        }
        current_state = next_state;
    }
}

// ==========================================
// 6. Heuristic Solution for Large Problems
// ==========================================

// Greedy lookahead heuristic with depth d
double evaluate_action(const ProblemData* data, const BeliefState* belief,
                       int action, int t, int prev_action, int depth) {
    if (t >= data->cfg.T || depth <= 0) {
        return 0.0;
    }
    
    // Compute immediate reward
    double immediate = 0.0;
    BeliefState next_belief;
    
    for (int j = 0; j < data->cfg.N; j++) {
        next_belief.prob[j] = 0.0;
    }
    
    for (int i = 0; i < data->cfg.N; i++) {
        if (belief->prob[i] < EPSILON) continue;
        for (int j = 0; j < data->cfg.N; j++) {
            double trans = data->transition_matrices[action][i][j];
            if (trans < EPSILON) continue;
            
            double prob = belief->prob[i] * trans;
            double reward = get_reward(data, j, t);
            int penalty = get_penalty(data, i, j, prev_action, action);
            
            immediate += prob * (reward - penalty);
            next_belief.prob[j] += prob;
        }
    }
    
    // Normalize next belief
    double sum = 0.0;
    for (int j = 0; j < data->cfg.N; j++) {
        sum += next_belief.prob[j];
    }
    if (sum > EPSILON) {
        for (int j = 0; j < data->cfg.N; j++) {
            next_belief.prob[j] /= sum;
        }
    }
    
    // Recursively evaluate future
    if (depth > 1 && t + 1 < data->cfg.T) {
        double best_future = -DBL_MAX;
        
        // Try all actions for next step (pruned for efficiency)
        for (int next_a = 0; next_a < data->cfg.M; next_a++) {
            double future = evaluate_action(data, &next_belief, next_a, 
                                           t + 1, action, depth - 1);
            if (future > best_future) {
                best_future = future;
            }
        }
        
        if (best_future > -DBL_MAX/2) {
            immediate += best_future;
        }
    }
    
    return immediate;
}

// Precompute expected rewards for each state and time
void precompute_rewards(const ProblemData* data, double rewards[MAX_T][MAX_N]) {
    for (int t = 0; t < data->cfg.T; t++) {
        for (int s = 0; s < data->cfg.N; s++) {
            rewards[t][s] = get_reward(data, s, t);
        }
    }
}

// Action selection with adaptive lookahead
int select_best_action(const ProblemData* data, const BeliefState* belief,
                       int t, int prev_action, double precomp_rewards[MAX_T][MAX_N]) {
    double best_value = -DBL_MAX;
    int best_action = 0;
    
    // Adaptive lookahead depth based on problem size
    int lookahead_depth;
    if (data->cfg.T <= 50 && data->cfg.N <= 10 && data->cfg.M <= 10) {
        lookahead_depth = 3;  // Deeper search for small problems
    } else if (data->cfg.N * data->cfg.M <= 1000) {
        lookahead_depth = 2;
    } else {
        lookahead_depth = 1;  // Greedy for very large problems
    }
    
    // Try all actions
    for (int a = 0; a < data->cfg.M; a++) {
        double value = 0.0;
        
        if (lookahead_depth == 1) {
            // Fast greedy evaluation
            for (int i = 0; i < data->cfg.N; i++) {
                if (belief->prob[i] < EPSILON) continue;
                for (int j = 0; j < data->cfg.N; j++) {
                    double trans = data->transition_matrices[a][i][j];
                    if (trans < EPSILON) continue;
                    
                    double prob = belief->prob[i] * trans;
                    double reward = precomp_rewards[t][j];
                    int penalty = get_penalty(data, i, j, prev_action, a);
                    
                    value += prob * (reward - penalty);
                }
            }
        } else {
            // Deeper lookahead
            value = evaluate_action(data, belief, a, t, prev_action, lookahead_depth);
        }
        
        if (value > best_value) {
            best_value = value;
            best_action = a;
        }
    }
    
    return best_action;
}

// ==========================================
// 7. Main Solver
// ==========================================

void solve(const ProblemData* data, int* result_actions) {
    int T = data->cfg.T;
    int N = data->cfg.N;
    int M = data->cfg.M;
    
    // Choose strategy based on problem size
    int use_exact = (T <= 100 && N <= 15 && M <= 15);
    
    if (use_exact) {
        // Use exact DP for small problems
        solve_small_exact(data, result_actions);
    } else {
        // Use heuristic for larger problems
        BeliefState belief;
        for (int i = 0; i < N; i++) {
            belief.prob[i] = 0.0;
        }
        belief.prob[0] = 1.0;  // Start in state 0
        
        // Precompute rewards for efficiency
        double precomp_rewards[MAX_T][MAX_N];
        precompute_rewards(data, precomp_rewards);
        
        int prev_action = 0;
        
        for (int t = 0; t < T; t++) {
            // Select best action
            result_actions[t] = select_best_action(data, &belief, t, 
                                                  prev_action, precomp_rewards);
            
            // Update belief state
            BeliefState new_belief;
            update_belief(data, &belief, result_actions[t], &new_belief);
            belief = new_belief;
            prev_action = result_actions[t];
        }
    }
}

// ==========================================
// 8. Output Formatting
// ==========================================

void print_output(const int* actions, int T) {
    printf("[");
    for (int i = 0; i < T; ++i) {
        printf("%d", actions[i]);
        if (i < T - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

// ==========================================
// 9. Main Function
// ==========================================

int main() {
    static ProblemData data;
    
    // Read input
    read_input(&data);
    
    // Solve
    int result_actions[MAX_T];
    solve(&data, result_actions);
    
    // Output
    print_output(result_actions, data.cfg.T);
    
    return 0;
}