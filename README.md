# 2025 Lord Kelvin Programming Competition Solution [VIP Track]

## Overview

This repository contains my solution to the 2025 Lord Kelvin Programming Competition. The problem involves a **Partially Observable Markov Decision Process (POMDP)** where an agent must select optimal actions over time to maximize rewards while managing uncertainty about the current state.

## Problem Description

The challenge involves controlling a system that:

- Operates over **T** time steps
- Has **N** possible hidden states
- Can execute **M** different sound effect actions
- Transitions between states probabilistically based on actions
- Provides time-varying rewards with optional periodic wandering (cosine-based with period **L**)
- May incur penalties for switching actions (depending on problem type)

### Key Features

- **Belief State Tracking**: Maintains probability distributions over possible states
- **Time-Varying Rewards**: Rewards oscillate according to a cosine function when wandering is enabled
- **Action Penalties**: Optional penalties for switching between different actions
- **Optimal Policy**: Finds action sequences that maximize expected cumulative reward

## Solution Approach

The solution employs a **hybrid strategy** that adapts based on problem size:

### 1. Exact Dynamic Programming (Small Problems)

For problems with **T ≤ 100**, **N ≤ 15**, and **M ≤ 15**:

- Uses backward induction with full state enumeration
- Computes optimal value function: `dp[time][state][prev_action]`
- Guarantees optimal solution within computational constraints

### 2. Adaptive Heuristic (Large Problems)

For larger instances:

- **Belief State Propagation**: Maintains probability distribution over states
- **Lookahead Search**: Evaluates actions with adaptive depth:
  - Depth 3 for small-medium problems
  - Depth 2 for medium problems
  - Depth 1 (greedy) for very large problems
- **Performance Optimization**: Precomputes rewards and prunes low-probability transitions

## Algorithm Details

### Core Components

1. **Belief Update**:

   ```
   belief'(s') = Σ_s belief(s) × P(s' | s, action)
   ```
2. **Reward Calculation**:

   ```
   R(s, t) = μ + (r_initial(s) - μ) × cos(2πt/L)
   ```
3. **Value Evaluation**:

   ```
   V(belief, t, action) = E[reward] - penalty + E[V(belief', t+1)]
   ```

## Implementation Highlights

- **Efficient Memory Management**: Static allocation for competition constraints
- **Numerical Stability**: Epsilon thresholding for floating-point comparisons
- **Performance Tuning**: Adaptive lookahead depth based on problem complexity
- **Modular Design**: Separated input parsing, belief updates, and policy computation

## Input Format

```
T N M L Type
r_0 r_1 ... r_{N-1}
penalty_matrix[N][N]
transition_matrix_0[N][N]
transition_matrix_1[N][N]
...
transition_matrix_{M-1}[N][N]
```

Where:

- **T**: Total time steps
- **N**: Number of states
- **M**: Number of actions
- **L**: Wandering cycle period (0 = no wandering)
- **Type**: Penalty type (0 = none, 1 = matrix-based)

## Output Format

```
[action_0, action_1, ..., action_{T-1}]
```

## Compilation and Execution

```bash
# Compile
gcc -o solution coding-challenge-2025.c -lm -O2

# Run
./solution < input.txt
```

For Windows:

```cmd
cl coding-challenge-2025.c /O2
coding-challenge-2025.exe < input.txt
```

## Algorithm Complexity

- **Space**: O(T × N × M) for DP table (exact method)
- **Time**:
  - Exact: O(T × N² × M²)
  - Heuristic: O(T × N² × M × d) where d is lookahead depth

## Key Design Decisions

1. **Hybrid Approach**: Balances optimality and computational efficiency
2. **Belief Representation**: Maintains full probability distribution for accuracy
3. **Adaptive Depth**: Automatically adjusts search depth based on problem size
4. **Precomputation**: Caches time-varying rewards for O(1) lookup

## Testing Considerations

The solution handles:

- Problems with and without wandering rewards
- Problems with and without action penalties
- Various problem sizes (small to large)
- Numerical stability with normalization and epsilon thresholding
