# -*- coding: utf-8 -*-
"""
# Author: liora1018@gmail.com
# Based on a skeleton provided by: ofersh@telhai.ac.il
"""

import numpy as np
import random

from MixedVariableObjectiveFunctions import setC
import MixedVariableObjectiveFunctions as f_mixed
import ellipsoidFunctions as Efunc


def weighted_random_choice(max_value):
    """
    Selects a random integer from the range [0, max_value] with a higher probability for larger values.

    Args:
        max_value (int): The maximum possible value to choose from.

    Returns:
        int: A randomly selected integer from [0, max_value], weighted towards larger values.
    """
    weights = [k + 1 for k in range(max_value + 1)]  # Create weights: [1, 2, 3, ..., i+1]
    return random.choices(range(max_value + 1), weights=weights)[0]  # Randomly choose based on weights


def small_step(sign, p_sign=2, p_not_sign=1):
    """
    Generates a small step in the integer space based on the current direction.

    Args:
        sign (int): The direction of movement (-1, 0, or 1).
        p_sign (float, optional): Probability weight for continuing in the same direction. Default is 2.
        p_not_sign (float, optional): Probability weight for switching direction. Default is 1.

    Returns:
        int: A step of -1, 0, or 1 based on weighted probability.
    """
    if sign == 0:
        return random.choices([-1, 0, 1], weights=[1, 2, 1])[0]
    else:
        return random.choices([sign, 0, -sign], weights=[p_sign, (p_sign + p_not_sign) / 2, p_not_sign])[0]


def large_step(max_step_size, dist):
    """
    Generates a large step in the integer space, ensuring movement in the correct direction.

    Args:
        max_step_size (int): The maximum step size allowed.
        dist (float): The current distance to the target, used for it's sign only.

    Returns:
        int: A step size within [-max_step_size, max_step_size], weighted to favor larger movements.
    """
    return weighted_random_choice(max_step_size) * np.sign(dist)


class Particle(object):
    """
    Represents a single particle in the swarm, maintaining its position, velocity, and best-known solutions.

    Attributes:
        n (int): The number of dimensions in the search space.
        positionfloats (np.ndarray): The current position in the floating-point subspace.
        positionints (np.ndarray): The current position in the integer subspace.
        velocityfloats (np.ndarray): The velocity in the floating-point subspace.
        velocityints (np.ndarray): The velocity in the integer subspace.
        x_pbest (np.ndarray): The best position found by this particle.
        f_pbest (float): The function value at the best position found.
        fval (float): The current function value.
    """

    def __init__(self, x0, seed=None):
        """
        Initializes a particle with a given initial position.

        Args:
            x0 (np.ndarray): The initial position of the particle.
            seed (int, optional): Random seed for reproducibility.
        """
        self.local_state = np.random.RandomState(seed)
        self.n = len(x0)
        self.positionfloats = np.copy(x0[:self.n // 2])
        self.positionints = np.copy(np.round(x0[self.n // 2:]))
        self.velocityfloats = self.local_state.uniform(size=self.n // 2) * 2 - 1
        self.velocityints = np.random.choice([1, 0, -1], size=self.n // 2, p=[1 / 3, 1 / 3, 1 / 3])
        self.x_pbest = np.empty([self.n])
        self.f_pbest = np.inf
        self.fval = np.inf

    def comb_pos(self):
        """
        Combines the floating-point and integer position components into a single vector.

        Returns:
            np.ndarray: The full position vector (floats first, then integers).
        """
        arr1 = np.array(self.positionfloats, dtype=float)
        arr2 = np.array(self.positionints, dtype=float)
        return np.concatenate((arr1, arr2))

    def evaluate(self, objFunc):
        """
        Evaluates the particle's objective function value at its current position.
        Updates the personal best if necessary.

        Args:
            objFunc (function): The objective function to evaluate.
        """
        x = self.comb_pos()
        f_x = (x.reshape(1, -1))
        self.fval = objFunc(f_x)
        if self.fval < self.f_pbest:
            self.x_pbest = np.copy(x)
            self.f_pbest = self.fval

    def update_position(self, lb, ub):
        """
        Updates the particle's position based on its velocity, enforcing boundary constraints.

        Args:
            lb (float): Lower bound of the search space.
            ub (float): Upper bound of the search space.
        """
        self.positionfloats = np.clip(self.positionfloats + self.velocityfloats, lb, ub)
        self.positionints = np.clip(self.positionints + self.velocityints, lb, ub)
        # np.clip() enforces boundary conditions

    # def update_ints_velocity(self, pos_best_g, c1, c2, max_step_size):
    #     """
    #     Updates the velocity of the integer part of the particle's position.
    #
    #     Args:
    #         pos_best_g (np.ndarray): The global best position.
    #         omega (float): The inertia weight (not directly used in the integer update).
    #         c1 (float): The cognitive coefficient, influencing the personal best step.
    #         c2 (float): The social coefficient, influencing the global best step.
    #         max_step_size (int): The maximum allowed step size for integer updates.
    #
    #     Description:
    #     - Loops through the integer dimensions (second half of the particle's vector).
    #     - Computes distances from the particle's personal best and from the global best.
    #     - If both distances are within `max_step_size`, performs a small step.
    #     - Otherwise:
    #       - Determines step size separately for `pbest` and `gbest`, choosing between small and large steps based on their distances.
    #       - Selects between the two step sizes probabilistically, using `random.choices()` with weights `c1` and `c2`.
    #     - Clips the final velocity to stay within `[-max_step_size, max_step_size]`.
    #     """
    #     # r1 - variable for personal best, r2 - variable for global best
    #     for d in range(self.n // 2):
    #         dist_pbest = self.x_pbest[self.n // 2 + d] - self.positionints[d]
    #         dist_gbest = pos_best_g[self.n // 2 + d] - self.positionints[d]
    #         # if both are close --> take steps of sizes [-1,0,1] + minimized prev velocity
    #         if (abs(dist_pbest) <= max_step_size and abs(dist_gbest) <= max_step_size):
    #             step_size = small_step(np.sign(dist_pbest), c1, c2)
    #             self.velocityints[d] = round(self.velocityints[d] / (max_step_size / 2)) + step_size
    #         # Otherwise, calculate the step sizes for both the global and personal steps, then select between them based on the percentages determined by r1 and r2
    #         if (np.sign(dist_pbest) == np.sign(dist_gbest)):
    #             p_step, g_step = 0, 0
    #             if (dist_pbest <= max_step_size):
    #                 p_step = small_step(np.sign(dist_pbest))
    #             else:
    #                 p_step = large_step(max_step_size, dist_pbest)
    #             if (dist_gbest <= max_step_size):
    #                 g_step = small_step(np.sign(dist_gbest))
    #             else:
    #                 p_step = large_step(max_step_size, dist_gbest)
    #             step_size = random.choices([p_step, g_step], weights=[c1, c2])[0]
    #             self.velocityints[d] = np.clip(step_size + self.velocityints[d], -max_step_size, max_step_size)
    def update_ints_velocity(self, pos_best_g):
        dist_pbest = np.sign(self.x_pbest[self.n // 2:] - self.positionints)
        dist_gbest = np.sign(pos_best_g[self.n // 2:] - self.positionints)
        self.velocityints = np.clip(self.velocityints + dist_pbest + dist_gbest, -1, 1)

    # def update_floats_velocity(self, pos_best_g, c1, c2, omega):
    #     r1 = self.local_state.uniform(size=self.n // 2)  # n-dimensional random vector ~U(0,1)
    #     r2 = self.local_state.uniform(size=self.n // 2)  # n-dimensional random vector ~U(0,1)
    #     # In the following, we use element-wise numpy array multiplication on r_i
    #     # NumPy performs operations element-by-element, so multiplying 2D arrays with * is not a matrix multiplication – it’s an element-by-element multiplication.
    #     pbest = (self.x_pbest)
    #     vel_cognitive = r1*(pbest[:self.n // 2] - self.positionfloats)
    #     vel_social = r2*(pos_best_g[:self.n // 2] - self.positionfloats)
    #     self.velocityfloats = omega*self.velocityfloats + c1*vel_cognitive + c2 * vel_social

    def update_floats_velocity(self, pos_best_g, c1, c2, omega, max_step_size):
        """
        Updates the velocity for the floating-point part of the position.

        Args:
            pos_best_g (np.ndarray): The global best position.
            c1 (float): The cognitive component weight.
            c2 (float): The social component weight.
            omega (float): The inertia weight.
            max_step_size (float): The maximum step size for floats.
        """
        r1 = self.local_state.uniform(size=self.n // 2)  # n-dimensional random vector ~U(0,1)
        r2 = self.local_state.uniform(size=self.n // 2)  # n-dimensional random vector ~U(0,1)
        # In the following, we use element-wise numpy array multiplication on r_i
        # NumPy performs operations element-by-element, so multiplying 2D arrays with * is not a matrix multiplication – it’s an element-by-element multiplication.
        vel_cognitive = r1 * (self.x_pbest[:self.n // 2] - self.positionfloats)
        vel_social = r2 * (pos_best_g[:self.n // 2] - self.positionfloats)
        self.velocityfloats = np.clip(omega * self.velocityfloats + c1 * vel_cognitive + c2 * vel_social,
                                      -max_step_size, max_step_size)
        # # Create occasional large steps to escape traps.
        # if np.random.rand() < 0.05:  # 5% chance of large movement
        #     self.velocityfloats += np.random.uniform(-0.5, 0.5, size=self.n // 2)


class ParticleSwarm(object):
    """
    Implements the Particle Swarm Optimization (PSO) algorithm for mixed-variable optimization.

    Attributes:
        n (int): The number of dimensions.
        lb (float): The lower bound of the search space.
        ub (float): The upper bound of the search space.
        num_particles (int): The number of particles in the swarm.
        maxEvals (int): The maximum number of function evaluations.
        objFunc (function): The objective function being optimized.
        swarm (list): List of Particle objects.
        f_best_g (float): The best global function value found.
        x_best_g (np.ndarray): The best global position found.
    """

    def __init__(self, n, lb, ub, num_particles, maxEvals, objFunc=lambda x: x.dot(x), seed=None):
        """
        Initializes the Particle Swarm Optimization (PSO) algorithm.

        Args:
            n (int): Number of dimensions.
            lb (float): Lower bound of search space.
            ub (float): Upper bound of search space.
            num_particles (int): Number of particles in the swarm.
            maxEvals (int): Maximum number of function evaluations.
            objFunc (function): The objective function.
            seed (int, optional): Random seed.
        """
        self.n = n
        self.f_best_g = np.inf
        self.local_state = np.random.RandomState(seed)
        self.x_best_g = np.array(self.local_state.uniform(size=n) * 200 - 100)  # best position for swarm
        self.maxEvals = maxEvals
        self.objFunc = objFunc
        self.lb = lb
        self.ub = ub
        self.num_particles = num_particles
        # construct the swarm
        self.swarm = []
        # default parameters
        self.omega = 0.7298  # default inertia weight
        self.c1 = 3  # 1.49618  # default cognitive coefficient
        self.c2 = 1.49618  # default social coefficient
        self.max_step_size = 5

    def get_gbest_from_neighbors(self, k):
        """
        Finds the best personal best (`pbest`) solution among a particle's two immediate neighbors.

        Args:
            k (int): The index of the current particle in the swarm.

        Returns:
            np.ndarray: The best `pbest` position among the two neighbors.
        """
        kminus1 = self.num_particles - 1 if k == 0 else k - 1
        kplus1 = 0 if k == self.num_particles - 1 else k + 1
        neighbors = [self.swarm[kminus1], self.swarm[kplus1]]
        return max(neighbors, key=lambda p: p.f_pbest).x_pbest

    def run(self, seed=None):
        """
        Runs the PSO algorithm until the evaluation budget is exhausted.

        Args:
            seed (int, optional): Random seed.

        Returns:
            tuple: (Best position found, Best function value, Improvement over initial best).
        """
        self.local_state = np.random.RandomState(seed)
        self.x0 = self.local_state.uniform(size=self.n) * (self.ub - self.lb) + self.lb
        for k in range(self.num_particles):
            self.swarm.append(Particle(self.x0, seed + k + 1))  # different seeds
        c = 0
        fstart = -1
        while c <= self.maxEvals:
            # # Starts high for exploration and lowers for fine-tuning
            # self.omega = 0.9 - (0.6 * c / self.maxEvals)
            # Cycle through particles in swarm and evaluate fitness
            for k in range(0, self.num_particles):
                self.swarm[k].evaluate(self.objFunc)
                if self.swarm[k].fval < self.f_best_g:
                    self.x_best_g = np.copy(self.swarm[k].comb_pos().flatten())
                    self.f_best_g = self.swarm[k].fval
                    # print(c, ": ", self.f_best_g)
                    if fstart == -1:
                        fstart = self.f_best_g
            c += self.num_particles
            # cycle through swarm and update velocities and position
            for k in range(0, self.num_particles):
                # print(self.x_best_g)
                self.swarm[k].update_floats_velocity(self.x_best_g, self.c1, self.c2, self.omega, self.max_step_size)
                self.swarm[k].update_ints_velocity(self.x_best_g)
                # # Second option for circular neighbors option
                # best_g_neighbors = self.get_gbest_from_neighbors(k)
                # self.swarm[k].update_floats_velocity(best_g_neighbors, self.c1, self.c2, self.omega, self.max_step_size)
                # self.swarm[k].update_ints_velocity(best_g_neighbors, self.omega, self.c1, self.c2, self.max_step_size)
                # if k == 0 and np.mod(c, int(self.maxEvals / 10)) == 0:
                #     print(c, "velocity: ", self.swarm[k].velocityfloats, "\n", self.swarm[k].velocityints)
                self.swarm[k].update_position(self.lb, self.ub)

            # print intermediate results
            # if np.mod(c, int(self.maxEvals / 10)) == 0:
            #     print(c, ": ", self.f_best_g)  # ," at position ",self.x_best_g)
        # end primary while loop
        diff = fstart - self.f_best_g
        return self.x_best_g, self.f_best_g, diff


if __name__ == "__main__":
    n = 64
    lb, ub = -100, 100
    dim = 64
    N = dim // 2
    c = 100
    setC(N)
    Nruns = 10
    objFunc = "MixedVarsEllipsoid"
    budget = 1e5
    # for index, funcName in enumerate(['genHcigar', 'genRotatedHellipse', 'genHadamardHellipse']):
    for index, funcName in enumerate(['genHadamardHellipse']):
        x_best_of_all = 0
        f_best_of_all = np.inf
        H = eval(f'Efunc.{funcName}')(dim, c)
        f = eval(f'f_mixed.{objFunc}')(d=dim, bid=0, ind=N, H=H, c=c, max_eval=budget)
        for i in range(Nruns):
            pso = ParticleSwarm(n, lb, ub, num_particles=100, maxEvals=budget, objFunc=f)
            xbest, fbest, diff = pso.run(seed=i)
            print("Run number", i, funcName, "value", fbest)
            if fbest < f_best_of_all:
                f_best_of_all = fbest
                x_best_of_all = xbest
        print("Best function", funcName, "value", f_best_of_all, " at location ", x_best_of_all)
