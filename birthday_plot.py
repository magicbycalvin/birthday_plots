#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 09:01:48 2022

@author: magicbycalvin
"""

import matplotlib.pyplot as plt
import numpy as np

from polynomial.bernstein import Bernstein
from stoch_opt import OctopusSearch
from stoch_opt.constraint_functions import CollisionAvoidance, MaximumSpeed, MaximumAngularRate
from stoch_opt.cost_functions import SumOfDistance, DistanceToGoal, SumOfAcceleration
from utils import setRCParams, resetRCParams


def cpts_from_initial_state(x, xdot, xddot, t0, tf, n):
    x = x.flatten()
    xdot = xdot.flatten()
    xddot = xddot.flatten()

    ndim = x.shape[0]
    cpts = np.empty((ndim, 3))

    cpts[:, 0] = x
    cpts[:, 1] = xdot*((tf-t0) / n) + cpts[:, 0]
    cpts[:, 2] = xddot*((tf-t0) / n)*((tf-t0) / (n-1)) + 2*cpts[:, 1] - cpts[:, 0]

    return cpts


if __name__ == '__main__':
    plt.close('all')

    obstacles = [(4.5, 6.5), (5, 6.4), (5.5, 6.4),
                 (4.25, 5.1), (4.75, 5.1), (5.25, 5.1), (5.75, 4.9),
                 (4.75, 3.75), (5.25, 3.75)
                 ]
    obstacles = [np.array(i) for i in obstacles]

    ### Mission Parameters
    # rng stuff
    SEED = 3
    STD = 1
    exploration_prob = 0.4
    nchildren = 4
    max_iter = 3e4
    ndim = 2
    n = 7
    t0 = 0
    tf = 10
    safe_dist = 0.33
    goal_eps = 0.1
    maximum_speed = 10  # m/s
    maximum_angular_rate = np.pi/4  # rad/s

    # obs = np.array([8, 4], dtype=float)     # Obstacle position (m)
    goal = np.array([8.5, 5], dtype=float)  # Goal position (m)

    # Initial state of vehicle
    x = np.array([1.5, 5], dtype=float)
    xdot = np.array([0, 0], dtype=float)
    xddot = np.array([0, 0], dtype=float)
    cpts = np.empty((ndim, n+1))

    cpts[:, :3] = cpts_from_initial_state(x, xdot, xddot, t0, tf, n)
    cpts[:, 3:] = np.vstack([np.linspace(cpts[0, 2], goal[0], n-1)[1:],
                             np.linspace(cpts[1, 2], goal[1], n-1)[1:]])

    # Create a Bernstein polynomial of the initial trajectory
    c = Bernstein(cpts, t0=t0, tf=tf)
    # assert np.array_equal(c.cpts[:, 0], x), 'Initial position incorrect.'
    # assert np.array_equal(c.diff().cpts[:, 0], xdot), 'Initial velocity incorrect.'
    # assert np.array_equal(c.diff().diff().cpts[:, 0], xddot), 'Initial acceleration incorrect.'

    cost_fn = DistanceToGoal(goal) + SumOfAcceleration()
    constraints = [
        CollisionAvoidance(safe_dist, obstacles, elev=1000),
        # MaximumSpeed(maximum_speed),
        # MaximumAngularRate(maximum_angular_rate)
        ]
    octopus = OctopusSearch(cost_fn, [c], constraints=constraints,
                            random_walk_std=STD, rng_seed=SEED, nsamples=nchildren, exploration_prob=exploration_prob,
                            max_iter=max_iter)
    qbest, feasible_trajs, infeasible_trajs = octopus.solve(print_iter=True)

    ### Plot the results
    setRCParams()
    fig, ax = plt.subplots()

    # if len(infeasible_trajs) > 0:
    #     infeasible_trajs[0][1][0].plot(ax, showCpts=False, color='red', alpha=0.5, label='infeasible')
    #     for traj in infeasible_trajs[1:]:
    #         traj[1][0].plot(ax, showCpts=False, color='red', alpha=0.3, label=None)

    if len(feasible_trajs) > 0:
        feasible_trajs[0][1][0].plot(ax, showCpts=False, color='green', alpha=0.5, label='feasible')
        for traj in feasible_trajs[1:]:
            traj[1][0].plot(ax, showCpts=False, color='green', alpha=0.3, label=None)

    qbest[0].plot(ax, showCpts=False, color='blue', lw=5, label='best')
    # ax.plot(goal[0], goal[1], marker=(5, 1), color='purple', ms=50)

    plt.text(5, 5, 'Happy\nBirthday\nTim!', fontsize=86, linespacing=2,
             horizontalalignment='center', verticalalignment='center', fontweight='heavy')

    # for obs in obstacles:
    #     ax.add_patch(plt.Circle(obs, radius=0.4, zorder=2))

    # plt.legend(loc='upper left')
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])

    txt = ax.get_children()[0]

    plt.show()
    resetRCParams()
