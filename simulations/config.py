"""This file contains the settings for the simulation"""
import argparse

def general_settings():
    parser = argparse.ArgumentParser(prog = 'NUV-DoA',\
                                     description = 'Dataset and tuning parameters')
    
    ### Dataset settings
    parser.add_argument('--sample', type=int, default=10, metavar='dataset-size',
                        help='input dataset size')
    parser.add_argument('--k', type=int, default=3, metavar='k',
                        help='number of sources')
    parser.add_argument('--n', type=int, default=16, metavar='n',
                        help='number of ULA elements')
    parser.add_argument('--m', type=int, default=360, metavar='m',
                        help='number of grids')
    parser.add_argument('--m_SF', type=int, default=101, metavar='m_SF',
                        help='number of grids in each Spatial Filter window')
    parser.add_argument('--gap', type=float, default=15, metavar='gap',
                        help='gap between sources')
    parser.add_argument('--x_var', type=float, default=0.5, metavar='x_var',
                        help='variance of source signals')
    parser.add_argument('--mean_c', type=float, default=2, metavar='mean_c',
                        help='mean of source signals')
    parser.add_argument('--r2', type=float, default=1e-1, metavar='r2',
                        help='ground truth variance of observation noise')
    parser.add_argument('--l', type=int, default=100, metavar='l',
                        help='number of snapshots')
    parser.add_argument('--doa_gt_range', type=float, default=75, metavar='doa_gt_range',
                        help='range of ground truth DOA')
    parser.add_argument('--doa_gt_increment', type=float, default=0.01, metavar='doa_gt_increment',
                        help='increment of ground truth DOA')
    parser.add_argument('--B1', type=float, default=-89, metavar='left_boundary',
                        help='left boundary of all windows')
    parser.add_argument('--B2', type=float, default=89, metavar='right_boundary',
                        help='right boundary of all windows')


    ### Tuning settings
    parser.add_argument('--use_cuda', type=bool, default=False, metavar='CUDA',
                        help='if True, use CUDA')
    parser.add_argument('--q_init', type=float, default=0.01, metavar='q_init',
                        help='initial guess of q')
    parser.add_argument('--max_iterations', type=int, default=10000, metavar='max_iterations',
                        help='maximum number of iterations')
    parser.add_argument('--convergence_threshold', type=float, default=4e-4, metavar='convergence_threshold',
                        help='convergence threshold')
    parser.add_argument('--resol', type=float, default=0.05, metavar='resol',
                        help='resolution of spatial filter')
    

    args = parser.parse_args()
    return args
