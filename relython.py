#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
relython - a Python library for structural reliability calculations

Mutahar Chalmers, 2015

"""

from sys import argv
import os

import finout as io       # Input/output functions
import solvers as sl      # Solvers
import ftrans as tr       # Transformations

__version__ = '0.21'
__author__  = 'Mutahar Chalmers'


def relython(inp):
    """
    Main function.
    """

    # Validate input
    err = io.validinp(inp)

    # Proceed if there are no validation errors
    if len(err) == 0:
        res = {} 
        # Parse and prepare input
        print('Parsing input...')
        gfuncs, xdists, cmat_x = io.parseinp(inp) 
        inp['xdists'] = xdists # So can show mus and sigs in pprint

        # Convert correlation matrix from X- to U-space
        print('Converting correlation matrix to U-space...')
        cmat_u = tr.corrmat_u(cmat_x, xdists)
        # Generate transformation matrix from standard to correlated U-space
        print('Generating transformation from standard to correlated U...')
        T = tr.utrans(cmat_u)

        for i, g in enumerate(gfuncs):
            res[i] = [] 
            for j, slvr in enumerate(inp['solver']):
                print('Solving: {0}...'.format(slvr))
                # Direct estimation methods
                if slvr.upper() == 'HLRF':
                    res[i].append(sl.hlrf(g, xdists, tr.u_to_x, T, 
                                          inp['maxitr'][j], inp['tol'], 
                                          inp['ftol'], inp['eps']))
                elif slvr.upper() == 'IHLRF':
                    res[i].append(sl.ihlrf(g, xdists, tr.u_to_x, T, 
                                           inp['maxitr'][j], inp['tol'], 
                                           inp['ftol'], inp['eps']))
                elif slvr.upper() == 'SLSQP':
                    res[i].append(sl.slsqp(g, xdists, tr.u_to_x, T, 
                                           inp['maxitr'][j], inp['tol'], 
                                           inp['ftol'], inp['eps'])) 
                # Monte Carlo simulation methods
                elif slvr.upper() == 'CMC':
                    res[i].append(sl.cmc(g, xdists, tr.u_to_x, T, inp['seed'], 
                                         inp['maxitr'][j]))
                elif slvr.upper() == 'ISMC':
                    res[i].append(sl.ismc(g, xdists, tr.u_to_x, T, inp['seed'],
                                          inp['maxitr'][j], inp['tol'], 
                                          inp['ftol'], inp['eps']))
                elif slvr.upper() == 'DSIM':
                    res[i].append(sl.dsim(g, xdists, tr.u_to_x, T, inp['seed'], 
                                          inp['maxitr'][j], inp['tol'], 
                                          inp['ftol'], inp['eps'])) 
                else:
                    continue

        output = io.pprint(inp, cmat_x, cmat_u, res, __version__, __author__)
        if 'outpath' in inp.keys():
            try:
                with open(inp['outpath'], 'w') as f:
                    f.write(output)
            except:
                print('\nUnable to write to {0}\n'.format(inp['outpath']))
        print(output)
        return res
    else:
        if inp['showresults'] == 1:
            print('\nErrors:\n{0}'.format('\n'.join(err)))
        return err


# Called with no arguments - process all files in input directory
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'input')
if (len(argv) == 1) and (__name__ == '__main__'):              
    # Read all files in the input directory which could be input files
    input_files = [os.path.join(path, fname) for fname in os.listdir(path)]
    for input_file in input_files:
        print('\nProcessing {0}...'.format(input_file))
        try:
            input_dict = io.loadfile(input_file)
            relython(input_dict)
        except Exception as e:
            print('* Error processing {0}'.format(os.path.join(path, input_file), e))
            continue
# Called with arguments
elif  __name__ == '__main__':
    for input_file in argv[1:]:
        print('\nProcessing {0}...'.format(input_file))
        try:
            input_dict = io.loadfile(os.path.abspath(input_file))
            relython(input_dict)
        except Exception as e:
            print('* Error processing {0}\n{1}'.format(os.path.join(path, input_file), e))
            continue
# Imported by other module or interpreter 
else:
    pass
    #print('relython v{0}'.format(__version__))
