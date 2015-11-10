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

__version__ = '0.2'


def relython(inp):
    """
    Main function.
    """

    # Validate input
    err = io.validinp(inp)

    # Proceed if there are no validation errors
    if len(err) == 0:
        results = []
        # Parse and prepare input
        print('Parsing input...')
        g, xdists, cmat_x = io.parseinp(inp) 
        inp['xdists'] = xdists # So can show mus and sigs in pprint
        # Convert correlation matrix from X- to U-space
        print('Converting correlation matrix to U-space...')
        cmat_u = tr.corrmat_u(cmat_x, xdists)
        # Generate transformation matrix from standard to correlated U-space
        print('Generating transformation from standard to correlated U...')
        T = tr.utrans(cmat_u)
        for i, slvr in enumerate(inp['solver']):
            print('Solving: {0}...'.format(slvr))
            # Direct estimation methods
            if slvr.upper() == 'HLRF':
                results.append(sl.hlrf(g, xdists, tr.u_to_x, T, inp['maxitr'][i],
                                       inp['tol'], inp['ftol'], inp['eps']))
            elif slvr.upper() == 'SLSQP':
                results.append(sl.slsqp(g, xdists, tr.u_to_x, T,inp['maxitr'][i],
                                        inp['tol'], inp['ftol'], inp['eps'])) 
            # Monte Carlo simulation methods
            elif slvr.upper() == 'CMC':
                results.append(sl.cmc(g, xdists, tr.u_to_x, T, inp['seed'], 
                                      inp['maxitr'][i]))
            elif slvr.upper() == 'ISMC':
                results.append(sl.ismc(g, xdists, tr.u_to_x, T, inp['seed'], 
                                       inp['maxitr'][i], inp['tol'], inp['ftol'],
                                       inp['eps'])) 
            elif slvr.upper() == 'DSIM':
                results.append(sl.dsim(g, xdists, tr.u_to_x, T, inp['seed'], 
                                       inp['maxitr'][i], inp['tol'], inp['ftol'],
                                       inp['eps'])) 
            else:
                continue

        if inp['showresults'] == 1:
            print(io.pprint(inp, cmat_x, cmat_u, results))

        print('Complete.\n')
        return results
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
        #try:
        input_dict = io.loadfile(os.path.abspath(input_file))
        relython(input_dict)
        #except Exception as e:
        #    print('* Error processing {0}\n{1}'.format(os.path.join(path, input_file), e))
        #    continue
# Imported by other module or interpreter 
else:
    pass
    #print('relython v{0}'.format(__version__))
