#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
relython - a Python library for structural reliability calculations

Mutahar Chalmers, 2015

"""

from sys import argv
import os

import finout as io       # Input/output functions
import solvers            # Solvers
import ftrans as tr       # Transformations

__version__ = '0.1'


def relython(inp):
    """
    Main function.
    """

    # Validate input
    err = io.validinp(inp)

    # Proceed if there are no validation errors
    if len(err) == 0:
        # Parse and prepare input
        print('Parsing input...')
        g, xdists, cmat_x = io.parseinp(inp) 
        # Convert correlation matrix from X- to U-spaceA
        print('Converting correlation matrix to U-space...')
        cmat_u = tr.corrmat_u(cmat_x, xdists)
        # Generate transformation matrix from standard to correlated U-space
        print('Generating transformation from standard to correlated U...')
        T = tr.utrans(cmat_u)
        # Direct calculation methods
        print('Solving...')
        if inp['solver'].upper() == 'HLRF':
            result = solvers.hlrf(g, xdists, tr.u_to_x, T, inp)
        elif inp['solver'].upper() == 'SLSQP':
            result = solvers.slsqp(g, xdists, tr.u_to_x, T, inp)
        # Monte Carlo simulation methods
        elif inp['solver'].upper() == 'CMC':
            result = solvers.cmc(g, xdists, tr.u_to_x, T, inp)
        elif inp['solver'].upper() == 'ISMC':
            #result = solvers.ismc(g, xdists, tr.u_to_x, T, inp) 
            print ('Importance Sampling not yet implemented\n')
            return None
        elif inp['solver'].upper() == 'DSIM':
            #result = solvers.dsim(g, xdists, tr.u_to_x, T, inp)
            print('Directional Simulation not yet implemented\n')
            return None
        else:
            print('Invalid/unsupported solver: {0}\n'.format(inp['solver']))
            return None

        if inp['showresults'] == 1:
            print(io.pprint(inp, cmat_x, cmat_u, result))

        print('Complete.\n')
        return result
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
