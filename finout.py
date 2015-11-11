# -*- coding: utf-8 -*-

from json import load
import re
import os
from numpy import eye
import dists 


def loadfile(fpath):
    """
    Load text file input.
    """
    
    with open(fpath) as f: 
        inp = load(f)
    return inp


def validinp(inp):
    """
    Validate input.
    """
    
    modpath = os.path.dirname(os.path.realpath(__file__))
    errmsgs = []

    non_dists = [var['dist'] for var in inp['vars'] if var['dist']+'.py' not in
                 os.listdir(os.path.join(modpath, 'dists'))]
    if len(non_dists)>0:
        errmsgs.append('\n*** Unsupported distribution(s): {0}'.format(
                       ', '.join(unsupp_dists)))

    # g-function variables
    vars_g = [re.findall(r'\b[a-zA-Z]+[a-zA-Z0-9]*\b', gstr) for gstr in inp['g']] 
    vars_m = [var['name'] for var in inp['vars']]
    gvm = '\n*** Warning: undefined variables in g-function '
    gvarsmsg = [gvm + str(i+1) for i, vg in enumerate(vars_g) 
                if not set(vg).issubset(set(vars_m))]

    #if inp['solver'].upper() in ['CMC','ISMC','DSIM'] and 'seed' not in inp.keys():
    #    errmsgs.append("""\n*** Specify seed in input file for CMC, ISMC and DSIM solvers""") 
    return errmsgs


def parseinp(inp):
    """
    Parse input dict into useful objects for computation.
    """
    
    # Parse g-function(s)
    gfuncs = [str_to_g(gstring, inp) for gstring in inp['g']]
    
    # Generate list of distribution objects
    xdists = [dists.__dict__[var['dist']](*var['params']) for var in inp['vars']]

    # Generate correlation matrix in X-space
    cmat_x = corrmat_x(inp)

    return gfuncs, xdists, cmat_x


def str_to_g(gstring, inp):
    """
    Convert string literal g-function into full function.
    """
    
    ldict = locals()    # Python 3 fix
    args = ','.join([var['name'] for var in inp['vars']])
    fstr = 'def g(x):{0} = x; return {1}'.format(args, gstring)
    exec (fstr, locals())
    return ldict['g']   # Python 3 fix


def corrmat_x(inp):
    """
    Parse input data and generate correlation matrices in X-space.
    """

    # Generate correlation matrix in basic variable space
    cmat_x = eye(len(inp['vars']))
    for corr in inp['correlation']:
        for i, vari in enumerate(inp['vars']):
            for j, varj  in enumerate(inp['vars']):
                if corr[0] == vari['name'] and corr[1] == varj['name']:
                    cmat_x[i, j], cmat_x[j, i] = corr[2], corr[2]
    return cmat_x


def pprint(inp, cmat_x, cmat_u, outputs):
    """
    Pretty print input and results of reliability calculation.
    """

    vars_m = [var['name'] for var in inp['vars']]
    nvars = len(vars_m)
    
    header = '\n{0}\n={2}=\n= Relython analysis{3}=\n={2}=\n{1}'.format('='*79, '='*79, ' '*77, ' '*59)
    pp = [header]

    pp.append('\n\n Calculation Notes\n {0}\n {1}'.format('-'*17, inp['notes']))

    pp.append('\n\n {0:6s}{1:<27s}{2:>4s}{3:>10s}{4:>10s}{5:>10s}{6:>10s}\n {7}'.format('Var','Description','Dist','Param1','Param2','Mean','StdDev','-'*77))
    for i, var in enumerate(inp['vars']): 
        vparams = [var['params'][j] if j<len(var['params']) else None for j in range(2)]
        pp.append('\n {0:6s}{1:<27s}{2:>4s}{3}'.format(var['name'], var['desc'], var['dist'], ''.join('{0:>10.2e}'.format(p) if p is not None else ' '*10 for p in vparams)))
        pp.append('{0:10.2e}{1:10.2e}'.format(inp['xdists'][i].mu, inp['xdists'][i].sig))
    
    if nvars <= 10: 
        pp.append('\n\n Correlation matrix in X-space\n {0}\n      {1}'.format(29*'-', ''.join('{0:>8s}'.format(var['name']) for var in inp['vars'])))
        for i, row in enumerate(cmat_x):
            pp.append('\n {0:6s}'.format(inp['vars'][i]['name']) + ''.join('{0:>-8.4f}'.format(elt) for elt in row))
    
        pp.append('\n\n Correlation matrix in U-space\n {0}\n       {1}'.format(29*'-', ''.join('{0:>8s}'.format(var['name']+'\'') for var in inp['vars'])))
        for i, row in enumerate(cmat_u):
            pp.append('\n {0:6s}'.format(inp['vars'][i]['name']+'\'') + ''.join('{0:>-8.4f}'.format(elt) for elt in row))
    else:
        pp.append('\n\n No correlation matrices shown - number of variables > 10')

    # Solver output 
    pp.append('\n\n\n Limit state functions\n {0}'.format('-'*21))
    for i, output in enumerate(outputs):
        slvix = i % len(inp['solver'])
        gix = i // len(inp['solver'])
        
        pp.append(llfmt('\n\n g({0}) = {1}'.format(','.join(vars_m), inp['g'][gix])))

        if inp['solver'][slvix] in ['HLRF', 'SLSQP']:
            subhead = '\n\n FORM results\n {0}'.format('-'*12)
        else:
            subhead = '\n\n Monte Carlo results\n {0}'.format('-'*19)
        pp.append(subhead)
        pp.append('\n {0:14s}{1:11.6f}\n {2:14s}{3:11.4e}\n {4:14s}{5:>11s}\n {6:14s}{7:>11s}\n'.format(
                  'Beta:', output['beta'], 'Pf:', output['Pf'], 'Transform:', inp['transform'],
                  'Solver:', inp['solver'][slvix]))
        if inp['solver'][slvix] in ['HLRF', 'SLSQP']:
            # FORM - additional info
            pp.append(' {0:14s}{1:11d}\n'.format('Iterations:', int(output['nitr'])))
            pp.append(' {0:14s}{1:-11.4e}\n'.format('g(x*):', output['g_beta']))
            pp.append(' {0:14s}{1:11.4e}\n'.format('Tolerance:', output['tol']))
            pp.append('\n {0:6s} {1:>10s} {2:>10s} {3:>10s} {4:>10s}\n {5}'.format('Var','x*','u*','alpha','a**2(%)','-'*50))
            for j, var in enumerate(inp['vars']):
               lineout = [var['name'], output['x_beta'][j], output['u_beta'][j], output['alpha'][j], output['alpha'][j]**2*100]
               pp.append('\n {0:6s} {1:>10.3e} {2:>10.3e} {3:>10.3e} {4:>10.2f}'.format(*lineout))
        else:
            # Monte Carlo - additional info
            pp.append(' {0:14s}{1:11d}\n'.format('Iterations:', inp['maxitr'][slvix]))
            pp.append(' {0:14s}{1:>11.3e}\n {2:14s}{3:>11.0f}'.format('MC s.e. CoV:', output['stdcv'], 'Seed:', inp['seed']))
 
    pp.append('\n\n{0}\n'.format('='*79))
    return ''.join(pp)


def llfmt(longline, maxlen=78):
    """
    Format a long output line to fit within 79 character limit.
    """

    out = ''
    if len(longline) < maxlen:
        return longline
    else:
        lstr = list(longline)
        spcix = [i for i, char in enumerate(lstr) if char==' ']
        for i in range(1, len(spcix), 1):
            if spcix[i]%maxlen < spcix[i-1]%maxlen:
                lstr[spcix[i-1]] = '\n '
    return ''.join(lstr)
