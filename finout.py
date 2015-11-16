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


def pprint(inp, cmat_x, cmat_u, res, version, author):
    """
    Pretty print input and results of reliability calculation.
    """

    vars_m = [var['name'] for var in inp['vars']]
    nvars = len(vars_m)
    
    header = '\n{0}\n= Relython v{1:4s}{2}=\n{0}'.format('='*79, version, ' '*62)
    pp = [header]
   
    # Summary table of Pf and betas (less detailed)
    fst3slvrs = inp['solver'][:3]
    summ_ = '\n Summary of results\n {0}\n'.format('-'*18)
    summ0 = '|     |' + '|'.join('{0:^22s}'.format(slv) for slv in fst3slvrs)+'|'
    summ1 = '| g_id|' + '|'.join('{0:^11s} {1:^10s}'.format('Pf', 'Beta') for slv in fst3slvrs)+'|'
    summ2 = '+-----+'+'+'.join('{0:22s}'.format('-'*22) for slv in fst3slvrs)+'+'
    summ = ['\n ' + '\n '.join((summ_, summ2, summ0, summ1, summ2)) + '\n ']
    for i in range(len(inp['g'])):
        summ3 = '|{0:^5d}|'.format(i)
        fst3slv = res[i][:3]
        summ4 = '|'.join('{0:11.4e} {1:10.6f}'.format(slv_i['Pf'], slv_i['beta']) for slv_i in fst3slv)
        summ.append(summ3 + summ4+'|\n ')
    summ.append(summ2)
    
    # Limit state functions
    gfunc_ = '\n\n Limit state functions\n {0}'.format('-'*22)
    gfunc = ['\n\n  g_id   Expression'+'\n '+'-'*18]
    for i in range(len(inp['g'])):
        gfunc.append('\n {0:^8d}'.format(i))
        gfunc.append(llfmt('g({0}) = {1}'.format(', '.join(vars_m), inp['g'][i])))
    
    # Print details of variables
    vs_ = '\n\n Variables\n {0}'.format('-'*9)
    vs = []
    vfmt = '\n\n {0:6s}{1:<24s}{2:>4s}{3:>10s}{4:>10s}{5:>10s}{6:>10s}\n {7}' 
    vs.append(vfmt.format('Var','Description','Dist','Param1','Param2','Mean','StdDev','-'*74))
    for i, var in enumerate(inp['vars']): 
        vparams = [var['params'][j] if j<len(var['params']) else None for j in range(2)]
        vs.append('\n {0:6s}{1:<24s}{2:>4s}'.format(var['name'], var['desc'], var['dist']))
        vs.append(''.join('{0:>10.2e}'.format(p) if p is not None else ' '*10 for p in vparams))
        if var['dist'] != 'cnst':
            vs.append('{0:10.2e}{1:10.2e}'.format(inp['xdists'][i].mu, inp['xdists'][i].sig))

    # Print correlation matrices
    cm = []
    if nvars <= 10: 
        cm.append('\n\n Correlation matrix in X-space\n {0}\n      '.format(29*'-'))
        cm.append(' '.join('{0:>7s}'.format(var['name']) for var in inp['vars']))
        for i, row in enumerate(cmat_x):
            cm.append('\n {0:6s}'.format(inp['vars'][i]['name']))
            cm.append(' '.join('{0:>-7.4f}'.format(elt) for elt in row))
    
        cm.append('\n\n Correlation matrix in U-space\n {0}\n      '.format(29*'-'))
        cm.append(''.join('{0:>7s}\''.format(var['name']) for var in inp['vars']))
        for i, row in enumerate(cmat_u):
            cm.append('\n {0:6s}'.format(inp['vars'][i]['name']+'\''))
            cm.append(' '.join('{0:>-7.4f}'.format(elt) for elt in row))
    else:
        cm.append('\n\n No correlation matrices shown - number of variables > 10')

    pp.extend([''.join(elem) for elem in [summ, gfunc, vs, cm]])

    # Detailed output 
    for i, output in res.items():           # Loop over g-functions
        for j in range(len(inp['solver'])):    # Loop over solvers
            if inp['solver'][j].upper() in ['HLRF', 'IHLRF', 'SLSQP']:
                subhead = '\n\n FORM results: g-func {0}\n {1}'.format(i, '-'*25)
            else:
                subhead = '\n\n MCS results: g-func {0}\n {1}'.format(i, '-'*25)
            dt = [subhead]
            dt.append('\n {0:14s}{1:11.6f}\n {2:14s}{3:11.4e}\n {4:14s}{5:>11s}\n {6:14s}{7:>11s}\n'.format(
                          'Beta:', output[j]['beta'], 'Pf:', output[j]['Pf'], 'Transform:', inp['transform'],
                          'Solver:', inp['solver'][j]))
            if inp['solver'][j].upper() in ['HLRF', 'IHLRF', 'SLSQP']:
                # FORM - additional info
                dt.append(' {0:14s}{1:11d}\n'.format('Iterations:', int(output[j]['nitr'])))
                dt.append(' {0:14s}{1:-11.4e}\n'.format('g(x*):', output[j]['g_beta']))
                dt.append(' {0:14s}{1:11.4e}\n'.format('Tolerance:', output[j]['tol']))
                dt.append('\n {0:6s} {1:>10s} {2:>10s} {3:>10s} {4:>10s}\n {5}'.format('Var','x*','u*','alpha','a**2(%)','-'*50))
                for k, var in enumerate(inp['vars']):
                   lineout = [var['name'], output[j]['x_beta'][k], output[j]['u_beta'][k], output[j]['alpha'][k], output[j]['alpha'][k]**2*100]
                   dt.append('\n {0:6s} {1:>10.3e} {2:>10.3e} {3:>10.3e} {4:>10.2f}'.format(*lineout))
            else:
                # Monte Carlo - additional info
                dt.append(' {0:14s}{1:11d}\n'.format('Iterations:', inp['maxitr'][j]))
                dt.append(' {0:14s}{1:>11.3e}\n {2:14s}{3:>11.0f}'.format('MC s.e. CoV:', output[j]['stdcv'], 'Seed:', inp['seed']))
            pp.append(''.join(dt)) 
    #pp.append((''.join(summ) + ''.join(gfunc) + ''.join(vs) + ''.join(cm)))
    pp.append('\n\n Calculation Notes\n {0}\n {1}'.format('-'*17, inp['notes']))
    pp.append('\n\n{0}\n'.format('='*79))
    return ''.join(pp)


def llfmt(longline, maxlen=68, shunt=9):
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
                lstr[spcix[i-1]] = '\n' + ' '*shunt
    return ''.join(lstr)
