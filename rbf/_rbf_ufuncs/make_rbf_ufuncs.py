'''
This script is used to write the code for some RBF ufuncs so that they can be
compiled at build-time rather than at run-time. It requires `rbf` to already be
installed.
'''
def power_string(base, exponent):
    'converts `base**exponent` to `base*base*base...`'
    return '(%s)' % '*'.join(base for _ in range(int(exponent)))
    
if __name__ == '__main__':
    import os
    import re
    import importlib
    import shutil
    import json
    from itertools import product
    from rbf.basis import _PREDEFINED

    # create precompiled functions for each of these derivatives for each
    # predefined RBF.
    diffs = [(0,), (0, 0), (0, 0, 0)]

    # clean up the current directory, removing all but this file
    for fn in os.listdir('.'):
        if fn != __file__:
           os.remove(fn)

    meta = []
    for (inst_name, inst), diff in product(_PREDEFINED.items(), diffs):
        inst._add_diff_to_cache(diff, tempdir='.')
        func = inst._cache[diff]
        # Get the name of the module containing `func` and the name of `func`
        # in that module. Oddly, `func.__name__` returns the module name.
        mod_name = func.__name__
        mod = importlib.import_module(mod_name)
        func_name, = (k for k, v in vars(mod).items() if v is func)
        index = mod_name.split('_')[-1]
        sources = [
            'rbf/_rbf_ufuncs/wrapper_module_%s.c' % index,
            'rbf/_rbf_ufuncs/wrapped_code_%s.c' % index
            ]

        meta += [
            {
                'rbf': inst_name,
                'diff': diff,
                'module': 'rbf._rbf_ufuncs.%s' % mod_name,
                'function': func_name,
                'sources': sources
                }
            ]

    for fn in os.listdir('.'):
        if fn.endswith('.so'):
            os.remove(fn)

    os.remove('setup.py')
    shutil.rmtree('build')
    with open('metadata.json', 'w') as f:
        json.dump(meta, f, indent=4)

    # Make wrapped_code_*.c more efficient by storing r and r**2 as variables
    for fn in os.listdir('.'):
        if not (fn.startswith('wrapped_code') and fn.endswith('.c')):
            continue

        with open(fn, 'r') as f:
            code = f.read()

        nvars = len(re.search('double autofunc0\((.*)\)', code).groups()[0].split(','))
        ndim = int((nvars - 1)/2)
        for i in range(ndim):
            code = code.replace('-c%d + x%d' % (i, i), 'd%d' % i)
            
        r2_code = ' + '.join('pow(d%d, 2)' % i for i in range(ndim))
        code = code.replace(r2_code, 'r2')
        r_code = 'sqrt(r2)'
        code = code.replace(r_code, 'r')
        old_decs = '   double autofunc0_result;'
        new_decs = '   double autofunc0_result;\n'
        for i in range(ndim):
            new_decs += '   double d%d = x%d - c%d;\n' % (i, i, i)
            
        new_decs += '   double r2 = %s;\n' % r2_code
        new_decs += '   double r = %s;' % r_code
        code = code.replace(old_decs, new_decs)

        # other hard coded optimizations...        
        code = code.replace('pow(r2, 3.0/2.0)', 'pow(r, 3)')
        code = code.replace('pow(r2, 5.0/2.0)', 'pow(r, 5)')
        code = code.replace('pow(r2, 7.0/2.0)', 'pow(r, 7)')
        code = re.sub('pow\((\w+), (\d)\)', lambda x: power_string(*x.groups()), code)
        with open(fn, 'w') as f:
            f.write(code)
