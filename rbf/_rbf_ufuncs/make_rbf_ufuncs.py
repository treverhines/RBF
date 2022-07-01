'''
This script is used to write the code for some RBF ufuncs so that they can be
compiled at build-time rather than at run-time. It requires `rbf` to already be
installed.
'''
if __name__ == '__main__':
    import os
    import importlib
    import shutil
    import json
    from itertools import product
    from rbf.basis import _PREDEFINED

    # create precompiled functions for each of these derivatives for each
    # predefined RBF.
    diffs = [(0,), (0, 0), (0, 0, 0)]

    # clean up the current directory, removing all but this file
    for f in os.listdir('.'):
        if f != __file__:
           os.remove(f)

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

    for f in os.listdir('.'):
        if f.endswith('.so'):
            os.remove(f)

    os.remove('setup.py')
    shutil.rmtree('build')
    with open('metadata.json', 'w') as f:
        json.dump(meta, f, indent=4)

