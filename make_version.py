#!/usr/bin/env python
if __name__ == '__main__':
    import subprocess as sp
    import os
    package = 'rbf'
    try:
        # Attempt to use git to get the version and hash
        git_hash = sp.check_output(['git', 'rev-parse', 'HEAD'])
        git_hash = git_hash.strip().decode()
        version = sp.check_output(['git', 'describe', '--always', '--dirty'])
        version = version.strip().decode()
    except Exception as exc:
        # We are not able to directly retrieve the hash and version from git.
        # Check if a _version.py exists. If one exists, keep it as is but warn
        # the user that it is not synchronized with git. Otherwise, create a
        # _version.py and set the git hash and version to "UNKOWN"
        print(
            'Encountered the following error while retrieving the current git '
            'hash and version:\n%s\n' % repr(exc))
        if os.path.exists('%s/_version.py' % package):
            print(
                'The git hash and version will be determined by the contents '
                'of `_version.py`, which may not accurately describe the '
                'package if modifications have been made since `_version.py` '
                'was written.')
            quit()    
        else:
            print(
                'The git hash and version will both be set to "UNKNOWN"')
            git_hash = 'UNKNOWN'
            version = 'UNKNOWN'

    with open('%s/_version.py' % package, 'w') as build_file:
        build_file.write('__git_hash__ = "%s"\n' % git_hash)
        build_file.write('__version__ = "%s"\n' % version)
