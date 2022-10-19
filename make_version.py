#!/usr/bin/env python
if __name__ == '__main__':
    import subprocess as sp
    import re
    from pathlib import Path

    version_file = Path('rbf/_version.py')
    version_info = {'__git_hash__':'',
                    '__version__':'0+unknown'}

    # if a `_version.py` file exists, then update `version_info` with its
    # content
    if version_file.exists():
        version_text = version_file.read_text()
        version_info.update(
            re.findall('(__[A-Za-z_]+__)\s*=\s*"([^"]+)"', version_text)
            )

    try:
        # Attempt to use git to get the version and hash
        git_hash = sp.check_output(['git', 'rev-parse', 'HEAD'])
        git_hash = git_hash.strip().decode()
        version_info['__git_hash__'] = git_hash

    except Exception:
        print('Unable to retrieve the current git hash')
        pass

    try:
        desc = sp.check_output(['git', 'describe', '--always', '--dirty'])
        desc_parts = desc.strip().decode().split('-')
        public_version = desc_parts[0]
        private_version = '.'.join(desc_parts[1:])
        if private_version:
            version = public_version + '+' + private_version
        else:
            version = public_version

        version_info['__version__'] = version

    except Exception:
        print('Unable to retrieve the current version from git')
        pass

    with version_file.open('w') as fb:
        fb.write('__git_hash__ = "%s"\n' % version_info['__git_hash__'])
        fb.write('__version__ = "%s"\n' % version_info['__version__'])
