try:
    from rbf._version import __git_hash__, __version__
except ImportError:
    print('Could not load the version information. This is likely because the '
          'package was not installed with setup.py')
    
