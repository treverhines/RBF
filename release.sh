git clean -xfd
python make_cython_extensions.py
python make_version.py
python -m build
twine upload -r pypi dist/*.tar.gz
