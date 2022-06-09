# Building and Testing locally
For now, please refer to `build_tools/build_source.sh`. TODO

# Releasing
First we create release files through github actions.
Then we will test them and finally upload them to PYPI

#### Creating Release files
1. Increase the version number in `CMakeLists.txt` as well.
2. Tag the same version `v*` on github or locally and push.
3. Wait for the github actions to build all the wheels and sdist.
4. Download these release files to some folder we will call `./release`

#### Checking installation worked
```bash
pip install ./release/pyrfr-<XXX>.tar.gz
python -c "import pyrfr"  # Should import with no problem
```

You could also try running the tests. TODO

#### Uploading to PYPI
```bash
pip install twine

# Make sure ./release has all the wheels and tar.gz you wish to push

# Make sure at this point you updated the version in CMakeLists.txt
twine upload ./release/*
```
