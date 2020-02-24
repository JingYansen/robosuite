cd ../..
rm -rf build dist robosuite.egg-info
python setup.py build & python setup.py install
cd robosuite/scripts