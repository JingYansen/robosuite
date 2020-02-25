cd ../..
rm -rf build dist robosuite.egg-info
python setup.py build & sudo python setup.py install
cd robosuite/scripts