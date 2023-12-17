# NOTE: on MacOS you need to add an addition flag: -undefined dynamic_lookup
default:
	c++ -O3 -Wl,-undefined,dynamic_lookup -shared -std=c++11 -fPIC $$(python3 -m pybind11 --includes) src/simple_ml_ext.cpp -o src/simple_ml_ext.so

test:
	c++ -std=c++11 -Wl,-undefined,dynamic_lookup $$(python3 -m pybind11 --includes) -o test src/simple_ml_ext.cpp src/test_simple_ext.cpp
