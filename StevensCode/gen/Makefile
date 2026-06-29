CXX      = g++
CXXFLAGS = -std=c++14 -O2 $(shell pythia8-config --cxxflags)
LDFLAGS  = $(shell pythia8-config --ldflags)

all: gen_Zqq gen_Hgg

gen_Zqq: gen_Zqq.cc
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

gen_Hgg: gen_Hgg.cc
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f gen_Zqq gen_Hgg
