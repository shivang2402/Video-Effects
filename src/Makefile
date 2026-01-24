# OSX compiler
#CC = clang++

# OSX compiler
CC = clang++

CXX = $(CC)

# OSX include paths (for MacPorts)
#CFLAGS = -I/opt/local/include -I../include

# OSX include paths (for homebrew, probably)
CFLAGS = -std=c++17 -Wall -I/opt/homebrew/opt/opencv/include/opencv4 -I/opt/homebrew/opt/onnxruntime/include -I../include

# Dwarf include paths
#CFLAGS = -I../include # opencv includes are in /usr/include
CXXFLAGS = $(CFLAGS)

# OSX Library paths (if you use MacPorts)
#LDFLAGS = -L/opt/local/lib

#OSX Library paths (if you use homebrew, probably)
LDFLAGS = -L/opt/homebrew/opt/opencv/lib -L/opt/homebrew/opt/onnxruntime/lib -Wl,-rpath,/opt/homebrew/opt/onnxruntime/lib

# Dwarf Library paths
#LDFLAGS = -L/opt/local/lib/opencv4/3rdparty -L/opt/local/lib # opencv libraries are here

# opencv libraries
LDLIBS = -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lonnxruntime

BINDIR = ../bin

img: imgDisplay.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

vid: vidDisplay.o filters.o faceDetect.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f *.o *~
