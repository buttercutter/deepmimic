BIN=Main
OBJ_DIR = objs
MKDIR = mkdir -p

# Reminder: these variables must be set as environment variables if the corresponding dev libs are not installed in the system: eg:
# export EIGEN_DIR = ../../libraries/eigen
# export BULLET_INC_DIR = ../../libraries/bullet3/src
# export BULLET_INC_DIR = third/bullet3-2.88/build_cmake/install/include/bullet/
# export BULLET_LIB_DIR = third/bullet3-2.88/build_cmake/install/lib
# export GLEW_INC_DIR = third/glew-2.1.0/install/usr/include/
# export GLEW_LIB_DIR = third/glew-2.1.0/lib/
# export FREEGLUT_INC_DIR = third/freeglut-3.0.0/install/include/
# export FREEGLUT_LIB_DIR = third/freeglut-3.0.0/install/lib

# Set library directories to /usr/lib/ where you found them
BULLET_LIB_DIR=/usr/lib
GLEW_LIB_DIR=/usr/lib  
FREEGLUT_LIB_DIR=/usr/lib
PYTHON_LIB=/usr/lib/python3.13

# Keep the include directories as before
EIGEN_DIR=/usr/include/eigen3
BULLET_INC_DIR=/usr/include/bullet
GLEW_INC_DIR=/usr/include
FREEGLUT_INC_DIR=/usr/include
PYTHON_INC=/usr/include/python3.13

PYTHON_INC = /usr/include/python3.13  #../libs/Python-3.7.16/local/include/python3.7m
PYTHON_LIB = /usr/lib/python3.13  #../libs/Python-3.7.16/local/lib -lpython3.7m

INC = -I./ \
	-I$(EIGEN_DIR) \
	-I$(BULLET_INC_DIR) \
	-I$(GLEW_INC_DIR) \
	-I$(FREEGLUT_INC_DIR) \
	-I$(BULLET_INC_DIR)

LIBDIRS = -L$(GLEW_LIB_DIR) -L$(FREEGLUT_LIB_DIR) -L$(BULLET_LIB_DIR)
LIBS = -lGLEW -lGL -lGLU -lglut -lBulletDynamics -lBulletCollision -lLinearMath -lm -lstdc++

CC = clang++
CFLAGS = -std=c++11 -O3 -Wall -fPIC 

SRC_DIRS = util/ \
	util/json/ \
	anim/ \
	sim/ \
	render/ \
	render/lodepng/ \
	scenes/ \
	scenes/arm/ \
	scenes/particle/ \
	scenes/pendulum/

SRC_FILES = $(BIN).cpp DeepMimicCore.cpp $(foreach dir, $(SRC_DIRS), $(wildcard $(dir)*.cpp))
OUT_DIRS = $(addprefix $(OBJ_DIR)/, $(SRC_DIRS))
OBJ = $(addprefix $(OBJ_DIR)/, $(patsubst %.cpp, %.o, $(SRC_FILES)))


default: build

build: $(BIN)

print:
	$(OUT_DIRS)

python: $(OUT_DIRS) $(OBJ)
	swig -c++ -python DeepMimicCore.i
	$(CC) -c -g $(CFLAGS) $(INC) $(LIBS) DeepMimicCore.cpp DeepMimicCore_wrap.cxx -I$(PYTHON_INC)
	$(CC) -shared $(CFLAGS) $(OBJ) $(LIBS) DeepMimicCore_wrap.o -o _DeepMimicCore.so -L$(PYTHON_LIB) $(LIBDIRS)

$(OBJ_DIR)/%.o: %.cpp
	@echo "Creating directory: $(dir $@)"
	@mkdir -p $(dir $@)
	@echo "Directory created, now compiling: $< -> $@"
	@echo "Checking if directory eists:"
	@ls -la $(dir $@)
	#$(CC) -c -g $(CFLAGS) $(INC) -o $@ $<
	clang++ -c -g -std=c++11 -O3 -Wall -fPIC -I./ -I/usr/include/eigen3 -I/usr/include/bullet -I/usr/include -o $@ $<

$(BIN): $(OUT_DIRS) $(OBJ)
	$(CC) -g $(CFLAGS) $(OBJ) $(LIBS) -o $@

$(OUT_DIRS):
	$(MKDIR) $(OUT_DIRS)
	
clean:
	- rm -f -r $(OUT_DIRS) $(BIN) $(OBJ) $(OBJ_DIR) DeepMimicCore_wrap.cxx _DeepMimicCore.so
