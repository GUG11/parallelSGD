CC=g++
MODE=release
ifeq ($(MODE), release)
	CFLAGS=-c -Wall -std=c++11 -O2
else
	CFLAGS=-c -Wall -g -std=c++11 
endif
LIBS=-larmadillo -lpthread
SRC0=loss_function.cpp sgd.cpp data_part.cpp io.cpp metrics.cpp simu_data.cpp
SRC1=$(SRC0) ser_gaussian.cpp 
SRC2=$(SRC0) par_random_gaussian.cpp 
SRC3=$(SRC0) eval_part.cpp
SRC4=$(SRC0) par_minibatch_gaussian.cpp
SRC5=$(SRC0) hogwild_gaussian.cpp

SRC6=$(SRC0) ser_mnist.cpp
SRC7=$(SRC0) eval_part_mnist.cpp
SRC8=$(SRC0) hogwild_mnist.cpp

OBJ1=$(SRC1:.cpp=.o)
OBJ2=$(SRC2:.cpp=.o)
OBJ3=$(SRC3:.cpp=.o)
OBJ4=$(SRC4:.cpp=.o)
OBJ5=$(SRC5:.cpp=.o)
OBJ6=$(SRC6:.cpp=.o)
OBJ7=$(SRC7:.cpp=.o)
OBJ8=$(SRC8:.cpp=.o)
EXE1=ser_gaussian 
EXE2=par_random_gaussian
EXE3=eval_part
EXE4=par_minibatch_gaussian
EXE5=hogwild_gaussian
EXE6=ser_mnist
EXE7=eval_part_mnist
EXE8=hogwild_mnist

all: $(EXE1) $(EXE2) $(EXE3) $(EXE4) $(EXE5) $(EXE6) $(EXE7) $(EXE8)

$(EXE1): $(OBJ1)
	$(CC) $(OBJ1) -o $@ $(LIBS)

$(EXE2): $(OBJ2)
	$(CC) $(OBJ2) -o $@ $(LIBS)

$(EXE3): $(OBJ3)
	$(CC) $(OBJ3) -o $@ $(LIBS)

$(EXE4): $(OBJ4)
	$(CC) $(OBJ4) -o $@ $(LIBS)

$(EXE5): $(OBJ5)
	$(CC) $(OBJ5) -o $@ $(LIBS)

$(EXE6): $(OBJ6)
	$(CC) $(OBJ6) -o $@ $(LIBS)

$(EXE7): $(OBJ7)
	$(CC) $(OBJ7) -o $@ $(LIBS)

$(EXE8): $(OBJ8)
	$(CC) $(OBJ8) -o $@ $(LIBS)

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

clean:
	-rm *.o 
