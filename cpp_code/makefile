CC=g++
CFLAGS=-c -Wall -std=c++11 -O2
#CFLAGS=-c -Wall -g -std=c++11 
LIBS=-larmadillo -lpthread
SRC0=loss_function.cpp sgd.cpp data_part.cpp
SRC1=$(SRC0) ser_gaussian.cpp 
SRC2=$(SRC0) par_random_gaussian.cpp 

OBJ1=$(SRC1:.cpp=.o)
OBJ2=$(SRC2:.cpp=.o)
EXE1=ser_gaussian 
EXE2=par_random_gaussian

all: $(EXE1) $(EXE2)

$(EXE1): $(OBJ1)
	$(CC) $(OBJ1) -o $@ $(LIBS)

$(EXE2): $(OBJ2)
	$(CC) $(OBJ2) -o $@ $(LIBS)

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

clean:
	-rm *.o 
