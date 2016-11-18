CC=g++
CFLAGS=-c -Wall -Werror -g -std=c++11 
LIBS=-larmadillo
SRC=loss_function.cpp sgd.cpp init_test.cpp
OBJ=$(SRC:.cpp=.o)
EXE=init_test

all: $(SRC) $(EXE)

$(EXE): $(OBJ)
	$(CC) $(OBJ) -o $@ $(LIBS)

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

clean:
	-rm *.o 
