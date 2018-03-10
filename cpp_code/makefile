CC=g++
MODE=release
ifeq ($(MODE), release)
	CFLAGS=-Wall -std=c++11 -O2
else
	CFLAGS=-Wall -g -std=c++11 
endif
LIBS=-larmadillo -lpthread

OBJIMP=loss_function.o sgd.o data_part.o io.o metrics.o simu_data.o

EXES := eval_part \
				eval_part_mnist \
				hogwild_gaussian \
				hogwild_mnist \
				par_minibatch_gaussian \
				par_random_gaussian \
				ser_gaussian \
				ser_mnist \

OBJS = $(EXES:%=%.o) $(OBJIMP)
			 
all: $(EXES)

./%: ./%.o $(OBJIMP)
	$(CC) $(CFLAGS) -o $@ $< $(OBJIMP) $(LIBS)

$(OBJS) : ./%.o : ./%.cpp
	$(CC) $(CFLAGS) $< -c -o $@ $(LIBS)

clean:
	-rm -rf $(OBJS)

remove:
	-rm $(EXES)
