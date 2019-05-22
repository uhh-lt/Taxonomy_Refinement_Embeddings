SRCS = graph.cpp agony.cpp read.cpp
OBJS = $(SRCS:.cpp=.o)


CC = g++
#LDFLAGS= -lm -O3 -lgsl -lblas -L/opt/local/lib
LDFLAGS= -Wl,--no-as-needed -lgsl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl


#CFLAGS= -g -o3 -Wall -Wextra -I/opt/local/include/ #-Werror
CFLAGS= -g -O3 -Wall -Wextra -I/opt/intel/compilers_and_libraries_2018.3.222/linux/mkl/include

all : agony 

agony: $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS)


%.o : %.cpp
	 $(CC) $(CFLAGS) -c $*.cpp

clean:
	-rm *.o agony

zip:
	zip agony.zip *.cpp *.h Makefile

depend:
	makedepend -Y -- $(CFLAGS) -- $(SRCS)
# DO NOT DELETE

graph.o: graph.h queue.h
agony.o: graph.h queue.h
