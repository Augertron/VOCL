CCPLUS=mpicxx
CC=gcc
NVCC=nvcc
OBJ=obj
INC=-Iinc -I/usr/local/cuda/include
SRC=src
LIB=

SRCFILES=$(SRC)/bandWidth.c $(SRC)/timeRec.c

HEADERFILES=$(INC)/functions.h $(INC)/timeRec.h
OBJFILES=$(OBJ)/bandWidth.o $(OBJ)/timeRec.o
CCFLAGS= -c -O3 $(INC)

all: bandwidth

bandwidth: $(OBJFILES)
	$(CCPLUS) -o $@ $? $(LIB) -lOpenCL

$(OBJ)/bandWidth.o: $(SRC)/bandWidth.c
	$(CCPLUS) $(CCFLAGS) -o $@ $?

$(OBJ)/timeRec.o: $(SRC)/timeRec.c
	$(CCPLUS) $(CCFLAGS) -o $@ $?

clean:
	rm -f bandwidth $(OBJ)/*

