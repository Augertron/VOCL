CCPLUS=g++
CC=gcc
NVCC=nvcc
OBJ=obj
INC=inc
SRC=src

SRCFILES=$(SRC)/alignments.cpp $(SRC)/prints.cpp $(SRC)/sequences.cpp $(SRC)/swat.cpp \
		 $(SRC)/param.cpp (SRC)/timeRec.cpp

HEADERFILES=$(INC)/functions.h $(INC)/param.h $(INC)/global.h $(INC)/timeRec.h
OBJFILES=$(OBJ)/alignments.o $(OBJ)/prints.o $(OBJ)/sequences.o $(OBJ)/swat.o \
		 $(OBJ)/param.o $(OBJ)/timeRec.o
CCFLAGS= -O3 -c -I$(INC) -I/usr/local/cuda/include

swatO: $(OBJFILES)
	$(CCPLUS) -lOpenCL -o swatO $(OBJFILES)

$(OBJ)/param.o: $(SRC)/param.cpp
	$(CCPLUS) $(CCFLAGS) -o $(OBJ)/param.o $(SRC)/param.cpp

$(OBJ)/timeRec.o: $(SRC)/timeRec.cpp
	$(CCPLUS) $(CCFLAGS) -o $(OBJ)/timeRec.o $(SRC)/timeRec.cpp

$(OBJ)/alignments.o: $(SRC)/alignments.cpp
	$(CCPLUS) $(CCFLAGS) -o $(OBJ)/alignments.o $(SRC)/alignments.cpp

$(OBJ)/prints.o: $(SRC)/prints.cpp
	$(CCPLUS) $(CCFLAGS) -o $(OBJ)/prints.o $(SRC)/prints.cpp

$(OBJ)/sequences.o: $(SRC)/sequences.cpp
	$(CCPLUS) $(CCFLAGS) -o $(OBJ)/sequences.o $(SRC)/sequences.cpp

$(OBJ)/swat.o: $(SRC)/swat.cpp
	$(CCPLUS) $(CCFLAGS) -o $(OBJ)/swat.o $(SRC)/swat.cpp

clean:
	rm -f swatO $(OBJ)/*

