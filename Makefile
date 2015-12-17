LIB=./lib
INCLUDE=./include
SRC=./src
OBJ=./obj
BIN=./bin

CC=gcc 

FLAGS=  -g -O0
CFLAGS=''

all: libDeep deep_generative_dbm deep_generative_dbn

libDeep: $(LIB)/libDeep.a
	echo "libDeep.a built..."

$(LIB)/libDeep.a: \
$(OBJ)/deep.o \
$(OBJ)/math_functions.o \
$(OBJ)/rbm.o \
$(OBJ)/auxiliary.o \
$(OBJ)/dbn.o \
$(OBJ)/regression.o \
$(OBJ)/logistic.o \
$(OBJ)/dbm.o \
$(OBJ)/dbn.o \

	ar csr $(LIB)/libDeep.a \
$(OBJ)/deep.o \
$(OBJ)/math_functions.o \
$(OBJ)/rbm.o \
$(OBJ)/auxiliary.o \
$(OBJ)/dbn.o \
$(OBJ)/regression.o \
$(OBJ)/logistic.o \
$(OBJ)/dbm.o \
$(OBJ)/dbn.o \

$(OBJ)/deep.o: $(SRC)/deep.c
	$(CC) $(FLAGS) -I $(INCLUDE) -I $(OPF_DIR)/include -I $(OPF_DIR)/include/util -I /usr/local/include -c $(SRC)/deep.c \
	-L /usr/local/lib -L $(OPF_DIR)/lib -lOPF -lgsl -lgslcblas -o $(OBJ)/deep.o `pkg-config --cflags --libs gsl`

$(OBJ)/math_functions.o: $(SRC)/math_functions.c
	$(CC) $(FLAGS) -I $(INCLUDE) -I $(OPF_DIR)/include -I $(OPF_DIR)/include/util -I /usr/local/include -c $(SRC)/math_functions.c \
	-L $(OPF_DIR)/lib -lOPF -o $(OBJ)/math_functions.o `pkg-config --cflags --libs gsl`

$(OBJ)/rbm.o: $(SRC)/rbm.c
	$(CC) $(FLAGS) -I $(INCLUDE) -I $(OPF_DIR)/include -I $(OPF_DIR)/include/util -I /usr/local/include -c $(SRC)/rbm.c \
	-L $(OPF_DIR)/lib -lOPF -o $(OBJ)/rbm.o `pkg-config --cflags --libs gsl`

$(OBJ)/auxiliary.o: $(SRC)/auxiliary.c
	$(CC) $(FLAGS) -I $(INCLUDE) -I $(OPF_DIR)/include -I $(OPF_DIR)/include/util -I /usr/local/include -c $(SRC)/auxiliary.c \
	-L $(OPF_DIR)/lib -lOPF -o $(OBJ)/auxiliary.o `pkg-config --cflags --libs gsl`

$(OBJ)/dbn.o: $(SRC)/dbn.c
	$(CC) $(FLAGS) -I $(INCLUDE) -I $(OPF_DIR)/include -I $(OPF_DIR)/include/util -I /usr/local/include -c $(SRC)/dbn.c \
	-L $(LIB) -L $(OPF_DIR)/lib -lOPF -o $(OBJ)/dbn.o `pkg-config --cflags --libs gsl`

$(OBJ)/regression.o: $(SRC)/regression.c
	$(CC) $(FLAGS) -I $(INCLUDE) -I $(OPF_DIR)/include -I $(OPF_DIR)/include/util -I /usr/local/include -c $(SRC)/regression.c \
	-L $(LIB) -L $(OPF_DIR)/lib -lOPF -o $(OBJ)/regression.o `pkg-config --cflags --libs gsl`

$(OBJ)/logistic.o: $(SRC)/logistic.c
	$(CC) $(FLAGS) -I $(INCLUDE) -I $(OPF_DIR)/include -I $(OPF_DIR)/include/util -I /usr/local/include -c $(SRC)/logistic.c \
	-L $(LIB) -L $(OPF_DIR)/lib -lOPF -o $(OBJ)/logistic.o `pkg-config --cflags --libs gsl -lm`

$(OBJ)/dbm.o: $(SRC)/dbm.c
	$(CC) $(FLAGS) -I $(INCLUDE) -I $(OPF_DIR)/include -I $(OPF_DIR)/include/util -I /usr/local/include -c $(SRC)/dbm.c \
	-L $(LIB) -L $(OPF_DIR)/lib -lOPF -o $(OBJ)/dbm.o `pkg-config --cflags --libs gsl`

deep_generative_dbm:
	$(CC) $(FLAGS) -I $(INCLUDE) -I $(OPF_DIR)/include -I $(OPF_DIR)/include/util -I /usr/local/include examples/deep_generative_dbm.c \
	-L $(LIB) -lDeep -L $(OPF_DIR)/lib -lOPF  -L /usr/local/lib   -lgsl -lgslcblas -o $(BIN)/deep_generative_dbm   -lm 

deep_generative_dbn:
	$(CC) $(FLAGS) -I $(INCLUDE) -I $(OPF_DIR)/include -I $(OPF_DIR)/include/util -I /usr/local/include examples/deep_generative_dbn.c \
	-L $(LIB) -lDeep -L $(OPF_DIR)/lib -lOPF  -L /usr/local/lib   -lgsl -lgslcblas -o $(BIN)/deep_generative_dbn   -lm 

clean:
	rm -f $(LIB)/lib*.a; rm -f $(OBJ)/*.o rm -f $(BIN)/*
