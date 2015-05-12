LIB=./lib
INCLUDE=./include
SRC=./src
OBJ=./obj

CC=gcc 

FLAGS=  -g -O0
CFLAGS=''

all: libDeep

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

	ar csr $(LIB)/libDeep.a \
$(OBJ)/deep.o \
$(OBJ)/math_functions.o \
$(OBJ)/rbm.o \
$(OBJ)/auxiliary.o \
$(OBJ)/dbn.o \
$(OBJ)/regression.o \
$(OBJ)/logistic.o \

$(OBJ)/deep.o: $(SRC)/deep.c
	$(CC) $(FLAGS) -I $(INCLUDE) -I $(OPF_DIR)/include -I $(OPF_DIR)/include/util -I /usr/local/include -c $(SRC)/deep.c \
	-L /usr/local/lib -L $(OPF_DIR)/lib -lopf -lgsl -lgslcblas -o $(OBJ)/deep.o `pkg-config --cflags --libs gsl`

$(OBJ)/math_functions.o: $(SRC)/math_functions.c
	$(CC) $(FLAGS) -I $(INCLUDE) -I $(OPF_DIR)/include -I $(OPF_DIR)/include/util -I /usr/local/include -c $(SRC)/math_functions.c \
	-L $(OPF_DIR)/lib -lopf -o $(OBJ)/math_functions.o `pkg-config --cflags --libs gsl`

$(OBJ)/rbm.o: $(SRC)/rbm.c
	$(CC) $(FLAGS) -I $(INCLUDE) -I $(OPF_DIR)/include -I $(OPF_DIR)/include/util -I /usr/local/include -c $(SRC)/rbm.c \
	-L $(OPF_DIR)/lib -lopf -o $(OBJ)/rbm.o `pkg-config --cflags --libs gsl`

$(OBJ)/auxiliary.o: $(SRC)/auxiliary.c
	$(CC) $(FLAGS) -I $(INCLUDE) -I $(OPF_DIR)/include -I $(OPF_DIR)/include/util -I /usr/local/include -c $(SRC)/auxiliary.c \
	-L $(OPF_DIR)/lib -lopf -o $(OBJ)/auxiliary.o `pkg-config --cflags --libs gsl`

$(OBJ)/dbn.o: $(SRC)/dbn.c
	$(CC) $(FLAGS) -I $(INCLUDE) -I $(OPF_DIR)/include -I $(OPF_DIR)/include/util -I /usr/local/include -c $(SRC)/dbn.c \
	-L $(LIB) -L $(OPF_DIR)/lib -lopf -o $(OBJ)/dbn.o `pkg-config --cflags --libs gsl`

$(OBJ)/regression.o: $(SRC)/regression.c
	$(CC) $(FLAGS) -I $(INCLUDE) -I $(OPF_DIR)/include -I $(OPF_DIR)/include/util -I /usr/local/include -c $(SRC)/regression.c \
	-L $(LIB) -L $(OPF_DIR)/lib -lopf -o $(OBJ)/regression.o `pkg-config --cflags --libs gsl`

$(OBJ)/logistic.o: $(SRC)/logistic.c
	$(CC) $(FLAGS) -I $(INCLUDE) -I $(OPF_DIR)/include -I $(OPF_DIR)/include/util -I /usr/local/include -c $(SRC)/logistic.c \
	-L $(LIB) -L $(OPF_DIR)/lib -lopf -o $(OBJ)/logistic.o `pkg-config --cflags --libs gsl`

clean:
	rm -f $(LIB)/lib*.a; rm -f $(OBJ)/*.o
