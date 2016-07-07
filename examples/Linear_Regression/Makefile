BIN=./bin

FLAGS=  -g -O0
INCLUDE= -I $(OPF_DIR)/include -I $(OPF_DIR)/include/util -I /usr/local/include -I $(LIBDEEP_DIR)/include -I $(OPT_DIR)/include
LIB= -L $(OPF_DIR)/lib -L $(OPT_DIR)/lib -L $(LIBDEEP_DIR)/lib -L /usr/local/lib

$@.c: $@.c
	gcc $(FLAGS) $@.c -o $(BIN)/$@ $(INCLUDE) $(LIB) -lDeep -lOPF -lm -lgsl -lgslcblas; \

clean:
	rm -rf $(BIN)/*;

