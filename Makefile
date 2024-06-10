DEBUG=1

SRC=src
BUILD=build

CXX=g++
CC=gcc
OPTS=-O3
LDFLAGS= 
CFLAGS=-Wall

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = Makefile src/*

all: $(BUILD) $(EXEC) main

main:
	$(CXX) $(CFLAGS) $(SRC)/main.cpp -o $(BUILD)/$@

$(BUILD):
	@mkdir -p $@

clean:
	@rm -rf $(EXECDIR)
.PHONY: clean
