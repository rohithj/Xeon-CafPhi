##******************************************************************************
##
##  Copyright (C) 2012-2014 Intel Corporation. All Rights Reserved.
##
##  The source code contained or described herein and all
##  documents related to the source code ("Material") are owned by
##  Intel Corporation or its suppliers or licensors. Title to the
##  Material remains with Intel Corporation or its suppliers and
##  licensors. The Material is protected by worldwide copyright
##  laws and treaty provisions.  No part of the Material may be
##  used, copied, reproduced, modified, published, uploaded,
##  posted, transmitted, distributed,  or disclosed in any way
##  except as expressly provided in the license provided with the
##  Materials.  No license under any patent, copyright, trade
##  secret or other intellectual property right is granted to or
##  conferred upon you by disclosure or delivery of the Materials,
##  either expressly, by implication, inducement, estoppel or
##  otherwise, except as expressly provided in the license
##  provided with the Materials.
##
##******************************************************************************
## Content:
##      Build for Sample shrd_sampleCPP
##******************************************************************************

NAME		=	forwardConv
CC		=	icc
CXX		=	icpc
LD		=	icpc
DEBUG		+=
CFLAGS		+=	-qopenmp $(DEBUG)
CXXFLAGS	+=	$(CFLAGS)
LDFLAGS		+=	

EXE		=	$(NAME).out
CFILES		=	$(wildcard *.c)
CXXFILES	=	$(wildcard *.cpp)
OBJECTS		=	$(CFILES:.c=.o) $(CXXFILES:.cpp=.o)

help:
	@echo "Usage:"
	@echo "    Build : make mic"
	@echo "    Run   : forwardConv.out"

mic : $(EXE)
	@echo "...."
	@echo "Build complete"
	@echo "Run : forwardConv.out"

no_mic : CXXFLAGS	+= -qoffload=none -diag-disable 161,2423,2571
no_mic : LDFLAGS	+= -loffload
no_mic : mic

mic_debug : DEBUG       += -g -traceback -O0
mic_debug : mic

$(EXE) : $(OBJECTS)
	$(LD) -V $^ $(CFLAGS) $(LDFLAGS) -o $@

%.o : %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o : %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean :	
	rm -fr $(EXE) $(OBJECTS) $(OBJECTS:.o=MIC.o)
