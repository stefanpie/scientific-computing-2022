SOURCES = $(wildcard *.dot)
OBJECTS = $(SOURCES:.dot=.doto)
IMAGES  = $(SOURCES:.dot=.png)

.SECONDARY: $(OBJS)

all: $(IMAGES)

%.png: %.doto
	dot -Tpng $< > $@

%.doto: %.dot
	cpp -P $< > $@
