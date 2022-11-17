SOURCES = $(wildcard *.dot)
OBJECTS = $(SOURCES:.dot=.doto)
IMAGES  = $(SOURCES:.dot=.png)

.SECONDARY: $(OBJS)

# log source files
$(info $(IMAGES))

%.png: %.doto
	dot -Tpng $< > $@

%.doto: %.dot
	cpp -P $< > $@

all: $(IMAGES)

