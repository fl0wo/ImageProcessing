CC=gcc
CFLAGS=-Wall -lm
CFLAGS_MAIN=-Wall -ansi --pedantic -lm -g3 -O3 -fsanitize=address -fsanitize=undefined -lasan -std=gnu89 -Wextra

a : src/main_iplib.c src/lib/bmp.o src/lib/ip_lib.o
	$(CC) $^ -o $@ $(CFLAGS_MAIN)

src/lib/ip_lib.o : src/ip_lib.c
	$(CC) $^ -c -o $@ $(CFLAGS)

src/lib/bmp.o : src/bmp.c
	$(CC) $^ -c -o $@ $(CFLAGS)

test : src/bmp.o lib/ip_lib.o src/test/test_mat.c
	$(CC) $^ -o test/$@ $(CFLAGS)

mandelbrot : lib/bmp.o lib/ip_lib.o test/test_bmp.c
	$(CC) $^ -o test/$@ $(CFLAGS)

clean:
	rm lib/* main
	rm test/test_bmp test/test_mat