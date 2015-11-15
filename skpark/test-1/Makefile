
TARGET=out.exe
SRC = main.cu
CC= nvcc.exe

$(TARGET) : $(SRC)
	$(CC) -w $(SRC) -o $(TARGET)

run : $(SRC)
	$(CC) -w $(SRC) -o $(TARGET)
	$(TARGET)

clean :
	rm -rf $(TARGET) *.exp *.obj *.lib



