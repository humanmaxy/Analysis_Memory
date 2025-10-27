# Makefile for X-ray Image Enhancement

# 编译器设置
CXX = g++
CXXFLAGS = -std=c++14 -Wall -Wextra -O3

# OpenCV设置
OPENCV_CFLAGS = `pkg-config --cflags opencv4` 2>/dev/null || `pkg-config --cflags opencv`
OPENCV_LIBS = `pkg-config --libs opencv4` 2>/dev/null || `pkg-config --libs opencv`

# 源文件
SOURCES = xray_enhancement.cpp
HEADERS = xray_enhancement.hpp
MAIN_SOURCE = main.cpp
TEST_SOURCE = test_enhancement.cpp

# 目标文件
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = xray_enhancement
TEST_TARGET = test_enhancement

# 默认目标
all: $(TARGET) $(TEST_TARGET)

# 主程序
$(TARGET): $(OBJECTS) main.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(OPENCV_LIBS)

# 测试程序
$(TEST_TARGET): $(OBJECTS) test_enhancement.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(OPENCV_LIBS)

# 编译对象文件
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

# 清理
clean:
	rm -f $(OBJECTS) main.o test_enhancement.o $(TARGET) $(TEST_TARGET)
	rm -f test_*.png

# 安装（可选）
install: $(TARGET)
	cp $(TARGET) /usr/local/bin/

# 运行测试
test: $(TEST_TARGET)
	./$(TEST_TARGET)

# 帮助
help:
	@echo "可用目标:"
	@echo "  all       - 编译所有程序"
	@echo "  $(TARGET) - 编译主程序"
	@echo "  $(TEST_TARGET) - 编译测试程序"
	@echo "  test      - 运行测试"
	@echo "  clean     - 清理编译文件"
	@echo "  install   - 安装到系统"
	@echo "  help      - 显示此帮助"
	@echo ""
	@echo "使用示例:"
	@echo "  make all"
	@echo "  make test"
	@echo "  ./xray_enhancement input.png output.png --combined"

.PHONY: all clean install test help