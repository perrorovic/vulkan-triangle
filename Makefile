# If clangd unable to read this file properly, just compile the commands with "bear -- make"
VULKAN_SDK_PATH = include/vulkansdk/1.4.321.1/x86_64
STB_INCLUDE_PATH = include/stb

CFLAGS = -std=c++17 -I$(STB_INCLUDE_PATH)
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

SOURCE = source
BUILD = build

PHONY = build, run, clean

compile: $(SOURCE)/vk_main.cpp
	clang++ $(CFLAGS) -o $(BUILD)/vk_main.x86_64 $(SOURCE)/vk_main.cpp $(LDFLAGS)

run: compile
	XDG_SESSION_TYPE=x11 GDK_BACKEND=x11 GLFW_PLATFORM=x11 ./$(BUILD)/vk_main.x86_64

clean:
	rm -f build/*