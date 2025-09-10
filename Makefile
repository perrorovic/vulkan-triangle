CFLAGS = -std=c++17 -O2
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

vk_extension: vk_ext.cpp
	g++ $(CFLAGS) -o vk_extension vk_ext.cpp $(LDFLAGS)

run_vk_extension: vk_extension
	./vk_extension

vk_main: vk_main.cpp
	g++ $(CFLAGS) -o vk_main vk_main.cpp $(LDFLAGS)

run: vk_main
	XDG_SESSION_TYPE=x11 GDK_BACKEND=x11 GLFW_PLATFORM=x11 ./vk_main

clean:
	rm -f vk_main vk_extension