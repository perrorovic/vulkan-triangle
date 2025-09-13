# Vulkan Triangle

Rite of pasage of learing Vulkan API, following along with [Vulkan Tutorial](https://vulkan-tutorial.com)

![Vulkan Triangle](markdown/66jJRSG.png)
Hours spent on this project: 11h (and counting...)

## Overview

Learning Vulkan at itself already something that is hard to get into, expecially with my minimum C++ knowledge, and running all of that on my Linux Wayland + Hyprland compositor?

Yeah, we got proper challange that will sink my morale

This is my journey through that pain and suffering with my personal note of hours wasted, mistake made and wayland shenanigans

My list of suffering along the way

- [Window creation](#window-creation)
- [Wayland support](#wayland-support)
- [Swap chain creation](#swap-chain-creation)
- [Swap chain recreation](#swap-chain-recreation)

## Window creation

Creating window is as simple as calling GLFW init function, but this window creation will not be displayed by Hyprland if it has no display content (empty draw buffer), which mean this GLFW window is exist but hidden by Hyprland

I acknowledge this issues after I tried to draw GLFW window with OpenGL and the window is shown by the compositor

## Wayland support

As I tried to use native support of Wayland in Vulkan as I know that extension exist in my GPU but I quickly found out that extension is disabled on `vulkaninfo`

```text
VK_KHR_wayland_surface: extension revision 6
GPU id : 0 (NVIDIA GeForce GTX 1060 6GB) [VK_KHR_wayland_surface]:
  Surface type = VK_KHR_wayland_surface
    VK_KHR_wayland_surface = false
```

Why it is disabled? I cannot tell myself, It might be because of my NVIDIA proprietary driver (ver. 580) or Hyprland simply said "No"

Therefore moving foward I will be using X11 running under XWayland

## Swap chain creation

This is some issues I encounter when doing `createSwapChain()` to fill requirement data for `VkSwapchainCreateInfoKHR()`

This happen because my dumbass forget to actually query the physical device capabilities with `vkGetPhysicalDeviceSurfaceCapabilitiesKHR()` in `querySwapChainSupport()`

The workaround below are fixed with the proper solution after I recognize this mistake, nevertheless this is my past insight on those issues

### Width and height

Vulkan use `chooseSwapExtent()` which return `VkExtent2D` information that hold width and height of the window, this might sometimes adjusted accordingly with high DPI monitor or system zoom scaling

This dynamic adjustment is absolutely broken on Hyprland as it return garbage/trash value for width/height from `VkSurfacecapabilitiesKHR` as the window itself aren't draw yet or what? I am not quite sure myself

My workaround this is to just get the information from GLFW framebuffer size and use those data as-is to create `VkExtent2D`

### Image count

For some reason image count query is also incorrect? My workaround is to hard-code image count value according to present mode

## Swap chain recreation

This is going to be something that I need to tackle soon!
