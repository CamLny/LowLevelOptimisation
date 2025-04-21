#!/bin/bash

g++ main.cpp glad/glad.c imgui/imgui*.cpp imgui/backends/imgui_impl_glfw.cpp imgui/backends/imgui_impl_opengl3.cpp -I. -I./imgui -I./imgui/backends -lglfw -lGL -O0 -g3 -o workshop

