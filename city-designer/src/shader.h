// src/shader.h
#pragma once
#include <GLFW/glfw3.h>
#include <iostream>
#include <string>

static GLuint compileShader(GLenum type, const char* src){
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, NULL);
    glCompileShader(sh);
    GLint ok = 0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if(!ok){
        GLint len = 0; glGetShaderiv(sh, GL_INFO_LOG_LENGTH, &len);
        std::string log(len, ' ');
        glGetShaderInfoLog(sh, len, NULL, &log[0]);
        std::cerr << "Shader compile error: " << log << "\n";
    }
    return sh;
}

static GLuint linkProgram(GLuint vs, GLuint fs){
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    GLint ok=0; glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if(!ok){
        GLint len=0; glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
        std::string log(len,' ');
        glGetProgramInfoLog(prog,len,NULL,&log[0]);
        std::cerr << "Program link error: " << log << "\n";
    }
    return prog;
}
