#include <stdio.h>
#include <vector>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "brainCpu.h"

GLuint CompileShader(const char* a_src, GLuint a_type)
{
  GLuint shader = glCreateShader(a_type);
  glShaderSource(shader, 1, &a_src, NULL);
  glCompileShader(shader);

  int result;
  glGetProgramiv(shader, GL_COMPILE_STATUS, &result);
  if (!result)
  {
    fprintf(stderr, "Error compiling shader\n");
    GLchar log[10240];
    GLsizei length;
    glGetShaderInfoLog(shader, 10239, &length, log);
    fprintf(stderr, "Linker log:\n%s\n", log);
  }

  return shader;
}

GLuint LinkProgram(std::vector<GLuint> a_shaders)
{
  GLuint program = glCreateProgram();
  for (GLuint shader : a_shaders)
    glAttachShader(program, shader);
  glLinkProgram(program);

  int result;
  glGetProgramiv(program, GL_LINK_STATUS, &result);
  if (!result) 
  {
    fprintf(stderr, "Error in linking compute shader program\n");
    GLchar log[10240];
    GLsizei length;
    glGetProgramInfoLog(program, 10239, &length, log);
    fprintf(stderr, "Linker log:\n%s\n", log);
  }

  for (GLuint shader : a_shaders)
  {
    glDetachShader(program, shader);
    glDeleteShader(shader);
  }
  return program;
}

int main()
{
  static const int width  = 40;
  static const int height = 40;
  static const int scale  = 8;

  BrainCpu brain;
  uint8_t* image = new uint8_t[width*height * 3];

#if 0
  brain.Dream(width, height, 1.0, image);
  FILE* file;
  fopen_s(&file, "output.raw", "wb");
  fwrite(image, 1, width*height * 3, file);
  fclose(file);
  return 0;
#endif

  // Init OpenGL and make a window via GLFW
  if (!glfwInit())
    return -1;

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
  //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

  GLFWwindow* window = glfwCreateWindow(width*scale, height*scale, "Hello World", NULL, NULL);
  if (!window)
  {
    glfwTerminate();
    return -1;
  }

  glfwMakeContextCurrent(window);


  // Load extensions with glad
  if (!gladLoadGL()) {
    printf("Something went wrong!\n");
    return -1;
  }

  GLuint error;
  error = glGetError();

  // Set up OpenGL base state
  glClearColor(1.0f, 1.0f, 0.0f, 0.0f);
  // Set up the texture
  GLuint texture;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
  // Set up the quad geometry
  GLfloat quad[] = { 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f };
  GLuint quadVBO;
  glGenBuffers(1, &quadVBO);
  glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
  // Set up a VAO
  GLuint vao = 0;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(0);
  // Set up the render shading program
  const char* vertShaderSrc = R"(#version 140
in vec2 iPosition;
out vec2 pos;
void main(void) {
  pos = iPosition * 0.5 + 0.5;
  gl_Position = vec4(iPosition, 0.0, 1.0);
})";
  const char* fragShaderSrc = R"(#version 140
in vec2 pos;
out vec4 oFragColor;
uniform sampler2D uTexture;
void main(void) {
  oFragColor = vec4(texture2D(uTexture, pos).rgb, 1.0);
})";
  error = glGetError();
  GLuint vertShader  = CompileShader(vertShaderSrc, GL_VERTEX_SHADER);
  GLuint fragShader  = CompileShader(fragShaderSrc, GL_FRAGMENT_SHADER);
  GLuint renderProg  = LinkProgram({ vertShader, fragShader });
  glUseProgram(renderProg);
  GLuint uTexture = glGetUniformLocation(renderProg, "uTexture");
  glUniform1i(uTexture, 0);

  // Main loop
  float bias = -1.0;
  while (!glfwWindowShouldClose(window))
  {
    brain.Dream(width, height, bias, image);
    bias += 0.01f;
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image);

    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }


  // Time to die.
  glfwTerminate();
  return 0;
}