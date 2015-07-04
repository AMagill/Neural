#include <stdio.h>
#include <vector>
#include <array>
#include <random>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "brainGpu.h"

GLuint CompileShader(const char* a_src, GLuint a_type)
{
  GLuint shader = glCreateShader(a_type);
  glShaderSource(shader, 1, &a_src, NULL);
  glCompileShader(shader);

  int result;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &result);
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
    fprintf(stderr, "Error linking shader program\n");
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
  static const int width  = 64;
  static const int height = 64;
  static const int scale  = 8;

  static const int nNeurons = 16;  // Neurons per layer
  static const int nInputs  = 3;   // Input channels  (x, y, ...)
  static const int nOutputs = 3;   // Output channels (r, g, b)
  static const int nHidden  = 8;   // Number of hidden layers

  BrainGpu brain;

  // Init OpenGL and make a window via GLFW
  if (!glfwInit())
    return -1;

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
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
  // Set up the output image texture
  GLuint imageTex;
  glGenTextures(1, &imageTex);
  glBindTexture(GL_TEXTURE_2D, imageTex);
  glBindImageTexture(0, imageTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
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
  error = glGetError();
  // Set up the render shaders
  const char* vertShaderSrc = R"(#version 440
in vec2 iPosition;
out vec2 pos;
void main(void) {
  pos = iPosition * 0.5 + 0.5;
  gl_Position = vec4(iPosition*0.9, 0.0, 1.0);
})";
  const char* fragShaderSrc = R"(#version 440
in vec2 pos;
out vec4 oFragColor;
layout(binding = 0) uniform sampler2D uTexture;
void main(void) {
  oFragColor = vec4(texture2D(uTexture, pos).rgb, 1.0);
  //oFragColor = vec4(1.0, 0.0, 0.0, 1.0);
})";
  GLuint vertShader = CompileShader(vertShaderSrc, GL_VERTEX_SHADER);
  GLuint fragShader = CompileShader(fragShaderSrc, GL_FRAGMENT_SHADER);
  GLuint renderProgram = LinkProgram({ vertShader, fragShader });
  glUseProgram(renderProgram);
  error = glGetError();
  // Set up the compute shader program
  const char* compShaderSrc = R"(#version 440
const uint nNeurons = 16;  // Neurons per layer
const uint nInputs  =  3;  // Input channels  (x, y, ...)
const uint nOutputs =  3;  // Output channels (r, g, b)
const uint nHidden  =  8;  // Number of hidden layers
layout(binding = 0) uniform writeonly image2D destTex;
layout(binding = 0, std140) buffer nn { float neuralNet[]; };
layout (local_size_x = 16, local_size_y = 16) in;
float scratchA[nNeurons];
float scratchB[nNeurons];

void multiply(uint layer, uint width, uint height) {
  uint los = layer * nNeurons * nNeurons;
  for (uint x = 0; x < nNeurons; x++) {
    float dot = 0.0;
    for (uint y = 0; y < nNeurons; y++) {
      float cell = neuralNet[los + (height*x) + y];
      dot += cell * scratchA[y];
    }
    scratchB[x] = dot;
  }
}

void arr_tanh() {
  for (uint i = 0; i < nNeurons; i++)
    scratchA[i] = tanh(scratchB[i]);
}

void arr_sigmoid() {
  for (uint i = 0; i < nNeurons; i++)
    scratchA[i] = 1.0 / (1.0 + exp(-scratchB[i]));
}

void main() {
  vec3 pos = vec3(gl_GlobalInvocationID) / (gl_NumWorkGroups * gl_WorkGroupSize);
  scratchA[0] = pos.x;
  scratchA[1] = pos.y;
  scratchA[2] = 1.0;
  multiply(0, nNeurons, nInputs);
  arr_tanh();
  
  for (int i = 1; i <= nHidden; i++) {
    multiply(i, nNeurons, nNeurons);
    arr_tanh();
  }

  multiply(nHidden+1, nOutputs, nNeurons);
  arr_sigmoid();

  vec4 color = vec4(scratchA[0], scratchA[1], scratchA[2], 1.0);

  color = vec4(scratchA[int(pos.x*nNeurons)]);
  imageStore(destTex, ivec2(gl_GlobalInvocationID.xy), color);
})";
  GLuint compShader  = CompileShader(compShaderSrc, GL_COMPUTE_SHADER);
  GLuint compProgram = LinkProgram({ compShader });
  glUseProgram(compProgram);
  glUniform1ui(glGetUniformLocation(compProgram, "nNeurons"), nNeurons);
  glUniform1ui(glGetUniformLocation(compProgram, "nInputs"),  nInputs);
  glUniform1ui(glGetUniformLocation(compProgram, "nOutputs"), nOutputs);
  glUniform1ui(glGetUniformLocation(compProgram, "nHidden"),  nHidden);
  error = glGetError();
  // Set up the neural net buffer
  GLuint neuralNetBuf;
  glGenBuffers(1, &neuralNetBuf);
  {
    const int totalSize = nNeurons * nNeurons * (nHidden + 2);  // Reserve two extra layers for input and output
    std::random_device rand;
    std::uniform_real_distribution<float> dist(-1, 1);
    std::array<float, totalSize> temp;
    for (int i = 0; i < totalSize; i++)
      temp[i] = dist(rand);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, neuralNetBuf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * totalSize, temp.data(), GL_STATIC_DRAW);
  }
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, neuralNetBuf);
  error = glGetError();

  // Main loop
  float bias = 0;
  while (!glfwWindowShouldClose(window))
  {
    glUseProgram(compProgram);
    glUniform1f(glGetUniformLocation(compProgram, "roll"), bias);
    glUniform1i(glGetUniformLocation(compProgram, "destTex"), 0);
    glDispatchCompute(width / 16, height / 16, 1);
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
    bias += 0.01f;

    glUseProgram(renderProgram);
    glUniform1i(glGetUniformLocation(renderProgram, "uTexture"), 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }


  // Time to die.
  glfwTerminate();
  return 0;
}