#include <stdio.h>
#include <vector>
#include <array>
#include <random>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

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

void onResize(GLFWwindow* a_window, int a_width, int a_height)
{
  glViewport(0, 0, a_width, a_height);
}

int main()
{
  static const int winWidth  = 256; // Initial window size
  static const int winHeight = 256;

  static const int width     = 256; // Size of the image produced
  static const int height    = 256;
  static const int nNeurons  = 16;  // Neurons per layer
  static const int nLayers   = 10;  // Number of layers

  // Init OpenGL and make a window via GLFW
  if (!glfwInit())
    return -1;

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

  GLFWwindow* window = glfwCreateWindow(winWidth, winHeight, "Neural", NULL, NULL);
  if (!window)
  {
    printf("Failed to create OpenGL context\n");
    glfwTerminate();
    return -1;
  }

  glfwMakeContextCurrent(window);
  glfwSetWindowSizeCallback(window, onResize);

  // Load extensions with glad
  if (!gladLoadGL()) {
    printf("Failed to load OpenGL extensions\n");
    return -1;
  }

  GLuint error;
  error = glGetError();

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
  // Set up the FSQ geometry
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
  gl_Position = vec4(iPosition, 0.0, 1.0);
})";
  const char* fragShaderSrc = R"(#version 440
const uint nNeurons = 16;
const uint nLayers  = 10;
const float freq[]  = {8.6, 7.5, 3.0, 9.8, 6.7, 5.3, 0.9, 8.6}; // Arbitrarily selected values
const float phase[] = {3.1, 4.1, 5.9, 2.6, 5.3, 5.8, 9.8, 1.2};
const float speed   = 0.05;

in vec2 pos;
out vec4 oFragColor;

layout(binding = 0) uniform sampler3D uNeuralNet;
uniform float uTime;

float scratchA[nNeurons];
float scratchB[nNeurons];

void multiply(uint layer) {
  for (uint x = 0; x < nNeurons; x++) {
    float dot = 0.0;
    for (uint y = 0; y < nNeurons; y++) {
      float cell = texture(uNeuralNet, vec3(float(x)/nNeurons, float(y)/nNeurons, float(layer)/nLayers)).r * 2.0 - 1.0;
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

void main(void) {
  scratchA[0] = pos.x * 4.0 - 2.0;
  scratchA[1] = pos.y * 4.0 - 2.0;
  for (int i = 0; i < 8; i++)
    scratchA[i+2] = sin(uTime*speed*freq[i]+phase[i]);
  for (int i = 10; i < nNeurons; i++)
    scratchA[i] = 0.0;

  for (int i = 0; i < nLayers-1; i++) {
    multiply(i);
    arr_tanh();
  }

  multiply(nLayers-1);
  arr_sigmoid();

  oFragColor = vec4(scratchA[0], scratchA[1], scratchA[2], 1.0);
  //oFragColor = vec4(vec3(scratchA[int(pos.x*nNeurons)]) * 0.5 + 0.5, 1.0);
  //uint foo = uint(pos.x * 16.0);
  //oFragColor = vec4(vec3(float(foo)/uint(16)), 1.0);
  //oFragColor = vec4(texture(uNeuralNet, vec3(pos.xy, fract(uTime*0.5))).rrr, 1.0);
})";
  GLuint vertShader = CompileShader(vertShaderSrc, GL_VERTEX_SHADER);
  GLuint fragShader = CompileShader(fragShaderSrc, GL_FRAGMENT_SHADER);
  GLuint renderProgram = LinkProgram({ vertShader, fragShader });
  glUseProgram(renderProgram);
  error = glGetError();
  // Set up the neural net buffer
  GLuint neuralNetTex;
  glGenTextures(1, &neuralNetTex);
  glBindTexture(GL_TEXTURE_3D, neuralNetTex);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
  {
    const int totalSize = nNeurons * nNeurons * nLayers;
    std::random_device rand;
    std::uniform_real_distribution<float> dist(-1, 1);
    std::array<float, totalSize> temp;
    for (int i = 0; i < totalSize; i++)
      temp[i] = dist(rand) * 0.5 + 0.5;
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RED, nNeurons, nNeurons, nLayers, 0, GL_RED, GL_FLOAT, temp.data());
  }
  error = glGetError();
  if (error)
    return 0;
  
  // Main loop
  while (!glfwWindowShouldClose(window))
  {
    // Draw the image to screen
    glUseProgram(renderProgram);
    glUniform1f(glGetUniformLocation(renderProgram, "uTime"), (float)glfwGetTime());
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glfwSwapBuffers(window);
    glfwPollEvents();
    if (glfwGetKey(window, GLFW_KEY_ESCAPE))
      break;
  }

  glfwTerminate();
  return 0;
}

// Starting point for the Windows subsystem, so we can disable the console
int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int)
{
  main();
}
