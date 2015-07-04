#pragma once
#include <array>
#include <vector>
#include <functional>
#include <cstdint>

template <typename T>
class Matrix
{
public:
  Matrix(int a_width = 0, int a_height = 0) :
    m_width(a_width), m_height(a_height), m_storage(a_width * a_height) {};

  Matrix(int a_width, int a_height, std::initializer_list<T> a_init) :
    m_width(a_width), m_height(a_height), m_storage(a_init) {};

  void Fill(std::function<T()> a_func)
  {
    for (int i = 0; i < m_width*m_height; i++)
      m_storage[i] = a_func();
  }

  Matrix<T> Multiply(Matrix<T> a_other)
  {
    if (m_height != a_other.m_width)
      throw std::invalid_argument("Incompatible array dimensions!");

    Matrix<T> result(m_width, a_other.m_height);

    for (int x = 0; x < m_width; x++)
    {
      for (int y = 0; y < a_other.m_height; y++)
      {
        T dot = 0;
        for (int k = 0; k < m_height; k++)
          dot += m_storage[m_height*x + k] * a_other.m_storage[a_other.m_height*k + y];
        result.m_storage[a_other.m_height*x + y] = dot;
      }
    }

    return result;
  }

  Matrix<T> Tanh()
  {
    Matrix<T> result(m_width, m_height);
    for (int i = 0; i < m_storage.size(); i++)
      result.m_storage[i] = tanh(m_storage[i]);
    return result;
  }

  Matrix<T> Sigmoid()
  {
    Matrix<T> result(m_width, m_height);
    for (int i = 0; i < m_storage.size(); i++)
      result.m_storage[i] = (T)1.0 / (1 + exp(-m_storage[i]));
    return result;
  }

  int m_width, m_height;
  std::vector<T> m_storage;
};

template <typename T>
class Pixel
{
public:
  std::array<T, 3> m_storage;
};

class BrainCpu
{
public:
  BrainCpu();

  Pixel<float> Think(float x, float y, float z);
  void Dream(int a_width, int a_height, float a_z, uint8_t* a_dest);

protected:
  static const int s_networkSize = 16;   // Neurons per layer
  static const int s_nIn         = 3;    // Input layer size
  static const int s_nHidden     = 8;    // Hidden layers
  static const int s_nOut        = 3;    // Output layer size

  Matrix<float> m_layerInput;
  std::array<Matrix<float>, s_nHidden> m_layersHidden;
  Matrix<float> m_layerOutput;
};
