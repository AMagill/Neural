#include "brainCpu.h"
#include <random>

BrainCpu::BrainCpu()
{
  // Allocate layers
  m_layerInput = Matrix<float>(s_networkSize, s_nIn);
  for (auto& layer : m_layersHidden)
    layer = Matrix<float>(s_networkSize, s_networkSize);
  m_layerOutput = Matrix<float>(s_nOut, s_networkSize);

  // Fill the layers with random numbers
  std::random_device rand;
  std::uniform_real_distribution<float> dist(-1, 1);

  m_layerInput.Fill([&]() { return dist(rand); });
  for (auto &layer : m_layersHidden)
    layer.Fill([&]() { return dist(rand); });
  m_layerOutput.Fill([&]() { return dist(rand); });
}

Pixel<float> BrainCpu::Think(float x, float y, float z)
{
  Matrix<float> input(3, 1, { x, y, z });

  Matrix<float> out = m_layerInput.Multiply(input).Tanh();
  for (auto& layer : m_layersHidden)
    out = layer.Multiply(out).Tanh();
  out = m_layerOutput.Multiply(out).Sigmoid();

  return Pixel<float> { out.m_storage[0], out.m_storage[1], out.m_storage[2] };
}

void BrainCpu::Dream(int a_width, int a_height, float a_z, uint8_t* a_dest)
{
  for (int y = 0; y < a_height; y++)
  {
    for (int x = 0; x < a_width; x++)
    {
      Pixel<float> color = Think((float)x / a_width - 0.5f, (float)y / a_height - 0.5f, a_z);
      *a_dest++ = (uint8_t)(color.m_storage[0] * 255.0);
      *a_dest++ = (uint8_t)(color.m_storage[1] * 255.0);
      *a_dest++ = (uint8_t)(color.m_storage[2] * 255.0);
    }
  }
}