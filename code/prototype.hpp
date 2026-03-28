#pragma once

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// MSVC 专属警告抑制（Debug 版 LibTorch 会产生大量 4251/4275）
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4251)
#pragma warning(disable: 4275)
#endif

namespace npy {

inline void save_f32(const std::string& path,
                     const float* data,
                     const std::vector<int64_t>& shape) {
  if (shape.empty()) {
    throw std::invalid_argument("npy::save_f32 shape must be non-empty");
  }

  std::ostringstream header;
  header << "{'descr': '<f4', 'fortran_order': False, 'shape': (";
  for (size_t i = 0; i < shape.size(); ++i) {
    header << shape[i];
    if (shape.size() == 1) {
      header << ",";  // 1-tuple needs a trailing comma
      break;
    }
    if (i + 1 < shape.size()) {
      header << ", ";
    }
  }
  header << "), }";

  std::string header_str = header.str();

  auto pad_header = [&](size_t prefix_len) {
    std::string h = header_str;
    size_t hlen = h.size();

    // Pad with spaces and a trailing newline so that the whole header is aligned to 16 bytes.
    h.push_back(' ');
    hlen += 1;
    while ((prefix_len + hlen + 1) % 16 != 0) {
      h.push_back(' ');
      hlen += 1;
    }
    h.push_back('\n');
    hlen += 1;
    return std::pair<std::string, size_t>(std::move(h), hlen);
  };

  // Try v1.0 first
  unsigned char version_major = 1;
  unsigned char version_minor = 0;
  size_t prefix_len = 10;  // magic(6) + version(2) + header_len(2)
  auto [padded_header, header_len] = pad_header(prefix_len);

  if (header_len > 0xFFFF) {
    // Fallback to v2.0
    version_major = 2;
    version_minor = 0;
    prefix_len = 12;  // magic(6) + version(2) + header_len(4)
    auto padded2 = pad_header(prefix_len);
    padded_header = std::move(padded2.first);
    header_len = padded2.second;
  }

  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("failed to open for write: " + path);
  }

  // Magic string + version
  const char magic[] = "\x93NUMPY";
  out.write(magic, 6);
  const unsigned char version[2] = {version_major, version_minor};
  out.write(reinterpret_cast<const char*>(version), 2);

  if (version_major == 1) {
    // Header length (uint16 little-endian)
    const uint16_t hlen = static_cast<uint16_t>(header_len);
    const unsigned char hlen_le[2] = {
        static_cast<unsigned char>(hlen & 0xFF),
        static_cast<unsigned char>((hlen >> 8) & 0xFF),
    };
    out.write(reinterpret_cast<const char*>(hlen_le), 2);
  } else {
    // Header length (uint32 little-endian)
    const uint32_t hlen = static_cast<uint32_t>(header_len);
    const unsigned char hlen_le[4] = {
        static_cast<unsigned char>(hlen & 0xFF),
        static_cast<unsigned char>((hlen >> 8) & 0xFF),
        static_cast<unsigned char>((hlen >> 16) & 0xFF),
        static_cast<unsigned char>((hlen >> 24) & 0xFF),
    };
    out.write(reinterpret_cast<const char*>(hlen_le), 4);
  }

  // Header payload
  out.write(padded_header.data(), static_cast<std::streamsize>(padded_header.size()));

  // Raw data payload
  size_t count = 1;
  for (auto s : shape) {
    if (s <= 0) {
      throw std::invalid_argument("npy::save_f32 shape entries must be > 0");
    }
    count *= static_cast<size_t>(s);
  }
  out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * sizeof(float)));
}

}  
#ifdef _MSC_VER
#pragma warning(pop)
#endif