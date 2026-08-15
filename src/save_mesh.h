// save_mesh.h
//
// Small helper used by the best_fit_* inverse-fit drivers to dump a simulated
// mesh to disk.  Writes an .off via fsim::saveOFF, creating the parent
// directory if it does not exist, and echoes the path so batch logs record
// where results went.

#pragma once

#include <fsim/util/io.h>
#include <fsim/util/typedefs.h>

#include <filesystem>
#include <iostream>
#include <string>

inline void saveMesh(const std::string& file,
                     const Eigen::Ref<const fsim::Mat3<double>> V,
                     const Eigen::Ref<const fsim::Mat3<int>> F)
{
  const std::filesystem::path p(file);
  if (p.has_parent_path())
    std::filesystem::create_directories(p.parent_path());

  fsim::saveOFF(file, V, F);
  std::cout << "  wrote " << file << "\n" << std::flush;
}
