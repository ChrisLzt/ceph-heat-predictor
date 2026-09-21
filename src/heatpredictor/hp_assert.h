#pragma once

#include <cstdio>
#include <cstdlib>

// Keep invariant checks active in release builds, without linking Ceph.
namespace hp_detail {
[[noreturn]] inline void assertion_failed(const char* expression,
                                         const char* file, int line) {
  std::fprintf(stderr, "heatpredictor assertion failed: %s (%s:%d)\n",
               expression, file, line);
  std::abort();
}
}
#define hp_assert(expression) \
  ((expression) ? static_cast<void>(0) : \
   hp_detail::assertion_failed(#expression, __FILE__, __LINE__))
