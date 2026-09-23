#pragma once
#include <cstdint>

// Host storage adapters classify observed data accesses as reads or writes.
enum class HpAccessType : uint8_t { Read, Write };
