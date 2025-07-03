/**
 * @file nvtxUtils.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief NVTX utilities for TensorRT-LLM, providing color cycling for ranges.
 * @version 0.1
 * @date 2025-07-03
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <array>
#include <nvtx3/nvtx3.hpp>

namespace tensorrt_llm::common::nvtx {
inline nvtx3::color nextColor() {
#if !defined(NVTX_DISABLE)
    constexpr std::array kColors{nvtx3::color{0xff00ff00}, nvtx3::color{0xff0000ff}, nvtx3::color{0xffffff00}, nvtx3::color{0xffff00ff},
                                 nvtx3::color{0xff00ffff}, nvtx3::color{0xffff0000}, nvtx3::color{0xffffffff}};
    constexpr auto numColors = kColors.size();

    static thread_local std::size_t colorId = 0;
    auto const color = kColors[colorId];
    colorId = colorId + 1 >= numColors ? 0 : colorId + 1;
    return color;
#else
    return nvtx3::color{0};
#endif
}

}  // namespace tensorrt_llm::common::nvtx

#define NVTX3_SCOPED_RANGE(range) ::nvtx3::scoped_range range##_range(::tensorrt_llm::common::nvtx::nextColor(), #range)
