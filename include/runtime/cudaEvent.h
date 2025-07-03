/**
 * @file cudaEvent.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief A wrapper for CUDA events.
 * @version 0.1
 * @date 2025-07-03
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <cuda_runtime_api.h>

#include <memory>

#include "common/assert.h"
#include "common/cudaUtils.h"
#include "common/logger.h"

namespace tensorrt_llm::runtime {

class CudaEvent {
   public:
    using pointer = cudaEvent_t;

    //! Creates a new cuda event. The event will be destroyed in the destructor.
    //!
    //! \param flags Flags for event creation. By default, event timing is disabled.
    explicit CudaEvent(unsigned int flags = cudaEventDisableTiming) {
        pointer event;
        TLLM_CUDA_CHECK(::cudaEventCreate(&event, flags));
        TLLM_LOG_TRACE("Created event %p", event);
        bool constexpr ownsEvent{true};
        mEvent = EventPtr{event, Deleter{ownsEvent}};
    }

    //! Pass an existing cuda event to this object.
    //!
    //! \param event The event to pass to this object.
    //! \param ownsEvent Whether this object owns the event and destroys it in the destructor.
    explicit CudaEvent(pointer event, bool ownsEvent = true) {
        TLLM_CHECK_WITH_INFO(event != nullptr, "event is nullptr");
        mEvent = EventPtr{event, Deleter{ownsEvent}};
    }

    //! Returns the event associated with this object.
    [[nodiscard]] pointer get() const { return mEvent.get(); }

    //! \brief Synchronizes the event.
    void synchronize() const { TLLM_CUDA_CHECK(::cudaEventSynchronize(get())); }

   private:
    class Deleter {
       public:
        explicit Deleter(bool ownsEvent) : mOwnsEvent{ownsEvent} {}

        explicit Deleter() : Deleter{true} {}

        constexpr void operator()(pointer event) const {
            if (mOwnsEvent && event != nullptr) {
                TLLM_CUDA_CHECK(::cudaEventDestroy(event));
                TLLM_LOG_TRACE("Destroyed event %p", event);
            }
        }

       private:
        bool mOwnsEvent;
    };

    using element_type = std::remove_pointer_t<pointer>;
    using EventPtr = std::unique_ptr<element_type, Deleter>;

    EventPtr mEvent;
};

}  // namespace tensorrt_llm::runtime
