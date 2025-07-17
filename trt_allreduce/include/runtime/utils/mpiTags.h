/**
 * @file mpiTags.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief This file defines the MPI tags used in the trt_allreduce project.
 * @version 0.1
 * @date 2025-07-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

namespace trt_allreduce::mpi {

enum class MpiTag : int {
    kDefault = 0,

    // DecoderStepAsyncSend
    kDecoderStepNewOutputTokensHost = 0,
    kDecoderStepFinishedSumHost = 1,
    kDecoderStepSequenceLengthsHost = 2,
    kDecoderStepCumLogProbsHost = 3,
    kDecoderStepLogProbsHost = 4,
    kDecoderStepCacheIndirectionOutput = 5,
    kDecoderStepAcceptedLengthsCumSumDevice = 6,
    kDecoderStepAcceptedPackedPathsDevice = 7,
    kDecoderStepFinishReasonsHost = 8,

    // DecoderSlotAsyncSend
    kDecoderSlotOutputIds = 9,
    kDecoderSlotSequenceLengths = 10,
    kDecoderSlotCumLogProbs = 11,
    kDecoderSlotLogProbs = 12,

    // CancelledRequestsAsyncSend
    kCancelledRequestsNumReq = 13,
    kCancelledRequestsIds = 14,

    // RequestWithIdAsyncSend
    kRequestWithIdNumReq = 15,
    kRequestWithIdVecSize = 16,
    kRequestWithIdPacked = 17,

    // Executor
    kExecutorNumActiveRequests = 18,
    kExecutorLowestPriorityActiveHasValue = 19,
    kExecutorLowestPriorityActive = 20,
    kExecutorShouldExit = 21,

    // TrtGptModelInflightBatching
    kTrtGptModelInflightBatchingContextLogits = 22,
    kTrtGptModelInflightBatchingGenerationLogits = 23,

    // Orchestrator
    kOrchestratorId = 127,
    kOrchestratorData = 1023,
    kOrchestratorStatsId = 128,
    kOrchestratorStatsData = 1024,

    // LogitsThread
    kSpecDecLogitsId = 129,
    kSpecDecLogitsData = 1025,
};

}  // namespace trt_allreduce::mpi