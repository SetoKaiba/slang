#pragma once

#include "slang-ir.h"

namespace Slang
{
class DiagnosticSink;
class TargetProgram;

void legalizeIRForWGSL(IRModule* module, TargetProgram* targetProgram, DiagnosticSink* sink);

bool lowerRawPointersForWGSL(IRModule* module, TargetProgram* targetProgram, DiagnosticSink* sink);

void invertYOfWGSLVertexPositionOutput(IRModule* module);

void specializeAddressSpaceForWGSL(IRModule* module);

} // namespace Slang
