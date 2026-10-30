#include "slang-ir-wgsl-legalize.h"

#include "slang-ir-insts.h"
#include "slang-ir-legalize-binary-operator.h"
#include "slang-ir-legalize-global-values.h"
#include "slang-ir-legalize-varying-params.h"
#include "slang-ir-layout.h"
#include "slang-ir-specialize-address-space.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

namespace Slang
{

// Returns true if the `argIndex`-th parameter of `call`'s callee is passed with
// copy-in/out (`out`/`inout`) semantics. Such an argument may be bridged through a
// `function`-space temporary, whereas a `ref`/borrow parameter aliases the caller's
// object (e.g. `workgroupUniformLoad` on groupshared memory) and must receive the
// real pointer. Returns false when the callee signature is unavailable, so an
// argument of unknown mode is left untouched.
static bool paramHasCopyInOutSemantics(IRCall* call, UInt argIndex)
{
    auto funcType = as<IRFuncType>(call->getCallee()->getFullType());
    if (!funcType || argIndex >= funcType->getParamCount())
        return false;
    auto paramType = funcType->getParamType(argIndex);
    return as<IROutParamType>(paramType) || as<IRBorrowInOutParamType>(paramType);
}

static void legalizeCall(IRCall* call)
{
    // WGSL does not allow forming a pointer to a sub part of a composite value.
    // For example, if we have
    // ```
    // struct S { float x; float y; };
    // void foo(inout float v) { v = 1.0f; }
    // void main() { S s; foo(s.x); }
    // ```
    // The call to `foo(s.x)` is illegal in WGSL because `s.x` is a sub part of `s`.
    // And trying to form `&s.x` in WGSL is illegal.
    // To work around this, we will create a local variable to hold the sub part of
    // the composite value.
    // And then pass the local variable to the function.
    // After the call, we will write back the local variable to the sub part of the
    // composite value.
    //
    IRBuilder builder(call);
    builder.setInsertBefore(call);
    struct WritebackPair
    {
        IRInst* dest;
        IRInst* value;
    };
    ShortList<WritebackPair> pendingWritebacks;

    for (UInt i = 0; i < call->getArgCount(); i++)
    {
        auto arg = call->getArg(i);
        auto ptrType = as<IRPtrTypeBase>(arg->getDataType());
        if (!ptrType)
            continue;

        // User buffer pointers (BDA / mega-buffer byte offsets) are passed as scalar offsets,
        // not stack references. Skip bridging them through function-space temporaries.
        if (isUserPointerType(ptrType) || ptrType->getAddressSpace() == AddressSpace::UserPointer)
            continue;

        auto argOp = arg->getOp();

        // A by-ref `Param` and a block-local `Var` are already in the `function` address
        // space, so they can be passed to a `ptr<function, T>` parameter directly.
        if (argOp == kIROp_Param)
            continue;
        if (argOp == kIROp_Var && as<IRBlock>(arg->getParent()))
            continue;

        // A module-scope object (`GlobalVar`/`GlobalParam`, or a `Var` outside a block)
        // lives in a module-scope space such as `private`. Passing it directly to a
        // `ptr<function, T>` (`out`/`inout`) parameter is the address-space mismatch WGSL
        // rejects (#12173), so bridge it through the copy-in/out temp below. A `ref`/borrow
        // parameter instead aliases the caller's real object (e.g. `workgroupUniformLoad`
        // on groupshared memory), so it must keep receiving that object directly.
        if (argOp == kIROp_Var || argOp == kIROp_GlobalVar || argOp == kIROp_GlobalParam)
        {
            if (!paramHasCopyInOutSemantics(call, i))
                continue;
        }
        // Otherwise the argument is a pointer to a sub-part of a composite value (e.g.
        // `s.x`), which WGSL cannot address directly — bridge it unconditionally, as before.

        // Create a local variable to hold the input argument.
        auto var = builder.emitVar(ptrType->getValueType(), AddressSpace::Function);

        // Store the input argument into the local variable.
        builder.emitStore(var, builder.emitLoad(arg));
        builder.replaceOperand(call->getArgs() + i, var);
        pendingWritebacks.add({arg, var});
    }

    // Perform writebacks after the call.
    builder.setInsertAfter(call);
    for (auto& pair : pendingWritebacks)
    {
        builder.emitStore(pair.dest, builder.emitLoad(pair.value));
    }
}

static void legalizeFunc(IRFunc* func)
{
    // Insert casts to convert integer return types
    auto funcReturnType = func->getResultType();
    if (isIntegralType(funcReturnType))
    {
        for (auto block : func->getBlocks())
        {
            if (auto returnInst = as<IRReturn>(block->getTerminator()))
            {
                auto returnedValue = returnInst->getOperand(0);
                auto returnedValueType = returnedValue->getDataType();
                if (isIntegralType(returnedValueType))
                {
                    IRBuilder builder(returnInst);
                    builder.setInsertBefore(returnInst);
                    auto newOp = builder.emitCast(funcReturnType, returnedValue);
                    builder.replaceOperand(returnInst->getOperands(), newOp);
                }
            }
        }
    }
}

static void legalizeSwitch(IRSwitch* switchInst)
{
    // WGSL Requires all switch statements to contain a default case.
    // If the switch statement does not contain a default case, we will add one.
    if (switchInst->getDefaultLabel() != switchInst->getBreakLabel())
        return;
    IRBuilder builder(switchInst);
    auto defaultBlock = builder.createBlock();
    builder.setInsertInto(defaultBlock);
    builder.emitBranch(switchInst->getBreakLabel());
    defaultBlock->insertBefore(switchInst->getBreakLabel());
    List<IRInst*> cases;
    for (UInt i = 0; i < switchInst->getCaseCount(); i++)
    {
        cases.add(switchInst->getCaseValue(i));
        cases.add(switchInst->getCaseLabel(i));
    }
    builder.setInsertBefore(switchInst);
    auto newSwitch = builder.emitSwitch(
        switchInst->getCondition(),
        switchInst->getBreakLabel(),
        defaultBlock,
        (UInt)cases.getCount(),
        cases.getBuffer());
    switchInst->transferDecorationsTo(newSwitch);
    switchInst->removeAndDeallocate();
}

static void processInst(IRInst* inst, TargetProgram* targetProgram, DiagnosticSink* sink)
{
    switch (inst->getOp())
    {
    case kIROp_Call:
        legalizeCall(static_cast<IRCall*>(inst));
        break;

    case kIROp_Switch:
        legalizeSwitch(as<IRSwitch>(inst));
        break;

    // For all binary operators, make sure both side of the operator have the same type
    // (vector-ness and matrix-ness).
    case kIROp_Add:
    case kIROp_Sub:
    case kIROp_Mul:
    case kIROp_Div:
    case kIROp_FRem:
    case kIROp_IRem:
    case kIROp_And:
    case kIROp_Or:
    case kIROp_BitAnd:
    case kIROp_BitOr:
    case kIROp_BitXor:
    case kIROp_Lsh:
    case kIROp_Rsh:
    case kIROp_Eql:
    case kIROp_Neq:
    case kIROp_Greater:
    case kIROp_Less:
    case kIROp_Geq:
    case kIROp_Leq:
        legalizeBinaryOp(inst, sink, targetProgram);
        break;

    case kIROp_Func:
        legalizeFunc(static_cast<IRFunc*>(inst));
        [[fallthrough]];
    default:
        for (auto child : inst->getModifiableChildren())
        {
            processInst(child, targetProgram, sink);
        }
    }
}

struct GlobalInstInliningContext : public GlobalInstInliningContextGeneric
{
    bool isLegalGlobalInstForTarget(IRInst* inst) override
    {
        switch (inst->getOp())
        {
        case kIROp_MakeStruct:
        case kIROp_MakeArray:
        case kIROp_MakeArrayFromElement:
        case kIROp_MakeVector:
        case kIROp_MakeMatrix:
        case kIROp_MakeMatrixFromScalar:
        case kIROp_MakeVectorFromScalar:
            return true;
        default:
            break;
        }
        // The global instructions that are generically considered legal are fine for
        // WGSL.
        return false;
    }

    bool isInlinableGlobalInstForTarget(IRInst* /* inst */) override
    {
        // The global instructions that are generically considered inlineable are fine
        // for WGSL.
        return false;
    }

    bool shouldBeInlinedForTarget(IRInst* /* user */) override
    {
        // WGSL doesn't do any extra inlining beyond what is generically done by default.
        return false;
    }

    IRInst* getOutsideASM(IRInst* beforeInst) override
    {
        // Not needed for WGSL, check e.g. the SPIR-V case to see why this is used.
        return beforeInst;
    }
};

void legalizeIRForWGSL(IRModule* module, TargetProgram* targetProgram, DiagnosticSink* sink)
{
    List<EntryPointInfo> entryPoints;
    for (auto inst : module->getGlobalInsts())
    {
        IRFunc* const func{as<IRFunc>(inst)};
        if (!func)
            continue;
        IREntryPointDecoration* const entryPointDecor =
            func->findDecoration<IREntryPointDecoration>();
        if (!entryPointDecor)
            continue;
        EntryPointInfo info;
        info.entryPointDecor = entryPointDecor;
        info.entryPointFunc = func;
        entryPoints.add(info);
    }

    legalizeEntryPointVaryingParamsForWGSL(module, sink, entryPoints);

    // Go through every instruction in the module and legalize them as needed.
    processInst(module->getModuleInst(), targetProgram, sink);

    // Some global insts are illegal, e.g. function calls.
    // We need to inline and remove those.
    GlobalInstInliningContext().inlineGlobalValuesAndRemoveIfUnused(module);
}

struct WGSLAddressSpaceAssigner : InitialAddressSpaceAssigner
{
    virtual bool tryAssignAddressSpace(IRInst* inst, AddressSpace& outAddressSpace) override
    {
        switch (inst->getOp())
        {
        case kIROp_Var:
            if (as<IRBlock>(inst->getParent()))
                outAddressSpace = AddressSpace::Function;
            else
                outAddressSpace = AddressSpace::ThreadLocal;
            return true;
        case kIROp_RWStructuredBufferGetElementPtr:
            outAddressSpace = AddressSpace::Global;
            return true;
        case kIROp_Load:
            {
                auto addrSpace = getAddressSpaceFromVarType(inst->getDataType());
                if (addrSpace != AddressSpace::Generic)
                {
                    outAddressSpace = addrSpace;
                    return true;
                }
            }
            return false;
        default:
            return false;
        }
    }

    virtual AddressSpace getAddressSpaceFromVarType(IRInst* type) override
    {
        if (as<IRUniformParameterGroupType>(type))
        {
            return AddressSpace::Uniform;
        }
        if (as<IRByteAddressBufferTypeBase>(type))
        {
            return AddressSpace::Global;
        }
        if (as<IRHLSLStructuredBufferTypeBase>(type))
        {
            return AddressSpace::Global;
        }
        if (as<IRGLSLShaderStorageBufferType>(type))
        {
            return AddressSpace::Global;
        }
        if (auto ptrType = as<IRPtrTypeBase>(type))
        {
            if (ptrType->hasAddressSpace())
                return ptrType->getAddressSpace();
            return AddressSpace::Generic;
        }
        return AddressSpace::Generic;
    }

    virtual AddressSpace getLeafInstAddressSpace(IRInst* inst) override
    {
        if (as<IRGroupSharedRate>(inst->getRate()))
            return AddressSpace::GroupShared;
        switch (inst->getOp())
        {
        case kIROp_RWStructuredBufferGetElementPtr:
            return AddressSpace::Global;
        case kIROp_Var:
            if (as<IRBlock>(inst->getParent()))
                return AddressSpace::Function;
            else
                return AddressSpace::ThreadLocal;
            break;
        default:
            break;
        }
        auto type = unwrapAttributedType(inst->getDataType());
        if (!type)
            return AddressSpace::Generic;
        return getAddressSpaceFromVarType(type);
    }
};

void specializeAddressSpaceForWGSL(IRModule* module)
{
    WGSLAddressSpaceAssigner wgslAddressSpaceAssigner;
    specializeAddressSpace(module, &wgslAddressSpaceAssigner);
}

static IRInst* _findOrCreateMegaBuffer(IRModule* module, bool isRW, int binding, int group)
{
    for (auto inst : module->getGlobalInsts())
    {
        if (auto globalParam = as<IRGlobalParam>(inst))
        {
            if (auto nameHint = globalParam->findDecoration<IRNameHintDecoration>())
            {
                if (nameHint->getName() == "__mega_buffer")
                    return globalParam;
            }
        }
    }

    IRBuilder builder(module);
    builder.setInsertBefore(module->getModuleInst()->getFirstChild());
    IRType* bufferType = (IRType*)builder.getType(
        isRW ? kIROp_HLSLRWByteAddressBufferType : kIROp_HLSLByteAddressBufferType);
    auto megaBuffer = builder.createGlobalParam(bufferType);
    builder.addNameHintDecoration(megaBuffer, toSlice("__mega_buffer"));

    auto resKind = isRW ? LayoutResourceKind::UnorderedAccess : LayoutResourceKind::ShaderResource;
    IRTypeLayout::Builder typeLayoutBuilder(&builder);
    typeLayoutBuilder.addResourceUsage(resKind, 1);
    auto typeLayout = typeLayoutBuilder.build();
    IRVarLayout::Builder varLayoutBuilder(&builder, typeLayout);
    varLayoutBuilder.findOrAddResourceInfo(LayoutResourceKind::RegisterSpace)->offset = group;
    varLayoutBuilder.findOrAddResourceInfo(resKind)->offset = binding;
    auto varLayout = varLayoutBuilder.build();
    builder.addLayoutDecoration(megaBuffer, varLayout);

    return megaBuffer;
}

static bool _isRawPointer(IRInst* inst)
{
    auto type = inst->getDataType();
    if (!type)
        return false;
    if (isUserPointerType(type))
        return true;
    if (type->getOp() == kIROp_RawPointerType)
        return true;
    if (auto ptrType = as<IRPtrTypeBase>(type))
    {
        if (ptrType->getAddressSpace() == AddressSpace::UserPointer)
            return true;
    }
    return false;
}

static IRInst* _getPointerByteOffset(
    IRBuilder& builder,
    IRInst* ptr,
    TargetProgram* targetProgram,
    Dictionary<IRInst*, IRInst*>& offsetCache)
{
    const bool inBlock = as<IRBlock>(ptr->getParent()) != nullptr;
    IRInst* cached = nullptr;
    if (inBlock && offsetCache.tryGetValue(ptr, cached))
        return cached;

    auto savedLoc = builder.getInsertLoc();
    if (inBlock)
    {
        setInsertAfterOrdinaryInst(&builder, ptr);
    }

    IRInst* result = nullptr;
    switch (ptr->getOp())
    {
    case kIROp_CastIntToPtr:
        result = builder.emitCast(builder.getUIntType(), ptr->getOperand(0));
        break;

    case kIROp_PtrCast:
    case kIROp_BitCast:
        result = _getPointerByteOffset(builder, ptr->getOperand(0), targetProgram, offsetCache);
        break;

    case kIROp_GetOffsetPtr:
    case kIROp_GetElementPtr:
        {
            auto basePtr = ptr->getOperand(0);
            auto index = ptr->getOperand(1);
            auto baseOffset = _getPointerByteOffset(builder, basePtr, targetProgram, offsetCache);
            auto ptrType = as<IRPtrTypeBase>(ptr->getDataType());
            SLANG_ASSERT(ptrType);
            auto elemType = (IRType*)ptrType->getValueType();
            IRSizeAndAlignment sizeAlignment;
            getNaturalSizeAndAlignment(targetProgram->getTargetReq(), elemType, &sizeAlignment);
            auto indexU32 = builder.emitCast(builder.getUIntType(), index);
            auto strideVal = builder.getIntValue(builder.getUIntType(), (uint32_t)sizeAlignment.getStride());
            auto delta = builder.emitMul(builder.getUIntType(), indexU32, strideVal);
            result = builder.emitAdd(builder.getUIntType(), baseOffset, delta);
        }
        break;

    case kIROp_FieldAddress:
        {
            auto basePtr = ptr->getOperand(0);
            auto fieldKey = ptr->getOperand(1);
            auto baseOffset = _getPointerByteOffset(builder, basePtr, targetProgram, offsetCache);
            auto ptrType = as<IRPtrTypeBase>(basePtr->getDataType());
            SLANG_ASSERT(ptrType);
            auto structType = as<IRStructType>(ptrType->getValueType());
            SLANG_ASSERT(structType);
            IRStructField* targetField = nullptr;
            for (auto field : structType->getFields())
            {
                if (field->getKey() == fieldKey)
                {
                    targetField = field;
                    break;
                }
            }
            IRIntegerValue fieldOffset = 0;
            if (targetField)
            {
                getNaturalOffset(targetProgram->getTargetReq(), targetField, &fieldOffset);
            }
            auto offsetVal = builder.getIntValue(builder.getUIntType(), (uint32_t)fieldOffset);
            result = builder.emitAdd(builder.getUIntType(), baseOffset, offsetVal);
        }
        break;

    case kIROp_Add:
        {
            IRInst* ptrOp = _isRawPointer(ptr->getOperand(0)) ? ptr->getOperand(0) : ptr->getOperand(1);
            IRInst* idxOp = (ptrOp == ptr->getOperand(0)) ? ptr->getOperand(1) : ptr->getOperand(0);
            auto baseOffset = _getPointerByteOffset(builder, ptrOp, targetProgram, offsetCache);
            auto ptrType = as<IRPtrTypeBase>(ptrOp->getDataType());
            auto elemType = (IRType*)ptrType->getValueType();
            IRSizeAndAlignment sizeAlignment;
            getNaturalSizeAndAlignment(targetProgram->getTargetReq(), elemType, &sizeAlignment);
            auto indexU32 = builder.emitCast(builder.getUIntType(), idxOp);
            auto strideVal = builder.getIntValue(builder.getUIntType(), (uint32_t)sizeAlignment.getStride());
            auto delta = builder.emitMul(builder.getUIntType(), indexU32, strideVal);
            result = builder.emitAdd(builder.getUIntType(), baseOffset, delta);
        }
        break;

    case kIROp_Sub:
        {
            auto baseOffset = _getPointerByteOffset(builder, ptr->getOperand(0), targetProgram, offsetCache);
            auto ptrType = as<IRPtrTypeBase>(ptr->getOperand(0)->getDataType());
            auto elemType = (IRType*)ptrType->getValueType();
            IRSizeAndAlignment sizeAlignment;
            getNaturalSizeAndAlignment(targetProgram->getTargetReq(), elemType, &sizeAlignment);
            auto indexU32 = builder.emitCast(builder.getUIntType(), ptr->getOperand(1));
            auto strideVal = builder.getIntValue(builder.getUIntType(), (uint32_t)sizeAlignment.getStride());
            auto delta = builder.emitMul(builder.getUIntType(), indexU32, strideVal);
            result = builder.emitSub(builder.getUIntType(), baseOffset, delta);
        }
        break;
    default:
        result = builder.emitCast(builder.getUIntType(), builder.emitCastPtrToInt(ptr));
        break;
    }

    builder.setInsertLoc(savedLoc);
    if (inBlock)
    {
        offsetCache.add(ptr, result);
    }
    return result;
}

bool lowerRawPointersForWGSL(
    IRModule* module,
    TargetProgram* targetProgram,
    DiagnosticSink* sink)
{
    SLANG_UNUSED(sink);
    if (!isWGPUTarget(targetProgram->getTargetReq()))
        return false;

    auto bdaOption = targetProgram->getOptionSet().options.tryGetValue(CompilerOptionName::WGSLBindBdaBuffer);
    if (!bdaOption || bdaOption->getCount() == 0)
        return false;
    const int binding = (*bdaOption)[0].intValue;
    const int group = (*bdaOption)[0].intValue2;

    HashSet<IRFunc*> reachableFuncs;
    List<IRFunc*> workList;
    for (auto inst : module->getGlobalInsts())
    {
        if (auto func = as<IRFunc>(inst))
        {
            if (func->findDecoration<IREntryPointDecoration>())
            {
                if (reachableFuncs.add(func))
                    workList.add(func);
            }
        }
    }
    if (workList.getCount() == 0)
    {
        for (auto inst : module->getGlobalInsts())
        {
            if (auto func = as<IRFunc>(inst))
            {
                if (reachableFuncs.add(func))
                    workList.add(func);
            }
        }
    }
    while (workList.getCount() > 0)
    {
        auto func = workList.getLast();
        workList.removeLast();
        for (auto block : func->getBlocks())
        {
            for (auto child : block->getChildren())
            {
                if (auto call = as<IRCall>(child))
                {
                    if (auto callee = as<IRFunc>(call->getCallee()))
                    {
                        if (reachableFuncs.add(callee))
                            workList.add(callee);
                    }
                }
            }
        }
    }

    List<IRLoad*> rawLoads;
    List<IRStore*> rawStores;
    bool hasRawPointerAccess = false;

    for (auto func : reachableFuncs)
    {
        for (auto block : func->getBlocks())
        {
            for (auto child : block->getChildren())
            {
                if (auto load = as<IRLoad>(child))
                {
                    if (_isRawPointer(load->getPtr()))
                    {
                        rawLoads.add(load);
                        hasRawPointerAccess = true;
                    }
                }
                else if (auto store = as<IRStore>(child))
                {
                    if (_isRawPointer(store->getPtr()))
                    {
                        rawStores.add(store);
                        hasRawPointerAccess = true;
                    }
                }
            }
        }
    }

    if (!hasRawPointerAccess)
        return false;

    bool isCompute = false;
    for (auto inst : module->getGlobalInsts())
    {
        if (auto func = as<IRFunc>(inst))
        {
            if (auto entryPointDecor = func->findDecoration<IREntryPointDecoration>())
            {
                if (entryPointDecor->getProfile().getStage() == Stage::Compute)
                {
                    isCompute = true;
                    break;
                }
            }
        }
    }
    const bool isRW = isCompute;

    auto megaBuffer = _findOrCreateMegaBuffer(module, isRW, binding, group);
    Dictionary<IRInst*, IRInst*> offsetCache;

    for (auto load : rawLoads)
    {
        IRBuilder builder(load);
        builder.setInsertBefore(load);
        auto byteOffset = _getPointerByteOffset(builder, load->getPtr(), targetProgram, offsetCache);
        auto alignment = builder.getIntValue(builder.getUIntType(), 0);
        IRInst* args[] = {megaBuffer, byteOffset, alignment};
        auto bufLoad = builder.emitIntrinsicInst(
            load->getDataType(),
            kIROp_ByteAddressBufferLoad,
            3,
            args);
        load->replaceUsesWith(bufLoad);
        load->removeAndDeallocate();
    }

    for (auto store : rawStores)
    {
        IRBuilder builder(store);
        builder.setInsertBefore(store);
        auto byteOffset = _getPointerByteOffset(builder, store->getPtr(), targetProgram, offsetCache);
        auto alignment = builder.getIntValue(builder.getUIntType(), 0);
        builder.emitByteAddressBufferStore(megaBuffer, byteOffset, alignment, store->getVal());
        store->removeAndDeallocate();
    }

    return true;
}

static IRInst* _invertYOfWGSLVector(IRBuilder& builder, IRInst* originalVector)
{
    auto vectorType = as<IRVectorType>(originalVector->getDataType());
    SLANG_ASSERT(vectorType);
    auto elemType = vectorType->getElementType();
    IRInst* one = builder.getFloatValue(elemType, 1.0);
    IRInst* negOne = builder.getFloatValue(elemType, -1.0);
    auto elemCountInst = as<IRIntLit>(vectorType->getElementCount());
    int elemCount = elemCountInst ? (int)elemCountInst->getValue() : 4;
    ShortList<IRInst*> elements;
    for (int i = 0; i < elemCount; i++)
    {
        elements.add(i == 1 ? negOne : one);
    }
    auto scale = builder.emitMakeVector(
        vectorType,
        (UInt)elements.getCount(),
        elements.getArrayView().getBuffer());
    return builder.emitMul(vectorType, originalVector, scale);
}

static bool _isPositionSemantic(UnownedStringSlice name)
{
    return name.caseInsensitiveEquals(toSlice("position")) ||
           name.caseInsensitiveEquals(toSlice("sv_position"));
}

static bool _isPositionKey(IRInst* key)
{
    if (auto sysValDecor = key->findDecoration<IRTargetSystemValueDecoration>())
    {
        if (_isPositionSemantic(sysValDecor->getSemantic()))
            return true;
    }
    if (auto semDecor = key->findDecoration<IRSemanticDecoration>())
    {
        if (_isPositionSemantic(semDecor->getSemanticName()))
            return true;
    }
    return false;
}

void invertYOfWGSLVertexPositionOutput(IRModule* module)
{
    for (auto globalInst : module->getGlobalInsts())
    {
        auto func = as<IRFunc>(globalInst);
        if (!func)
            continue;
        auto entryPointDecor = func->findDecoration<IREntryPointDecoration>();
        if (!entryPointDecor || entryPointDecor->getProfile().getStage() != Stage::Vertex)
            continue;

        IRBuilder builder(module);
        if (as<IRVectorType>(func->getResultType()))
        {
            for (auto block : func->getBlocks())
            {
                if (auto ret = as<IRReturn>(block->getTerminator()))
                {
                    builder.setInsertBefore(ret);
                    auto originalVal = ret->getVal();
                    auto invertedVal = _invertYOfWGSLVector(builder, originalVal);
                    builder.replaceOperand(ret->getOperands(), invertedVal);
                }
            }
        }
        else if (auto structType = as<IRStructType>(func->getResultType()))
        {
            Index positionFieldIndex = -1;
            Index currentIndex = 0;
            IRStructKey* positionKey = nullptr;
            for (auto field : structType->getFields())
            {
                if (_isPositionKey(field->getKey()))
                {
                    positionFieldIndex = currentIndex;
                    positionKey = field->getKey();
                    break;
                }
                currentIndex++;
            }
            if (positionFieldIndex != -1)
            {
                for (auto block : func->getBlocks())
                {
                    for (auto inst : block->getChildren())
                    {
                        if (auto store = as<IRStore>(inst))
                        {
                            if (auto fieldAddr = as<IRFieldAddress>(store->getPtr()))
                            {
                                if (fieldAddr->getField() == positionKey)
                                {
                                    builder.setInsertBefore(store);
                                    auto originalVal = store->getVal();
                                    if (as<IRVectorType>(originalVal->getDataType()))
                                    {
                                        auto invertedVal = _invertYOfWGSLVector(builder, originalVal);
                                        builder.replaceOperand(&store->val, invertedVal);
                                    }
                                }
                            }
                        }
                        else if (inst->getOp() == kIROp_MakeStruct && inst->getDataType() == structType)
                        {
                            if ((UInt)positionFieldIndex < inst->getOperandCount())
                            {
                                auto posVal = inst->getOperand(positionFieldIndex);
                                if (as<IRVectorType>(posVal->getDataType()))
                                {
                                    builder.setInsertBefore(inst);
                                    auto invertedVal = _invertYOfWGSLVector(builder, posVal);
                                    builder.replaceOperand(
                                        inst->getOperands() + positionFieldIndex,
                                        invertedVal);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace Slang
