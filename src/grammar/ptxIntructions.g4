parser grammar ptxInstructions;

options {
    tokenVocab = ptxLexer;
}

import ptxOperands;

funcBody
    : LEFT_BRACE instructionList RIGHT_BRACE
    ;
instructionList
    : instruction*
    ;

instruction
    : controlFlowInst
    | arithmeticInst
    | logicalInst
    | dataMovementInst
    | parallelSyncInst
    | atomicInst
    | warpLevelInst
    | textureSurfaceInst
    | reductionPrefetchInst
    | matrixInst
    | videoSimdInst
    | newInst
    | SEMI
    ;

controlFlowInst
    : BRA (UNI | voteMode)? labelOperand SEMI
    | BRX (UNI | voteMode)? addressExpr COMMA labelOperand SEMI
    | CALL callParams? labelOperand callArgs? SEMI
    | RET SEMI
    | EXIT SEMI
    | TRAP SEMI
    | BRK (DECIMAL_INT)? SEMI
    | BRKPT SEMI
    ;

labelOperand : ID ;
voteMode : BALLOT | ANY | ALL | UNI | EQV ;
callParams : paramList ;
callArgs : LEFT_PAREN operand (COMMA operand)* RIGHT_PAREN ;

arithmeticInst
    : ADD roundingMode? satFlag? typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI
    | SUB roundingMode? satFlag? typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI
    | MUL roundingMode? satFlag? ftzFlag? hiLoWide? typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI
    | DIV roundingMode? ftzFlag? approxFlag? typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI
    | REM roundingMode? typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI
    | ABS ftzFlag? typeSpecifier vectorSpec? operand COMMA operand SEMI
    | NEG ftzFlag? typeSpecifier vectorSpec? operand COMMA operand SEMI
    | SQRT roundingMode? ftzFlag? approxFlag? typeSpecifier vectorSpec? operand COMMA operand SEMI
    | RSQRT roundingMode? ftzFlag? approxFlag? typeSpecifier vectorSpec? operand COMMA operand SEMI
    | SIN approxFlag? typeSpecifier vectorSpec? operand COMMA operand SEMI
    | COS approxFlag? typeSpecifier vectorSpec? operand COMMA operand SEMI
    | LG2 roundingMode? ftzFlag? approxFlag? typeSpecifier vectorSpec? operand COMMA operand SEMI
    | EX2 roundingMode? ftzFlag? approxFlag? typeSpecifier vectorSpec? operand COMMA operand SEMI
    | POPC typeSpecifier vectorSpec? operand COMMA operand SEMI
    | CLZ typeSpecifier vectorSpec? operand COMMA operand SEMI
    | MAD roundingMode? satFlag? ftzFlag? typeSpecifier vectorSpec? operand COMMA operand COMMA operand COMMA operand SEMI
    | FMA roundingMode? ftzFlag? typeSpecifier vectorSpec? operand COMMA operand COMMA operand COMMA operand SEMI
    | ADDC roundingMode? satFlag? ftzFlag? typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI
    | SUBC roundingMode? satFlag? ftzFlag? typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI
    | MUL24 typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI
    | MAD24 roundingMode? satFlag? typeSpecifier vectorSpec? operand COMMA operand COMMA operand COMMA operand SEMI
    | MADC roundingMode? satFlag? ftzFlag? typeSpecifier vectorSpec? operand COMMA operand COMMA operand COMMA operand SEMI
    | MIN ftzFlag? typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI
    | MAX ftzFlag? typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI
    | SAD ftzFlag? typeSpecifier vectorSpec? operand COMMA operand COMMA operand COMMA operand SEMI
    | COPYSIGN ftzFlag? typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI
    | TESTP testProperty predType? operand COMMA operand SEMI
    | TANH approxFlag? typeSpecifier vectorSpec? operand COMMA operand SEMI
    ;

roundingMode : RN | RZ | RM | RP | RS | RNI | RZI | RMI | RPI ;
satFlag : SAT ;
ftzFlag : FTZ ;
approxFlag : APPROX ;
hiLoWide : HI | LO | WIDE ;
testProperty : NAN | FINITE | INFINITY | NUMBER | NORMAL | SUBNORMAL ;
predType : U16 | U32 | U64 | S16 | S32 | S64 | F16 | F32 | F64 | BF16 ;

logicalInst
    : AND typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI
    | OR typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI
    | XOR typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI
    | NOT typeSpecifier vectorSpec? operand COMMA operand SEMI
    | SELP typeSpecifier vectorSpec? operand COMMA operand COMMA operand COMMA operand SEMI
    | SETP compareOp typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI
    | SET compareOp typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI
    | SLCT roundingMode? typeSpecifier vectorSpec? operand COMMA operand COMMA operand COMMA operand SEMI
    | CNOT typeSpecifier vectorSpec? operand COMMA operand SEMI
    | LOP3 typeSpecifier vectorSpec? operand COMMA operand COMMA operand COMMA operand COMMA operand SEMI
    | SHL typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI
    | SHR typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI
    | SHF shiftMode typeSpecifier vectorSpec? operand COMMA operand COMMA operand COMMA operand SEMI
    ;

shiftMode : LEFT_SHIFT | RIGHT_SHIFT | WRAP | CLAMP ;
compareOp : EQ | NE | LT | LE | GT | GE | LTU | LEU | GTU | GEU | FEQ | FNE | FLT | FLE | FGT | FGE ;

dataMovementInst
    : MOV typeSpecifier vectorSpec? operand COMMA operand SEMI
    | LD ldQualifiers typeSpecifier vectorSpec? operand COMMA addressExpr SEMI
    | ST stQualifiers typeSpecifier vectorSpec? addressExpr COMMA operand SEMI
    | CVT roundingMode? satFlag? ftzFlag? typeSpecifier vectorSpec? operand COMMA typeSpecifier vectorSpec? operand SEMI
    | CVTA genericOrSpecificSpace? toAddrSpace? operand COMMA operand SEMI
    | RCP roundingMode? ftzFlag? approxFlag? typeSpecifier vectorSpec? operand COMMA operand SEMI
    | PRMT typeSpecifier vectorSpec? operand COMMA operand COMMA operand COMMA operand SEMI
    | ISSPACEP addrSpaceQuery? typeSpecifier vectorSpec? operand COMMA operand SEMI
    | MAPA typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI
    | ALLOCA alignClause? typeSpecifier vectorSpec? operand COMMA operand SEMI
    | CP_ASYNC cpAsyncSpace? cacheOperator* typeSpecifier vectorSpec? addressExpr COMMA addressExpr (COMMA operand)? SEMI
    ;

ldQualifiers
    : spaceQualifier cacheOperator* VOLATILE?
    | VOLATILE spaceQualifier cacheOperator*
    ;

stQualifiers
    : spaceQualifier cacheOperator* VOLATILE?
    | VOLATILE spaceQualifier cacheOperator*
    ;

spaceQualifier
    : GLOBAL | CONSTANT | PARAM | SHARED | LOCAL | GENERIC_SPACE
    | WL_SPACE | WU_SPACE | TEX_SPACE | SURF_SPACE | LDG_SPACE
    ;

cacheOperator
    : (WB | WT | CG | CS | CV | CA | NC) (EV | LU)? | MMIO
    ;

cpAsyncSpace : GLOBAL | SHARED ;
genericOrSpecificSpace : GENERIC_SPACE | GLOBAL | SHARED | CONST ;
toAddrSpace : (GLOBAL | SHARED)? ;
addrSpaceQuery : GENERIC_SPACE | GLOBAL | SHARED | CONST | LOCAL ;

parallelSyncInst
    : BAR barrierOp? (DECIMAL_INT operand?)? SEMI
    | MEMBAR membarScope? SEMI
    | FENCE fenceQualifiers? SEMI
    | REDUX_SYNC reduxOp typeSpecifier vectorSpec? operand COMMA operand SEMI
    | MBARRIER_INIT mbarrierSpace typeSpecifier vectorSpec? operand COMMA operand SEMI
    | MBARRIER_ARRIVE mbarrierSpace? arriveFlags? typeSpecifier vectorSpec? operand (COMMA operand)? SEMI
    | MBARRIER_TRY_WAIT mbarrierSpace? typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI
    ;

barrierOp : SYNC | ARRIVE | RED ;
membarScope : CTX | CTA_SCOPE | GL_SCOPE | SYS_SCOPE | COHERENT ;
fenceQualifiers : (memoryOrderQualifier | scopeQualifier)* ;
memoryOrderQualifier : STRONG | ACQUIRE | RELEASE | RELAXED | ACQ_REL ;
scopeQualifier : GPU_SCOPE | GL_SCOPE | SYS_SCOPE | CTA_SCOPE | CLUSTER_SCOPE | CTX | COHERENT ;
reduxOp : ADD_ATOM | MIN_ATOM | MAX_ATOM ;
mbarrierSpace : STATE | SYNC ;
arriveFlags : (RELAXED | ACQ_REL)? ;

atomicInst
    : ATOM atomQualifiers atomOp typeSpecifier vectorSpec? operand COMMA addressExpr COMMA operand (COMMA operand)? SEMI
    ;

atomQualifiers
    : spaceQualifier? cacheOperator* memoryOrderQualifier? scopeQualifier?
    ;

atomOp
    : ADD_ATOM | AND_ATOM | OR_ATOM | XOR_ATOM | INC_ATOM | DEC_ATOM
    | EXCH_ATOM | MIN_ATOM | MAX_ATOM | CAS_ATOM
    ;

warpLevelInst
    : VOTE voteMode PRED? operand (COMMA operand)? SEMI
    | SHFL shuffleMode PRED? typeSpecifier vectorSpec? operand COMMA operand COMMA operand (COMMA operand)? SEMI
    ;

shuffleMode : UP | DOWN | BFLY | IDX ;

textureSurfaceInst
    : TEX texQualifiers typeSpecifier vectorSpec? operand (COMMA operand)* SEMI
    | SURF surfQualifiers typeSpecifier vectorSpec? operand (COMMA operand)* SEMI
    | TEX_LDG texQualifiers typeSpecifier vectorSpec? operand (COMMA operand)* SEMI
    | TEX_GRAD texQualifiers typeSpecifier vectorSpec? operand (COMMA operand)* SEMI
    | TEX_LOD texQualifiers typeSpecifier vectorSpec? operand (COMMA operand)* SEMI
    | TXQ typeSpecifier vectorSpec? operand COMMA operand SEMI
    | SULD surfQualifiers typeSpecifier vectorSpec? operand (COMMA operand)* SEMI
    | SUST surfQualifiers typeSpecifier vectorSpec? operand (COMMA operand)* SEMI
    | SUQ typeSpecifier vectorSpec? operand COMMA operand SEMI
    ;

texQualifiers : TEX_SPACE | LDG_SPACE ;
surfQualifiers : SURF_SPACE ;

reductionPrefetchInst
    : RED redQualifiers redOp typeSpecifier vectorSpec? operand COMMA addressExpr COMMA operand SEMI
    | PREFETCH prefetchSpace? cacheOperator* addressExpr SEMI
    | PREFETCHU prefetchSpace? cacheOperator* addressExpr SEMI
    ;

redQualifiers
    : spaceQualifier? cacheOperator* memoryOrderQualifier? scopeQualifier?
    ;

prefetchSpace : GLOBAL | TEX_SPACE ;
redOp : ADD_ATOM | MIN_ATOM | MAX_ATOM ;

matrixInst
    : WMMA wmmaOp wmmaLayout? wmmaShape? wmmaKind? typeSpecifier vectorSpec? operand (COMMA operand)* SEMI
    ;

wmmaOp : MMA | LOAD | STORE | FILL ;
wmmaLayout : ROW | COL ;
wmmaShape : M8N8K4 | M16N16K16 | M32N8K16 | M16N8K16 ;
wmmaKind : KIND COLONCOLON MXF4NVF4 ;

videoSimdInst
    : VADD4 satFlag? typeSpecifier vectorSpec? operand COMMA operand COMMA operand COMMA operand SEMI
    | VSUB4 satFlag? typeSpecifier vectorSpec? operand COMMA operand COMMA operand COMMA operand SEMI
    | VAVRG4 typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI
    | VABSDIFF4 typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI
    | VMIN4 typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI
    | VMAX4 typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI
    | VSET4 vsetOp typeSpecifier vectorSpec? operand COMMA operand COMMA operand COMMA operand SEMI
    | DP4A typeSpecifier vectorSpec? operand COMMA operand COMMA operand COMMA operand SEMI
    | DP2A typeSpecifier vectorSpec? operand COMMA operand COMMA operand COMMA operand SEMI
    ;

vsetOp : EQ | NE | LT | LE | GT | GE | LTU | LEU | GTU | GEU ;

newInst
    : ST_ASYNC stAsyncQualifiers typeSpecifier vectorSpec? addressExpr COMMA operand SEMI
    | RED_ASYNC redAsyncQualifiers redOp typeSpecifier vectorSpec? operand COMMA addressExpr COMMA operand SEMI
    | TCGEN_ALLOC tcgenSpace typeSpecifier vectorSpec? operand COMMA operand SEMI
    | TCGEN_DEALLOC tcgenSpace typeSpecifier vectorSpec? operand SEMI
    | TCGEN_RELINQUISH tcgenSpace typeSpecifier vectorSpec? operand SEMI
    | TCGEN_CP tcgenSpace typeSpecifier vectorSpec? operand COMMA operand SEMI
    | TCGEN_SHIFT tcgenSpace typeSpecifier vectorSpec? operand COMMA operand SEMI
    | TCGEN_MMA tcgenSpace typeSpecifier vectorSpec? operand (COMMA operand)* SEMI
    | TCGEN_COMMIT tcgenSpace typeSpecifier vectorSpec? operand SEMI
    | TENSORMAP_REPLACE tensormapSpace typeSpecifier vectorSpec? operand COMMA operand SEMI
    | ST_BULK stBulkQualifiers typeSpecifier vectorSpec? addressExpr COMMA operand SEMI
    ;

stAsyncQualifiers
    : spaceQualifier? cacheOperator* memoryOrderQualifier? scopeQualifier?
    ;

redAsyncQualifiers
    : spaceQualifier? cacheOperator* memoryOrderQualifier? scopeQualifier?
    ;

tcgenSpace : GLOBAL | SHARED ;
tensormapSpace : GLOBAL ;
stBulkQualifiers : spaceQualifier? cacheOperator* ;