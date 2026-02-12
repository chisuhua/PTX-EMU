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
    | tcgenInst
    | SEMI
    ;

predicate
    : AT operand
    | BANG operand
    ;

braInst
    : predicate? BRA (UNI | voteMode)? labelOperand SEMI
    ;

brxInst
    : predicate? BRX (UNI | voteMode)? addressExpr COMMA labelOperand SEMI
    ;

callInst
    : predicate? CALL callParams? labelOperand callArgs? SEMI
    ;

retInst
    : predicate? RET SEMI
    ;

exitInst
    : predicate? EXIT SEMI
    ;

trapInst
    : predicate? TRAP SEMI
    ;

brkInst
    : predicate? BRK (DECIMAL_INT)? SEMI
    ;

brkptInst
    : predicate? BRKPT SEMI
    ;

controlFlowInst
    : braInst
    | brxInst
    | callInst
    | retInst
    | exitInst
    | trapInst
    | brkInst
    | brkptInst
    ;

labelOperand : ID ;
voteMode : BALLOT | ANY | ALL | UNI | EQV ;
callParams : paramList ;
callArgs : LEFT_PAREN operand (COMMA operand)* RIGHT_PAREN ;

// Arithmetic instruction rules
addInst: ADD roundingMode? satFlag? typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI;
subInst: SUB roundingMode? satFlag? typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI;
mulInst: MUL roundingMode? satFlag? ftzFlag? hiLoWide? typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI;
divInst: DIV roundingMode? ftzFlag? approxFlag? typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI;
remInst: REM roundingMode? typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI;
absInst: ABS ftzFlag? typeSpecifier vectorSpec? operand COMMA operand SEMI;
negInst: NEG ftzFlag? typeSpecifier vectorSpec? operand COMMA operand SEMI;
sqrtInst: SQRT roundingMode? ftzFlag? approxFlag? typeSpecifier vectorSpec? operand COMMA operand SEMI;
rsqrtInst: RSQRT roundingMode? ftzFlag? approxFlag? typeSpecifier vectorSpec? operand COMMA operand SEMI;
sinInst: SIN approxFlag? typeSpecifier vectorSpec? operand COMMA operand SEMI;
cosInst: COS approxFlag? typeSpecifier vectorSpec? operand COMMA operand SEMI;
lg2Inst: LG2 roundingMode? ftzFlag? approxFlag? typeSpecifier vectorSpec? operand COMMA operand SEMI;
ex2Inst: EX2 roundingMode? ftzFlag? approxFlag? typeSpecifier vectorSpec? operand COMMA operand SEMI;
popcInst: POPC typeSpecifier vectorSpec? operand COMMA operand SEMI;
clzInst: CLZ typeSpecifier vectorSpec? operand COMMA operand SEMI;
madInst: MAD roundingMode? satFlag? ftzFlag? typeSpecifier vectorSpec? operand COMMA operand COMMA operand COMMA operand SEMI;
fmaInst: FMA roundingMode? ftzFlag? typeSpecifier vectorSpec? operand COMMA operand COMMA operand COMMA operand SEMI;
addcInst: ADDC roundingMode? satFlag? ftzFlag? typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI;
subcInst: SUBC roundingMode? satFlag? ftzFlag? typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI;
mul24Inst: MUL24 typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI;
mad24Inst: MAD24 roundingMode? satFlag? typeSpecifier vectorSpec? operand COMMA operand COMMA operand COMMA operand SEMI;
madcInst: MADC roundingMode? satFlag? ftzFlag? typeSpecifier vectorSpec? operand COMMA operand COMMA operand COMMA operand SEMI;
minInst: MIN ftzFlag? typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI;
maxInst: MAX ftzFlag? typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI;
sadInst: SAD ftzFlag? typeSpecifier vectorSpec? operand COMMA operand COMMA operand COMMA operand SEMI;
copysignInst: COPYSIGN ftzFlag? typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI;
testpInst: TESTP testProperty predType? operand COMMA operand SEMI;
tanhInst: TANH approxFlag? typeSpecifier vectorSpec? operand COMMA operand SEMI;

arithmeticInst
    : addInst
    | subInst
    | mulInst
    | divInst
    | remInst
    | absInst
    | negInst
    | sqrtInst
    | rsqrtInst
    | sinInst
    | cosInst
    | lg2Inst
    | ex2Inst
    | popcInst
    | clzInst
    | madInst
    | fmaInst
    | addcInst
    | subcInst
    | mul24Inst
    | mad24Inst
    | madcInst
    | minInst
    | maxInst
    | sadInst
    | copysignInst
    | testpInst
    | tanhInst
    ;

roundingMode : RN | RZ | RM | RP | RS | RNI | RZI | RMI | RPI ;
satFlag : SAT ;
ftzFlag : FTZ ;
approxFlag : APPROX ;
hiLoWide : HI | LO | WIDE ;
testProperty : NAN | FINITE | INFINITY | NUMBER | NORMAL | SUBNORMAL ;
predType : U16 | U32 | U64 | S16 | S32 | S64 | F16 | F32 | F64 | BF16 ;

// Logical instruction rules
andInst: AND typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI;
orInst: OR typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI;
xorInst: XOR typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI;
notInst: NOT typeSpecifier vectorSpec? operand COMMA operand SEMI;
selpInst: SELP typeSpecifier vectorSpec? operand COMMA operand COMMA operand COMMA operand SEMI;
setpInst: SETP compareOp typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI;
setInst: SET compareOp typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI;
slctInst: SLCT roundingMode? typeSpecifier vectorSpec? operand COMMA operand COMMA operand COMMA operand SEMI;
cnotInst: CNOT typeSpecifier vectorSpec? operand COMMA operand SEMI;
lop3Inst: LOP3 typeSpecifier vectorSpec? operand COMMA operand COMMA operand COMMA operand COMMA operand SEMI;
shlInst: SHL typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI;
shrInst: SHR typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI;
shfInst: SHF shiftMode typeSpecifier vectorSpec? operand COMMA operand COMMA operand COMMA operand SEMI;

logicalInst
    : andInst
    | orInst
    | xorInst
    | notInst
    | selpInst
    | setpInst
    | setInst
    | slctInst
    | cnotInst
    | lop3Inst
    | shlInst
    | shrInst
    | shfInst
    ;

shiftMode : LEFT_SHIFT | RIGHT_SHIFT | WRAP | CLAMP ;
compareOp : EQ | NE | LT | LE | GT | GE | LTU | LEU | GTU | GEU | FEQ | FNE | FLT | FLE | FGT | FGE ;

dataMovementInst
    : movInst
    | ldInst
    | stInst
    | cvtInst
    | cvtaInst
    | rcpInst
    | prmtInst
    | isspacepInst
    | mapaInst
    | allocaInst
    | cpAsyncInst
    ;

movInst
    : MOV typeSpecifier vectorSpec? operand COMMA operand SEMI
    ;

ldInst
    : LD ldQualifiers typeSpecifier vectorSpec? operand COMMA addressExpr SEMI
    ;

stInst
    : ST stQualifiers typeSpecifier vectorSpec? addressExpr COMMA operand SEMI
    ;

cvtInst
    : CVT roundingMode? satFlag? ftzFlag? typeSpecifier vectorSpec? operand COMMA typeSpecifier vectorSpec? operand SEMI
    ;

cvtaInst
    : CVTA genericOrSpecificSpace? toAddrSpace? operand COMMA operand SEMI
    ;

rcpInst
    : RCP roundingMode? ftzFlag? approxFlag? typeSpecifier vectorSpec? operand COMMA operand SEMI
    ;

prmtInst
    : PRMT typeSpecifier vectorSpec? operand COMMA operand COMMA operand COMMA operand SEMI
    ;

isspacepInst
    : ISSPACEP addrSpaceQuery? typeSpecifier vectorSpec? operand COMMA operand SEMI
    ;

mapaInst
    : MAPA typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI
    ;

allocaInst
    : ALLOCA alignClause? typeSpecifier vectorSpec? operand COMMA operand SEMI
    ;

cpAsyncInst
    : CP_ASYNC cpAsyncSpace? cacheOperator* typeSpecifier vectorSpec? addressExpr COMMA addressExpr (COMMA operand)? SEMI
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

// Parallel sync instruction rules
barInst: BAR barrierOp? (DECIMAL_INT operand?)? SEMI;
membarInst: MEMBAR membarScope? SEMI;
fenceInst: FENCE fenceQualifiers? SEMI;
reduxSyncInst: REDUX_SYNC reduxOp typeSpecifier vectorSpec? operand COMMA operand SEMI;
mbarrierInitInst: MBARRIER_INIT mbarrierSpace typeSpecifier vectorSpec? operand COMMA operand SEMI;
mbarrierArriveInst: MBARRIER_ARRIVE mbarrierSpace? arriveFlags? typeSpecifier vectorSpec? operand (COMMA operand)? SEMI;
mbarrierTryWaitInst: MBARRIER_TRY_WAIT mbarrierSpace? typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI;

parallelSyncInst
    : barInst
    | membarInst
    | fenceInst
    | reduxSyncInst
    | mbarrierInitInst
    | mbarrierArriveInst
    | mbarrierTryWaitInst
    ;

barrierOp : SYNC | ARRIVE | RED ;
membarScope : CTX | CTA_SCOPE | GL_SCOPE | SYS_SCOPE | COHERENT ;
fenceQualifiers : (memoryOrderQualifier | scopeQualifier)* ;
memoryOrderQualifier : STRONG | ACQUIRE | RELEASE | RELAXED | ACQ_REL ;
scopeQualifier : GPU_SCOPE | GL_SCOPE | SYS_SCOPE | CTA_SCOPE | CLUSTER_SCOPE | CTX | COHERENT ;
reduxOp : ADD_ATOM | MIN_ATOM | MAX_ATOM ;
mbarrierSpace : STATE | SYNC ;
arriveFlags : (RELAXED | ACQ_REL)? ;

// Atomic instruction rules
atomInst: ATOM atomQualifiers atomOp typeSpecifier vectorSpec? operand COMMA addressExpr COMMA operand (COMMA operand)? SEMI;

atomicInst
    : atomInst
    ;

atomQualifiers
    : spaceQualifier? cacheOperator* memoryOrderQualifier? scopeQualifier?
    ;

atomOp
    : ADD_ATOM | AND_ATOM | OR_ATOM | XOR_ATOM | INC_ATOM | DEC_ATOM
    | EXCH_ATOM | MIN_ATOM | MAX_ATOM | CAS_ATOM
    ;

// Warp level instruction rules
voteInst: VOTE voteMode PRED? operand (COMMA operand)? SEMI;
shflInst: SHFL shuffleMode PRED? typeSpecifier vectorSpec? operand COMMA operand COMMA operand (COMMA operand)? SEMI;

warpLevelInst
    : voteInst
    | shflInst
    ;

shuffleMode : UP | DOWN | BFLY | IDX ;

// Texture surface instruction rules
texInst: TEX texQualifiers typeSpecifier vectorSpec? operand (COMMA operand)* SEMI;
surfInst: SURF surfQualifiers typeSpecifier vectorSpec? operand (COMMA operand)* SEMI;
texLdgInst: TEX_LDG texQualifiers typeSpecifier vectorSpec? operand (COMMA operand)* SEMI;
texGradInst: TEX_GRAD texQualifiers typeSpecifier vectorSpec? operand (COMMA operand)* SEMI;
texLodInst: TEX_LOD texQualifiers typeSpecifier vectorSpec? operand (COMMA operand)* SEMI;
txqInst: TXQ typeSpecifier vectorSpec? operand COMMA operand SEMI;
suldInst: SULD surfQualifiers typeSpecifier vectorSpec? operand (COMMA operand)* SEMI;
sustInst: SUST surfQualifiers typeSpecifier vectorSpec? operand (COMMA operand)* SEMI;
suqInst: SUQ typeSpecifier vectorSpec? operand COMMA operand SEMI;

textureSurfaceInst
    : texInst
    | surfInst
    | texLdgInst
    | texGradInst
    | texLodInst
    | txqInst
    | suldInst
    | sustInst
    | suqInst
    ;

texQualifiers : TEX_SPACE | LDG_SPACE ;
surfQualifiers : SURF_SPACE ;

// Reduction prefetch instruction rules
redInst: RED redQualifiers redOp typeSpecifier vectorSpec? operand COMMA addressExpr COMMA operand SEMI;
prefetchInst: PREFETCH prefetchSpace? cacheOperator* addressExpr SEMI;
prefetchuInst: PREFETCHU prefetchSpace? cacheOperator* addressExpr SEMI;

reductionPrefetchInst
    : redInst
    | prefetchInst
    | prefetchuInst
    ;

redQualifiers
    : spaceQualifier? cacheOperator* memoryOrderQualifier? scopeQualifier?
    ;

prefetchSpace : GLOBAL | TEX_SPACE ;
redOp : ADD_ATOM | MIN_ATOM | MAX_ATOM ;

// Matrix instruction rules
wmmaInst: WMMA wmmaOp wmmaLayout? wmmaShape? wmmaKind? typeSpecifier vectorSpec? operand (COMMA operand)* SEMI;

matrixInst
    : wmmaInst
    ;

wmmaOp : MMA | LOAD | STORE | FILL ;
wmmaLayout : ROW | COL ;
wmmaShape : M8N8K4 | M16N16K16 | M32N8K16 | M16N8K16 ;
wmmaKind : KIND COLONCOLON MXF4NVF4 ;

// Video SIMD instruction rules
vadd4Inst: VADD4 satFlag? typeSpecifier vectorSpec? operand COMMA operand COMMA operand COMMA operand SEMI;
vsub4Inst: VSUB4 satFlag? typeSpecifier vectorSpec? operand COMMA operand COMMA operand COMMA operand SEMI;
vavrg4Inst: VAVRG4 typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI;
vabsdiff4Inst: VABSDIFF4 typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI;
vmin4Inst: VMIN4 typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI;
vmax4Inst: VMAX4 typeSpecifier vectorSpec? operand COMMA operand COMMA operand SEMI;
vset4Inst: VSET4 vsetOp typeSpecifier vectorSpec? operand COMMA operand COMMA operand COMMA operand SEMI;
dp4aInst: DP4A typeSpecifier vectorSpec? operand COMMA operand COMMA operand COMMA operand SEMI;
dp2aInst: DP2A typeSpecifier vectorSpec? operand COMMA operand COMMA operand COMMA operand SEMI;

videoSimdInst
    : vadd4Inst
    | vsub4Inst
    | vavrg4Inst
    | vabsdiff4Inst
    | vmin4Inst
    | vmax4Inst
    | vset4Inst
    | dp4aInst
    | dp2aInst
    ;

vsetOp : EQ | NE | LT | LE | GT | GE | LTU | LEU | GTU | GEU ;

// New instruction rules
stAsyncInst: ST_ASYNC stAsyncQualifiers typeSpecifier vectorSpec? addressExpr COMMA operand SEMI;
redAsyncInst: RED_ASYNC redAsyncQualifiers redOp typeSpecifier vectorSpec? operand COMMA addressExpr COMMA operand SEMI;
tcgenAllocInst: TCGEN_ALLOC tcgenSpace typeSpecifier vectorSpec? operand COMMA operand SEMI;
tcgenDeallocInst: TCGEN_DEALLOC tcgenSpace typeSpecifier vectorSpec? operand SEMI;
tcgenRelinquishInst: TCGEN_RELINQUISH tcgenSpace typeSpecifier vectorSpec? operand SEMI;
tcgenCpInst: TCGEN_CP tcgenSpace typeSpecifier vectorSpec? operand COMMA operand SEMI;
tcgenShiftInst: TCGEN_SHIFT tcgenSpace typeSpecifier vectorSpec? operand COMMA operand SEMI;
tcgenMmaInst: TCGEN_MMA tcgenSpace typeSpecifier vectorSpec? operand (COMMA operand)* SEMI;
tcgenCommitInst: TCGEN_COMMIT tcgenSpace typeSpecifier vectorSpec? operand SEMI;
tensormapReplaceInst: TENSORMAP_REPLACE tensormapSpace typeSpecifier vectorSpec? operand COMMA operand SEMI;
stBulkInst: ST_BULK stBulkQualifiers typeSpecifier vectorSpec? addressExpr COMMA operand SEMI;

tcgenInst
    : stAsyncInst
    | redAsyncInst
    | tcgenAllocInst
    | tcgenDeallocInst
    | tcgenRelinquishInst
    | tcgenCpInst
    | tcgenShiftInst
    | tcgenMmaInst
    | tcgenCommitInst
    | tensormapReplaceInst
    | stBulkInst
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
