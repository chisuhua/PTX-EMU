parser grammar ptxParser;

options { tokenVocab = ptxLexer; }

ast
    : versionDes targetDes addressSizeDes (fileDes | sectionDes)* globalDecls? kernels EOF
    ;

// --- Version & Target ---
versionDes      : VERSION decimalLiteral DOT decimalLiteral SEMI ;
targetDes       : TARGET smTarget SEMI ;
addressSizeDes  : ADDRESS_SIZE decimalLiteral SEMI ;
fileDes         : FILE (decimalLiteral | STRING) SEMI ;
sectionDes      : SECTION ID SEMI ;

smTarget        : SM_ IMMEDIATE (SM_SUFFIX_F | SM_SUFFIX_A)? ;

// --- Global Declarations ---
globalDecls
    : globalDecl+
    ;

globalDecl
    : constDecl
    | globalVarDecl
    | externFuncDecl
    | externSharedDecl
    | abiPreserveDirective
    ;

constDecl
    : CONST (ALIGN decimalLiteral)? dataType ID (arraySize)? (initList)? SEMI
    ;

globalVarDecl
    : GLOBAL (ALIGN decimalLiteral)? dataType ID (arraySize)? (initList)? SEMI
    ;

externFuncDecl
    : EXTERN FUNC (returnParam)? ID LEFT_PAREN formalParams? RIGHT_PAREN SEMI
    ;

externSharedDecl
    : EXTERN SHARED (ALIGN decimalLiteral)? dataType ID arrayBrackets? SEMI
    ;

abiPreserveDirective
    : ABI_PRESERVE decimalLiteral SEMI
    | ABI_PRESERVE_CTRL decimalLiteral SEMI
    ;

arraySize       : LEFT_BRACK decimalLiteral RIGHT_BRACK ;
arrayBrackets   : LEFT_BRACK RIGHT_BRACK ;
initList        : ASSIGN LEFT_BRACE (IMMEDIATE (COMMA IMMEDIATE)*)? RIGHT_BRACE ;

// --- Kernels and Functions ---
kernels
    : kernelOrFunc+
    ;

kernelOrFunc
    : VISIBLE? ENTRY ID LEFT_PAREN formalParams? RIGHT_PAREN perfTunings? funcBody
    | VISIBLE? FUNC ID LEFT_PAREN formalParams? RIGHT_PAREN funcBody
    ;

perfTunings
    : perfTuning+
    ;

perfTuning
    : MAXNTID decimalLiteral COMMA decimalLiteral COMMA decimalLiteral
    | MINNCTAPERSM decimalLiteral
    | MAXNREG decimalLiteral
    | REQNTID decimalLiteral COMMA decimalLiteral COMMA decimalLiteral
    ;

formalParams
    : formalParam (COMMA formalParam)*
    ;

formalParam
    : (PARAM | CONST PARAM)? dataType ID (arraySize)?
    ;

returnParam
    : LEFT_PAREN formalParam RIGHT_PAREN
    ;

funcBody
    : LEFT_BRACE statements? RIGHT_BRACE
    ;

// --- Statements ---
statements
    : statement+
    ;

statement
    : labelDef
    | regDecl
    | localVarDecl
    | sharedVarDecl
    | paramVarDecl
    | controlFlow
    | arithmetic
    | logical
    | memory
    | parallel
    | warp
    | texSurf
    | video
    | misc
    | pragma
    | compoundStatement
    | asyncStore
    | asyncReduction
    | tcgenInstr
    | tensormapInstr
    | stBulkInstr
    ;

labelDef        : DOLLAR ID COLON ;
compoundStatement : LEFT_BRACE statements? RIGHT_BRACE ;

regDecl         : REG (ALIGN decimalLiteral)? (dataType | vectorType) ID (arraySize)? SEMI ;
localVarDecl    : LOCAL (ALIGN decimalLiteral)? dataType ID (arraySize)? SEMI ;
sharedVarDecl   : SHARED (ALIGN decimalLiteral)? dataType ID (arraySize)? SEMI ;
paramVarDecl    : PARAM (ALIGN decimalLiteral)? dataType ID (arraySize)? SEMI ;

// --- Control Flow ---
controlFlow
    : BRA (UNI | predicateExpr)? DOLLAR ID SEMI
    | BRX IDX UNI? operand2 SEMI
    | CALL callArgs SEMI
    | RET SEMI
    | EXIT SEMI
    | TRAP SEMI
    | BRK SEMI
    | BRKPT SEMI
    ;

predicateExpr
    : operand (EQ | NE | LT | LE | GT | GE | LTU | LEU | GTU | GEU) operand
    ;

callArgs
    : (LEFT_PAREN operand (COMMA operand)* RIGHT_PAREN COMMA)? operand COMMA LEFT_PAREN (operand (COMMA operand)*)? RIGHT_BRACE
    ;

// --- Arithmetic ---
arithmetic
    : ADD  typeWithMods operand3 SEMI
    | SUB  typeWithMods operand3 SEMI
    | MUL  typeWithMods operand3 SEMI
    | MAD  typeWithMods operand4 SEMI
    | FMA  typeWithMods operand4 SEMI
    | DIV  typeWithMods operand3 SEMI
    | MIN  typeWithMods operand3 SEMI
    | MAX  typeWithMods operand3 SEMI
    | SAD  dataType operand4 SEMI
    | ABS  dataType operand2 SEMI
    | NEG  dataType operand2 SEMI
    | SQRT typeWithMods operand2 SEMI
    | SIN  typeWithMods operand2 SEMI
    | COS  typeWithMods operand2 SEMI
    | LG2  typeWithMods operand2 SEMI
    | EX2  typeWithMods operand2 SEMI
    | RSQRT typeWithMods operand2 SEMI
    | TANH typeWithMods operand2 SEMI
    | COPYSIGN dataType operand3 SEMI
    | TESTP testpQualifier dataType operand2 SEMI
    | POPC dataType operand2 SEMI
    | CLZ  dataType operand2 SEMI
    | ADDC dataType CC? operand3 SEMI
    | SUBC dataType CC? operand3 SEMI
    | MADC dataType CC? operand4 SEMI
    | MUL24 dataType operand3 SEMI
    | MAD24 dataType operand4 SEMI
    | REM  dataType operand3 SEMI
    ;

typeWithMods
        : (roundingMode | FTZ | SAT | APPROX | HI | LO | WIDE | CC)* dataType
            (roundingMode | FTZ | SAT | APPROX | HI | LO | WIDE | CC)*
    ;

roundingMode
    : RN | RZ | RM | RP
    ;

// --- Logical ---
logical
    : AND  intType operand3 SEMI
    | OR   intType operand3 SEMI
    | XOR  intType operand3 SEMI
    | NOT  intType operand2 SEMI
    | CNOT intType operand2 SEMI
    | SHL  intType operand3 SEMI
    | SHR  intType operand3 SEMI
    | SHF  shfDirection shfMode B32 operand4 SEMI
    | LOP3 LUT? B32 operand5 SEMI
    | PRMT B32 operand4 SEMI
    | SELP dataType operand4 SEMI
    | SLCT dataType operand4 SEMI
    | SET  comparisonOp dataType dataType operand3 SEMI
    | SETP comparisonOp dataType operand3 SEMI
    | MOV  (dataType | vectorType) operand2 SEMI
    ;

intType     : U8 | U16 | U32 | U64 | S8 | S16 | S32 | S64 | B8 | B16 | B32 | B64 ;
predType    : PRED ;

comparisonOp
    : EQ | NE | LT | LE | GT | GE | LO | LS | HI | HS | LTU | LEU | GTU | GEU
    ;

// --- Memory ---
memory
    : LD  space? (dataType | vectorType) (VOLATILE | NC)? operand2 SEMI
    | ST  space? (dataType | vectorType) (VOLATILE)? operand2 SEMI
    | CVT  cvtMods dataType TO dataType operand2 SEMI
    | CVTA space? (U32 | U64) TO (U32 | U64) operand2 SEMI
    | RCP  typeWithMods operand2 SEMI
    | ATOM atomSpace atomOp (dataType | F32 | F64 | E4M3 | E5M2) operand3or4 SEMI
    | CP_ASYNC qualifiers? operandList SEMI
    | ISSPACEP space operand2 SEMI
    | MAPA (U64 | U32) operand3 SEMI
    | ALLOCA space? (ALIGN decimalLiteral)? (U32 | U64) operand2 SEMI
    ;

space       : CONST | PARAM | GLOBAL | LOCAL | SHARED | GENERIC ;
atomSpace   : GLOBAL | SHARED ;
atomOp      : ADD_ATOM | AND_ATOM | OR_ATOM | XOR_ATOM | INC_ATOM | DEC_ATOM | EXCH_ATOM | MIN_ATOM | MAX_ATOM | CAS_ATOM ;

cvtMods
    : (FTZ | SAT | APPROX | roundingMode | RS | RNI | RZI | RMI | RPI)*
    ;

// === Async Instructions (PTX 8.7+) ===
asyncStore
    : ST_ASYNC space?
      (dataType | vectorType)
      (VOLATILE | NC | MMIO | RELEASE | LEVEL_CACHE_HINT | scopeQualifier)?
      operand2 SEMI
    ;

asyncReduction
    : RED_ASYNC space?
      reductionOp
      (dataType | F32 | F64 | F16 | E4M3 | E5M2)
      (MMIO | RELEASE | LEVEL_CACHE_HINT | scopeQualifier)?
      operand3or4 SEMI
    ;

scopeQualifier
    : CLUSTER_SCOPE
    | CTA_SCOPE
    ;

reductionOp
    : ADD_ATOM | MIN_ATOM | MAX_ATOM | AND_ATOM | OR_ATOM | XOR_ATOM
    ;

// === TCGEN05 Instructions (Hopper+) ===
tcgenInstr
    : TCGEN_ALLOC qualifiers? operand SEMI
    | TCGEN_DEALLOC qualifiers? operand SEMI
    | TCGEN_RELINQUISH qualifiers? operand SEMI
    | TCGEN_CP qualifiers? operandList SEMI
    | TCGEN_SHIFT qualifiers? operandList SEMI
    | TCGEN_MMA qualifiers? operandList SEMI
    | TCGEN_COMMIT qualifiers? operandList SEMI
    ;

// === Tensor Map ===
tensormapInstr
    : TENSORMAP_REPLACE qualifiers? operandList SEMI
    ;

// === st.bulk (PTX 9.0) ===
stBulkInstr
    : ST_BULK space? (dataType | vectorType) operand3 SEMI
    ;

qualifiers
    : qualifier+
    ;

qualifier
    : RN | RZ | RM | RP
    | FTZ | SAT | APPROX
    | HI | LO | WIDE
    | VOLATILE | NC | MMIO | RELEASE | ACQUIRE | STRONG | RELAXED | ACQ_REL
    | SYNC | GPU | GL | CA | CG
    | GENERIC
    | ROW | COL | ALIGNED
    | M8N8K4 | M16N16K16 | M32N8K16 | M16N8K16
    | KIND_MXF4NVF4
    | SCALE_VEC_SIZE
    | BLOCK_SCALE
    | LEVEL_CACHE_HINT
    | CLUSTER_SCOPE | CTA_SCOPE
    ;

shfDirection
    : LEFT_SHIFT | RIGHT_SHIFT
    ;

shfMode
    : WRAP | CLAMP
    ;

reduxModifier
    : ABS_MOD | NAN
    ;

testpQualifier
    : FINITE | INFINITY | NUMBER | NAN | NORMAL | SUBNORMAL
    ;

videoType
    : (U32 | S32) SAT?
    ;

vset4Mods
    : (PRED | comparisonOp | MERGE | ADD_ATOM | ASEL | BSEL)+
    ;

operandList
    : operand (COMMA operand)*
    ;

// --- Parallel ---
parallel
    : BAR barModifier? decimalLiteral SEMI
    | MEMBAR membarScope? SEMI
    | RED space reductionOp (dataType | F32 | F64 | F16 | E4M3 | E5M2) operand3or4 SEMI
    | REDUX_SYNC reductionOp reduxModifier? (dataType | F32 | F16) operand3 SEMI
    | FENCE qualifiers? SEMI
    | MBARRIER_INIT qualifiers? operand2 SEMI
    | MBARRIER_ARRIVE qualifiers? operand2 SEMI
    | MBARRIER_TRY_WAIT qualifiers? operand2 SEMI
    | PREFETCH  (GLOBAL | CONST) cacheLevel? operand2 SEMI
    | PREFETCHU (GLOBAL | CONST) cacheLevel? operand2 SEMI
    ;

barModifier : SYNC | CTA_SCOPE | GPU ;
membarScope : CTA_SCOPE | GL | SYS | COHERENT | STRONG | ACQUIRE | RELEASE ;
cacheLevel  : L1 | L2 ;

// --- Warp ---
warp
    : VOTE (ANY | ALL | UNI | BALLOT | EQ | EQV) PRED? operand2 SEMI
    | SHFL SYNC? (UP | DOWN | BFLY | IDX) (dataType | F32 | F16) operand4 SEMI
    ;

// --- Texture / Surface ---
texSurf
    : TEX  texMods texInst SEMI
    | TEX_LDG texMods texInst SEMI
    | TEX_GRAD texMods texInst SEMI
    | TEX_LOD texMods texInst SEMI
    | TXQ txqOp dataType operand2 SEMI
    | SULD surfMods texInst SEMI
    | SUST surfMods texInst SEMI
    | SUQ suqOp dataType operand2 SEMI
    | SURF surfMods texInst SEMI
    ;

texMods     : (CONST | GLOBAL) (dataType | F32 | F16 | E4M3 | E5M2)* ;
surfMods    : (GLOBAL) (dataType | F32 | F16)* ;
texInst     : operandList ;
surfInst    : operandList ;

txqOp
    : DOT ID
    ;

suqOp
    : DOT ID
    ;

// --- Misc ---
pragma          : PRAGMA STRING SEMI ;

video
    : VADD4 videoType operand3 SEMI
    | VSUB4 videoType operand3 SEMI
    | VAVRG4 videoType operand3 SEMI
    | VABSDIFF4 videoType operand3 SEMI
    | VMIN4 videoType operand3 SEMI
    | VMAX4 videoType operand3 SEMI
    | VSET4 vset4Mods? videoType operand4 SEMI
    | DP4A dataType dataType operand4 SEMI
    | DP2A dataType dataType operand4 SEMI
    ;

misc
    : BRANCHTARGETS ID COMMA LEFT_BRACE ID (COMMA ID)* RIGHT_BRACE SEMI
    ;

// --- Operands ---
operand2        : operand COMMA operand ;
operand3        : operand COMMA operand COMMA operand ;
operand4        : operand COMMA operand COMMA operand COMMA operand ;
operand5        : operand COMMA operand COMMA operand COMMA operand COMMA operand ;
operand3or4     : operand4 | operand3 ;

operand
    : IMMEDIATE
    | register
    | ID
    | addressExpr
    | vectorLit
    ;

register
    : PERCENT ID (DOT (ID | IMMEDIATE))?
    ;

addressExpr
    : LEFT_BRACK (register | ID) (PLUS decimalLiteral)? RIGHT_BRACK
    ;

vectorLit
    : LEFT_BRACE register (COMMA register)* RIGHT_BRACE
    ;

// --- Types ---
dataType
    : U8 | U16 | U32 | U64
    | S8 | S16 | S32 | S64
    | F8 | F16 | F16X2 | BF16 | BF16X2 | F32 | F64
    | E4M3 | E5M2 | E3M2 | E2M3 | E2M1
    | S16X2 | U16X2
    | B8 | B16 | B32 | B64 | B128
    | PRED
    ;

vectorType
    : V2 dataType
    | V4 dataType
    | E2M1X4
    | E4M3X4
    | E5M2X4
    | E3M2X4
    | E2M3X4
    ;

decimalLiteral  : IMMEDIATE ;