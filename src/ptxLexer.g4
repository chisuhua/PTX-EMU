lexer grammar ptxLexer;

// --- Directives ---
VERSION         : '.version';
TARGET          : '.target';
ADDRESS_SIZE    : '.address_size';
FILE            : '.file';
SECTION         : '.section';
VISIBLE         : '.visible';
EXTERN          : '.extern';
FUNC            : '.func';
ENTRY           : '.entry';
REG             : '.reg';
PARAM           : '.param';
CONST           : '.const';
GLOBAL          : '.global';
LOCAL           : '.local';
SHARED          : '.shared';
ALIGN           : '.align';
MAXNTID         : '.maxntid';
MINNCTAPERSM    : '.minnctapersm';
MAXNREG         : '.maxnreg';
REQNTID         : '.reqntid';
PRAGMA          : '.pragma';

// New ABI directives (PTX 9.0+)
ABI_PRESERVE       : '.abi_preserve';
ABI_PRESERVE_CTRL  : '.abi_preserve_control';

// --- Data Types ---
U8   : '.u8';   U16  : '.u16';  U32  : '.u32';  U64  : '.u64';
S8   : '.s8';   S16  : '.s16';  S32  : '.s32';  S64  : '.s64';
F16  : '.f16';  F32  : '.f32';  F64  : '.f64';
F8   : '.f8';   // legacy alias
F16X2 : '.f16x2';
BF16  : '.bf16';
BF16X2 : '.bf16x2';

// New FP8 types (PTX 8.7+)
E4M3   : '.e4m3';
E5M2   : '.e5m2';
E3M2   : '.e3m2';
E2M3   : '.e2m3';
E2M1   : '.e2m1';

// Packed vector FP8 types (PTX 8.7+)
E2M1X4 : '.e2m1x4';
E4M3X4 : '.e4m3x4';
E5M2X4 : '.e5m2x4';
E3M2X4 : '.e3m2x4';
E2M3X4 : '.e2m3x4';

B8   : '.b8';   B16  : '.b16';  B32  : '.b32';  B64  : '.b64';  B128 : '.b128';
PRED : '.pred';
S16X2 : '.s16x2';
U16X2 : '.u16x2';

// --- Vector Types ---
V2 : '.v2';
V4 : '.v4';

// --- Instruction Modifiers ---

// Rounding
RN  : '.rn';  RZ  : '.rz';  RM  : '.rm';  RP  : '.rp';
RS  : '.rs';
RNI : '.rni';
RZI : '.rzi';
RMI : '.rmi';
RPI : '.rpi';

// Comparison
EQ  : '.eq';  NE  : '.ne';  LT  : '.lt';  LE  : '.le';
GT  : '.gt';  GE  : '.ge';  LTU : '.ltu'; LEU : '.leu';
GTU : '.gtu'; GEU : '.geu';

FEQ     : '.feq';
FNE     : '.fne';
FLT     : '.flt';
FLE     : '.fle';
FGT     : '.fgt';
FGE     : '.fge';

// Saturation / Flush-to-zero
SAT : '.sat';
FTZ : '.ftz';
ABS_MOD : '.abs';

// Approximation
APPROX : '.approx';

// Carry / Condition Code
CC  : '.cc';

// High/Low parts
LO  : '.lo';
HI  : '.hi';
WIDE: '.wide';
LS  : '.ls';
HS  : '.hs';

// Memory ordering and access
VOLATILE: '.volatile';
NC      : '.nc';        // non-coherent
STRONG  : '.strong';
ACQUIRE : '.acquire';
RELEASE : '.release';   // new in PTX 8.7+
MMIO    : '.mmio';      // new in PTX 8.7+
RELAXED : '.relaxed';
ACQ_REL : '.acq_rel';
SYNC    : '.sync';
GPU     : '.gpu';
GL      : '.gl';

// Scope qualifiers (PTX 7.8+)
SCOPE_QUAL : '.scope';
CLUSTER_SCOPE : '.cluster';
CTA_SCOPE     : '.cta';

// Cache hints (PTX 7.4+)
LEVEL_CACHE_HINT : '.level::cache_hint';
CA      : '.ca';
CG      : '.cg';

// Atomic operations — NOTE: .add, .min etc. are shared with reductions
ADD_ATOM : '.add'; AND_ATOM : '.and'; OR_ATOM  : '.or';
XOR_ATOM : '.xor'; INC_ATOM : '.inc'; DEC_ATOM : '.dec';
EXCH_ATOM: '.exch'; MIN_ATOM : '.min'; MAX_ATOM : '.max';
CAS_ATOM : '.cas';

// Shuffle modes
UP     : '.up';
DOWN   : '.down';
BFLY   : '.bfly';
IDX    : '.idx';

// WMMA layouts
ROW    : '.row';
COL    : '.col';
ALIGNED: '.aligned';

// WMMA shapes (including new ones)
M8N8K4        : '.m8n8k4';
M16N16K16     : '.m16n16k16';
M32N8K16      : '.m32n8k16';
M16N8K16      : '.m16n8k16';  // new in PTX 8.7

// WMMA kind and scale (PTX 8.7+)
KIND_MXF4NVF4   : '.kind::mxf4nvf4';
SCALE_VEC_SIZE  : '.scale_vec_size';
BLOCK_SCALE     : '.block_scale';

// Cache levels
L1 : '.l1';
L2 : '.l2';
LEFT_SHIFT  : '.left';
RIGHT_SHIFT : '.right';
WRAP        : '.wrap';
CLAMP       : '.clamp';
LUT         : '.LUT' | '.lut';
NAN         : '.NaN' | '.nan';
FINITE      : '.finite';
INFINITY    : '.infinity';
NUMBER      : '.number';
NORMAL      : '.normal';
SUBNORMAL   : '.subnormal';
MERGE       : '.merge';
ASEL        : '.asel';
BSEL        : '.bsel';
GENERIC     : '.generic';

// Membar scopes
CTX       : '.ctx';
SYS       : '.sys';
COHERENT  : '.coherent';

// Vote modes
BALLOT : '.ballot';
ANY    : '.any';
ALL    : '.all';
UNI    : '.uni';
EQV    : '.eqv';

// --- Special Registers ---
TID         : '%tid';
NTID        : '%ntid';
CTAID       : '%ctaid';
NCTAID      : '%nctaid';
LANEID      : '%laneid';
CLOCK       : '%clock';
CLOCK64     : '%clock64';
LANEMASK_EQ : '%lanemask_eq';
LANEMASK_LE : '%lanemask_le';
LANEMASK_LT : '%lanemask_lt';
LANEMASK_GE : '%lanemask_ge';
LANEMASK_GT : '%lanemask_gt';
PM0 : '%pm0'; PM1 : '%pm1'; PM2 : '%pm2'; PM3 : '%pm3';
PM4 : '%pm4'; PM5 : '%pm5'; PM6 : '%pm6'; PM7 : '%pm7';
SP  : '%sp';

// --- Instructions ---

// Control flow
BRA     : 'bra';
BRX     : 'brx';
CALL    : 'call';
RET     : 'ret';
EXIT    : 'exit';
TRAP    : 'trap';
BRK     : 'brk';
BRKPT   : 'brkpt';

// Arithmetic
ADD     : 'add';    SUB     : 'sub';    MUL     : 'mul';
DIV     : 'div';    REM     : 'rem';    ABS     : 'abs';
NEG     : 'neg';    SQRT    : 'sqrt';   RSQRT   : 'rsqrt';
SIN     : 'sin';    COS     : 'cos';    LG2     : 'lg2';
EX2     : 'ex2';    POPC    : 'popc';   CLZ     : 'clz';
MAD     : 'mad';    FMA     : 'fma';    ADDC    : 'addc';
SUBC    : 'subc';   MUL24   : 'mul24';  MAD24   : 'mad24';
MADC    : 'madc';
MIN     : 'min';    MAX     : 'max';    SAD     : 'sad';
COPYSIGN: 'copysign';
TESTP   : 'testp';
TANH    : 'tanh';

POPC_SUFFIX    : '.popc';
CLZ_SUFFIX     : '.clz';
FFS_SUFFIX     : '.ffs';
BREV_SUFFIX    : '.brev';
SBFE_SUFFIX    : '.sbfe';
BFI_SUFFIX     : '.bfi';
SEL_SUFFIX     : '.sel';
SHL_SUFFIX     : '.shl';
SHR_SUFFIX     : '.shr';
SAD_SUFFIX     : '.sad';
MADC_SUFFIX    : '.madc';
MSAD_SUFFIX    : '.msad';
FMA_SUFFIX     : '.fma';
DFMA_SUFFIX    : '.dfma';
FNMA_SUFFIX    : '.fnma';
DNMA_SUFFIX    : '.dnma';
BMOV_SUFFIX    : '.bmov';
BDEP_SUFFIX    : '.bdep';
BEXT_SUFFIX    : '.bext';
CS2R_SUFFIX    : '.cs2r';
CS2T_SUFFIX    : '.cs2t';
CTRSIG_SUFFIX  : '.ctrsig';
GETLMEMBASE_SUFFIX : '.getlmembase';
LOP_SUFFIX     : '.lop';
LOP3_SUFFIX    : '.lop3';
VADD_SUFFIX    : '.vadd';
VSUB_SUFFIX    : '.vsub';
VABS_SUFFIX    : '.vabs';
VNEG_SUFFIX    : '.vneg';
VMIN_SUFFIX    : '.vmin';
VMAX_SUFFIX    : '.vmax';
VSET_SUFFIX    : '.vset';
VSHL_SUFFIX    : '.vshl';
VSHR_SUFFIX    : '.vshr';
VAND_SUFFIX    : '.vand';
VOR_SUFFIX     : '.vor';
VXOR_SUFFIX    : '.vxor';
VANY_SUFFIX    : '.vany';
VALL_SUFFIX    : '.vall';
VEQ_SUFFIX     : '.veq';
VOTE_SUFFIX    : '.vote';
VLANECOUNT_SUFFIX : '.vlane.count';
VLANEPROFMASK_SUFFIX : '.vlane.prof.mask';
VLANEPROFINC_SUFFIX : '.vlane.prof.inc';
VLANEPROFTEST_SUFFIX : '.vlane.prof.test';

// Logical
AND     : 'and';    OR      : 'or';     XOR     : 'xor';
NOT     : 'not';    SELP    : 'selp';   SETP    : 'setp';
SET     : 'set';
SLCT    : 'slct';
CNOT    : 'cnot';
LOP3_LOGICAL : 'lop3';
SHL_LOGICAL : 'shl';
SHR_LOGICAL : 'shr';
SHF     : 'shf';

// Data movement
MOV     : 'mov';    LD      : 'ld';     ST      : 'st';
CVT     : 'cvt';    CVTA    : 'cvta';   RCP     : 'rcp';
PRMT    : 'prmt';
ISSPACEP: 'isspacep';
MAPA    : 'mapa';
ALLOCA  : 'alloca';
CP_ASYNC: 'cp.async';

LD_SUFFIX      : '.ld';
ST_SUFFIX      : '.st';
CVT_SUFFIX     : '.cvt';
CVTA_SUFFIX    : '.cvta';
RCP_SUFFIX     : '.rcp';
ATOM_SUFFIX    : '.atom';
CP_ASYNC_SUFFIX : '.cp.async';
ISSPACEP_SUFFIX : '.isspacep';
MAPA_SUFFIX    : '.mapa';
ALLOCA_SUFFIX  : '.alloca';
ADD_SUFFIX     : '.add';
SUB_SUFFIX     : '.sub';
MUL_SUFFIX     : '.mul';
DIV_SUFFIX     : '.div';
REM_SUFFIX     : '.rem';
ABS_SUFFIX     : '.abs';
NEG_SUFFIX     : '.neg';
MIN_SUFFIX     : '.min';
MAX_SUFFIX     : '.max';
LT_SUFFIX      : '.lt';
GT_SUFFIX      : '.gt';
EQ_SUFFIX      : '.eq';
NE_SUFFIX      : '.ne';
LE_SUFFIX      : '.le';
GE_SUFFIX      : '.ge';

// Parallel sync
BAR     : 'bar';
MEMBAR  : 'membar';
FENCE   : 'fence';
REDUX_SYNC : 'redux.sync';
MBARRIER_INIT : 'mbarrier.init';
MBARRIER_ARRIVE : 'mbarrier.arrive';
MBARRIER_TRY_WAIT : 'mbarrier.try_wait';

// Atomics
ATOM_ATOM    : 'atom';

// Warp-level
VOTE_WARP    : 'vote';
SHFL    : 'shfl';

VOTEQ   : '.voteq';
VOTEU   : '.voteu';
VOTES   : '.votes';

// Texture / Surface
TEX_TEX     : 'tex';
SURF    : 'surf';
TEX_LDG : 'tex.ldg';
TEX_GRAD: 'tex.grad';
TEX_LOD : 'tex.lod';
TXQ     : 'txq';
SULD    : 'suld';
SUST    : 'sust';
SUQ     : 'suq';

// Reduction
RED     : 'red';

// Prefetch
PREFETCH  : 'prefetch';
PREFETCHU : 'prefetchu';

// Matrix (WMMA)
WMMA    : 'wmma';

// Video / SIMD
VADD4     : 'vadd4';
VSUB4     : 'vsub4';
VAVRG4    : 'vavrg4';
VABSDIFF4 : 'vabsdiff4';
VMIN4     : 'vmin4';
VMAX4     : 'vmax4';
VSET4     : 'vset4';
DP4A      : 'dp4a';
DP2A      : 'dp2a';

// === NEW INSTRUCTIONS (PTX 8.7–9.1) ===
ST_ASYNC  : 'st.async';
RED_ASYNC : 'red.async';

TCGEN_ALLOC        : 'tcgen05.alloc';
TCGEN_DEALLOC      : 'tcgen05.dealloc';
TCGEN_RELINQUISH   : 'tcgen05.relinquish_alloc_permit';
TCGEN_CP           : 'tcgen05.cp';
TCGEN_SHIFT        : 'tcgen05.shift';
TCGEN_MMA          : 'tcgen05.mma';
TCGEN_COMMIT       : 'tcgen05.commit';

TENSORMAP_REPLACE  : 'tensormap.replace';
ST_BULK            : 'st.bulk';

BRANCHTARGETS      : '.branchtargets';

// --- Memory Space and Scope ---
GLOBAL_SPACE : '.global';
CONSTANT     : '.constant';
PARAM_MEMORY : '.param';
SHARED_MEMORY : '.shared';
LOCAL_MEMORY  : '.local';
GENERIC_MEM   : '.generic';
SYNC_SCOPE    : '.sync';
ASYNC_SCOPE   : '.async';
UNIFORM       : '.uniform';
L1_CACHE      : '.L1';
L2_CACHE      : '.L2';
TEX_MEMORY    : '.tex';
LDG_MEMORY    : '.ldg';
CA_MEMORY     : '.ca';
CG_MEMORY     : '.cg';
CS_MEMORY     : '.cs';
CV_MEMORY     : '.cv';
EV_MEMORY     : '.ev';
LU_MEMORY     : '.lu';
WL_MEMORY     : '.wl';
WU_MEMORY     : '.wu';
WU_ACC_MEMORY : '.wu.acc';
WU_RED_MEMORY : '.wu.red';
NV_MEMORY     : '.nv';
SC_MEMORY     : '.sc';
ACQUIRE_MEM   : '.acquire';
RELEASE_MEM   : '.release';
ACQ_REL_MEM   : '.acq_rel';
RELAXED_MEM   : '.relaxed';
STRONG_MEM    : '.strong';
WEAK          : '.weak';
GPU_SCOPE     : '.gpu';
GL_SCOPE      : '.gl';
SYS_SCOPE     : '.sys';
COHERENT_MEM  : '.coherent';

// --- Address Spaces ---
SPACE_64        : '.space64';
SPACE_32        : '.space32';
SPACE_16        : '.space16';
SPACE_8         : '.space8';
SPACE_4         : '.space4';
SPACE_2         : '.space2';
SPACE_1         : '.space1';
SPACE_PRED      : '.space.pred';
SPACE_U8        : '.space.u8';
SPACE_U16       : '.space.u16';
SPACE_U32       : '.space.u32';
SPACE_U64       : '.space.u64';
SPACE_S8        : '.space.s8';
SPACE_S16       : '.space.s16';
SPACE_S32       : '.space.s32';
SPACE_S64       : '.space.s64';
SPACE_F16       : '.space.f16';
SPACE_F32       : '.space.f32';
SPACE_F64       : '.space.f64';
SPACE_B8        : '.space.b8';
SPACE_B16       : '.space.b16';
SPACE_B32       : '.space.b32';
SPACE_B64       : '.space.b64';

// --- Architecture Types ---
SM_ : 'sm_';
SM_SUFFIX_F : 'f';
SM_SUFFIX_A : 'a';

// --- Punctuation ---
PLUS        : '+';
MINUS       : '-';
STAR        : '*';
SLASH       : '/';
PERCENT     : '%';
DOT         : '.';
COMMA       : ',';
SEMI        : ';';
COLON       : ':';
ASSIGN      : '=';
LEFT_PAREN  : '(';
RIGHT_PAREN : ')';
LEFT_BRACE  : '{';
RIGHT_BRACE : '}';
LEFT_BRACK  : '[';
RIGHT_BRACK : ']';
LESS        : '<';
GREATER     : '>';
AT          : '@';
DOLLAR      : '$';  // No space!

// --- Literals ---
fragment HEX_DIGIT : [0-9a-fA-F];
HEX_LITERAL : '0x' HEX_DIGIT+;

fragment FLOAT_LITERAL
    : DIGIT+ '.' DIGIT* ([eE] [+-]? DIGIT+)? [fF]?
    | '.' DIGIT+ ([eE] [+-]? DIGIT+)? [fF]?
    | DIGIT+ [eE] [+-]? DIGIT+ [fF]?
    ;

IMMEDIATE
    : MINUS? (HEX_LITERAL | FLOAT_LITERAL | DIGIT+)
    ;

STRING
    : '"' (~["\\\r\n] | '\\' .)* '"'
    ;

// Identifier: allow $ and _ at start, digits after
ID
    : [a-zA-Z_$] [a-zA-Z_0-9$]*
    ;

fragment DIGIT : [0-9];

// --- Whitespace & Comments ---
WS          : [ \t\r\n]+ -> skip;
BLOCK_COMMENT : '/*' .*? '*/' -> skip;
LINE_COMMENT  : '//' ~[\r\n]* -> skip;