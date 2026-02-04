lexer grammar ptxLexer;

// ============================================================================
// FRAGMENTS (must be defined FIRST)
// ============================================================================
fragment DIGIT      : [0-9];
fragment HEX_DIGIT  : [0-9a-fA-F];

// ============================================================================
// WHITESPACE & COMMENTS
// ============================================================================
WS              : [ \t\r\n]+ -> skip;
BLOCK_COMMENT   : '/*' .*? '*/' -> skip;
LINE_COMMENT    : '//' ~[\r\n]* -> skip;

// ============================================================================
// PUNCTUATION
// ============================================================================
PLUS            : '+';
MINUS           : '-';
STAR            : '*';
SLASH           : '/';
PERCENT         : '%';
DOT             : '.';
COMMA           : ',';
SEMI            : ';';
COLON           : ':';
COLONCOLON      : '::';  // Required for .level::cache_hint, .kind::mxf4nvf4
ASSIGN          : '=';
LEFT_PAREN      : '(';
RIGHT_PAREN     : ')';
LEFT_BRACE      : '{';
RIGHT_BRACE     : '}';
LEFT_BRACK      : '[';
RIGHT_BRACK     : ']';
LESS            : '<';
GREATER         : '>';
AT              : '@';
DOLLAR          : '$';

// ============================================================================
// DIRECTIVES (longer prefixes FIRST to avoid fragmentation)
// ============================================================================
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
CONSTANT        : '.constant';  // MUST come before .const
CONST           : '.const';
GLOBAL          : '.global';
LOCAL           : '.local';
SHARED          : '.shared';
ALIGN           : '.align';
MAXNTID         : '.maxntid';   // MUST come before .max
MAXNREG         : '.maxnreg';   // MUST come before .max
MINNCTAPERSM    : '.minnctapersm'; // MUST come before .min
REQNTID         : '.reqntid';
PRAGMA          : '.pragma';
BRANCHTARGETS   : '.branchtargets';

// ABI directives (PTX 9.0+)
ABI_PRESERVE       : '.abi_preserve';
ABI_PRESERVE_CTRL  : '.abi_preserve_control';

// ============================================================================
// DATA TYPES (vector types BEFORE scalar prefixes)
// ============================================================================
// Vector types (longer FIRST)
F16X2   : '.f16x2';   // BEFORE .f16
BF16X2  : '.bf16x2';  // BEFORE .bf16
U16X2   : '.u16x2';   // BEFORE .u16
S16X2   : '.s16x2';   // BEFORE .s16
E2M1X4  : '.e2m1x4';  // BEFORE .e2m1
E4M3X4  : '.e4m3x4';  // BEFORE .e4m3
E5M2X4  : '.e5m2x4';  // BEFORE .e5m2
E3M2X4  : '.e3m2x4';  // BEFORE .e3m2
E2M3X4  : '.e2m3x4';  // BEFORE .e2m3

// Scalar types
U8      : '.u8';    U16     : '.u16';   U32     : '.u32';   U64     : '.u64';
S8      : '.s8';    S16     : '.s16';   S32     : '.s32';   S64     : '.s64';
F16     : '.f16';   F32     : '.f32';   F64     : '.f64';
F8      : '.f8';    // legacy alias
BF16    : '.bf16';
E4M3    : '.e4m3';
E5M2    : '.e5m2';
E3M2    : '.e3m2';
E2M3    : '.e2m3';
E2M1    : '.e2m1';
B8      : '.b8';    B16     : '.b16';   B32     : '.b32';   B64     : '.b64';   B128    : '.b128';
PRED    : '.pred';

// ============================================================================
// VECTOR TYPE SPECIFIERS
// ============================================================================
V2      : '.v2';
V4      : '.v4';

// ============================================================================
// ROUNDING MODES
// ============================================================================
RN      : '.rn';    RZ      : '.rz';    RM      : '.rm';    RP      : '.rp';
RS      : '.rs';
RNI     : '.rni';
RZI     : '.rzi';
RMI     : '.rmi';
RPI     : '.rpi';

// ============================================================================
// COMPARISON OPERATORS
// ============================================================================
EQ      : '.eq';    NE      : '.ne';    LT      : '.lt';    LE      : '.le';
GT      : '.gt';    GE      : '.ge';    LTU     : '.ltu';   LEU     : '.leu';
GTU     : '.gtu';   GEU     : '.geu';
FEQ     : '.feq';
FNE     : '.fne';
FLT     : '.flt';
FLE     : '.fle';
FGT     : '.fgt';
FGE     : '.fge';

// ============================================================================
// INSTRUCTION MODIFIERS
// ============================================================================
SAT     : '.sat';
FTZ     : '.ftz';
ABS_MOD : '.abs';
APPROX  : '.approx';
CC      : '.cc';
LO      : '.lo';
HI      : '.hi';
WIDE    : '.wide';
LS      : '.ls';
HS      : '.hs';

// ============================================================================
// MEMORY SPACES (used in ld/st and variable declarations)
// NOTE: These share tokens with directives - parser handles context
// ============================================================================
GENERIC_SPACE   : '.generic';   // AFTER data types to avoid .gen* conflicts
WL_SPACE        : '.wl';        // PTX 8.0+: weakly-ordered load space
WU_SPACE        : '.wu';        // PTX 8.0+: weakly-ordered store space
TEX_SPACE       : '.tex';
SURF_SPACE      : '.surf';
LDG_SPACE       : '.ldg';       // PTX 5.0+: cached global load

// ============================================================================
// CACHE OPERATORS (for ld/st .cache)
// MUST come before shorter prefixes (e.g., .ca before .c*)
// ============================================================================
WB      : '.wb';        // write-back (default for global)
WT      : '.wt';        // write-through
CG      : '.cg';        // cache at global level
CS      : '.cs';        // cache streaming (likely accessed once)
CV      : '.cv';        // cache volatile (no write combining)
CA      : '.ca';        // cache at all levels (synonym for .wb in PTX 8.0+)
NC      : '.nc';        // non-coherent
VOLATILE: '.volatile';  // volatile semantics (also used in atomics)
MMIO    : '.mmio';      // PTX 8.7+: memory-mapped I/O space

// ============================================================================
// CACHE HINTS (suffix modifiers for cache operators)
// ============================================================================
EV      : '.ev';        // evict after use (e.g., .wb.ev)
LU      : '.lu';        // last use hint (e.g., .cg.lu)

// ============================================================================
// MEMORY ORDER QUALIFIERS (for atomics/fence)
// ============================================================================
STRONG      : '.strong';
ACQUIRE     : '.acquire';
RELEASE     : '.release';
RELAXED     : '.relaxed';
ACQ_REL     : '.acq_rel';
SYNC        : '.sync';

// ============================================================================
// SCOPE QUALIFIERS (for membar/fence/atomics)
// ============================================================================
GPU_SCOPE   : '.gpu';
GL_SCOPE    : '.gl';
SYS_SCOPE   : '.sys';
CTX         : '.ctx';
COHERENT    : '.coherent';
SCOPE_QUAL  : '.scope';
CLUSTER_SCOPE: '.cluster';
CTA_SCOPE   : '.cta';
LEVEL       : '.level';          // For .level::cache_hint (split token)

// ============================================================================
// ATOMIC OPERATIONS (AFTER directives containing these prefixes)
// ============================================================================
ADD_ATOM    : '.add';
AND_ATOM    : '.and';
OR_ATOM     : '.or';
XOR_ATOM    : '.xor';
INC_ATOM    : '.inc';
DEC_ATOM    : '.dec';
EXCH_ATOM   : '.exch';
MIN_ATOM    : '.min';   // AFTER .minnctapersm
MAX_ATOM    : '.max';   // AFTER .maxntid/.maxnreg
CAS_ATOM    : '.cas';

// ============================================================================
// SHUFFLE MODES
// ============================================================================
UP      : '.up';
DOWN    : '.down';
BFLY    : '.bfly';
IDX     : '.idx';

// ============================================================================
// WMMA LAYOUTS & SHAPES
// ============================================================================
ROW         : '.row';
COL         : '.col';
ALIGNED     : '.aligned';
M8N8K4      : '.m8n8k4';
M16N16K16   : '.m16n16k16';
M32N8K16    : '.m32n8k16';
M16N8K16    : '.m16n8k16';  // PTX 8.7+

// ============================================================================
// WMMA KIND & SCALE (PTX 8.7+)
// ============================================================================
KIND        : '.kind';           // For .kind::mxf4nvf4 (split token)
MXF4NVF4    : 'mxf4nvf4';        // Requires parser handling with COLONCOLON
SCALE_VEC_SIZE  : '.scale_vec_size';
BLOCK_SCALE     : '.block_scale';

// ============================================================================
// OTHER MODIFIERS
// ============================================================================
LEFT_SHIFT  : '.left';
RIGHT_SHIFT : '.right';
WRAP        : '.wrap';
CLAMP       : '.clamp';
LUT         : '.lut';            // lowercase as per PTX spec
NAN         : '.nan';            // lowercase
FINITE      : '.finite';
INFINITY    : '.infinity';
NUMBER      : '.number';
NORMAL      : '.normal';
SUBNORMAL   : '.subnormal';
MERGE       : '.merge';
ASEL        : '.asel';
BSEL        : '.bsel';

// ============================================================================
// VOTE MODES & MODIFIERS
// ============================================================================
BALLOT      : '.ballot';
ANY         : '.any';
ALL         : '.all';
UNI         : '.uni';
EQV         : '.eqv';
VOTEQ       : '.voteq';
VOTEU       : '.voteu';
VOTES       : '.votes';

// ============================================================================
// SPECIAL REGISTERS
// ============================================================================
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

// ============================================================================
// INSTRUCTIONS (Keywords without leading dot)
// ============================================================================

// Control Flow
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

// Logical
AND     : 'and';    OR      : 'or';     XOR     : 'xor';
NOT     : 'not';    SELP    : 'selp';   SETP    : 'setp';
SET     : 'set';
SLCT    : 'slct';
CNOT    : 'cnot';
LOP3    : 'lop3';   // Removed _LOGICAL suffix
SHL     : 'shl';    // Removed _LOGICAL suffix
SHR     : 'shr';    // Removed _LOGICAL suffix
SHF     : 'shf';

// Data Movement
MOV     : 'mov';    LD      : 'ld';     ST      : 'st';
CVT     : 'cvt';    CVTA    : 'cvta';   RCP     : 'rcp';
PRMT    : 'prmt';
ISSPACEP: 'isspacep';
MAPA    : 'mapa';
ALLOCA  : 'alloca';
CP_ASYNC: 'cp.async';

// Parallel Sync
BAR         : 'bar';
MEMBAR      : 'membar';
FENCE       : 'fence';
REDUX_SYNC  : 'redux.sync';
MBARRIER_INIT       : 'mbarrier.init';
MBARRIER_ARRIVE     : 'mbarrier.arrive';
MBARRIER_TRY_WAIT   : 'mbarrier.try_wait';

// Atomics
ATOM    : 'atom';   // Renamed from ATOM_ATOM

// Warp-level
VOTE    : 'vote';   // Renamed from VOTE_WARP
SHFL    : 'shfl';

// Texture / Surface
TEX     : 'tex';    // Renamed from TEX_TEX
SURF    : 'surf';
TEX_LDG : 'tex.ldg';
TEX_GRAD: 'tex.grad';
TEX_LOD : 'tex.lod';
TXQ     : 'txq';
SULD    : 'suld';
SUST    : 'sust';
SUQ     : 'suq';

// Reduction & Prefetch
RED         : 'red';
PREFETCH    : 'prefetch';
PREFETCHU   : 'prefetchu';

// Matrix (WMMA)
WMMA        : 'wmma';

// Video / SIMD
VADD4       : 'vadd4';
VSUB4       : 'vsub4';
VAVRG4      : 'vavrg4';
VABSDIFF4   : 'vabsdiff4';
VMIN4       : 'vmin4';
VMAX4       : 'vmax4';
VSET4       : 'vset4';
DP4A        : 'dp4a';
DP2A        : 'dp2a';

// NEW INSTRUCTIONS (PTX 8.7–9.1)
ST_ASYNC    : 'st.async';
RED_ASYNC   : 'red.async';
TCGEN_ALLOC        : 'tcgen05.alloc';
TCGEN_DEALLOC      : 'tcgen05.dealloc';
TCGEN_RELINQUISH   : 'tcgen05.relinquish_alloc_permit';
TCGEN_CP           : 'tcgen05.cp';
TCGEN_SHIFT        : 'tcgen05.shift';
TCGEN_MMA          : 'tcgen05.mma';
TCGEN_COMMIT       : 'tcgen05.commit';
TENSORMAP_REPLACE  : 'tensormap.replace';
ST_BULK            : 'st.bulk';

// ============================================================================
// ARCHITECTURE TARGETS (combined token)
// ============================================================================
SM_TARGET   : 'sm_' DIGIT+ [a-zA-Z]?;  // Matches sm_90, sm_90a, sm_52f, etc.

// ============================================================================
// LITERALS (corrected to avoid fragment misuse)
// ============================================================================
// IMMEDIATE covers all numeric literals (integer/float/hex) without fragment dependency
IMMEDIATE
    : DIGIT+ '.' DIGIT* ([eE] [+-]? DIGIT+)? [fF]?   // 1.0, 1., 1.0e10, 1.0f
    | '.' DIGIT+ ([eE] [+-]? DIGIT+)? [fF]?          // .5, .5e10, .5F
    | DIGIT+ [eE] [+-]? DIGIT+ [fF]?                 // 1e10, 1e+10f
    | '0x' HEX_DIGIT+                                // 0x1a, 0XFF
    | DIGIT+                                         // 123, 0
    ;

STRING
    : '"' ( ~["\\\r\n] | '\\' [nrt0"'\\] | '\\' 'x' HEX_DIGIT HEX_DIGIT )* '"'
    ;

// ============================================================================
// IDENTIFIERS (labels, variable names, function names)
// ============================================================================
ID
    : [a-zA-Z_$] [a-zA-Z_0-9$]*
    ;