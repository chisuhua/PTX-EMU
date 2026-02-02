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

// --- Vector Types ---
V2 : '.v2';
V4 : '.v4';

// --- Instruction Modifiers ---

// Rounding
RN  : '.rn';  RZ  : '.rz';  RM  : '.rm';  RP  : '.rp';

// Comparison
EQ  : '.eq';  NE  : '.ne';  LT  : '.lt';  LE  : '.le';
GT  : '.gt';  GE  : '.ge';  LTU : '.ltu'; LEU : '.leu';
GTU : '.gtu'; GEU : '.geu';

// Saturation / Flush-to-zero
SAT : '.sat';
FTZ : '.ftz';

// Approximation
APPROX : '.approx';

// Carry / Condition Code
CC  : '.cc';

// High/Low parts
LO  : '.lo';
HI  : '.hi';
WIDE: '.wide';

// Memory ordering and access
VOLATILE: '.volatile';
NC      : '.nc';        // non-coherent
STRONG  : '.strong';
ACQUIRE : '.acquire';
RELEASE : '.release';   // new in PTX 8.7+
MMIO    : '.mmio';      // new in PTX 8.7+

// Scope qualifiers (PTX 7.8+)
SCOPE_QUAL : '.scope';
CLUSTER_SCOPE : '.cluster';
CTA_SCOPE     : '.cta';

// Cache hints (PTX 7.4+)
LEVEL_CACHE_HINT : '.level::cache_hint';

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

// --- Instructions ---

// Control flow
BRA     : 'bra';
CALL    : 'call';
RET     : 'ret';
EXIT    : 'exit';
TRAP    : 'trap';
BRK     : 'brk';

// Arithmetic
ADD     : 'add';    SUB     : 'sub';    MUL     : 'mul';
DIV     : 'div';    REM     : 'rem';    ABS     : 'abs';
NEG     : 'neg';    SQRT    : 'sqrt';   RSQRT   : 'rsqrt';
SIN     : 'sin';    COS     : 'cos';    LG2     : 'lg2';
EX2     : 'ex2';    POPC    : 'popc';   CLZ     : 'clz';
MAD     : 'mad';    FMA     : 'fma';    ADDC    : 'addc';
SUBC    : 'subc';   MUL24   : 'mul24';  MAD24   : 'mad24';

// Logical
AND     : 'and';    OR      : 'or';     XOR     : 'xor';
NOT     : 'not';    SELP    : 'selp';   SETP    : 'setp';

// Data movement
MOV     : 'mov';    LD      : 'ld';     ST      : 'st';
CVT     : 'cvt';    CVTA    : 'cvta';   RCP     : 'rcp';

// Parallel sync
BAR     : 'bar';
MEMBAR  : 'membar';

// Atomics
ATOM    : 'atom';

// Warp-level
VOTE    : 'vote';
SHFL    : 'shfl';

// Texture / Surface
TEX     : 'tex';
SURF    : 'surf';

// Reduction
RED     : 'red';

// Prefetch
PREFETCH  : 'prefetch';
PREFETCHU : 'prefetchu';

// Matrix (WMMA)
WMMA    : 'wmma';

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