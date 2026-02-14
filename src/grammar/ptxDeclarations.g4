parser grammar ptxDeclarations;

options {
    tokenVocab = ptxLexer;
}

// Import instruction grammar to access instructionList (used in function bodies)
import ptxOperands, ptxInstructions;

// --- Top-level declarations ---
declaration
    : versionDirective
    | targetDirective
    | addressSizeDirective
    | fileDirective
    | sectionDirective
    | variableDecl
    | functionDecl
    | pragmaDirective
    | abiPreserveDirective
    ;

// --- Version ---
versionDirective
    : VERSION IMMEDIATE DOT IMMEDIATE SEMI
    ;

// --- Target ---
targetDirective
    : TARGET SM_TARGET (COMMA SM_TARGET)* SEMI
    ;

// --- Address Size ---
addressSizeDirective
    : ADDRESS_SIZE IMMEDIATE SEMI
    ;

// --- File ---
fileDirective
    : FILE DECIMAL_INT STRING SEMI
    ;

// --- Section ---
sectionDirective
    : SECTION ID SEMI
    ;

// --- Pragma ---
pragmaDirective
    : PRAGMA ID (ASSIGN STRING)? SEMI
    ;

// --- ABI Preserve (PTX 9.0+) ---
abiPreserveDirective
    : ABI_PRESERVE_CTRL? ABI_PRESERVE ID SEMI
    ;

// --- Variable Declarations ---
variableDecl
    : visibility? storageClass typeSpecifier? vectorSpec? ID arraySize? alignClause? initializer? SEMI
    ;

visibility
    : VISIBLE
    | EXTERN
    ;

// NOTE: .constant is NOT a valid storage class for variables (only .const)
storageClass
    : REG
    | PARAM
    | CONST          // .const only (not .constant)
    | GLOBAL
    | LOCAL
    | SHARED
    ;

typeSpecifier
    : U8 | U16 | U32 | U64
    | S8 | S16 | S32 | S64
    | F16 | F32 | F64
    | BF16
    | E4M3 | E5M2 | E3M2 | E2M3 | E2M1
    | B8 | B16 | B32 | B64 | B128
    | PRED
    ;

vectorSpec
    : V2 | V4
    ;

// Support multi-dimensional arrays: [10][20]
arraySize
    : (LEFT_BRACK IMMEDIATE RIGHT_BRACK)+
    ;

// Align value must be power-of-two (validated in semantic analysis)
alignClause
    : ALIGN IMMEDIATE
    ;

initializer
    : ASSIGN initializerValue
    ;

initializerValue
    : IMMEDIATE                          // e.g., = 42
    | STRING                             // e.g., = "hello"
    | ID                                 // label or variable reference
    | specialRegister                    // e.g., = %tid.x (defined in ptxInstructions)
    | LEFT_BRACE initializerList RIGHT_BRACE  // array/struct init
    ;

initializerList
    : initializerValue (COMMA initializerValue)*
    ;

// --- Function Declarations ---
functionDecl
    : visibility? FUNC functionHeader funcBody
    | visibility? ENTRY functionHeader funcBody
    ;

functionHeader
    : ID paramList? functionAttribute*
    ;

paramList
    : LEFT_PAREN paramDecl (COMMA paramDecl)* RIGHT_PAREN
    ;

// NOTE: Parameters are implicitly in .param space; no PARAM token here
paramDecl
    : typeSpecifier? vectorSpec? ID
    ;

// Function attributes (PTX §6.1)
functionAttribute
    : MAXNREG IMMEDIATE
    | REQNTID threadDim
    | MINNCTAPERSM IMMEDIATE
    ;

threadDim
    : IMMEDIATE (COMMA IMMEDIATE (COMMA IMMEDIATE)?)?
    ;

