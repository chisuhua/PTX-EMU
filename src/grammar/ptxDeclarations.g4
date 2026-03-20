parser grammar ptxDeclarations;

options {
    tokenVocab = ptxLexer;
}

// Import ONLY ptxOperands (NOT ptxInstructions to avoid circular dependency)
// funcBody and functionDecl are now defined in ptxInstructions.g4
import ptxOperands;

// --- Top-level declarations ---
declaration
    : versionDirective
    | targetDirective
    | addressSizeDirective
    | fileDirective
    | sectionDirective
    | variableDecl
    | pragmaDirective
    | abiPreserveDirective
    ;

// ---
versionDirective
    : VERSION anyVersion SEMI?
    ;

anyVersion
    : IMMEDIATE DOT IMMEDIATE
    | IMMEDIATE
    | ID
    ;

// ---
targetDirective
    : TARGET SM_TARGET (COMMA SM_TARGET)* SEMI?
    ;

// ---
addressSizeDirective
    : ADDRESS_SIZE IMMEDIATE SEMI?
    ;


// --- File ---
fileDirective
    : FILE IMMEDIATE STRING SEMI
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
    : visibility? storageClass alignClause? typeSpecifier? vectorSpec? ID arraySize? initializer? SEMI
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

arraySize
    : (LEFT_BRACK IMMEDIATE RIGHT_BRACK)+
    | (LESS IMMEDIATE GREATER)+
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
threadDim
    : IMMEDIATE (COMMA IMMEDIATE (COMMA IMMEDIATE)?)?
    ;
