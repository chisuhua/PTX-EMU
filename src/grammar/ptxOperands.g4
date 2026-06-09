parser grammar ptxOperands;

options {
    tokenVocab = ptxLexer;
}

operand
: register
| immediate
| address
| specialRegister
| vectorRegister
| ID
;

// Vector register list for store instructions: {%r5, %r1, %r2, %r3}
// virtRegister allows bare ID for CUTLASS-generated PTX (e.g., {tmp, %r2}).
vectorRegister
    : LEFT_BRACE virtRegister (COMMA virtRegister)+ RIGHT_BRACE
    ;

// virtRegister is a register OR a bare identifier (used in vector register lists).
virtRegister
    : register
    | ID
    ;

// Register: %ID (virtual) or $ID (label/string alias).
// NOTE: bare ID is NOT a register here — keep this rule strict to avoid
// misclassifying long PTX symbol names (e.g., _Z14..._param_1) as registers.
register
    : PERCENT ID
    | DOLLAR ID
    ;

// 修正1: 使用统一的 IMMEDIATE token（匹配修正后的 lexer）
immediate
    : MINUS? IMMEDIATE
    ;

specialRegister
    : TID component?
    | NTID component?
    | CTAID component?
    | NCTAID component?
    | LANEID
    | CLOCK
    | CLOCK64
    | LANEMASK_EQ
    | LANEMASK_LE
    | LANEMASK_LT
    | LANEMASK_GE
    | LANEMASK_GT
    | PM0 | PM1 | PM2 | PM3
    | PM4 | PM5 | PM6 | PM7
    | SP
    ;

component
    : DOT (X_COMP | Y_COMP | Z_COMP | W_COMP)
    ;

address
    : LEFT_BRACK addressExpr RIGHT_BRACK
    ;

addressExpr
    : operand (PLUS immediate)?
    ;
