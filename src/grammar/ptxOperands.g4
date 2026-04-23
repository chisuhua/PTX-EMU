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

// Vector register pair for store instructions: {%r5, %r1}
vectorRegister
    : LEFT_BRACE register COMMA register RIGHT_BRACE
    ;

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
