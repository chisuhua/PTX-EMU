parser grammar ptxOperands;

options {
    tokenVocab = ptxLexer;
}

operand
    : register
    | immediate
    | address
    | specialRegister
    | ID
    ;

register
    : DOLLAR ID
    | PERCENT ID
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
    : DOT ('x' | 'y' | 'z' | 'w')
    ;

address
    : LEFT_BRACK addressExpr RIGHT_BRACK
    ;

addressExpr
    : operand (PLUS immediate)?
    ;