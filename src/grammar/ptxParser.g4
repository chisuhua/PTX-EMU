parser grammar ptxParser;

options {
    tokenVocab = ptxLexer;
}

import ptxDeclarations, ptxInstructions;

ptxFile
    : declaration* EOF
    ;