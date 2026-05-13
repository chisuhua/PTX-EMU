# PTX Lane Execution Path Review Report

## Metadata

| Item | Value |
|------|-------|
| File | `test_divergence_sync_standalone.1.sm_100.ptx` |
| Kernel | `_Z27test_divergence_sync_kernelIiEvPT_` |
| Start Line | 25 |
| Analyzed Lanes | 32 (tid.x 0-31) |
| Unique Paths | 2 |

## Register & Predicate Analysis (LLM)

- **%p1**: lane_id < 16 (from setp.lt.u32 %r2, 16 at PC where %r2 = lane_id)
- **%p2**: loop_iteration < loop_iterations (from setp.ne.s32 %r88, -15 in loop)
- **%p3**: tid.x != 0 (from setp.ne.s32 %r1, 0)

## Path Summary

| Path | tid.x Range | Lanes | Total PC |
|------|-------------|-------|----------|
| Path 1 | 0-15 | 16 | 85 |
| Path 2 | 16-31 | 16 | 94 |

## Execution Matrix (Lane × PC)

*Only showing PC ranges that differ between paths.*

| PC | Path1 | Path2 |
|---|---|---|
| 7 | L124: add.s32 %r15, %r2, -1; | L31: neg.s32 %r88, %r2; |
| 8 | L125: mul.wide.u32 %rd3, %r15, %r2; | L32: mov.b32 %r90, 1; |
| 9 | L126: shr.u64 %rd4, %rd3, 1; | L33: mov.b32 %r87, 15; |
| 10 | L127: cvt.u32.u64 %r16, %rd4; | L35: add.s32 %r14, %r87, -14; |
| 11 | L128: add.s32 %r90, %r2, %r16; | L36: mul.lo.s32 %r90, %r14, %r90; |
| 12 | L129: bra.uni $L__BB0_3; | L37: add.s32 %r88, %r88, 1; |
| 13 | L42: shl.b32 %r17, %r2, 2; | L38: add.s32 %r87, %r87, 1; |
| 14 | L43: mov.u32 %r18, _ZZ27test_divergence_... | L39: setp.ne.s32 %p2, %r88, -15; |
| 15 | L44: add.s32 %r19, %r18, %r17; | L40: @%p2 bra $L__BB0_2; |
| 16 | L45: st.shared.u32 [%r19], %r90; | L35: add.s32 %r14, %r87, -14; |
| 17 | L46: bar.sync 0; | L36: mul.lo.s32 %r90, %r14, %r90; |
| 18 | L47: setp.ne.s32 %p3, %r1, 0; | L37: add.s32 %r88, %r88, 1; |
| 19 | L48: @%p3 bra $L__BB0_5; | L38: add.s32 %r87, %r87, 1; |
| 20 | L49: ld.shared.u32 %r24, [_ZZ27test_dive... | L39: setp.ne.s32 %p2, %r88, -15; |
| 21 | L50: ld.shared.u32 %r25, [_ZZ27test_dive... | L40: @%p2 bra $L__BB0_2; |
| 22 | L51: add.s32 %r26, %r25, %r24; | L42: shl.b32 %r17, %r2, 2; |
| 23 | L52: ld.shared.u32 %r27, [_ZZ27test_dive... | L43: mov.u32 %r18, _ZZ27test_divergence_... |
| 24 | L53: add.s32 %r28, %r27, %r26; | L44: add.s32 %r19, %r18, %r17; |
| 25 | L54: ld.shared.u32 %r29, [_ZZ27test_dive... | L45: st.shared.u32 [%r19], %r90; |
| 26 | L55: add.s32 %r30, %r29, %r28; | L46: bar.sync 0; |
| 27 | L56: ld.shared.u32 %r31, [_ZZ27test_dive... | L47: setp.ne.s32 %p3, %r1, 0; |
| 28 | L57: add.s32 %r32, %r31, %r30; | L48: @%p3 bra $L__BB0_5; |
| 29 | L58: ld.shared.u32 %r33, [_ZZ27test_dive... | L49: ld.shared.u32 %r24, [_ZZ27test_dive... |
| 30 | L59: add.s32 %r34, %r33, %r32; | L50: ld.shared.u32 %r25, [_ZZ27test_dive... |
| 31 | L60: ld.shared.u32 %r35, [_ZZ27test_dive... | L51: add.s32 %r26, %r25, %r24; |
| 32 | L61: add.s32 %r36, %r35, %r34; | L52: ld.shared.u32 %r27, [_ZZ27test_dive... |
| 33 | L62: ld.shared.u32 %r37, [_ZZ27test_dive... | L53: add.s32 %r28, %r27, %r26; |
| 34 | L63: add.s32 %r38, %r37, %r36; | L54: ld.shared.u32 %r29, [_ZZ27test_dive... |
| 35 | L64: ld.shared.u32 %r39, [_ZZ27test_dive... | L55: add.s32 %r30, %r29, %r28; |
| 36 | L65: add.s32 %r40, %r39, %r38; | L56: ld.shared.u32 %r31, [_ZZ27test_dive... |
| 37 | L66: ld.shared.u32 %r41, [_ZZ27test_dive... | L57: add.s32 %r32, %r31, %r30; |
| 38 | L67: add.s32 %r42, %r41, %r40; | L58: ld.shared.u32 %r33, [_ZZ27test_dive... |
| 39 | L68: ld.shared.u32 %r43, [_ZZ27test_dive... | L59: add.s32 %r34, %r33, %r32; |
| 40 | L69: add.s32 %r44, %r43, %r42; | L60: ld.shared.u32 %r35, [_ZZ27test_dive... |
| 41 | L70: ld.shared.u32 %r45, [_ZZ27test_dive... | L61: add.s32 %r36, %r35, %r34; |
| 42 | L71: add.s32 %r46, %r45, %r44; | L62: ld.shared.u32 %r37, [_ZZ27test_dive... |
| 43 | L72: ld.shared.u32 %r47, [_ZZ27test_dive... | L63: add.s32 %r38, %r37, %r36; |
| 44 | L73: add.s32 %r48, %r47, %r46; | L64: ld.shared.u32 %r39, [_ZZ27test_dive... |
| 45 | L74: ld.shared.u32 %r49, [_ZZ27test_dive... | L65: add.s32 %r40, %r39, %r38; |
| 46 | L75: add.s32 %r50, %r49, %r48; | L66: ld.shared.u32 %r41, [_ZZ27test_dive... |
| 47 | L76: ld.shared.u32 %r51, [_ZZ27test_dive... | L67: add.s32 %r42, %r41, %r40; |
| 48 | L77: add.s32 %r52, %r51, %r50; | L68: ld.shared.u32 %r43, [_ZZ27test_dive... |
| 49 | L78: ld.shared.u32 %r53, [_ZZ27test_dive... | L69: add.s32 %r44, %r43, %r42; |
| 50 | L79: add.s32 %r54, %r53, %r52; | L70: ld.shared.u32 %r45, [_ZZ27test_dive... |
| 51 | L80: ld.shared.u32 %r55, [_ZZ27test_dive... | L71: add.s32 %r46, %r45, %r44; |
| 52 | L81: add.s32 %r56, %r55, %r54; | L72: ld.shared.u32 %r47, [_ZZ27test_dive... |
| 53 | L82: ld.shared.u32 %r57, [_ZZ27test_dive... | L73: add.s32 %r48, %r47, %r46; |
| 54 | L83: add.s32 %r58, %r57, %r56; | L74: ld.shared.u32 %r49, [_ZZ27test_dive... |
| 55 | L84: ld.shared.u32 %r59, [_ZZ27test_dive... | L75: add.s32 %r50, %r49, %r48; |
| 56 | L85: add.s32 %r60, %r59, %r58; | L76: ld.shared.u32 %r51, [_ZZ27test_dive... |
| 57 | L86: ld.shared.u32 %r61, [_ZZ27test_dive... | L77: add.s32 %r52, %r51, %r50; |
| 58 | L87: add.s32 %r62, %r61, %r60; | L78: ld.shared.u32 %r53, [_ZZ27test_dive... |
| 59 | L88: ld.shared.u32 %r63, [_ZZ27test_dive... | L79: add.s32 %r54, %r53, %r52; |
| 60 | L89: add.s32 %r64, %r63, %r62; | L80: ld.shared.u32 %r55, [_ZZ27test_dive... |
| 61 | L90: ld.shared.u32 %r65, [_ZZ27test_dive... | L81: add.s32 %r56, %r55, %r54; |
| 62 | L91: add.s32 %r66, %r65, %r64; | L82: ld.shared.u32 %r57, [_ZZ27test_dive... |
| 63 | L92: ld.shared.u32 %r67, [_ZZ27test_dive... | L83: add.s32 %r58, %r57, %r56; |
| 64 | L93: add.s32 %r68, %r67, %r66; | L84: ld.shared.u32 %r59, [_ZZ27test_dive... |
| 65 | L94: ld.shared.u32 %r69, [_ZZ27test_dive... | L85: add.s32 %r60, %r59, %r58; |
| 66 | L95: add.s32 %r70, %r69, %r68; | L86: ld.shared.u32 %r61, [_ZZ27test_dive... |
| 67 | L96: ld.shared.u32 %r71, [_ZZ27test_dive... | L87: add.s32 %r62, %r61, %r60; |
| 68 | L97: add.s32 %r72, %r71, %r70; | L88: ld.shared.u32 %r63, [_ZZ27test_dive... |
| 69 | L98: ld.shared.u32 %r73, [_ZZ27test_dive... | L89: add.s32 %r64, %r63, %r62; |
| 70 | L99: add.s32 %r74, %r73, %r72; | L90: ld.shared.u32 %r65, [_ZZ27test_dive... |
| 71 | L100: ld.shared.u32 %r75, [_ZZ27test_dive... | L91: add.s32 %r66, %r65, %r64; |
| 72 | L101: add.s32 %r76, %r75, %r74; | L92: ld.shared.u32 %r67, [_ZZ27test_dive... |
| 73 | L102: ld.shared.u32 %r77, [_ZZ27test_dive... | L93: add.s32 %r68, %r67, %r66; |
| 74 | L103: add.s32 %r78, %r77, %r76; | L94: ld.shared.u32 %r69, [_ZZ27test_dive... |
| 75 | L104: ld.shared.u32 %r79, [_ZZ27test_dive... | L95: add.s32 %r70, %r69, %r68; |
| 76 | L105: add.s32 %r80, %r79, %r78; | L96: ld.shared.u32 %r71, [_ZZ27test_dive... |
| 77 | L106: ld.shared.u32 %r81, [_ZZ27test_dive... | L97: add.s32 %r72, %r71, %r70; |
| 78 | L107: add.s32 %r82, %r81, %r80; | L98: ld.shared.u32 %r73, [_ZZ27test_dive... |
| 79 | L108: ld.shared.u32 %r83, [_ZZ27test_dive... | L99: add.s32 %r74, %r73, %r72; |
| 80 | L109: add.s32 %r84, %r83, %r82; | L100: ld.shared.u32 %r75, [_ZZ27test_dive... |
| 81 | L110: ld.shared.u32 %r85, [_ZZ27test_dive... | L101: add.s32 %r76, %r75, %r74; |
| 82 | L111: add.s32 %r86, %r85, %r84; | L102: ld.shared.u32 %r77, [_ZZ27test_dive... |
| 83 | L112: st.global.u32 [%rd1], %r86; | L103: add.s32 %r78, %r77, %r76; |
| 84 | L113: bra.uni $L__BB0_6; | L104: ld.shared.u32 %r79, [_ZZ27test_dive... |
| 85 | L122: ret; | L105: add.s32 %r80, %r79, %r78; |
| 86 | - | L106: ld.shared.u32 %r81, [_ZZ27test_dive... |
| 87 | - | L107: add.s32 %r82, %r81, %r80; |
| 88 | - | L108: ld.shared.u32 %r83, [_ZZ27test_dive... |
| 89 | - | L109: add.s32 %r84, %r83, %r82; |
| 90 | - | L110: ld.shared.u32 %r85, [_ZZ27test_dive... |
| 91 | - | L111: add.s32 %r86, %r85, %r84; |
| 92 | - | L112: st.global.u32 [%rd1], %r86; |
| 93 | - | L113: bra.uni $L__BB0_6; |
| 94 | - | L122: ret; |

## Path 1 Detail

**Lanes**: tid.x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

**Total Instructions**: 85

| PC | Line | Instruction |
|----|------|-------------|
| 1 | 25 | `ld.param.u64 %rd2, [_Z27test_divergence_sync_kernelIiEvPT__param_0];` |
| 2 | 26 | `cvta.to.global.u64 %rd1, %rd2;` |
| 3 | 27 | `mov.u32 %r1, %tid.x;` |
| 4 | 28 | `and.b32 %r2, %r1, 31;` |
| 5 | 29 | `setp.lt.u32 %p1, %r2, 16;` |
| 6 | 30 | `@%p1 bra $L__BB0_7;` |
| 7 | 124 | `add.s32 %r15, %r2, -1;` |
| 8 | 125 | `mul.wide.u32 %rd3, %r15, %r2;` |
| 9 | 126 | `shr.u64 %rd4, %rd3, 1;` |
| 10 | 127 | `cvt.u32.u64 %r16, %rd4;` |
| 11 | 128 | `add.s32 %r90, %r2, %r16;` |
| 12 | 129 | `bra.uni $L__BB0_3;` |
| 13 | 42 | `shl.b32 %r17, %r2, 2;` |
| 14 | 43 | `mov.u32 %r18, _ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data;` |
| 15 | 44 | `add.s32 %r19, %r18, %r17;` |
| 16 | 45 | `st.shared.u32 [%r19], %r90;` |
| 17 | 46 | `bar.sync 0;` |
| 18 | 47 | `setp.ne.s32 %p3, %r1, 0;` |
| 19 | 48 | `@%p3 bra $L__BB0_5;` |
| 20 | 49 | `ld.shared.u32 %r24, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data];` |
| 21 | 50 | `ld.shared.u32 %r25, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+4];` |
| 22 | 51 | `add.s32 %r26, %r25, %r24;` |
| 23 | 52 | `ld.shared.u32 %r27, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+8];` |
| 24 | 53 | `add.s32 %r28, %r27, %r26;` |
| 25 | 54 | `ld.shared.u32 %r29, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+12];` |
| 26 | 55 | `add.s32 %r30, %r29, %r28;` |
| 27 | 56 | `ld.shared.u32 %r31, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+16];` |
| 28 | 57 | `add.s32 %r32, %r31, %r30;` |
| 29 | 58 | `ld.shared.u32 %r33, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+20];` |
| 30 | 59 | `add.s32 %r34, %r33, %r32;` |
| 31 | 60 | `ld.shared.u32 %r35, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+24];` |
| 32 | 61 | `add.s32 %r36, %r35, %r34;` |
| 33 | 62 | `ld.shared.u32 %r37, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+28];` |
| 34 | 63 | `add.s32 %r38, %r37, %r36;` |
| 35 | 64 | `ld.shared.u32 %r39, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+32];` |
| 36 | 65 | `add.s32 %r40, %r39, %r38;` |
| 37 | 66 | `ld.shared.u32 %r41, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+36];` |
| 38 | 67 | `add.s32 %r42, %r41, %r40;` |
| 39 | 68 | `ld.shared.u32 %r43, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+40];` |
| 40 | 69 | `add.s32 %r44, %r43, %r42;` |
| 41 | 70 | `ld.shared.u32 %r45, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+44];` |
| 42 | 71 | `add.s32 %r46, %r45, %r44;` |
| 43 | 72 | `ld.shared.u32 %r47, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+48];` |
| 44 | 73 | `add.s32 %r48, %r47, %r46;` |
| 45 | 74 | `ld.shared.u32 %r49, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+52];` |
| 46 | 75 | `add.s32 %r50, %r49, %r48;` |
| 47 | 76 | `ld.shared.u32 %r51, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+56];` |
| 48 | 77 | `add.s32 %r52, %r51, %r50;` |
| 49 | 78 | `ld.shared.u32 %r53, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+60];` |
| 50 | 79 | `add.s32 %r54, %r53, %r52;` |
| 51 | 80 | `ld.shared.u32 %r55, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+64];` |
| 52 | 81 | `add.s32 %r56, %r55, %r54;` |
| 53 | 82 | `ld.shared.u32 %r57, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+68];` |
| 54 | 83 | `add.s32 %r58, %r57, %r56;` |
| 55 | 84 | `ld.shared.u32 %r59, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+72];` |
| 56 | 85 | `add.s32 %r60, %r59, %r58;` |
| 57 | 86 | `ld.shared.u32 %r61, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+76];` |
| 58 | 87 | `add.s32 %r62, %r61, %r60;` |
| 59 | 88 | `ld.shared.u32 %r63, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+80];` |
| 60 | 89 | `add.s32 %r64, %r63, %r62;` |
| 61 | 90 | `ld.shared.u32 %r65, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+84];` |
| 62 | 91 | `add.s32 %r66, %r65, %r64;` |
| 63 | 92 | `ld.shared.u32 %r67, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+88];` |
| 64 | 93 | `add.s32 %r68, %r67, %r66;` |
| 65 | 94 | `ld.shared.u32 %r69, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+92];` |
| 66 | 95 | `add.s32 %r70, %r69, %r68;` |
| 67 | 96 | `ld.shared.u32 %r71, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+96];` |
| 68 | 97 | `add.s32 %r72, %r71, %r70;` |
| 69 | 98 | `ld.shared.u32 %r73, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+100];` |
| 70 | 99 | `add.s32 %r74, %r73, %r72;` |
| 71 | 100 | `ld.shared.u32 %r75, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+104];` |
| 72 | 101 | `add.s32 %r76, %r75, %r74;` |
| 73 | 102 | `ld.shared.u32 %r77, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+108];` |
| 74 | 103 | `add.s32 %r78, %r77, %r76;` |
| 75 | 104 | `ld.shared.u32 %r79, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+112];` |
| 76 | 105 | `add.s32 %r80, %r79, %r78;` |
| 77 | 106 | `ld.shared.u32 %r81, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+116];` |
| 78 | 107 | `add.s32 %r82, %r81, %r80;` |
| 79 | 108 | `ld.shared.u32 %r83, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+120];` |
| 80 | 109 | `add.s32 %r84, %r83, %r82;` |
| 81 | 110 | `ld.shared.u32 %r85, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+124];` |
| 82 | 111 | `add.s32 %r86, %r85, %r84;` |
| 83 | 112 | `st.global.u32 [%rd1], %r86;` |
| 84 | 113 | `bra.uni $L__BB0_6;` |
| 85 | 122 | `ret;` |

## Path 2 Detail

**Lanes**: tid.x = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

**Total Instructions**: 94

| PC | Line | Instruction |
|----|------|-------------|
| 1 | 25 | `ld.param.u64 %rd2, [_Z27test_divergence_sync_kernelIiEvPT__param_0];` |
| 2 | 26 | `cvta.to.global.u64 %rd1, %rd2;` |
| 3 | 27 | `mov.u32 %r1, %tid.x;` |
| 4 | 28 | `and.b32 %r2, %r1, 31;` |
| 5 | 29 | `setp.lt.u32 %p1, %r2, 16;` |
| 6 | 30 | `@%p1 bra $L__BB0_7;` |
| 7 | 31 | `neg.s32 %r88, %r2;` |
| 8 | 32 | `mov.b32 %r90, 1;` |
| 9 | 33 | `mov.b32 %r87, 15;` |
| 10 | 35 | `add.s32 %r14, %r87, -14;` |
| 11 | 36 | `mul.lo.s32 %r90, %r14, %r90;` |
| 12 | 37 | `add.s32 %r88, %r88, 1;` |
| 13 | 38 | `add.s32 %r87, %r87, 1;` |
| 14 | 39 | `setp.ne.s32 %p2, %r88, -15;` |
| 15 | 40 | `@%p2 bra $L__BB0_2;` |
| 16 | 35 | `add.s32 %r14, %r87, -14;` |
| 17 | 36 | `mul.lo.s32 %r90, %r14, %r90;` |
| 18 | 37 | `add.s32 %r88, %r88, 1;` |
| 19 | 38 | `add.s32 %r87, %r87, 1;` |
| 20 | 39 | `setp.ne.s32 %p2, %r88, -15;` |
| 21 | 40 | `@%p2 bra $L__BB0_2;` |
| 22 | 42 | `shl.b32 %r17, %r2, 2;` |
| 23 | 43 | `mov.u32 %r18, _ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data;` |
| 24 | 44 | `add.s32 %r19, %r18, %r17;` |
| 25 | 45 | `st.shared.u32 [%r19], %r90;` |
| 26 | 46 | `bar.sync 0;` |
| 27 | 47 | `setp.ne.s32 %p3, %r1, 0;` |
| 28 | 48 | `@%p3 bra $L__BB0_5;` |
| 29 | 49 | `ld.shared.u32 %r24, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data];` |
| 30 | 50 | `ld.shared.u32 %r25, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+4];` |
| 31 | 51 | `add.s32 %r26, %r25, %r24;` |
| 32 | 52 | `ld.shared.u32 %r27, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+8];` |
| 33 | 53 | `add.s32 %r28, %r27, %r26;` |
| 34 | 54 | `ld.shared.u32 %r29, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+12];` |
| 35 | 55 | `add.s32 %r30, %r29, %r28;` |
| 36 | 56 | `ld.shared.u32 %r31, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+16];` |
| 37 | 57 | `add.s32 %r32, %r31, %r30;` |
| 38 | 58 | `ld.shared.u32 %r33, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+20];` |
| 39 | 59 | `add.s32 %r34, %r33, %r32;` |
| 40 | 60 | `ld.shared.u32 %r35, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+24];` |
| 41 | 61 | `add.s32 %r36, %r35, %r34;` |
| 42 | 62 | `ld.shared.u32 %r37, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+28];` |
| 43 | 63 | `add.s32 %r38, %r37, %r36;` |
| 44 | 64 | `ld.shared.u32 %r39, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+32];` |
| 45 | 65 | `add.s32 %r40, %r39, %r38;` |
| 46 | 66 | `ld.shared.u32 %r41, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+36];` |
| 47 | 67 | `add.s32 %r42, %r41, %r40;` |
| 48 | 68 | `ld.shared.u32 %r43, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+40];` |
| 49 | 69 | `add.s32 %r44, %r43, %r42;` |
| 50 | 70 | `ld.shared.u32 %r45, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+44];` |
| 51 | 71 | `add.s32 %r46, %r45, %r44;` |
| 52 | 72 | `ld.shared.u32 %r47, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+48];` |
| 53 | 73 | `add.s32 %r48, %r47, %r46;` |
| 54 | 74 | `ld.shared.u32 %r49, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+52];` |
| 55 | 75 | `add.s32 %r50, %r49, %r48;` |
| 56 | 76 | `ld.shared.u32 %r51, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+56];` |
| 57 | 77 | `add.s32 %r52, %r51, %r50;` |
| 58 | 78 | `ld.shared.u32 %r53, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+60];` |
| 59 | 79 | `add.s32 %r54, %r53, %r52;` |
| 60 | 80 | `ld.shared.u32 %r55, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+64];` |
| 61 | 81 | `add.s32 %r56, %r55, %r54;` |
| 62 | 82 | `ld.shared.u32 %r57, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+68];` |
| 63 | 83 | `add.s32 %r58, %r57, %r56;` |
| 64 | 84 | `ld.shared.u32 %r59, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+72];` |
| 65 | 85 | `add.s32 %r60, %r59, %r58;` |
| 66 | 86 | `ld.shared.u32 %r61, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+76];` |
| 67 | 87 | `add.s32 %r62, %r61, %r60;` |
| 68 | 88 | `ld.shared.u32 %r63, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+80];` |
| 69 | 89 | `add.s32 %r64, %r63, %r62;` |
| 70 | 90 | `ld.shared.u32 %r65, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+84];` |
| 71 | 91 | `add.s32 %r66, %r65, %r64;` |
| 72 | 92 | `ld.shared.u32 %r67, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+88];` |
| 73 | 93 | `add.s32 %r68, %r67, %r66;` |
| 74 | 94 | `ld.shared.u32 %r69, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+92];` |
| 75 | 95 | `add.s32 %r70, %r69, %r68;` |
| 76 | 96 | `ld.shared.u32 %r71, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+96];` |
| 77 | 97 | `add.s32 %r72, %r71, %r70;` |
| 78 | 98 | `ld.shared.u32 %r73, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+100];` |
| 79 | 99 | `add.s32 %r74, %r73, %r72;` |
| 80 | 100 | `ld.shared.u32 %r75, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+104];` |
| 81 | 101 | `add.s32 %r76, %r75, %r74;` |
| 82 | 102 | `ld.shared.u32 %r77, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+108];` |
| 83 | 103 | `add.s32 %r78, %r77, %r76;` |
| 84 | 104 | `ld.shared.u32 %r79, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+112];` |
| 85 | 105 | `add.s32 %r80, %r79, %r78;` |
| 86 | 106 | `ld.shared.u32 %r81, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+116];` |
| 87 | 107 | `add.s32 %r82, %r81, %r80;` |
| 88 | 108 | `ld.shared.u32 %r83, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+120];` |
| 89 | 109 | `add.s32 %r84, %r83, %r82;` |
| 90 | 110 | `ld.shared.u32 %r85, [_ZZ27test_divergence_sync_kernelIiEvPT_E11shared_data+124];` |
| 91 | 111 | `add.s32 %r86, %r85, %r84;` |
| 92 | 112 | `st.global.u32 [%rd1], %r86;` |
| 93 | 113 | `bra.uni $L__BB0_6;` |
| 94 | 122 | `ret;` |

## Divergence Analysis

### Path 1 Branches

| PC | Line | Branch Instruction | Condition |
|----|------|--------------------|-----------|
| 6 | 30 | `@%p1 bra $L__BB0_7;` | lane_id < 16 |
| 19 | 48 | `@%p3 bra $L__BB0_5;` | tid.x != 0 |

### Path 2 Branches

| PC | Line | Branch Instruction | Condition |
|----|------|--------------------|-----------|
| 6 | 30 | `@%p1 bra $L__BB0_7;` | lane_id < 16 |
| 15 | 40 | `@%p2 bra $L__BB0_2;` | loop_iteration < loop_iterations |
| 21 | 40 | `@%p2 bra $L__BB0_2;` | loop_iteration < loop_iterations |
| 28 | 48 | `@%p3 bra $L__BB0_5;` | tid.x != 0 |
