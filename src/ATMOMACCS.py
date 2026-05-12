# $Id$
#
# Copyright (C) 2001-2011 greg Landrum and Rational Discovery LLC
# Modified by Linus Lind (2025). changes are subject to the BSD license of RDKit:
#
#   @@ All Rights Reserved @@
#  This file is part of the RDKit.
#  The contents are covered by the terms of the BSD license
#  which is included in the file license.txt, found at the root
#  of the RDKit source tree.
#  https://github.com/rdkit/rdkit/blob/master/license.txt
'''
BSD 3-Clause License

Copyright (c) 2006-2015, Rational Discovery LLC, Greg Landrum, and Julie Penzotti and others
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
#############################################################################
""" SMARTS definitions for the publicly available MACCS keys
and a MACCS fingerprinter

I compared the MACCS fingerprints generated here with those from two
other packages (not MDL, unfortunately). Of course there are
disagreements between the various fingerprints still, but I think
these definitions work pretty well. Some notes:

1) most of the differences have to do with aromaticity
2) there's a discrepancy sometimes because the current RDKit
definitions do not require multiple matches to be distinct. e.g. the
SMILES C(=O)CC(=O) can match the (hypothetical) key O=CC twice in my
definition. It's not clear to me what the correct behavior is.
3) Some keys are not fully defined in the MDL documentation
4) Two keys, 125 and 166, have to be done outside of SMARTS.
5) Key 1 (ISOTOPE) isn't defined

Rev history:
2006 (gl): Original open-source release
May 2011 (gl): Update some definitions based on feedback from Andrew Dalke

"""


"""
CHANGES 28. June 2024 by Linus Lind:
- Add ATMO keys, which contain relevant fragments for atmospheric compounds
- Add MACCS + ATMO concatenation
- Small change in _InitKeys method to fit ATMO implementation
- Implement nitrophenol counting.
"""


from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from numpy import binary_repr

# these are SMARTS patterns corresponding to the MDL MACCS keys
smartsPatts = {
  1: ('?', 0),  # ISOTOPE
  #2:('[#104,#105,#106,#107,#106,#109,#110,#111,#112]',0),  # atomic num >103 Not complete
  2: ('[#104]', 0),  # limit the above def'n since the RDKit only accepts up to #104
  3: ('[#32,#33,#34,#50,#51,#52,#82,#83,#84]', 0),  # Group IVa,Va,VIa Rows 4-6 
  4: ('[Ac,Th,Pa,U,Np,Pu,Am,Cm,Bk,Cf,Es,Fm,Md,No,Lr]', 0),  # actinide
  5: ('[Sc,Ti,Y,Zr,Hf]', 0),  # Group IIIB,IVB (Sc...)  
  6: ('[La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu]', 0),  # Lanthanide
  7: ('[V,Cr,Mn,Nb,Mo,Tc,Ta,W,Re]', 0),  # Group VB,VIB,VIIB
  8: ('[!#6;!#1]1~*~*~*~1', 0),  # QAAA@1
  9: ('[Fe,Co,Ni,Ru,Rh,Pd,Os,Ir,Pt]', 0),  # Group VIII (Fe...)
  10: ('[Be,Mg,Ca,Sr,Ba,Ra]', 0),  # Group IIa (Alkaline earth)
  11: ('*1~*~*~*~1', 0),  # 4M Ring
  12: ('[Cu,Zn,Ag,Cd,Au,Hg]', 0),  # Group IB,IIB (Cu..)
  13: ('[#8]~[#7](~[#6])~[#6]', 0),  # ON(C)C
  14: ('[#16]-[#16]', 0),  # S-S
  15: ('[#8]~[#6](~[#8])~[#8]', 0),  # OC(O)O
  16: ('[!#6;!#1]1~*~*~1', 0),  # QAA@1
  17: ('[#6]#[#6]', 0),  #CTC
  18: ('[#5,#13,#31,#49,#81]', 0),  # Group IIIA (B...) 
  19: ('*1~*~*~*~*~*~*~1', 0),  # 7M Ring
  20: ('[#14]', 0),  #Si
  21: ('[#6]=[#6](~[!#6;!#1])~[!#6;!#1]', 0),  # C=C(Q)Q
  22: ('*1~*~*~1', 0),  # 3M Ring
  23: ('[#7]~[#6](~[#8])~[#8]', 0),  # NC(O)O
  24: ('[#7]-[#8]', 0),  # N-O
  25: ('[#7]~[#6](~[#7])~[#7]', 0),  # NC(N)N
  26: ('[#6]=;@[#6](@*)@*', 0),  # C$=C($A)$A
  27: ('[I]', 0),  # I
  28: ('[!#6;!#1]~[CH2]~[!#6;!#1]', 0),  # QCH2Q
  29: ('[#15]', 0),  # P
  30: ('[#6]~[!#6;!#1](~[#6])(~[#6])~*', 0),  # CQ(C)(C)A
  31: ('[!#6;!#1]~[F,Cl,Br,I]', 0),  # QX
  32: ('[#6]~[#16]~[#7]', 0),  # CSN
  33: ('[#7]~[#16]', 0),  # NS
  34: ('[CH2]=*', 0),  # CH2=A
  35: ('[Li,Na,K,Rb,Cs,Fr]', 0),  # Group IA (Alkali Metal)
  36: ('[#16R]', 0),  # S Heterocycle
  37: ('[#7]~[#6](~[#8])~[#7]', 0),  # NC(O)N
  38: ('[#7]~[#6](~[#6])~[#7]', 0),  # NC(C)N
  39: ('[#8]~[#16](~[#8])~[#8]', 0),  # OS(O)O
  40: ('[#16]-[#8]', 0),  # S-O
  41: ('[#6]#[#7]', 0),  # CTN
  42: ('F', 0),  # F
  43: ('[!#6;!#1;!H0]~*~[!#6;!#1;!H0]', 0),  # QHAQH
  44: ('[!#1;!#6;!#7;!#8;!#9;!#14;!#15;!#16;!#17;!#35;!#53]', 0),  # OTHER
  45: ('[#6]=[#6]~[#7]', 0),  # C=CN
  46: ('Br', 0),  # BR
  47: ('[#16]~*~[#7]', 0),  # SAN
  48: ('[#8]~[!#6;!#1](~[#8])(~[#8])', 0),  # OQ(O)O
  49: ('[!+0]', 0),  # CHARGE  
  50: ('[#6]=[#6](~[#6])~[#6]', 0),  # C=C(C)C
  51: ('[#6]~[#16]~[#8]', 0),  # CSO
  52: ('[#7]~[#7]', 0),  # NN
  53: ('[!#6;!#1;!H0]~*~*~*~[!#6;!#1;!H0]', 0),  # QHAAAQH
  54: ('[!#6;!#1;!H0]~*~*~[!#6;!#1;!H0]', 0),  # QHAAQH
  55: ('[#8]~[#16]~[#8]', 0),  #OSO
  56: ('[#8]~[#7](~[#8])~[#6]', 0),  # ON(O)C
  57: ('[#8R]', 0),  # O Heterocycle
  58: ('[!#6;!#1]~[#16]~[!#6;!#1]', 0),  # QSQ
  59: ('[#16]!:*:*', 0),  # Snot%A%A
  60: ('[#16]=[#8]', 0),  # S=O
  61: ('*~[#16](~*)~*', 0),  # AS(A)A
  62: ('*@*!@*@*', 0),  # A$!A$A
  63: ('[#7]=[#8]', 0),  # N=O
  64: ('*@*!@[#16]', 0),  # A$A!S
  65: ('c:n', 0),  # C%N
  66: ('[#6]~[#6](~[#6])(~[#6])~*', 0),  # CC(C)(C)A
  67: ('[!#6;!#1]~[#16]', 0),  # QS
  68: ('[!#6;!#1;!H0]~[!#6;!#1;!H0]', 0),  # QHQH (&...) SPEC Incomplete
  69: ('[!#6;!#1]~[!#6;!#1;!H0]', 0),  # QQH
  70: ('[!#6;!#1]~[#7]~[!#6;!#1]', 0),  # QNQ
  71: ('[#7]~[#8]', 0),  # NO
  72: ('[#8]~*~*~[#8]', 0),  # OAAO
  73: ('[#16]=*', 0),  # S=A
  74: ('[CH3]~*~[CH3]', 0),  # CH3ACH3
  75: ('*!@[#7]@*', 0),  # A!N$A
  76: ('[#6]=[#6](~*)~*', 0),  # C=C(A)A
  77: ('[#7]~*~[#7]', 0),  # NAN
  78: ('[#6]=[#7]', 0),  # C=N
  79: ('[#7]~*~*~[#7]', 0),  # NAAN
  80: ('[#7]~*~*~*~[#7]', 0),  # NAAAN
  81: ('[#16]~*(~*)~*', 0),  # SA(A)A
  82: ('*~[CH2]~[!#6;!#1;!H0]', 0),  # ACH2QH
  83: ('[!#6;!#1]1~*~*~*~*~1', 0),  # QAAAA@1
  84: ('[NH2]', 0),  #NH2
  85: ('[#6]~[#7](~[#6])~[#6]', 0),  # CN(C)C
  86: ('[C;H2,H3][!#6;!#1][C;H2,H3]', 0),  # CH2QCH2
  87: ('[F,Cl,Br,I]!@*@*', 0),  # X!A$A
  88: ('[#16]', 0),  # S
  89: ('[#8]~*~*~*~[#8]', 0),  # OAAAO
  90:
  ('[$([!#6;!#1;!H0]~*~*~[CH2]~*),$([!#6;!#1;!H0;R]1@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~[R]1@[R]@[CH2;R]1)]',
   0),  # QHAACH2A
  91:
  ('[$([!#6;!#1;!H0]~*~*~*~[CH2]~*),$([!#6;!#1;!H0;R]1@[R]@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~[R]1@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~*~[R]1@[R]@[CH2;R]1)]',
   0),  # QHAAACH2A
  92: ('[#8]~[#6](~[#7])~[#6]', 0),  # OC(N)C
  93: ('[!#6;!#1]~[CH3]', 0),  # QCH3
  94: ('[!#6;!#1]~[#7]', 0),  # QN
  95: ('[#7]~*~*~[#8]', 0),  # NAAO
  96: ('*1~*~*~*~*~1', 0),  # 5 M ring
  97: ('[#7]~*~*~*~[#8]', 0),  # NAAAO
  98: ('[!#6;!#1]1~*~*~*~*~*~1', 0),  # QAAAAA@1
  99: ('[#6]=[#6]', 0),  # C=C
  100: ('*~[CH2]~[#7]', 0),  # ACH2N
  101:
  ('[$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1)]',
   0),  # 8M Ring or larger. This only handles up to ring sizes of 14
  102: ('[!#6;!#1]~[#8]', 0),  # QO
  103: ('Cl', 0),  # CL
  104: ('[!#6;!#1;!H0]~*~[CH2]~*', 0),  # QHACH2A
  105: ('*@*(@*)@*', 0),  # A$A($A)$A
  106: ('[!#6;!#1]~*(~[!#6;!#1])~[!#6;!#1]', 0),  # QA(Q)Q
  107: ('[F,Cl,Br,I]~*(~*)~*', 0),  # XA(A)A
  108: ('[CH3]~*~*~*~[CH2]~*', 0),  # CH3AAACH2A
  109: ('*~[CH2]~[#8]', 0),  # ACH2O
  110: ('[#7]~[#6]~[#8]', 0),  # NCO
  111: ('[#7]~*~[CH2]~*', 0),  # NACH2A
  112: ('*~*(~*)(~*)~*', 0),  # AA(A)(A)A
  113: ('[#8]!:*:*', 0),  # Onot%A%A
  114: ('[CH3]~[CH2]~*', 0),  # CH3CH2A
  115: ('[CH3]~*~[CH2]~*', 0),  # CH3ACH2A
  116: ('[$([CH3]~*~*~[CH2]~*),$([CH3]~*1~*~[CH2]1)]', 0),  # CH3AACH2A
  117: ('[#7]~*~[#8]', 0),  # NAO
  118: ('[$(*~[CH2]~[CH2]~*),$(*1~[CH2]~[CH2]1)]', 1),  # ACH2CH2A > 1
  119: ('[#7]=*', 0),  # N=A
  120: ('[!#6;R]', 1),  # Heterocyclic atom > 1 (&...) Spec Incomplete
  121: ('[#7;R]', 0),  # N Heterocycle
  122: ('*~[#7](~*)~*', 0),  # AN(A)A
  123: ('[#8]~[#6]~[#8]', 0),  # OCO
  124: ('[!#6;!#1]~[!#6;!#1]', 0),  # QQ
  125: ('?', 0),  # Aromatic Ring > 1
  126: ('*!@[#8]!@*', 0),  # A!O!A
  127: ('*@*!@[#8]', 1),  # A$A!O > 1 (&...) Spec Incomplete
  128:
  ('[$(*~[CH2]~*~*~*~[CH2]~*),$([R]1@[CH2;R]@[R]@[R]@[R]@[CH2;R]1),$(*~[CH2]~[R]1@[R]@[R]@[CH2;R]1),$(*~[CH2]~*~[R]1@[R]@[CH2;R]1)]',
   0),  # ACH2AAACH2A
  129: ('[$(*~[CH2]~*~*~[CH2]~*),$([R]1@[CH2]@[R]@[R]@[CH2;R]1),$(*~[CH2]~[R]1@[R]@[CH2;R]1)]',
        0),  # ACH2AACH2A
  130: ('[!#6;!#1]~[!#6;!#1]', 1),  # QQ > 1 (&...)  Spec Incomplete
  131: ('[!#6;!#1;!H0]', 1),  # QH > 1
  132: ('[#8]~*~[CH2]~*', 0),  # OACH2A
  133: ('*@*!@[#7]', 0),  # A$A!N
  134: ('[F,Cl,Br,I]', 0),  # X (HALOGEN)
  135: ('[#7]!:*:*', 0),  # Nnot%A%A
  136: ('[#8]=*', 1),  # O=A>1 
  137: ('[!C;!c;R]', 0),  # Heterocycle
  138: ('[!#6;!#1]~[CH2]~*', 1),  # QCH2A>1 (&...) Spec Incomplete
  139: ('[O;!H0]', 0),  # OH
  140: ('[#8]', 3),  # O > 3 (&...) Spec Incomplete
  141: ('[CH3]', 2),  # CH3 > 2  (&...) Spec Incomplete
  142: ('[#7]', 1),  # N > 1
  143: ('*@*!@[#8]', 0),  # A$A!O
  144: ('*!:*:*!:*', 0),  # Anot%A%Anot%A
  145: ('*1~*~*~*~*~*~1', 1),  # 6M ring > 1
  146: ('[#8]', 2),  # O > 2
  147: ('[$(*~[CH2]~[CH2]~*),$([R]1@[CH2;R]@[CH2;R]1)]', 0),  # ACH2CH2A
  148: ('*~[!#6;!#1](~*)~*', 0),  # AQ(A)A
  149: ('[C;H3,H4]', 1),  # CH3 > 1
  150: ('*!@*@*!@*', 0),  # A!A$A!A
  151: ('[#7;!H0]', 0),  # NH
  152: ('[#8]~[#6](~[#6])~[#6]', 0),  # OC(C)C
  153: ('[!#6;!#1]~[CH2]~*', 0),  # QCH2A
  154: ('[#6]=[#8]', 0),  # C=O
  155: ('*!@[CH2]!@*', 0),  # A!CH2!A
  156: ('[#7]~*(~*)~*', 0),  # NA(A)A
  157: ('[#6]-[#8]', 0),  # C-O
  158: ('[#6]-[#7]', 0),  # C-N
  159: ('[#8]', 1),  # O>1
  160: ('[C;H3,H4]', 0),  #CH3
  161: ('[#7]', 0),  # N
  162: ('a', 0),  # Aromatic
  163: ('*1~*~*~*~*~*~1', 0),  # 6M Ring
  164: ('[#8]', 0),  # O
  165: ('[R]', 0),  # Ring
  166: ('?', 0),  # Fragments  FIX: this can't be done in SMARTS
}

# The ATMO keys that are matched with SMARTS
atmoFragments = {
  1: '[C][NX3;H2;!$(NC=O)]([H])[H]', # amine, primary
  2: '[C][NX3;H;!$(NC=O)]([C])[H]', # amine, secondary
  3: '[C][NX3;H0;!$(NC=O);!$(N=O)]([C])[C]', # amine tertiary
  4: '[CX4][H]', # alkane CH
  5: '[CX3;$(C=C)][H]', # alkene CH
  6: '[c][H]', # aromatic CH
  7: '[#6,#1][CX3](=O)[#6,#1]', # carbonyl
  8: '[C;!$(C=O)][OX2H][H]', # hydroxyl (alkyl) 
  9: '[CX3](=O)[OX2H][H]', # carboxylic acid
  10: '[CX3H1,CX3](=O)[OX2H0][#6]', # ester, all
  11: '[OD2]([C;!R;!$(C=O)])[C;!R;!$(C=O)]', # ether
  12: '[#6][OD2][OD2,OD1][#6]', # peroxide
  13: '[#6][$([NX3](=O)=O),$([NX3+](=O)[O-])](~[O])(~[O])', # nitro
  14: '[c;!$(C=O)][OX2H][H]', # aromatic hydroxyl
  15: '[#6;!$(C=O)][OD2][OX2H,OD1][#1]', # hydroperoxide
  16: '[CX3](=O)[NX3;!$(N=O)]([#6,#1])[#6,#1]', # amide
  17: '[#6][O][NX3;$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)](~[O])(~[O])', # nitrate
  18: '[#6][O][SX4;$([SX4](=O)(=O)(O)O),$([SX4+2]([O-])([O-])(O)O)](~[O])(~[O])(~[O])', # organosulfate
  19: '[C](=O)OO[N](~O)~[O]', # carbonylperoxynitrate
  20: '[CX3;$(C([#6])(=[O])[#6])](=[O;!$([O][O])])', # ketone
  21: '[CX3;$(C([#6,#1])(=[O])[#6,#1])]([#6,#1])(=[O;!$([O][O])])[H]', # aldehyde
  22: '[CX3;$(C(=[O])[NX3;!$(N=O)])](=[O])[N]([#1])[#1]', # amide, primary
  23: '[CX3;$(C(=[O])[NX3;!$(N=O)]([#6])[#1])](=[O])[N][#1]', # amide, secondary
  24: '[CX3;$(C(=[O])[NX3;!$(N=O)]([#6])[#6])](=[O])[N]', # amide, tertiary
  25: '[C](=O)O[O][H]', # carbonylperoxyacid
  26: '[#6][O;!$(OOC(=O))][O;!$(OOC(=O))][N](~O)~[O]', # peroxy nitrate
  27: 'c~[O,o]~[c,C&!$(C=O)]', # ether, aromatic
  28: '[OD2;R]([C;!$(C=O);R])[C;!$(C=O);R]', # ether (alicyclic)
  29: '[N;!$(NC=O);!$(N=O);$(Na)]', # amine, aromatic
  30: '[#6][OX2H0][CX3,CX3H1](=O)[C;$(C[N](~[O])~[O]),$(CC[N](~[O])~[O]),$(CCC[N](~[O])~[O]),$(CCCC[N](~[O])~[O]),$(CCCCC[N](~[O])~[O])]', # nitroester
  31: '[$(C=CC=O);A;R]', # C=C-C=O in non-aromatic ring
  32: 'C=C', # C=C (non-aromatic)
  33: '[C;$(C[NX3][CH,CC](=O)),$(CC[NX3][CH,CC](=O)),$(CCC[NX3][CH,CC](=O)),$(CCCC[NX3][CH,CC](=O)),$(CCCCC[NX3][CH,CC](=O))]', # nC-OHside-a
  34: 'Aromatic ring', # Aromatic Ring (NO SMARTS)
  35: 'Non-aromatic ring', # Non-aromatic Ring (NO SMARTS)
  36: 'Nitrophenol' # Nitrophenol (NO SMARTS)
}

# Unused variable but left in for clarity
atmoFragmentsBinEnum = {
  666: '[#6]', # Carbon number
  888: '[#8]'  # Oxygen number
} 

no_smarts = ['?', 'Aromatic ring', 'Non-aromatic ring', 'Nitrophenol']
'''
These patterns may not be representable by SMARTS. '?' is for maccs key logic
'''

maccsKeys = None
atmoKeys = None
atmoSmartsPatts = None
atmoVersion = 4

def _initAtmoDict(version: int) -> dict:
  """
  Initializes the atmo keys dictionary, according to ATMOMACCS version
  """
  atmoDict = dict()
  if version >= 1: # single enum keys
    for key, value in atmoFragments.items():
      atmoDict[key] = (value, 0)
  if version >= 2: # single enum keys
    for key, value in atmoFragments.items():
      atmoDict[key + len(atmoFragments.items())] = (value, 1)
      atmoDict[key + 2 * len(atmoFragments.items())] = (value, 2)
  if version >= 3: # binary enum carbon key
    atmoDict[len(atmoDict) + 1] = ('[#6]', 'binary') # "append" dict
  if version >= 4: # binary enum oxygen key
    atmoDict[len(atmoDict) + 1] = ('[#8]', 'binary') # "append" dict
  return atmoDict

def _InitKeys(keyList: list, keyDict: dict) -> None:
  """ *Internal Use Only*

   generates SMARTS patterns for the keys, run once

   Mutates the given keyList and fills it with Chem.rdchem.Mol objects

  """
  assert len(keyList) == len(keyDict.keys()), 'length mismatch! keyList: ' +\
      f'{len(keyList)}, keyDict.keys(): {len(keyDict.keys())}'
  for key in keyDict.keys():
    patt, count = keyDict[key]
    if patt not in no_smarts:
      sma = Chem.MolFromSmarts(patt)
      assert sma, 'SMARTS parser error for key #%d: %s' % (key, patt)
      keyList[key - 1] = sma, count
    else:
      keyList[key - 1] = patt, count # patts not represented by SMARTS

def pyGenMACCSKeys(mol, **kwargs):
  """ generates the MACCS fingerprint for a molecules

   **Arguments**

     - mol: the molecule to be fingerprinted

     - any extra keyword arguments are ignored
     
   **Returns**

      a _DataStructs.SparseBitVect_ containing the fingerprint.

  >>> m = Chem.MolFromSmiles('CNO')
  >>> bv = pyGenMACCSKeys(m)
  >>> tuple(bv.GetOnBits())
  (24, 68, 69, 71, 93, 94, 102, 124, 131, 139, 151, 158, 160, 161, 164)
  >>> bv = pyGenMACCSKeys(Chem.MolFromSmiles('CCC'))
  >>> tuple(bv.GetOnBits())
  (74, 114, 149, 155, 160)

  """
  global maccsKeys
  if maccsKeys is None:
    maccsKeys = [(None, 0)] * len(smartsPatts.keys())
    _InitKeys(maccsKeys, smartsPatts)
  ctor = kwargs.get('ctor', DataStructs.SparseBitVect)
  res = ctor(len(maccsKeys) + 1)
  for i, (patt, count) in enumerate(maccsKeys):
    if patt not in no_smarts:
      if count == 0:
        res[i + 1] = mol.HasSubstructMatch(patt)
      else:
        matches = mol.GetSubstructMatches(patt)
        if len(matches) > count:
          res[i + 1] = 1
    elif (i + 1) == 125:
      # special case: num aromatic rings > 1
      ri = mol.GetRingInfo()
      nArom = 0
      res[125] = 0
      for ring in ri.BondRings():
        isArom = True
        for bondIdx in ring:
          if not mol.GetBondWithIdx(bondIdx).GetIsAromatic():
            isArom = False
            break
        if isArom:
          nArom += 1
          if nArom > 1:
            res[125] = 1
            break
    elif (i + 1) == 166:
      res[166] = 0
      # special case: num frags > 1
      if len(Chem.GetMolFrags(mol)) > 1:
        res[166] = 1

  return res

def _initNoneKeys(keyDict: dict) -> list[tuple[None, int]]:
  '''
  Initializes list of (None, 0), which is mutated by _InitKeys
  '''
  numPatts: int = len(keyDict)
  nones: list[tuple[None, int]] = [(None, 0)]
  return nones * numPatts

def _encodeToBinary(num: int, bits: int) -> list[int]:
  """ *Interal Use Only*

    arguments
    num: The number (in decimal system) to be converted to binary 
    bits: The bit width to allocate. If num exceeds the bit allocation, an 
              assertion exception is thrown.
  
    Returns: Binary representation (list[int]) with len(bits). Significance in
              descending order (Most significant elements first).

    Calculates the binary representation of a number

    Example:

    >>> _encodeToBinary(5, 6)
    [0, 0, 0, 1, 0, 1]
  """
  assert bits > 0, f'Bit width must be greater than zero'
  assert num < 2**bits, f'{bits}-bit can only represent numbers up to ' + \
    f'{2**bits - 1}, but received {num}. Increase the bit allocation.'
  return [int(bit) for bit in binary_repr(num, width=bits)]

def _isVersionValid(version: int) -> bool:
  '''
  Is the user provided version number for ATMO valid?

  return version in [1, 2, 3, 4]
  '''
  return version in [1, 2, 3, 4]

def _matchExists(matches: int) -> bool:
  '''
  Does fragment exist in molecule?

  return matches > 0
  '''
  return matches > 0

def _matchesExactlyOnce(matches: int) -> bool:
  '''
  Does fragment appear exactly once?

  return matches == 1
  '''
  return matches == 1

def _matchesExactlyTwice(matches: int) -> bool:
  '''
  Does fragment appear exactly Twice?

  return matches == 2
  '''
  return matches == 2

def _matchesMoreThanTwice(matches: int) -> bool:
  '''
  Does fragment appear more than twice?

  return matches > 1
  '''
  return matches > 2

def _bespe(matches: int, count: int, version: int) -> int:
  '''
  binary encoded smarts pattern enumeration asks one of the questions depending
  on count and version.
  count = 0 ver1: Does fragment exist in molecule?
  count = 0 ver2-4: Does fragment appear exactly once?
  count = 1 ver2-4: Does fragment appear exactly twice?
  count = 2 ver2-4: Does fragment appear more than twice?

  returns binary integer to binary question
  '''
  if version == 1:
    return int(_matchExists(matches))
  else:
    if count == 0:
      return int(_matchesExactlyOnce(matches))
    if count == 1:
      return int(_matchesExactlyTwice(matches))
    if count == 2:
      return int(_matchesMoreThanTwice(matches))
  raise Exception('Unexpected count value given to _bespe')

def _countNitrophenols(mol) -> int:
  '''
  Counts the number of nitrophenols in a mol object
  '''
  # Match SMARTS
  phenol = Chem.MolFromSmarts('[OX2H][cX3]:[c]')
  nitro = Chem.MolFromSmarts('[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]')
  phenolMatches = mol.GetSubstructMatches(phenol)
  nitroMatches = mol.GetSubstructMatches(nitro)
  if len(phenolMatches) == 0 or len(nitroMatches) == 0:
    return 0
  
  # Get rings
  rings = mol.GetRingInfo().AtomRings()
  aromaticRings = []
  for ring in rings:
    if len(ring) > 2:
      is_aromatic = True
      for atom in ring:
        if not mol.GetAtomWithIdx(atom).GetIsAromatic():
          is_aromatic = False
          break
      if is_aromatic:
        aromaticRings.append(ring)
  
  nitroPhenols={}
  # iterate rings to match phenols and nitros in rings
  for i, ring in enumerate(aromaticRings):
    ringPhenols = [] # store atoms, which match to phenol
    ringNitros = []  # store atoms, which match to nitro
    
    for phenolMatch in phenolMatches:
      if any(atom in ring for atom in phenolMatch):
        ringPhenols.append(phenolMatch)
      
    for nitroMatch in nitroMatches:
      if any(atom in ring for atom in nitroMatch):
        ringNitros.append(nitroMatch)

    # if match found for both nitro and phenol in same ring
    if ringNitros != [] and ringPhenols != []: 
      nitroPhenols[i] = (ringPhenols, ringNitros)
  return len(nitroPhenols.keys())

def pyGenATMOKeys(mol, *, version: int = 4, bit_width: int = 6, **kwargs):
  """ generates the ATMO fingerprint for a molecule

   **Arguments**

     - mol: the molecule to be fingerprinted
     - version: the version of ATMOMACCS to compute

        version 1: Single enumeration of atmoFragments

        version 2: Multi enumeration of atmoFragments (1, 2 and >2) 

        version 3: version 2 + binary encode carbon

        version 4: version 2 + binary encode carbon and oxygen
     - bit_width: For versions 3 and 4, the number of bits to allocate for 
                  binary encoding carbons and oxygens.

     - any extra keyword arguments are ignored
     
   **Returns**

      a _DataStructs.SparseBitVect_ containing the fingerprint.
  >>> from ATMOMACCS import pyGenATMOKeys
  >>> from rdkit import Chem
  >>> mol = Chem.MolFromSmiles('CC1(C)C(CC1C(=O)CC=O)ON(=O)=O')
  >>> fp = pyGenATMOKeys(mol)
  >>> list(fp.GetOnBits())
  [16, 19, 20, 34, 42, 75, 110, 113, 117, 119]
  >>> len(fp)
  120
  >>> fp = pyGenATMOKeys(mol, version = 1)
  >>> list(fp)
  [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

  """
  molh = Chem.AddHs(mol) # Add hydrogens because some patterns need them explicitly
  assert _isVersionValid(version), 'Invalid version input.'\
    +' Expected int in interval [1, 4]'
  global atmoVersion
  global atmoKeys
  global atmoSmartsPatts
  # initialize dict and keys if not initialized already
  if atmoKeys == None or atmoVersion != version:
    atmoSmartsPatts = _initAtmoDict(version)
    atmoVersion = version
    atmoKeys = _initNoneKeys(atmoSmartsPatts.keys()) # init atmoKeys list
    _InitKeys(atmoKeys, atmoSmartsPatts)
  
  ctor = kwargs.get('ctor', DataStructs.SparseBitVect)
  if version >= 4:
    res = ctor(len(atmoKeys) + 2 * bit_width - 2)
  elif version == 3:
    res = ctor(len(atmoKeys) + bit_width - 1)
  else: 
    res = ctor(len(atmoKeys))
  
  i = 0
  for patt, count in atmoKeys: 
    if patt not in no_smarts:
      matches = len(molh.GetSubstructMatches(patt))
      if count == 'binary':
        for bit in _encodeToBinary(matches, bit_width):
          res[i] = bit
          i += 1
        continue # after encoding binary bits, jump to next key
      else:
        res[i] = _bespe(matches, count, version)
    else: # separate logic for no-smarts
      if patt == 'Aromatic ring':
        matches = rdMolDescriptors.CalcNumAromaticRings(molh)
        res[i] = _bespe(matches, count, version)
      elif patt == 'Non-aromatic ring':
        matches = rdMolDescriptors.CalcNumRings(molh) # calculate total number of rings
        matches = max(0, matches - rdMolDescriptors.CalcNumAromaticRings(molh)) # subtract aromatic rings
        res[i] = _bespe(matches, count, version)
      elif patt == 'Nitrophenol':
        matches = _countNitrophenols(molh)
        res[i] = _bespe(matches, count, version)
    i += 1
  return res

def pyGenATMOMACCS(mol, *, version: int = 4, bit_width: int = 6, **kwargs):
  '''
  Simple method to concatenate MACCS keys and ATMO keys. Creates the ATMOMACCS
  descriptor fingerprint.

  **arguments**
     - mol: the molecule to be fingerprinted
     - version: the version of ATMOMACCS to compute
     - bit_width: The number of bits to allocate for binary encoding carbons
                  and oxygens.

  see pyGenATMOKeys and pyGenMACCSKeys methods for details.

  Example:
  >>> from ATMOMACCS import pyGenATMOMACCS
  >>> from rdkit import Chem
  >>> mol = Chem.MolFromSmiles('CC1(C)C(CC1C(=O)CC=O)ON(=O)=O')
  >>> fp = pyGenATMOMACCS(mol)
  >>> list(fp.GetOnBits())
  [11, 24, 48, 49, 63, 66, 70, 71, 74, 89, 94, 102, 106, 108, 112, 116, 119, 122, 124, 126, 127, 129, 130, 132, 136, 140, 143, 146, 148, 149, 150, 152, 154, 155, 157, 159, 160, 161, 164, 165, 183, 186, 187, 201, 209, 242, 277, 280, 284, 286]
  >>> len(fp)
  287
  '''
  maccs = pyGenMACCSKeys(mol)
  atmo = pyGenATMOKeys(mol, version=version, bit_width=bit_width)
  ctor = kwargs.get('ctor', DataStructs.SparseBitVect)
  fp = ctor(len(maccs) + len(atmo)) # init fingerprint
  i = 0
  for val in maccs:
    fp[i] = val
    i += 1
  for val in atmo:
    fp[i] = val
    i += 1
  return fp

#------------------------------------
#
#  doctest boilerplate
#
def _test():
  import doctest
  import sys
  return doctest.testmod(sys.modules["__main__"])


if __name__ == '__main__':
  import sys
  failed, tried = _test()
  sys.exit(failed)
