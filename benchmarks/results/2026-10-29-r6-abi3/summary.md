| case | main stat (ns) | main range | abi3 stat (ns) | abi3 range | vs stat (ns) | vs range | flags |
|---|---|---|---|---|---|---|---|
| r6.one_point.tiny@t1 | 3103 | 683 | 834 | 82 | 828 | 101 |  |
| r6.one_point.tiny@t8 | 3050 | 625 | 852 | 477 | 810 | 207 |  |
| r6.property.tiny@t1 | 68 | 5 | 88 | 6 | 89 | 7 | abi3,vs |
| r6.property.tiny@t8 | 65 | 4 | 87 | 4 | 88 | 5 | abi3,vs |
| r6.scalar.tiny@t1 | 2460 | 176 | 98 | 8 | 97 | 9 |  |
| r6.scalar.tiny@t8 | 2450 | 1295 | 101 | 51 | 97 | 6 |  |
| r6.tiny_array.tiny@t1 | 3499 | 669 | 887 | 93 | 887 | 112 |  |
| r6.tiny_array.tiny@t8 | 3484 | 452 | 886 | 50 | 866 | 166 |  |

| r6 case | abi3 > main stat + range | abi3 - versioned > main range | trigger |
|---|---|---|---|
| r6.one_point.tiny@t1 | False | False | False |
| r6.one_point.tiny@t8 | False | False | False |
| r6.property.tiny@t1 | True | False | False |
| r6.property.tiny@t8 | True | False | False |
| r6.scalar.tiny@t1 | False | False | False |
| r6.scalar.tiny@t8 | False | False | False |
| r6.tiny_array.tiny@t1 | False | False | False |
| r6.tiny_array.tiny@t8 | False | False | False |
