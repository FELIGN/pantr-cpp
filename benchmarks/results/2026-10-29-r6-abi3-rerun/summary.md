| case | main stat (ns) | main range | abi3 stat (ns) | abi3 range | vs stat (ns) | vs range | flags |
|---|---|---|---|---|---|---|---|
| r6.one_point.tiny@t1 | 3035 | 393 | 833 | 91 | 795 | 88 |  |
| r6.one_point.tiny@t8 | 3158 | 282 | 898 | 474 | 811 | 78 |  |
| r6.property.tiny@t1 | 65 | 3 | 87 | 9 | 88 | 3 | abi3,vs |
| r6.property.tiny@t8 | 67 | 6 | 87 | 8 | 88 | 5 | abi3,vs |
| r6.scalar.tiny@t1 | 2512 | 285 | 99 | 53 | 92 | 10 |  |
| r6.scalar.tiny@t8 | 2545 | 1332 | 100 | 54 | 96 | 9 |  |
| r6.tiny_array.tiny@t1 | 3483 | 349 | 877 | 90 | 871 | 93 |  |
| r6.tiny_array.tiny@t8 | 3480 | 178 | 904 | 101 | 872 | 92 |  |

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
