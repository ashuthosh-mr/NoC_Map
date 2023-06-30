#include <stdbool.h> 
#define ver 889
#define edg 5828

int h_graph_nodes[ver+1]={0,
3,11,20,22,30,34,38,63,75,76,78,80,96,104,105,111,112,115,117,128,133,148,153,157,169,173,174,176,178,179,181,185,190,193,201,203,206,207,209,211,212,214,215,216,221,222,230,231,239,249,262,263,273,276,287,292,314,320,336,341,346,347,354,356,358,360,363,366,368,374,375,378,381,389,392,394,395,401,402,403,405,407,408,410,411,412,414,430,431,438,442,443,447,448,452,456,457,474,475,477,
478,480,481,492,501,505,506,518,530,539,542,544,561,570,574,575,577,590,592,598,611,615,635,640,646,648,649,659,661,673,674,698,706,711,714,719,722,724,734,737,741,742,752,753,754,756,758,769,770,771,774,783,785,788,793,803,805,813,814,818,824,830,840,849,852,865,867,869,871,937,943,945,950,953,957,959,960,966,970,973,975,976,978,979,981,984,988,989,998,1002,1010,1011,1023,1024,1030,1036,1043,1044,1070,1071,
1072,1076,1078,1133,1152,1155,1162,1163,1168,1174,1177,1196,1199,1206,1211,1225,1227,1231,1235,1237,1240,1241,1254,1271,1277,1281,1285,1294,1297,1330,1337,1340,1360,1362,1363,1368,1390,1392,1410,1415,1419,1422,1428,1430,1434,1436,1452,1456,1460,1462,1464,1465,1466,1473,1481,1488,1499,1510,1532,1553,1557,1559,1565,1567,1568,1587,1593,1594,1596,1601,1608,1611,1703,1705,1713,1715,1718,1723,1724,1732,1745,1747,1748,1753,1775,1778,1779,1781,1794,1805,1807,1809,1810,1811,1812,1813,1818,1822,1838,1840,
1846,1847,1851,1852,1853,1862,1864,1870,1873,1875,1883,1885,1886,1894,1896,1903,1921,1924,1932,1933,1938,1957,1959,1966,1968,1972,1977,1986,1987,1990,1992,1993,1996,2000,2004,2009,2025,2026,2028,2029,2034,2038,2042,2051,2055,2056,2062,2067,2069,2075,2090,2092,2095,2099,2100,2116,2117,2123,2125,2126,2137,2143,2145,2159,2161,2162,2165,2176,2181,2197,2200,2213,2216,2218,2222,2225,2232,2233,2236,2271,2282,2283,2285,2288,2297,2319,2322,2328,2334,2337,2341,2355,2377,2378,2382,2396,2397,2410,2466,2467,
2469,2470,2477,2492,2497,2506,2512,2530,2531,2552,2555,2559,2567,2569,2585,2634,2643,2651,2656,2660,2674,2675,2681,2683,2697,2708,2712,2722,2724,2734,2836,2837,2861,2878,2881,2885,2897,2904,2911,2914,2919,2929,2931,2933,2939,2940,2972,3008,3034,3037,3049,3068,3094,3096,3112,3126,3128,3149,3159,3167,3177,3186,3196,3207,3217,3235,3237,3240,3247,3254,3273,3275,3276,3279,3282,3283,3288,3289,3293,3294,3305,3339,3344,3356,3363,3370,3382,3389,3391,3404,3408,3416,3417,3421,3422,3436,3438,3443,3455,3462,
3463,3470,3481,3512,3516,3534,3535,3544,3552,3555,3567,3568,3581,3582,3590,3591,3595,3603,3607,3608,3610,3612,3626,3628,3634,3646,3648,3655,3665,3667,3676,3681,3682,3686,3688,3748,3755,3782,3793,3802,3803,3805,3818,3823,3847,3861,3879,3900,3905,3955,3964,3981,3983,3986,3993,3997,4013,4019,4022,4036,4039,4070,4071,4085,4086,4090,4097,4124,4125,4135,4141,4147,4156,4162,4172,4175,4176,4177,4182,4183,4185,4190,4191,4202,4204,4212,4216,4235,4236,4237,4239,4245,4254,4255,4256,4261,4271,4273,4275,4276,
4280,4286,4287,4290,4291,4309,4311,4316,4325,4331,4339,4340,4348,4352,4355,4356,4388,4389,4400,4403,4404,4409,4410,4414,4419,4425,4433,4434,4438,4442,4443,4444,4445,4448,4485,4490,4499,4502,4509,4511,4513,4515,4518,4520,4527,4545,4548,4549,4551,4554,4558,4567,4572,4577,4585,4586,4592,4593,4608,4627,4629,4631,4634,4635,4644,4664,4665,4679,4682,4688,4693,4697,4705,4706,4709,4710,4715,4726,4730,4731,4740,4742,4746,4747,4748,4749,4761,4763,4764,4788,4802,4815,4822,4824,4825,4826,4863,4864,4867,4868,
4869,4883,4884,4887,4893,4896,4916,4919,4941,4955,4956,4958,4959,4963,4968,4970,4977,4981,4983,4986,4987,4989,4995,5001,5006,5007,5009,5010,5015,5020,5021,5022,5023,5025,5033,5076,5078,5097,5098,5103,5104,5110,5114,5115,5119,5121,5123,5124,5127,5150,5158,5159,5167,5168,5170,5171,5179,5183,5184,5185,5188,5231,5247,5250,5259,5260,5262,5266,5287,5292,5302,5306,5314,5317,5318,5325,5336,5339,5342,5347,5364,5365,5367,5376,5386,5394,5396,5398,5401,5402,5408,5409,5410,5412,5416,5417,5418,5420,5423,5427,
5434,5444,5462,5473,5476,5479,5498,5499,5501,5502,5503,5504,5506,5507,5509,5518,5522,5529,5541,5549,5550,5563,5569,5571,5575,5580,5583,5585,5597,5598,5600,5608,5610,5614,5615,5616,5621,5624,5641,5651,5664,5668,5677,5678,5686,5689,5693,5694,5695,5698,5702,5707,5723,5726,5731,5755,5756,5761,5769,5771,5773,5775,5777,5785,5787,5789,5790,5791,5794,5796,5797,5801,5804,5808,5809,5810,5811,5813,5814,5815,5816,5819,5820,5822,5824,5825,5826,5827,5828};

int h_graph_edges[edg]={7,
18,35,2,3,5,7,11,15,19,20,1,3,7,10,15,19,22,23,59,1,2,87,117,163,305,368,446,607,656,1,7,27,47,8,15,17,25,0,1,2,5,8,10,16,17,18,19,20,22,25,26,29,31,32,33,36,38,39,41,43,44,45,6,7,15,19,31,33,48,54,57,58,87,116,22,2,7,1,49,15,27,32,40,63,70,77,94,95,126,162,165,189,195,206,229,19,46,54,55,62,
87,97,109,19,1,2,6,8,12,59,7,6,7,33,0,7,1,2,7,8,13,14,23,24,32,65,78,1,7,32,36,272,35,39,66,69,87,97,103,122,157,206,305,407,535,554,616,2,7,9,32,36,2,19,31,44,19,52,54,74,87,89,99,111,123,145,149,150,6,7,30,51,7,5,12,49,63,7,25,86,7,8,23,44,7,12,19,20,22,7,8,17,42,97,105,147,194,211,256,272,
0,21,7,20,22,198,7,49,7,21,12,7,49,34,7,7,23,31,49,68,7,13,49,53,59,66,81,90,96,5,8,65,87,89,120,160,189,192,11,28,38,41,44,46,75,81,85,86,201,206,211,213,229,230,252,310,341,414,419,481,495,25,24,58,104,113,116,122,124,128,165,166,46,59,68,8,13,24,58,60,87,104,134,136,139,140,13,73,90,99,100,58,60,92,93,97,103,107,113,120,
121,122,124,142,155,157,159,162,169,170,175,178,180,8,94,103,104,124,160,8,52,54,56,62,73,92,104,105,113,122,124,137,144,162,169,2,15,46,53,90,54,56,120,150,162,113,13,58,77,134,150,155,210,12,28,203,395,19,48,21,46,82,203,265,300,44,53,21,98,109,201,215,257,12,163,229,495,97,307,430,55,58,120,125,129,136,171,186,24,94,111,49,80,559,12,62,95,165,228,229,
19,80,75,79,46,49,66,89,195,151,49,30,49,4,8,13,21,24,48,54,88,89,95,106,110,114,115,117,120,87,24,48,83,87,107,161,169,46,55,59,414,97,56,58,122,141,56,12,57,74,117,12,77,87,229,46,13,21,34,56,72,91,103,104,107,108,117,122,136,137,162,169,194,69,24,55,55,108,159,194,21,56,57,97,113,122,155,161,166,169,170,52,54,57,58,97,117,122,127,168,
34,58,177,188,87,56,89,97,108,113,120,121,140,157,170,182,188,97,101,107,120,121,122,182,188,211,213,223,229,13,69,120,122,129,169,222,305,327,87,114,134,24,74,325,412,430,478,514,545,550,561,569,633,636,665,689,706,708,714,737,52,56,58,61,103,107,120,170,175,87,110,122,127,87,8,52,4,87,94,97,104,122,169,177,211,297,446,470,605,222,237,158,258,310,316,349,363,48,56,60,
73,87,107,108,109,113,125,183,184,185,56,107,108,122,21,52,56,58,92,97,103,104,108,109,114,117,121,123,124,162,168,170,171,173,24,122,124,145,167,52,56,57,58,122,123,73,120,12,104,114,129,142,161,169,170,190,194,204,52,177,73,109,127,130,169,196,223,228,232,236,276,277,129,203,272,360,384,391,398,409,415,441,447,452,454,455,460,481,487,503,505,522,535,559,561,607,616,142,169,186,
204,223,238,239,258,169,246,288,330,350,54,62,110,689,821,831,833,854,54,73,97,58,97,230,238,323,430,436,446,452,454,455,457,54,229,327,54,107,155,219,92,56,127,132,162,186,204,206,216,239,253,355,58,24,123,556,716,34,162,198,238,253,256,266,272,304,308,313,327,24,24,60,62,84,155,156,188,192,204,213,214,215,215,247,224,226,305,187,198,224,272,336,56,62,103,140,151,162,188,195,
207,209,151,214,21,56,107,169,188,203,272,321,119,56,101,165,169,48,57,165,229,326,327,89,103,127,172,194,233,12,56,58,60,97,122,142,147,155,204,4,71,229,373,379,380,398,465,495,169,336,350,12,52,77,159,160,169,174,177,193,214,223,229,247,52,103,123,216,104,122,56,58,89,97,103,109,117,127,129,132,133,157,159,164,165,190,191,192,194,198,201,202,203,204,208,211,213,215,225,232,
243,246,257,258,259,262,263,270,271,272,274,276,277,279,284,288,297,298,302,305,307,316,321,323,329,333,334,340,348,355,359,360,362,363,364,366,56,103,107,113,122,127,73,122,161,246,262,266,317,122,177,197,165,199,204,239,56,113,388,105,117,128,165,173,229,56,272,288,333,236,263,264,56,255,355,107,108,120,120,185,120,184,251,73,132,142,223,154,105,107,108,151,155,157,206,213,220,12,48,192,
235,127,169,217,265,272,321,398,401,169,48,151,169,189,235,270,279,332,343,350,355,358,165,34,97,102,127,161,169,12,83,155,232,236,244,129,212,246,272,284,316,334,173,37,147,154,169,203,205,211,224,232,238,246,255,258,259,265,268,272,280,284,298,315,316,324,327,336,350,174,204,50,69,169,272,169,221,64,67,131,157,169,198,208,229,230,236,238,254,265,272,305,342,346,355,360,363,369,373,379,
386,387,391,392,398,403,406,409,415,420,430,432,436,446,447,451,452,454,455,457,460,470,487,490,507,525,530,535,542,554,559,564,127,132,142,151,162,169,174,200,208,213,214,236,253,257,258,259,260,261,262,198,305,379,12,21,50,142,188,220,229,155,169,203,204,336,354,155,232,256,257,259,260,62,227,236,34,50,108,117,169,198,222,223,253,259,265,280,292,293,327,331,372,398,485,196,215,220,50,108,
151,169,188,204,226,151,156,165,204,229,69,151,152,169,212,223,232,236,247,257,258,269,274,285,142,167,190,238,535,709,229,246,266,281,140,285,188,206,212,202,109,118,211,232,259,265,284,302,322,352,367,403,452,108,129,132,165,186,211,215,232,236,239,246,253,257,259,266,272,284,153,154,198,226,280,340,169,226,296,340,153,213,224,225,210,236,272,323,360,385,414,415,473,77,129,231,12,50,71,77,
95,108,139,160,163,165,177,203,206,214,218,280,300,327,335,357,385,392,394,395,409,432,447,459,482,485,502,508,512,50,138,203,429,430,454,491,228,277,334,129,169,195,198,209,215,222,223,236,246,247,256,258,259,265,269,272,283,284,303,161,260,322,189,192,318,343,350,129,179,195,203,204,210,215,223,227,232,246,259,278,289,310,323,388,403,454,463,490,507,118,610,132,138,147,198,203,217,241,254,265,
272,316,319,336,348,371,419,430,435,132,142,174,223,259,340,341,343,356,238,244,430,395,542,607,616,670,714,169,316,195,241,432,501,265,385,133,169,172,196,198,218,223,232,236,257,259,272,274,279,287,289,152,165,215,232,249,250,256,266,248,250,248,249,185,50,142,147,204,211,223,262,290,203,238,360,613,629,637,664,750,180,198,259,271,280,289,324,34,147,209,232,248,257,266,272,287,289,308,69,169,
204,209,215,223,246,256,269,274,284,119,132,169,198,204,215,232,259,268,269,274,280,288,289,316,318,335,349,350,376,398,409,169,198,204,209,211,222,223,232,236,239,246,255,258,262,265,274,285,289,298,314,316,204,209,233,273,204,549,169,172,204,253,259,295,169,179,179,67,190,198,203,211,222,232,238,245,259,284,288,298,316,321,336,398,404,463,147,172,218,223,248,256,269,198,258,215,232,257,258,267,
169,192,284,290,318,336,350,169,255,272,20,34,131,147,154,157,169,178,190,196,198,201,203,223,227,232,238,246,256,271,279,284,289,299,307,309,316,321,346,347,360,363,369,379,380,384,385,387,388,391,395,397,398,407,414,415,425,427,428,429,430,433,440,441,446,447,448,457,459,461,462,464,470,474,482,486,491,493,495,498,504,518,526,527,535,537,546,547,549,551,559,567,569,585,616,690,729,735,749,750,
769,833,260,283,169,215,246,257,258,259,297,298,298,316,129,169,313,129,169,231,300,334,236,169,192,246,272,284,288,336,371,198,211,224,229,255,258,281,289,291,298,316,327,341,218,280,371,232,273,363,415,430,169,196,198,222,223,232,257,265,270,272,279,321,336,343,350,357,363,369,378,380,385,398,215,219,259,375,246,256,133,169,178,258,265,279,318,321,346,363,371,394,398,236,246,255,256,258,259,272,
280,298,316,349,253,270,280,467,211,211,635,262,225,415,447,481,517,117,169,274,332,169,198,259,265,274,275,280,289,299,316,335,346,349,363,394,409,272,298,67,229,277,302,321,385,361,169,222,300,343,232,147,4,21,109,153,169,203,205,355,431,314,388,72,169,272,308,316,357,147,256,307,272,499,50,119,236,321,407,414,435,484,312,541,311,147,276,318,336,350,355,357,374,259,306,198,360,416,459,535,
567,589,119,169,196,198,238,243,258,259,265,272,275,280,289,298,307,343,350,363,172,361,418,235,258,270,288,313,350,355,363,238,430,545,549,556,603,157,169,190,265,272,284,288,300,310,349,371,385,398,415,420,427,437,438,444,222,234,138,169,227,236,385,452,461,198,255,112,708,771,801,160,394,416,417,419,109,139,148,160,198,211,229,280,372,403,169,407,420,133,355,211,192,297,337,169,178,343,369,169,
196,231,277,229,258,298,409,489,154,164,198,208,238,265,270,279,284,313,346,350,355,358,382,409,332,430,593,404,169,224,225,240,341,50,240,280,340,203,491,609,708,192,235,240,284,302,316,333,350,355,651,800,803,806,481,203,272,288,298,336,355,272,430,457,559,612,169,238,119,258,289,298,321,364,133,164,192,198,235,258,270,284,313,316,318,336,343,355,381,605,638,222,367,379,379,447,503,510,208,143,
169,181,192,203,305,313,318,330,336,343,346,350,372,398,405,240,229,284,307,313,385,477,192,336,169,131,169,203,227,254,272,315,425,430,432,473,301,317,395,509,512,538,169,423,119,169,203,272,283,284,288,298,316,318,369,398,423,436,169,349,547,169,398,624,222,352,376,415,430,470,483,484,498,501,507,4,407,447,547,549,203,272,284,333,363,387,392,424,430,446,465,481,493,569,585,654,398,424,716,238,
279,282,288,321,385,398,411,432,436,440,446,469,211,327,355,163,203,313,382,512,559,286,640,674,258,367,395,414,446,501,524,414,284,411,440,163,203,205,272,352,353,390,415,430,434,437,455,458,461,483,503,505,508,510,517,533,537,539,544,546,547,556,567,597,608,610,614,625,656,662,163,272,284,385,398,411,424,446,567,605,616,350,336,374,461,490,509,131,272,433,448,454,457,503,553,562,227,229,245,272,
284,300,321,323,357,371,380,404,409,411,416,429,436,446,470,489,495,505,203,412,535,203,272,369,430,466,569,176,236,272,306,398,479,551,566,672,379,391,535,618,131,203,272,390,392,409,430,446,454,455,457,465,497,538,203,229,369,391,398,407,414,415,416,425,427,448,450,457,463,469,481,495,497,507,522,535,502,229,288,298,326,64,229,242,272,361,376,410,415,430,452,473,489,495,496,533,272,407,424,449,
470,498,505,530,535,548,552,585,618,131,163,190,203,211,258,265,272,284,288,321,355,363,366,370,371,380,388,392,415,416,420,426,427,430,433,435,437,441,447,452,457,461,463,464,465,468,486,489,493,503,505,508,518,525,530,535,537,554,559,561,565,567,575,612,620,416,430,525,190,544,549,605,626,682,684,696,203,222,236,328,415,427,428,430,447,448,503,507,510,517,519,265,339,385,430,439,355,424,535,537,
549,569,635,690,696,203,430,481,483,541,597,21,272,310,329,368,392,397,416,433,446,452,470,481,512,535,547,605,624,418,131,203,229,258,298,335,336,385,391,429,432,447,448,450,452,462,470,489,502,512,567,395,445,528,371,378,380,385,112,386,508,549,662,686,740,771,556,670,50,90,227,272,310,376,377,392,417,427,432,441,461,468,484,507,131,203,227,272,283,296,321,367,379,392,395,398,403,420,427,429,
430,441,447,448,450,452,454,455,458,460,462,463,465,481,483,485,487,498,502,503,505,509,510,511,516,517,522,535,538,550,561,583,584,315,326,385,392,398,399,407,425,427,326,414,468,480,481,555,557,611,317,408,432,729,744,50,238,326,425,203,321,329,398,415,430,446,448,452,457,460,463,528,537,524,438,503,549,554,586,653,362,363,369,370,380,397,405,470,535,547,565,605,608,636,701,769,272,360,392,416,
419,432,452,457,467,480,490,398,470,535,567,272,321,392,398,403,414,415,416,444,467,272,403,230,272,385,409,415,430,437,441,444,449,72,112,138,203,230,238,241,272,283,320,338,347,360,367,369,379,387,391,395,398,400,403,404,406,415,420,429,439,443,446,447,451,454,455,459,461,462,465,471,480,486,487,491,492,497,498,500,503,523,525,528,531,535,537,539,543,545,546,547,549,557,560,574,585,590,592,595,
603,605,606,612,618,621,671,689,690,696,706,735,737,749,756,761,762,764,780,784,798,802,810,819,821,822,825,833,836,839,842,852,854,867,868,305,203,229,244,360,371,409,414,418,425,436,442,446,448,452,454,457,462,484,495,514,516,527,543,601,272,384,398,407,436,446,447,464,470,489,495,502,505,522,537,542,559,379,654,678,238,310,398,463,138,203,363,371,385,432,433,460,464,476,489,495,321,379,398,429,
464,535,567,321,422,572,659,708,709,722,404,430,547,272,371,378,464,465,131,272,398,414,415,429,459,491,505,584,432,803,430,529,321,427,429,491,573,641,410,4,117,138,203,272,369,371,376,380,385,391,407,420,430,432,433,452,453,455,457,462,464,465,481,482,501,502,503,505,510,516,526,131,203,229,272,296,353,368,398,403,409,415,430,433,448,452,455,457,458,463,481,496,497,502,503,505,510,522,528,542,
547,561,567,573,583,592,596,272,384,392,403,409,415,420,432,447,450,454,455,457,459,462,482,495,498,503,518,520,525,535,538,560,577,397,429,535,392,409,415,448,457,465,470,512,538,559,560,561,203,430,578,658,667,686,761,784,785,792,819,824,838,849,850,857,862,871,885,131,138,203,222,323,395,398,407,409,415,420,425,432,446,447,455,462,474,488,498,522,530,535,537,555,557,446,554,131,138,203,230,236,
384,391,415,430,432,448,460,462,503,518,559,131,138,203,379,391,415,430,446,447,448,452,460,471,505,549,703,138,203,272,347,384,391,392,398,420,425,432,446,447,448,450,465,497,498,535,542,543,379,415,447,503,510,522,535,549,583,596,229,272,315,430,441,448,470,503,131,203,415,420,436,454,455,495,501,503,272,323,379,383,398,414,430,499,501,272,409,415,430,432,446,448,452,454,517,236,265,392,398,415,
420,435,447,489,528,538,272,398,433,436,437,440,446,485,489,559,163,369,391,398,415,430,440,446,450,457,498,502,503,516,535,537,538,542,387,656,291,425,427,398,414,417,524,696,701,809,371,392,481,535,537,549,701,117,203,272,367,385,397,407,409,424,426,433,450,459,482,488,512,535,618,651,430,455,549,227,360,395,272,452,542,538,436,491,592,634,696,357,112,642,729,778,388,417,425,430,505,535,547,561,
567,573,581,604,50,131,296,345,369,392,406,407,415,417,446,447,469,491,503,505,508,522,525,535,542,544,545,548,549,561,567,583,587,590,591,596,619,635,229,272,446,448,470,367,379,406,415,535,539,634,659,665,701,703,709,310,367,414,432,501,507,508,211,229,415,464,505,530,537,272,398,430,537,539,542,547,561,567,571,581,612,131,203,415,430,714,782,870,452,470,335,385,395,398,409,433,436,463,464,530,
535,537,573,203,236,383,425,230,272,342,430,441,444,476,481,430,272,369,398,561,761,50,71,163,272,385,392,395,432,433,436,448,460,502,559,395,447,391,392,430,447,457,272,367,397,415,430,448,452,457,465,538,568,579,309,461,527,556,729,735,745,430,244,367,376,446,460,461,484,229,393,409,415,433,446,447,465,495,529,559,131,353,379,384,398,403,415,422,430,446,447,448,454,458,459,460,465,481,505,507,
510,512,517,522,528,535,543,567,581,583,596,272,506,549,634,131,379,385,397,398,415,433,441,446,447,455,480,481,485,503,535,547,558,504,203,236,367,392,403,414,484,503,510,229,379,398,412,481,484,513,514,361,383,415,353,379,403,415,446,447,458,503,507,522,583,596,415,229,361,374,407,409,450,470,503,535,558,696,785,818,508,112,432,508,549,582,587,616,630,664,415,432,446,465,296,379,403,415,462,503,
535,574,272,398,448,454,403,448,591,616,723,131,392,415,433,447,452,458,481,503,510,525,567,583,623,430,567,376,421,468,549,665,768,203,398,400,430,448,481,522,528,535,561,567,591,272,446,272,432,499,544,616,645,680,410,420,430,447,463,503,525,537,542,595,443,502,203,397,398,452,485,489,535,542,585,430,556,819,821,841,785,379,396,539,549,537,624,21,131,203,217,272,315,386,390,392,397,398,405,407,
415,424,426,430,437,448,449,452,457,458,465,469,470,480,481,483,489,503,505,512,517,525,530,537,538,544,545,547,549,561,565,567,569,571,574,575,576,578,591,592,600,605,608,618,629,638,649,735,761,780,822,838,863,879,272,379,398,405,420,430,433,452,465,469,485,486,489,528,534,535,547,557,561,567,570,571,574,592,618,623,631,361,391,415,448,450,463,465,475,498,535,539,379,430,483,486,533,538,544,634,
644,625,311,406,203,242,433,447,457,465,474,481,486,528,530,549,618,430,432,457,503,549,379,402,481,527,535,539,545,546,548,549,550,587,610,612,618,619,625,628,638,645,652,659,671,682,112,320,430,481,535,544,546,549,556,583,587,596,616,625,272,379,430,544,545,549,587,595,610,612,616,626,638,644,646,652,654,659,272,365,368,379,407,424,430,439,447,480,486,505,535,537,561,571,601,608,610,646,649,397,
481,544,595,610,261,272,320,368,402,405,412,422,430,456,458,469,472,481,504,514,524,533,535,542,543,544,545,546,550,551,556,561,565,578,583,586,600,616,621,624,634,653,659,664,680,681,689,690,696,701,704,708,725,737,112,415,544,549,555,595,609,637,652,272,389,549,605,616,634,638,659,665,669,734,735,752,775,776,793,801,397,661,384,572,601,21,203,398,422,453,561,645,417,452,550,557,146,320,379,413,
499,531,545,549,563,607,616,650,654,695,709,735,417,430,452,537,555,559,505,512,567,76,131,203,272,347,374,398,433,450,454,464,495,502,557,430,448,450,112,131,398,415,447,450,480,481,486,493,525,535,537,547,549,554,567,571,573,574,575,580,585,587,588,592,599,615,618,621,628,384,556,634,658,664,677,686,696,701,734,735,737,744,750,771,203,398,424,535,549,389,634,691,696,698,762,770,272,315,379,380,
398,409,426,437,447,480,481,486,503,522,523,525,535,537,558,561,571,574,578,580,587,603,618,498,112,272,369,387,405,535,574,585,769,841,537,572,616,654,689,767,486,535,537,547,561,567,438,553,570,601,609,633,634,665,680,444,447,480,489,561,641,430,517,535,537,561,567,569,581,592,618,398,535,561,535,448,451,535,549,567,634,498,561,567,480,486,503,574,585,514,415,447,458,481,503,510,522,545,549,587,
596,415,441,272,369,397,430,530,561,569,581,422,549,677,750,481,514,544,545,546,561,567,583,596,605,608,616,650,652,659,660,671,680,707,561,315,430,481,481,520,525,535,614,644,430,447,476,535,537,561,574,636,651,338,610,430,528,546,548,550,447,458,481,503,510,545,583,587,652,675,379,406,665,738,561,535,549,690,701,432,547,553,572,626,649,616,320,430,567,480,117,351,380,402,407,424,430,535,551,587,
608,619,626,627,677,690,749,769,430,621,4,131,242,556,713,379,424,535,547,587,605,626,629,639,342,550,572,616,659,680,237,379,544,546,547,548,594,626,417,347,398,430,486,544,546,737,761,254,645,757,855,379,591,655,561,21,131,242,272,380,514,521,527,545,546,549,551,556,570,587,602,609,645,661,677,680,691,701,704,708,709,726,734,735,749,750,770,674,390,397,430,470,535,537,542,544,561,567,574,481,
544,605,398,430,549,561,606,624,665,522,537,633,635,366,407,534,549,621,379,540,544,545,654,674,402,546,601,605,608,610,639,681,605,544,561,660,664,254,535,608,662,514,537,635,112,572,623,476,483,504,539,549,551,563,566,572,578,645,658,665,667,676,680,689,697,701,704,705,706,709,713,717,722,724,734,735,737,754,761,768,776,779,785,786,294,405,481,623,632,112,424,592,646,659,678,686,696,747,254,550,
664,351,535,544,546,551,644,659,608,626,375,888,444,573,478,768,770,691,728,539,546,591,638,654,667,677,527,544,554,613,616,634,654,665,689,696,706,708,722,735,748,762,763,799,546,547,636,788,668,704,535,547,601,556,587,665,673,344,470,592,659,672,693,716,762,833,544,546,550,587,596,422,549,670,696,722,369,434,546,556,570,625,644,645,614,4,379,466,678,709,733,708,451,563,634,659,678,689,711,735,
777,790,817,840,844,852,855,438,483,544,546,549,551,587,609,636,638,651,658,672,682,690,709,734,743,745,587,628,552,616,379,412,629,671,254,515,549,563,628,637,735,757,812,112,483,524,551,572,598,622,634,645,650,667,677,701,703,709,730,735,753,761,783,682,451,634,644,665,669,677,680,716,719,735,744,749,801,818,648,722,727,551,667,719,737,761,770,242,413,653,692,734,430,544,587,663,389,651,659,696,
701,728,775,794,650,375,617,625,596,634,680,690,696,721,563,586,605,616,644,665,667,705,717,735,752,434,636,656,658,838,527,549,572,587,609,616,634,667,676,549,626,402,544,659,666,689,402,737,412,451,563,636,687,689,696,708,742,750,828,850,686,883,735,112,135,430,549,570,634,645,658,683,686,708,756,761,762,768,772,806,816,817,821,831,844,852,855,272,405,430,549,600,605,659,676,691,704,734,755,769,
811,566,616,643,690,706,735,749,751,783,789,794,803,816,670,742,768,824,831,838,852,651,742,737,556,402,405,430,468,476,512,549,563,566,636,645,653,672,676,686,709,713,719,735,737,749,757,762,768,776,783,801,804,815,816,817,818,828,831,836,845,851,634,566,699,778,698,722,424,468,469,483,549,563,600,616,634,665,672,707,713,737,708,456,483,665,549,616,634,648,690,709,634,677,737,112,430,634,645,691,
708,726,735,761,762,780,803,821,838,841,844,851,855,859,876,587,701,718,112,325,342,438,549,616,645,657,686,689,702,706,723,735,749,762,768,772,791,801,803,806,217,438,483,556,616,634,656,659,665,696,704,720,723,724,818,658,712,711,607,634,696,701,112,242,487,770,782,735,819,146,370,651,667,723,772,775,634,677,735,768,707,723,667,669,696,709,676,855,438,634,645,653,668,700,521,708,709,716,718,744,
634,709,790,825,835,549,616,706,668,643,672,760,780,790,272,418,478,499,733,665,749,739,656,729,551,563,616,634,659,670,690,786,272,430,499,536,551,556,563,616,634,645,658,664,665,667,677,688,691,696,706,708,715,717,737,748,749,752,761,762,763,768,772,775,776,779,785,788,800,802,803,815,821,836,842,855,877,112,430,549,563,612,634,669,685,694,696,701,705,735,749,752,770,801,803,826,598,732,762,802,
806,845,412,749,761,772,776,830,834,686,692,693,884,659,418,563,667,723,499,659,815,826,636,645,735,759,272,430,605,616,667,691,696,708,731,735,737,741,761,762,776,790,800,801,803,805,806,815,824,254,272,563,586,616,686,790,797,691,551,677,735,737,761,762,776,828,665,634,802,690,430,689,757,806,836,839,840,855,613,664,696,756,780,748,728,783,794,430,451,494,536,612,634,665,669,689,706,735,741,749,
752,762,764,767,768,772,773,776,780,784,799,800,801,806,815,818,821,822,826,831,832,838,840,842,844,845,852,854,855,873,430,566,645,651,689,696,706,708,735,739,749,752,761,768,802,814,645,735,783,430,761,777,802,827,838,858,866,872,806,818,843,570,761,768,821,524,634,642,689,692,696,708,717,735,761,762,767,770,780,795,802,806,821,828,836,839,272,424,569,605,690,566,616,642,669,714,737,768,806,812,
837,325,412,563,829,689,708,716,735,741,761,780,839,761,814,824,806,551,672,716,735,776,815,877,551,634,696,735,741,749,752,761,775,801,828,658,764,838,478,698,837,634,735,780,815,818,430,536,706,728,758,761,768,772,779,787,806,816,821,840,842,852,855,806,487,714,665,691,696,760,763,790,804,806,808,430,451,761,837,844,855,857,863,872,874,451,512,532,634,735,855,863,873,634,734,780,802,647,735,828,
691,658,724,728,749,750,783,708,451,551,818,672,691,760,802,768,797,750,796,430,852,863,645,761,828,855,344,735,749,761,802,806,839,325,551,667,696,708,737,749,761,776,828,430,735,739,754,762,764,768,787,794,800,803,817,819,822,825,832,838,864,344,442,691,706,708,735,737,749,802,815,820,696,783,858,749,806,830,344,689,708,739,749,756,761,765,768,770,774,780,781,783,800,805,813,818,839,808,783,807,
468,430,690,664,770,806,762,773,696,735,746,749,761,775,779,803,817,689,691,696,780,658,689,696,802,815,827,858,512,667,696,710,761,766,779,793,806,819,847,855,430,451,531,715,802,818,846,858,803,135,430,531,689,706,735,761,767,768,780,861,862,863,430,536,761,802,839,842,844,887,451,692,749,773,430,724,802,839,852,737,746,761,764,817,686,696,752,768,776,788,799,801,839,846,848,850,771,741,805,135,
689,692,696,761,839,842,855,761,802,135,272,430,651,741,724,430,696,735,756,768,770,778,784,451,536,679,692,706,761,764,777,802,841,842,852,853,854,857,859,860,430,756,768,772,800,806,822,825,828,831,658,756,761,780,846,849,852,854,855,857,858,864,881,531,569,706,838,430,735,761,780,822,831,838,852,853,766,658,689,706,761,784,823,868,878,696,739,761,819,828,840,880,818,828,451,840,886,451,686,828,
869,696,706,852,856,871,430,658,689,692,761,780,798,825,838,840,842,851,853,855,865,871,838,842,852,135,430,761,838,840,613,658,689,706,721,736,756,761,780,784,785,799,818,831,840,852,857,858,863,865,873,875,881,882,851,451,784,838,840,855,764,804,817,819,840,855,868,872,706,838,838,871,821,881,451,821,536,784,785,798,821,855,869,873,802,840,852,855,764,430,430,844,858,850,863,487,451,851,852,860,
764,784,858,761,785,855,863,784,855,706,736,775,844,536,846,840,855,861,855,687,884,742,883,451,849,823,640};

int h_graph_active[ver]={0};

int h_updating_graph_active[ver]={0};


int h_cost[ver]={-1};
