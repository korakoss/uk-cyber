"""
Build tidy DataFrames of IC3 "crime type by complaint count" and "crime type by
loss" tables across years (2015-2025; 2011-2014 either missing or lacking a
unified crime-type table in the source report, see IC3_SUMMARY.md).

Data hand-transcribed from data/raw/ic3/text/*_IC3Report.txt (pdftotext -layout
extractions) during manual reading of each report. This is reference/context
data (US complaint statistics), not the UK CSBS survey - see NOTES.md "New
data source found: ic3.zip".

Category names shift across years (e.g. "Phishing/Vishing/Smishing/Pharming"
-> "Phishing" -> "Phishing/Spoofing" after the 2023 Phishing+Spoofing merge;
"BEC/EAC" -> "BEC" -> "Business Email Compromise"). `canonicalize()` maps raw
per-year labels onto a stable set of category names for cross-year comparison,
but keeps the raw label too so the merge/split points stay visible.
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Raw per-year data, transcribed as printed in each report.
# ---------------------------------------------------------------------------

COUNTS = {
    2015: {
        "Non-Payment/Non-Delivery": 67375, "419/Overpayment": 30855,
        "Identity Theft": 21949, "Auction": 21510, "Other": 19963,
        "Personal Data Breach": 19632, "Employment": 18758, "Extortion": 17804,
        "Credit Card Fraud": 17172, "Phishing/Vishing/Smishing/Pharming": 16594,
        "Advanced Fee": 16445, "Harassment/Threats of Violence": 14812,
        "Confidence Fraud/Romance": 12509, "No Lead Value": 12187,
        "Government Impersonation": 11832, "Real Estate/Rental": 11562,
        "Business Email Compromise": 7837, "Misrepresentation": 5458,
        "Lottery/Sweepstakes": 5324, "Malware/Scareware": 3294,
        "Corporate Data Breach": 2499, "Ransomware": 2453,
        "IPR/Copyright and Counterfeit": 1931, "Investment": 1806,
        "Crimes Against Children": 1348, "Civil Matter": 1148,
        "Re-shipping": 1073, "Denial of Service": 1020, "Virus": 971,
        "Health Care Related": 465, "Charity": 411, "Terrorism": 361,
        "Hacktivist": 211, "Gambling": 131, "Criminal Forums": 62,
    },
    2016: {
        "Non-Payment/Non-Delivery": 81029, "Personal Data Breach": 27573,
        "419/Overpayment": 25716, "Phishing/Vishing/Smishing/Pharming": 19465,
        "Employment": 17387, "Extortion": 17146, "Identity Theft": 16878,
        "Harassment/Threats of Violence": 16385, "Credit Card Fraud": 15895,
        "Advanced Fee": 15075, "Confidence Fraud/Romance": 14546,
        "No Lead Value": 13794, "Other": 12619, "Real Estate/Rental": 12574,
        "Government Impersonation": 12344, "BEC/EAC": 12005,
        "Tech Support": 10850, "Misrepresentation": 5436,
        "Lottery/Sweepstakes": 4231, "Corporate Data Breach": 3403,
        "Malware/Scareware": 2783, "Ransomware": 2673,
        "IPR/Copyright and Counterfeit": 2572, "Investment": 2197,
        "Virus": 1498, "Crimes Against Children": 1230, "Civil Matter": 1070,
        "Denial of Service": 979, "Re-shipping": 893, "Charity": 437,
        "Health Care Related": 369, "Terrorism": 295, "Gambling": 137,
        "Hacktivist": 113,
    },
    2017: {
        "Non-Payment/Non-Delivery": 84079, "Personal Data Breach": 30904,
        "Phishing/Vishing/Smishing/Pharming": 25344, "Overpayment": 23135,
        "No Lead Value": 20241, "Identity Theft": 17636, "Advanced Fee": 16368,
        "Harassment/Threats of Violence": 16194, "Employment": 15784,
        "BEC/EAC": 15690, "Confidence Fraud/Romance": 15372,
        "Credit Card Fraud": 15220, "Extortion": 14938, "Other": 14023,
        "Tech Support": 10949, "Real Estate/Rental": 9645,
        "Government Impersonation": 9149, "Misrepresentation": 5437,
        "Corporate Data Breach": 3785, "Investment": 3089,
        "Malware/Scareware/Virus": 3089, "Lottery/Sweepstakes": 3012,
        "IPR/Copyright and Counterfeit": 2644, "Ransomware": 1783,
        "Crimes Against Children": 1300, "Denial of Service/TDoS": 1201,
        "Civil Matter": 1057, "Re-shipping": 1025, "Charity": 436,
        "Health Care Related": 406, "Gambling": 203, "Terrorism": 177,
        "Hacktivist": 158,
    },
    2018: {
        "Non-Payment/Non-Delivery": 65116, "Extortion": 51146,
        "Personal Data Breach": 50642, "No Lead Value": 36936,
        "Phishing/Vishing/Smishing/Pharming": 26379, "BEC/EAC": 20373,
        "Confidence Fraud/Romance": 18493, "Harassment/Threats of Violence": 18415,
        "Advanced Fee": 16362, "Identity Theft": 16128, "Spoofing": 15569,
        "Overpayment": 15512, "Credit Card Fraud": 15210, "Employment": 14979,
        "Tech Support": 14408, "Real Estate/Rental": 11300,
        "Government Impersonation": 10978, "Other": 10826,
        "Lottery/Sweepstakes": 7146, "Misrepresentation": 5959,
        "Investment": 3693, "Malware/Scareware/Virus": 2811,
        "Corporate Data Breach": 2480, "IPR/Copyright and Counterfeit": 2249,
        "Denial of Service/TDoS": 1799, "Ransomware": 1493,
        "Crimes Against Children": 1394, "Re-shipping": 907,
        "Civil Matter": 768, "Charity": 493, "Health Care Related": 337,
        "Gambling": 181, "Terrorism": 120, "Hacktivist": 77,
    },
    2019: {
        "Phishing/Vishing/Smishing/Pharming": 114702, "Non-Payment/Non-Delivery": 61832,
        "Extortion": 43101, "Personal Data Breach": 38218, "Spoofing": 25789,
        "BEC/EAC": 23775, "Confidence Fraud/Romance": 19473, "Identity Theft": 16053,
        "Harassment/Threats of Violence": 15502, "Overpayment": 15395,
        "Advanced Fee": 14607, "Employment": 14493, "Credit Card Fraud": 14378,
        "Government Impersonation": 13873, "Tech Support": 13633,
        "Real Estate/Rental": 11677, "Other": 10842,
        "Lottery/Sweepstakes/Inheritance": 7767, "Misrepresentation": 5975,
        "Investment": 3999, "IPR/Copyright and Counterfeit": 3892,
        "Malware/Scareware/Virus": 2373, "Ransomware": 2047,
        "Corporate Data Breach": 1795, "Denial of Service/TDoS": 1353,
        "Crimes Against Children": 1312, "Re-shipping": 929, "Civil Matter": 908,
        "Health Care Related": 657, "Charity": 407, "Gambling": 262,
        "Terrorism": 61, "Hacktivist": 39,
    },
    2020: {
        "Phishing/Vishing/Smishing/Pharming": 241342, "Non-Payment/Non-Delivery": 108869,
        "Extortion": 76741, "Personal Data Breach": 45330, "Identity Theft": 43330,
        "Spoofing": 28218, "Misrepresentation": 24276, "Confidence Fraud/Romance": 23751,
        "Harassment/Threats of Violence": 20604, "BEC/EAC": 19369,
        "Credit Card Fraud": 17614, "Employment": 16879, "Tech Support": 15421,
        "Real Estate/Rental": 13638, "Advanced Fee": 13020,
        "Government Impersonation": 12827, "Overpayment": 10988, "Other": 10372,
        "Investment": 8788, "Lottery/Sweepstakes/Inheritance": 8501,
        "IPR/Copyright and Counterfeit": 4213, "Crimes Against Children": 3202,
        "Corporate Data Breach": 2794, "Ransomware": 2474,
        "Denial of Service/TDoS": 2018, "Malware/Scareware/Virus": 1423,
        "Health Care Related": 1383, "Civil Matter": 968, "Re-shipping": 883,
        "Charity": 659, "Gambling": 391, "Terrorism": 65, "Hacktivist": 52,
    },
    2021: {
        "Phishing/Vishing/Smishing/Pharming": 323972, "Non-Payment/Non-Delivery": 82478,
        "Personal Data Breach": 51829, "Identity Theft": 51629, "Extortion": 39360,
        "Confidence Fraud/Romance": 24299, "Tech Support": 23903, "Investment": 20561,
        "BEC/EAC": 19954, "Spoofing": 18522, "Credit Card Fraud": 16750,
        "Employment": 15253, "Other": 12346, "Terrorism/Threats of Violence": 12346,
        "Real Estate/Rental": 11578, "Government Impersonation": 11335,
        "Advanced Fee": 11034, "Overpayment": 6108,
        "Lottery/Sweepstakes/Inheritance": 5991, "IPR/Copyright and Counterfeit": 4270,
        "Ransomware": 3729, "Crimes Against Children": 2167,
        "Corporate Data Breach": 1287, "Civil Matter": 1118,
        "Denial of Service/TDoS": 1104, "Computer Intrusion": 979,
        "Malware/Scareware/Virus": 810, "Health Care Related": 578,
        "Re-shipping": 516, "Gambling": 395,
        # NOTE: source PDF prints "Other" and "Terrorism/Threats of Violence"
        # with the *same* value (12,346) - flagged by extraction as a likely
        # pdftotext column-alignment artifact, not verified against the raw
        # PDF layout. Kept as-is rather than guessing a correction.
    },
    2022: {
        "Phishing": 300497, "Personal Data Breach": 58859,
        "Non-Payment/Non-Delivery": 51679, "Extortion": 39416, "Tech Support": 32538,
        "Investment": 30529, "Identity Theft": 27922, "Credit Card/Check Fraud": 22985,
        "BEC": 21832, "Spoofing": 20649, "Confidence/Romance": 19021,
        "Employment": 14946, "Harassment/Stalking": 11779, "Real Estate": 11727,
        "Government Impersonation": 11554, "Advanced Fee": 11264, "Other": 9966,
        "Overpayment": 6183, "Lottery/Sweepstakes/Inheritance": 5650,
        "Data Breach": 2795, "Crimes Against Children": 2587, "Ransomware": 2385,
        "Threats of Violence": 2224, "IPR/Copyright and Counterfeit": 2183,
        "SIM Swap": 2026, "Malware": 762, "Botnet": 568,
    },
    2023: {
        "Phishing/Spoofing": 298878, "Personal Data Breach": 55851,
        "Non-payment/Non-Delivery": 50523, "Extortion": 48223, "Investment": 39570,
        "Tech Support": 37560, "BEC": 21489, "Identity Theft": 19778,
        "Confidence/Romance": 17823, "Employment": 15443,
        "Government Impersonation": 14190, "Credit Card/Check Fraud": 13718,
        "Harassment/Stalking": 9587, "Real Estate": 9521, "Other": 8808,
        "Advanced Fee": 8045, "Lottery/Sweepstakes/Inheritance": 4168,
        "Overpayment": 4144, "Data Breach": 3727, "Ransomware": 2825,
        "Crimes Against Children": 2361, "Threats of Violence": 1697,
        "IPR/Copyright and Counterfeit": 1498, "SIM Swap": 1075,
        "Malware": 659, "Botnet": 540,
    },
    2024: {
        "Phishing/Spoofing": 193407, "Extortion": 86415, "Personal Data Breach": 64882,
        "Non-Payment/Non-Delivery": 49572, "Investment": 47919, "Tech Support": 36002,
        "Business Email Compromise": 21442, "Identity Theft": 21403, "Employment": 20044,
        "Confidence/Romance": 17910, "Government Impersonation": 17367,
        "Credit Card/Check Fraud": 12876, "Other": 12318, "Harassment/Stalking": 11672,
        "Real Estate": 9359, "Advanced Fee": 7097, "Crimes Against Children": 4472,
        "Lottery/Sweepstakes/Inheritance": 3690, "Data Breach": 3204, "Ransomware": 3156,
        "Overpayment": 2705, "IPR/Copyright and Counterfeit": 1583,
        "Threats of Violence": 1360, "SIM Swap": 982, "Botnet": 587, "Malware": 441,
    },
    2025: {
        "Phishing/Spoofing": 191561, "Extortion": 89129, "Investment": 72984,
        "Personal Data Breach": 67456, "Non-Payment/Non-Delivery": 56478,
        "Tech/Customer Support": 47794, "Government Impersonation": 32424,
        "Identity Theft": 31675, "Business Email Compromise": 24768,
        "Employment": 24688, "Confidence/Romance": 23159, "Harassment/Stalking": 21557,
        "Other": 20031, "Credit Card/Check Fraud": 18774, "Real Estate": 12368,
        "Advanced Fee": 7762, "Lottery/Sweepstakes/Inheritance": 5623,
        "Threats of Violence": 4826, "Data Breach": 3963, "Ransomware": 3611,
        "IPR/Copyright and Counterfeit": 2386, "Overpayment": 2194, "SIM Swap": 971,
        "Malware": 893, "Botnet": 715, "Charity": 662,
    },
}

LOSSES = {
    2015: {
        "Business Email Compromise": 246226016, "Confidence Fraud/Romance": 203390531,
        "Non-Payment/Non-Delivery": 121329122, "Investment": 119177899,
        "Identity Theft": 57294589, "Other": 56153977, "Advanced Fee": 50721226,
        "419/Overpayment": 49217119, "Personal Data Breach": 43477526,
        "Credit Card Fraud": 41503502, "Real Estate/Rental": 41417647,
        "Corporate Data Breach": 38800430, "Employment": 33890824,
        "Lottery/Sweepstakes": 19365223, "Auction": 18906416,
        "Misrepresentation": 17974014, "Extortion": 14799705,
        "Harassment/Threats of Violence": 13126123, "Government Impersonation": 12090159,
        "Civil Matter": 9946345, "Phishing/Vishing/Smishing/Pharming": 8174316,
        "IPR/Copyright and Counterfeit": 7230803, "Re-shipping": 3831957,
        "Malware/Scareware": 2912628, "Denial of Service": 2770978,
        "Ransomware": 1620814, "Charity": 1328153, "Virus": 1230812,
        "Gambling": 955360, "Health Care Related": 906343, "Hacktivist": 171601,
        "Crimes Against Children": 97584, "Terrorism": 65789,
        "Criminal Forums": 55996, "No Lead Value": 0,
    },
    2016: {
        "BEC/EAC": 360513961, "Confidence Fraud/Romance": 219807760,
        "Non-Payment/Non-Delivery": 138228282, "Investment": 123407997,
        "Corporate Data Breach": 95869990, "Other": 73092101,
        "Advanced Fee": 60484573, "Personal Data Breach": 59139152,
        "Identity Theft": 58917398, "Civil Matter": 57688555,
        "419/Overpayment": 56004836, "Credit Card Fraud": 48187993,
        "Real Estate/Rental": 47875765, "Employment": 40517605,
        "Phishing/Vishing/Smishing/Pharming": 31679451,
        "Harassment/Threats of Violence": 22005655, "Lottery/Sweepstakes": 21283769,
        "Extortion": 15811837, "Misrepresentation": 13725233,
        "Government Impersonation": 12278714, "Denial of Service": 11213566,
        "Tech Support": 7806416, "IPR/Copyright and Counterfeit": 6829467,
        "Malware/Scareware": 3853351, "Ransomware": 2431261,
        "Re-shipping": 1932021, "Charity": 1660452, "Virus": 1635321,
        "Health Care Related": 995659, "Gambling": 290693, "Terrorism": 219935,
        "Crimes Against Children": 79173, "Hacktivist": 55500, "No Lead Value": 0,
    },
    2017: {
        "BEC/EAC": 676151185, "Confidence Fraud/Romance": 211382989,
        "Non-Payment/Non-Delivery": 141110441, "Investment": 96844144,
        "Personal Data Breach": 77134865, "Identity Theft": 66815298,
        "Corporate Data Breach": 60942306, "Advanced Fee": 57861324,
        "Credit Card Fraud": 57207248, "Real Estate/Rental": 56231333,
        "Overpayment": 53450830, "Employment": 38883616,
        "Phishing/Vishing/Smishing/Pharming": 29703421, "Other": 23853704,
        "Lottery/Sweepstakes": 16835001, "Extortion": 15302792,
        "Tech Support": 14810080, "Misrepresentation": 14580907,
        "Harassment/Threats of Violence": 12569185, "Government Impersonation": 12467380,
        "Civil Matter": 5766550, "IPR/Copyright and Counterfeit": 5536912,
        "Malware/Scareware/Virus": 5003434, "Ransomware": 2344365,
        "Denial of Service/TDoS": 1466195, "Charity": 1405460,
        "Health Care Related": 925849, "Re-Shipping": 809746, "Gambling": 598853,
        "Crimes Against Children": 46411, "Hacktivist": 20147, "Terrorism": 18926,
        "No Lead Value": 0,
    },
    2018: {
        "BEC/EAC": 1297803489, "Confidence Fraud/Romance": 362500761,
        "Investment": 252955320, "Non-Payment/Non-Delivery": 183826809,
        "Real Estate/Rental": 149458114, "Personal Data Breach": 148892403,
        "Corporate Data Breach": 117711989, "Identity Theft": 100429691,
        "Advanced Fee": 92271682, "Credit Card Fraud": 88991436,
        "Extortion": 83357901, "Spoofing": 70000248,
        "Government Impersonation": 64211765, "Other": 63126929,
        "Lottery/Sweepstakes": 60214814, "Overpayment": 53225507,
        "Phishing/Vishing/Smishing/Pharming": 48241748, "Employment": 45487120,
        "Tech Support": 38697026, "Harassment/Threats of Violence": 21903829,
        "Misrepresentation": 20000713, "IPR/Copyright and Counterfeit": 15802011,
        "Civil Matter": 15172692, "Malware/Scareware/Virus": 7411651,
        "Health Care Related": 4474792, "Ransomware": 3621857,
        "Denial of Service/TDoS": 2052340, "Re-Shipping": 1684179,
        "Charity": 1006379, "Gambling": 926953, "Crimes Against Children": 265996,
        "Hacktivist": 77612, "Terrorism": 10193, "No Lead Value": 0,
    },
    2019: {
        "BEC/EAC": 1776549688, "Confidence Fraud/Romance": 475014032,
        "Spoofing": 300478433, "Investment": 222186195,
        "Real Estate/Rental": 221365911, "Non-Payment/Non-Delivery": 196563497,
        "Identity Theft": 160305789, "Government Impersonation": 124292606,
        "Personal Data Breach": 120102501, "Credit Card Fraud": 111491163,
        "Extortion": 107498956, "Advanced Fee": 100602297, "Other": 66223160,
        "Phishing/Vishing/Smishing/Pharming": 57836379, "Overpayment": 55820212,
        "Tech Support": 54041053, "Corporate Data Breach": 53398278,
        "Lottery/Sweepstakes/Inheritance": 48642332, "Employment": 42618705,
        "Civil Matter": 20242867, "Harassment/Threats of Violence": 19866654,
        "Misrepresentation": 12371573, "IPR/Copyright and Counterfeit": 10293307,
        "Ransomware": 8965847, "Denial of Service/TDoS": 7598198,
        "Charity": 2214383, "Malware/Scareware/Virus": 2009119,
        "Re-shipping": 1772692, "Gambling": 1458118, "Health Care Related": 1128838,
        "Crimes Against Children": 975311, "Hacktivist": 129000, "Terrorism": 49589,
        "No Lead Value": 0,
    },
    2020: {
        "BEC/EAC": 1866642107, "Confidence Fraud/Romance": 600249821,
        "Investment": 336469000, "Non-Payment/Non-Delivery": 265011249,
        "Identity Theft": 219484699, "Spoofing": 216513728,
        "Real Estate/Rental": 213196082, "Personal Data Breach": 194473055,
        "Tech Support": 146477709, "Credit Card Fraud": 129820792,
        "Corporate Data Breach": 128916648, "Government Impersonation": 109938030,
        "Other": 101523082, "Advanced Fee": 83215405, "Extortion": 70935939,
        "Employment": 62314015, "Lottery/Sweepstakes/Inheritance": 61111319,
        "Phishing/Vishing/Smishing/Pharming": 54241075, "Overpayment": 51039922,
        "Ransomware": 29157405, "Health Care Related": 29042515,
        "Civil Matter": 24915958, "Misrepresentation": 19707242,
        "Malware/Scareware/Virus": 6904054, "Harassment/Threats of Violence": 6547449,
        "IPR/Copyright/Counterfeit": 5910617, "Charity": 4428766, "Gambling": 3961508,
        "Re-shipping": 3095265, "Crimes Against Children": 660044,
        "Denial of Service/TDoS": 512127, "Hacktivist": 50, "Terrorism": 0,
    },
    2021: {
        "BEC/EAC": 2395953296, "Investment": 1455943193,
        "Confidence Fraud/Romance": 956039740, "Personal Data Breach": 517021289,
        "Real Estate/Rental": 350328166, "Tech Support": 347657432,
        "Non-Payment/Non-Delivery": 337493071, "Identity Theft": 278267918,
        "Credit Card Fraud": 172998385, "Corporate Data Breach": 151568225,
        "Government Impersonation": 142643253,  # source printed "$142,643.253" (typo, treated as full dollars)
        "Advanced Fee": 98694137, "Civil Matter": 85049939, "Spoofing": 82169806,
        "Other": 75837524, "Lottery/Sweepstakes/Inheritance": 71289089,
        "Extortion": 60577741, "Ransomware": 49207908, "Employment": 47231023,
        "Phishing/Vishing/Smishing/Pharming": 44213707, "Overpayment": 33407671,
        "Computer Intrusion": 19603037, "IPR/Copyright/Counterfeit": 16365011,
        "Health Care Related": 7042942, "Malware/Scareware/Virus": 5596889,
        "Terrorism/Threats of Violence": 4390720, "Gambling": 1940237,
        "Re-shipping": 631466, "Denial of Service/TDoS": 217981,
        "Crimes Against Children": 198950,
    },
    2022: {
        "Investment": 3311742206, "BEC": 2742354049, "Tech Support": 806551993,
        "Personal Data Breach": 742438136, "Confidence/Romance": 735882192,
        "Data Breach": 459321859, "Real Estate": 396932821,
        "Non-Payment/Non-Delivery": 281770073, "Credit Card/Check Fraud": 264148905,
        "Government Impersonation": 240553091, "Identity Theft": 189205793,
        "Other": 117686789, "Spoofing": 107926252, "Advanced Fee": 104325444,
        "Lottery/Sweepstakes/Inheritance": 83602376, "SIM Swap": 72652571,
        "Extortion": 54335128, "Employment": 52204269, "Phishing": 52089159,
        "Overpayment": 38335772, "Ransomware": 34353237, "Botnet": 17099378,
        "Malware": 9326482, "Harassment/Stalking": 5621402,
        "Threats of Violence": 4972099, "IPR/Copyright/Counterfeit": 4591177,
        "Crimes Against Children": 577464,
    },
    2023: {
        "Investment": 4570275683, "BEC": 2946830270, "Tech Support": 924512658,
        "Personal Data Breach": 744219879, "Confidence/Romance": 652544805,
        "Data Breach": 534397222, "Government Impersonation": 394050518,
        "Non-payment/Non-Delivery": 309648416, "Other": 240053059,
        "Credit Card/Check Fraud": 173627614, "Real Estate": 145243348,
        "Advanced Fee": 134516577, "Identity Theft": 126203809,
        "Lottery/Sweepstakes/Inheritance": 94502836, "Extortion": 74821835,
        "Employment": 70234079, "Ransomware": 59641384, "SIM Swap": 48798103,
        "Overpayment": 27955195, "Botnet": 22422708, "Phishing/Spoofing": 18728550,
        "Threats of Violence": 13531178, "Harassment/Stalking": 9677332,
        "IPR/Copyright and Counterfeit": 7555329, "Crimes Against Children": 2031485,
        "Malware": 1213317,
    },
    2024: {
        "Investment": 6570639864, "Business Email Compromise": 2770151146,
        "Tech Support": 1464755976, "Personal Data Breach": 1453296303,
        "Non-Payment/Non-Delivery": 785436888, "Confidence/Romance": 672009052,
        "Government Impersonation": 405624084, "Data Breach": 364855818,
        "Other": 280278325, "Employment": 264223271,
        "Credit Card/Check Fraud": 199889841, "Identity Theft": 174354745,
        "Real Estate": 173586820, "Extortion": 143185736,
        "Lottery/Sweepstakes/Inheritance": 102212250, "Advanced Fee": 102074512,
        "Phishing/Spoofing": 70013036, "SIM Swap": 25983946, "Overpayment": 21452521,
        "Ransomware": 12473156, "Harassment/Stalking": 10611223, "Botnet": 8860202,
        "IPR/Copyright and Counterfeit": 8715512, "Threats of Violence": 1842186,
        "Malware": 1365945, "Crimes Against Children": 519424,
    },
    2025: {
        "Investment": 8648617756, "Business Email Compromise": 3046598558,
        "Tech/Customer Support": 2134675818, "Personal Data Breach": 1314923988,
        "Confidence/Romance": 929287469, "Government Impersonation": 797943193,
        "Other": 512146819, "Non-Payment/Non-Delivery": 503373587,
        "Data Breach": 435240992, "Employment": 362934762,
        "Credit Card/Check Fraud": 282670235, "Real Estate": 275110419,
        "Phishing/Spoofing": 215843126, "Lottery/Sweepstakes/Inheritance": 194147851,
        "Identity Theft": 185832657, "Advanced Fee": 155910852,
        "Extortion": 122499133, "Ransomware": 32320105, "Harassment/Stalking": 27707167,
        "IPR/Copyright and Counterfeit": 26667006, "Overpayment": 22898075,
        "Malware": 19370572, "SIM Swap": 17366758, "Botnet": 13859049,
        "Threats of Violence": 9509532, "Charity": 7907609,
    },
}

# ---------------------------------------------------------------------------
# Canonicalization: map each year's raw category label to a stable name so
# the same underlying crime type lines up across years despite relabeling.
# Categories that were merged/split at a specific year keep that fact in the
# canonical name rather than being silently blended.
# ---------------------------------------------------------------------------

def canonicalize(raw: str) -> str:
    r = raw.lower()
    if "phishing" in r:
        return "Phishing/Spoofing (merged 2023+)" if "spoofing" in r else "Phishing"
    if r == "spoofing":
        return "Spoofing (standalone, 2016-2022)"
    if "bec" in r or "business email compromise" in r:
        return "BEC/EAC"
    if "non-payment" in r or "non-delivery" in r:
        return "Non-Payment/Non-Delivery"
    if r in ("overpayment", "419/overpayment"):
        return "Overpayment/419"
    if r == "personal data breach":
        return "Personal Data Breach"
    if r == "corporate data breach":
        return "Corporate Data Breach"
    if r == "data breach":
        return "Data Breach (merged category, 2022+)"
    if "identity theft" in r:
        return "Identity Theft"
    if r == "extortion":
        return "Extortion"
    if "tech support" in r or "tech/customer support" in r:
        return "Tech Support"
    if r == "investment":
        return "Investment"
    if "confidence" in r or "romance" in r:
        return "Confidence Fraud/Romance"
    if "credit card" in r:
        return "Credit Card/Check Fraud"
    if "government impersonation" in r:
        return "Government Impersonation"
    if "real estate" in r:
        return "Real Estate/Rental"
    if "advanced fee" in r:
        return "Advanced Fee"
    if r == "employment":
        return "Employment"
    if "malware" in r or r == "virus":
        return "Malware/Scareware/Virus"
    if "denial of service" in r:
        return "Denial of Service/TDoS"
    if "crimes against children" in r:
        return "Crimes Against Children"
    if "civil matter" in r:
        return "Civil Matter"
    if "re-ship" in r or "re-shipping" in r:
        return "Re-shipping"
    if r == "charity":
        return "Charity"
    if "health care" in r:
        return "Health Care Related"
    if r == "gambling":
        return "Gambling"
    if r == "terrorism":
        return "Terrorism (standalone, pre-2021)"
    if "hacktivist" in r:
        return "Hacktivist"
    if "ipr" in r or "copyright" in r:
        return "IPR/Copyright and Counterfeit"
    if "lottery" in r:
        return "Lottery/Sweepstakes/Inheritance"
    if "harassment" in r or "threats of violence" in r or "terrorism/threats" in r:
        return "Harassment/Threats of Violence (merged w/ Terrorism, 2021+)"
    if r == "auction":
        return "Auction (2015 only)"
    if "no lead value" in r:
        return "No Lead Value"
    if "misrepresentation" in r:
        return "Misrepresentation"
    if "sim swap" in r:
        return "SIM Swap"
    if "botnet" in r:
        return "Botnet"
    if "ransomware" in r:
        return "Ransomware"
    if "computer intrusion" in r:
        return "Computer Intrusion (2021 only)"
    if "criminal forums" in r:
        return "Criminal Forums (2015 only)"
    return raw


def build_long_df(data: dict, value_name: str) -> pd.DataFrame:
    rows = []
    for year, categories in data.items():
        for raw_label, value in categories.items():
            rows.append({
                "year": year,
                "category_raw": raw_label,
                "category": canonicalize(raw_label),
                value_name: value,
            })
    return pd.DataFrame(rows)


def analyze(counts_long: pd.DataFrame, losses_long: pd.DataFrame) -> None:
    """Derived metrics: avg loss/complaint by category, growth 2015->2025,
    and volume-vs-loss divergence. Aggregates category_raw duplicates within
    a (year, category) group first (relevant for the 2021 Other/Terrorism tie)."""
    counts_agg = counts_long.groupby(["year", "category"], as_index=False)["count"].sum()
    losses_agg = losses_long.groupby(["year", "category"], as_index=False)["loss_usd"].sum()
    merged = pd.merge(counts_agg, losses_agg, on=["year", "category"], how="inner")
    merged["avg_loss_per_complaint"] = merged["loss_usd"] / merged["count"]

    print("\n=== Average loss per complaint, by category, first vs last year observed ===")
    first_last = []
    for cat, g in merged.groupby("category"):
        g = g.sort_values("year")
        if len(g) < 2:
            continue
        first, last = g.iloc[0], g.iloc[-1]
        first_last.append({
            "category": cat,
            "first_year": int(first["year"]), "first_avg_loss": round(first["avg_loss_per_complaint"]),
            "last_year": int(last["year"]), "last_avg_loss": round(last["avg_loss_per_complaint"]),
            "count_growth_x": round(last["count"] / first["count"], 2) if first["count"] else None,
            "loss_growth_x": round(last["loss_usd"] / first["loss_usd"], 2) if first["loss_usd"] else None,
        })
    fl_df = pd.DataFrame(first_last).sort_values("loss_growth_x", ascending=False)
    print(fl_df.to_string(index=False))

    print("\n=== 2025 avg loss per complaint, all categories, sorted descending ===")
    last_year = merged["year"].max()
    print(
        merged[merged["year"] == last_year]
        .sort_values("avg_loss_per_complaint", ascending=False)
        [["category", "count", "loss_usd", "avg_loss_per_complaint"]]
        .to_string(index=False)
    )

    print("\n=== Year-over-year count for Non-Payment/Non-Delivery and Extortion (non-monotonic categories) ===")
    for cat in ["Non-Payment/Non-Delivery", "Extortion"]:
        g = counts_agg[counts_agg["category"] == cat].sort_values("year")
        print(f"\n{cat}:")
        print(g[["year", "count"]].to_string(index=False))


def growth_family(canonical: str) -> str:
    """Coarser grouping than canonicalize(), for growth-decomposition only:
    merges lineages that were renamed/reshuffled but are clearly the same
    underlying line item (unlike canonicalize(), which deliberately keeps
    pre/post Phishing+Spoofing merge separate for transparency)."""
    if canonical in (
        "Phishing", "Phishing/Spoofing (merged 2023+)",
        "Spoofing (standalone, 2016-2022)",
    ):
        return "Phishing/Spoofing (family)"
    if canonical in ("Corporate Data Breach", "Data Breach (merged category, 2022+)"):
        return "Data Breach - Corporate (family)"
    return canonical


def decompose_growth(losses_long: pd.DataFrame, counts_long: pd.DataFrame,
                      year_start: int, year_end: int) -> None:
    print(f"\n{'='*90}\nGROWTH DECOMPOSITION: {year_start} -> {year_end}\n{'='*90}")

    for label, long_df, value_col in [("LOSS ($)", losses_long, "loss_usd"),
                                        ("COUNT", counts_long, "count")]:
        df = long_df.copy()
        df["family"] = df["category"].map(growth_family)
        agg = df.groupby(["year", "family"], as_index=False)[value_col].sum()
        piv = agg.pivot(index="family", columns="year", values=value_col)

        if year_start not in piv.columns or year_end not in piv.columns:
            continue
        both = piv[[year_start, year_end]].dropna()
        both["delta"] = both[year_end] - both[year_start]
        both["growth_x"] = both[year_end] / both[year_start].replace(0, pd.NA)
        total_delta = both["delta"].sum()
        both["pct_of_total_growth"] = 100 * both["delta"] / total_delta
        both = both.sort_values("delta", ascending=False)

        new_families = piv[piv[year_start].isna() & piv[year_end].notna()].index.tolist()
        dropped_families = piv[piv[year_start].notna() & piv[year_end].isna()].index.tolist()

        print(f"\n--- {label}: total change among categories present in both years "
              f"= {total_delta:,.0f} ---")
        print(both[[year_start, year_end, "delta", "growth_x", "pct_of_total_growth"]]
              .round(2).to_string())

        n = len(both)
        cum = both["pct_of_total_growth"].cumsum()
        n_for_80 = int((cum < 80).sum()) + 1
        print(f"\n{n} categories compared. Top {n_for_80} of {n} account for "
              f"~80% of the {label} growth.")
        declining_or_flat = both[both["growth_x"] < 1.2]
        print(f"Categories with growth factor < 1.2x over the period "
              f"({len(declining_or_flat)} of {n}): {declining_or_flat.index.tolist()}")
        if new_families:
            print(f"New categories in {year_end} (not present in {year_start}): {new_families}")
        if dropped_families:
            print(f"Categories present in {year_start} but not {year_end}: {dropped_families}")


def stagnant_category_detail(losses_long: pd.DataFrame, counts_long: pd.DataFrame,
                              year_start: int, year_end: int) -> None:
    """For categories whose TOTAL loss barely moved (growth < 1.2x) between
    year_start and year_end, show what count and avg-loss-per-complaint did
    separately - a flat total can hide count falling while severity rises,
    or count rising while severity falls."""
    losses_f = losses_long.copy()
    losses_f["family"] = losses_f["category"].map(growth_family)
    counts_f = counts_long.copy()
    counts_f["family"] = counts_f["category"].map(growth_family)

    loss_agg = losses_f.groupby(["year", "family"], as_index=False)["loss_usd"].sum()
    count_agg = counts_f.groupby(["year", "family"], as_index=False)["count"].sum()

    loss_piv = loss_agg.pivot(index="family", columns="year", values="loss_usd")
    count_piv = count_agg.pivot(index="family", columns="year", values="count")

    both = loss_piv[[year_start, year_end]].dropna()
    loss_growth = both[year_end] / both[year_start]
    stagnant = loss_growth[loss_growth < 1.2].index.tolist()

    print(f"\n--- Stagnant-loss categories ({year_start}->{year_end}), "
          f"decomposed into count growth x avg-loss-per-complaint growth ---")
    rows = []
    for fam in stagnant:
        c0, c1 = count_piv.loc[fam, year_start], count_piv.loc[fam, year_end]
        l0, l1 = loss_piv.loc[fam, year_start], loss_piv.loc[fam, year_end]
        if pd.isna(c0) or pd.isna(c1) or c0 == 0 or c1 == 0:
            continue
        count_x = c1 / c0
        avg0, avg1 = l0 / c0, l1 / c1
        avg_x = avg1 / avg0
        rows.append({
            "family": fam, "loss_growth_x": round(l1 / l0, 2),
            "count_growth_x": round(count_x, 2),
            "avg_loss_per_complaint_growth_x": round(avg_x, 2),
            f"avg_{year_start}": round(avg0), f"avg_{year_end}": round(avg1),
        })
    print(pd.DataFrame(rows).to_string(index=False))


# Categories that showed up as top contributors to LOSS growth in BOTH the
# 2020->2025 and 2022->2025 decompositions (see decompose_growth output):
# Investment and Tech Support are the two consistent #1/#2 in both windows;
# Personal Data Breach and Government Impersonation are consistently top-5;
# BEC/EAC is included because it's one of the largest absolute-dollar
# categories throughout even though its growth rate slowed after 2022.
GROWTH_LEADERS = [
    "Investment", "Tech Support", "BEC/EAC",
    "Personal Data Breach", "Government Impersonation",
]


def pooled_rest_dynamics(losses_long: pd.DataFrame, counts_long: pd.DataFrame) -> None:
    print(f"\n{'='*90}\nPOOLED 'REST' DYNAMICS (all categories EXCLUDING {GROWTH_LEADERS})\n{'='*90}")

    for label, long_df, value_col in [("LOSS ($)", losses_long, "loss_usd"),
                                        ("COUNT", counts_long, "count")]:
        df = long_df.copy()
        df["family"] = df["category"].map(growth_family)
        agg = df.groupby(["year", "family"], as_index=False)[value_col].sum()

        totals = agg.groupby("year")[value_col].sum().rename("total")
        leaders = (agg[agg["family"].isin(GROWTH_LEADERS)]
                   .groupby("year")[value_col].sum().rename("leaders"))
        out = pd.concat([totals, leaders], axis=1).fillna(0)
        out["rest"] = out["total"] - out["leaders"]
        out["rest_share_of_total_pct"] = 100 * out["rest"] / out["total"]

        print(f"\n--- {label}: total vs. leaders vs. rest, by year ---")
        print(out.round(1).to_string())

        first_year, last_year = out.index.min(), out.index.max()
        rest_growth = out.loc[last_year, "rest"] / out.loc[first_year, "rest"]
        total_growth = out.loc[last_year, "total"] / out.loc[first_year, "total"]
        leaders_growth = out.loc[last_year, "leaders"] / out.loc[first_year, "leaders"]
        print(f"\n{first_year}->{last_year}: TOTAL grew {total_growth:.2f}x, "
              f"LEADERS grew {leaders_growth:.2f}x, REST (pooled) grew {rest_growth:.2f}x")

        if value_col == "loss_usd":
            counts_df = counts_long.copy()
            counts_df["family"] = counts_df["category"].map(growth_family)
            c_agg = counts_df.groupby(["year", "family"], as_index=False)["count"].sum()
            c_totals = c_agg.groupby("year")["count"].sum().rename("total_count")
            c_leaders = (c_agg[c_agg["family"].isin(GROWTH_LEADERS)]
                         .groupby("year")["count"].sum().rename("leaders_count"))
            c_out = pd.concat([c_totals, c_leaders], axis=1).fillna(0)
            c_out["rest_count"] = c_out["total_count"] - c_out["leaders_count"]
            rest_avg = (out["rest"] / c_out["rest_count"]).rename("rest_avg_loss_per_complaint")
            print("\n--- Pooled 'rest': avg loss per complaint, by year ---")
            print(rest_avg.round(0).to_string())
            print(f"\n{first_year}->{last_year}: pooled-rest avg loss/complaint grew "
                  f"{rest_avg.loc[last_year] / rest_avg.loc[first_year]:.2f}x")


# "Investment fraud with cryptocurrency reference" / "Crypto Investment Fraud"
# loss figures, called out narratively (not in the main crime-type table) in
# each report's dedicated Investment/Crypto section. Figures are as printed
# (mostly rounded to 2-3 sig figs in the source text, e.g. "$5.8 billion");
# cross-checked via each report's own YoY % claim against the prior year's
# figure from that prior year's own report - they agree to within rounding,
# see conversation notes. 2021 figure sourced from the 2022 report's
# retrospective comparison (2021 report itself did not break out crypto
# specifically within Investment).
CRYPTO_INVESTMENT_LOSS_USD = {
    2021: 907_000_000,
    2022: 2_570_000_000,
    2023: 3_960_000_000,
    2024: 5_800_000_000,
    2025: 7_228_000_000,
}


def crypto_investment_share(losses_long: pd.DataFrame) -> None:
    print(f"\n{'='*90}\nIS INVESTMENT FRAUD'S GROWTH DRIVEN BY CRYPTO?\n{'='*90}")
    total_investment = (
        losses_long[losses_long["category"] == "Investment"]
        .groupby("year")["loss_usd"].sum()
    )
    rows = []
    for year, crypto in sorted(CRYPTO_INVESTMENT_LOSS_USD.items()):
        total = total_investment.get(year)
        if total is None:
            continue
        non_crypto = total - crypto
        rows.append({
            "year": year, "total_investment_loss": total,
            "crypto_investment_loss": crypto, "non_crypto_investment_loss": non_crypto,
            "crypto_share_pct": round(100 * crypto / total, 1),
        })
    df = pd.DataFrame(rows).set_index("year")
    print(df.to_string())

    y0, y1 = df.index.min(), df.index.max()
    total_delta = df.loc[y1, "total_investment_loss"] - df.loc[y0, "total_investment_loss"]
    crypto_delta = df.loc[y1, "crypto_investment_loss"] - df.loc[y0, "crypto_investment_loss"]
    non_crypto_delta = df.loc[y1, "non_crypto_investment_loss"] - df.loc[y0, "non_crypto_investment_loss"]
    print(f"\n{y0}->{y1}: total Investment loss grew "
          f"{df.loc[y1,'total_investment_loss']/df.loc[y0,'total_investment_loss']:.2f}x "
          f"(+${total_delta:,.0f}).")
    print(f"  Crypto-specific grew "
          f"{df.loc[y1,'crypto_investment_loss']/df.loc[y0,'crypto_investment_loss']:.2f}x "
          f"(+${crypto_delta:,.0f}, {100*crypto_delta/total_delta:.1f}% of the total increase)")
    print(f"  Non-crypto grew "
          f"{df.loc[y1,'non_crypto_investment_loss']/df.loc[y0,'non_crypto_investment_loss']:.2f}x "
          f"(+${non_crypto_delta:,.0f}, {100*non_crypto_delta/total_delta:.1f}% of the total increase)")


# US CPI-U annual average index (US city average, all items, 1982-84=100).
# Source: BLS-derived historical table via usinflationcalculator.com, fetched
# 2026-07-09. Used to deflate nominal dollar figures to constant 2025 dollars.
CPI_U_ANNUAL_AVERAGE = {
    2015: 237.017, 2016: 240.007, 2017: 245.120, 2018: 251.107,
    2019: 255.657, 2020: 258.811, 2021: 270.970, 2022: 292.655,
    2023: 304.702, 2024: 313.689, 2025: 321.943,
}


def inflation_adjusted_rest_growth(losses_long: pd.DataFrame) -> None:
    print(f"\n{'='*90}\nPOOLED 'REST' LOSS GROWTH, INFLATION-ADJUSTED (constant 2025 USD)\n{'='*90}")

    df = losses_long.copy()
    df["family"] = df["category"].map(growth_family)
    agg = df.groupby(["year", "family"], as_index=False)["loss_usd"].sum()

    totals = agg.groupby("year")["loss_usd"].sum().rename("total_nominal")
    leaders = (agg[agg["family"].isin(GROWTH_LEADERS)]
               .groupby("year")["loss_usd"].sum().rename("leaders_nominal"))
    out = pd.concat([totals, leaders], axis=1).fillna(0)
    out["rest_nominal"] = out["total_nominal"] - out["leaders_nominal"]

    base_cpi = CPI_U_ANNUAL_AVERAGE[2025]
    out["cpi"] = out.index.map(CPI_U_ANNUAL_AVERAGE)
    out["deflator_to_2025"] = base_cpi / out["cpi"]
    out["rest_real_2025usd"] = out["rest_nominal"] * out["deflator_to_2025"]
    out["leaders_real_2025usd"] = out["leaders_nominal"] * out["deflator_to_2025"]
    out["total_real_2025usd"] = out["total_nominal"] * out["deflator_to_2025"]

    print(out[["rest_nominal", "deflator_to_2025", "rest_real_2025usd"]].round(1).to_string())

    y0, y1 = out.index.min(), out.index.max()
    cum_inflation_pct = 100 * (CPI_U_ANNUAL_AVERAGE[y1] / CPI_U_ANNUAL_AVERAGE[y0] - 1)
    print(f"\nCumulative inflation {y0}->{y1} (CPI-U): {cum_inflation_pct:.1f}%")

    for series, label in [("rest", "POOLED REST"), ("leaders", "LEADERS"), ("total", "TOTAL")]:
        nominal_growth = out.loc[y1, f"{series}_nominal"] / out.loc[y0, f"{series}_nominal"]
        real_growth = out.loc[y1, f"{series}_real_2025usd"] / out.loc[y0, f"{series}_real_2025usd"]
        print(f"{label}: nominal growth {y0}->{y1} = {nominal_growth:.2f}x   "
              f"real (inflation-adjusted) growth = {real_growth:.2f}x   "
              f"(${out.loc[y0, f'{series}_real_2025usd']:,.0f} -> "
              f"${out.loc[y1, f'{series}_real_2025usd']:,.0f} in constant 2025 dollars)")


def main():
    counts_long = build_long_df(COUNTS, "count")
    losses_long = build_long_df(LOSSES, "loss_usd")

    counts_wide = counts_long.pivot_table(
        index="year", columns="category", values="count", aggfunc="sum"
    )
    losses_wide = losses_long.pivot_table(
        index="year", columns="category", values="loss_usd", aggfunc="sum"
    )

    out_dir = "data/raw/ic3"
    counts_long.sort_values(["year", "count"], ascending=[True, False]).to_csv(
        f"{out_dir}/crime_type_counts_long.csv", index=False
    )
    losses_long.sort_values(["year", "loss_usd"], ascending=[True, False]).to_csv(
        f"{out_dir}/crime_type_losses_long.csv", index=False
    )
    counts_wide.to_csv(f"{out_dir}/crime_type_counts_wide.csv")
    losses_wide.to_csv(f"{out_dir}/crime_type_losses_wide.csv")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 12)

    print("=== Crime type COUNTS, wide (year x canonical category), core columns ===")
    core_cols = [
        "Phishing", "Phishing/Spoofing (merged 2023+)", "BEC/EAC",
        "Non-Payment/Non-Delivery", "Extortion", "Investment",
        "Confidence Fraud/Romance", "Ransomware", "Tech Support",
    ]
    print(counts_wide[[c for c in core_cols if c in counts_wide.columns]])

    print("\n=== Crime type LOSSES ($), wide (year x canonical category), core columns ===")
    print(losses_wide[[c for c in core_cols if c in losses_wide.columns]].round(0))

    print(f"\nFull long-format tables written to {out_dir}/crime_type_counts_long.csv "
          f"and {out_dir}/crime_type_losses_long.csv")
    print(f"Full wide (pivoted) tables written to {out_dir}/crime_type_counts_wide.csv "
          f"and {out_dir}/crime_type_losses_wide.csv")
    print(f"\n{len(counts_long)} count rows, {len(losses_long)} loss rows, "
          f"years covered: {sorted(COUNTS.keys())}")
    print("2011-2013 excluded: no unified crime-type table in those reports "
          "(see IC3_SUMMARY.md). 2014 report missing from archive.")

    analyze(counts_long, losses_long)

    decompose_growth(losses_long, counts_long, 2020, 2025)
    decompose_growth(losses_long, counts_long, 2022, 2025)
    stagnant_category_detail(losses_long, counts_long, 2020, 2025)
    stagnant_category_detail(losses_long, counts_long, 2022, 2025)
    pooled_rest_dynamics(losses_long, counts_long)
    crypto_investment_share(losses_long)
    inflation_adjusted_rest_growth(losses_long)


if __name__ == "__main__":
    main()
