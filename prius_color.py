# import the necessary packages
import argparse
import json
import os

import cv2
import imutils
import numpy as np
from ColorLabeler import ColorLabeler
from imutils import contours

# construct the argument parse and parse the arguments


prius_list = [
	"surfie green [2]",  ##0C7A79
	"ming [2]",  ##407577
	"mosque [2]",  ##036A6E
	"muse",  ##2E696D
	"paradiso [2]",  ##317D82
	"retro",  ##1B5256
	"deep teal [3]",  ##00555A
	"PMS3165",  ##00565B
	"PMS5473",  ##26686D
	"deep aqua",  ##08787F
	"william [2]",  ##3A686C
	"casal [2]",  ##2F6168
	"petrol",  ##005F6A
	"PMS315",  ##006B77
	"blue lagoon [3]",  ##017987
	"metallic seaweed",  ##0A7E8C
	"deep arctic blue",  ##004E59
	"dauntless",  ##166F7F
	"maestro",  ##005E6D
	"boomtown",  ##346672
	"kitsch",  ##006C7F
	"breaker bay [2]",  ##5DA19F
	"cyan [5]",  ##008B8B
	"dark cyan [3]",  ##008B8B
	"strong cyan",  ##00A8A8
	"patriot",  ##4F9292
	"java [2]",  ##259797
	"moderate cyan",  ##4AA8A8
	"blue chill [2]",  ##408F90
	"viridian green [3]",  ##009698
	"PMS5483",  ##609191
	"juniper [2]",  ##6D9292
	"PMS320",  ##009EA0
	"desaturated cyan",  ##669999
	"cadet blue [10]",  ##5F9F9F
	"grayish cyan",  ##7DA8A8
	"PMS321",  ##008789
	"steel teal",  ##5F8A8B
	"bounce",  ##679394
	"cadet [2]",  ##5F9EA0
	"turquoise [8]",  ##00868B
	"kumutoto",  ##78AFB2
	"half baked [2]",  ##558F93
	"wishlist",  ##659295
	"neptune [2]",  ##77A8AB
	"fountain blue [2]",  ##65ADB2
	"paradiso [2]",  ##488084
	"PMS3145",  ##00848E
	"ziggurat [2]",  ##81A6AA
	"hullabaloo",  ##008B97
	"yabbadabbadoo",  ##008B97
	"such fun",  ##489EA8
	"PMS3125",  ##00B7C6
	"retreat",  ##39909B
	"gumbo [2]",  ##7CA1A6
	"PMS3135",  ##009BAA
	"moderate arctic blue",  ##4A9CA8
	"turquoise blue [5]",  ##06B1C4
	"PMS631",  ##54B7C6
	"cerulean [5]",  ##05B8CC
	"seeker",  ##0092A5
	"wot eva",  ##4495A4
	"pelorous [2]",  ##3EABBF
	"strong arctic blue",  ##0093A8
	"scooter [2]",  ##308EA0
	"eastern blue [2]",  ##1E9AB0
	"PMS312",  ##00ADC6
	"glacier [2]",  ##78B1BF
	"toto",  ##519DAF
	"PMS632",  ##00A0BA
	"teal blue [4]",  ##01889F
	"viking [2]",  ##4DB1C8
	"onepoto",  ##81D3D1
	"PMS318",  ##93DDDB
	"java [2]",  ##1FC2C2
	"robin's egg blue [2]",  ##00CCCC
	"robin egg blue [4]",  ##00CCCC
	"cyan [5]",  ##00CDCD
	"vivid cyan",  ##00E7E7
	"dark slate grey [5]",  ##79CDCD
	"medium turquoise [3]",  ##70DBDB
	"brilliant cyan",  ##51E7E7
	"pale turquoise [7]",  ##96CDCD
	"PMS629",  ##B2D8D8
	"light cyan [6]",  ##8BE7E7
	"pale light grayish cyan",  ##B8E7E7
	"turquoise [8]",  ##ADEAEA
	"dark turquoise [5]",  ##00CED1
	"turquoise dark",  ##00CED1
	"PMS319",  ##4CCED1
	"bright light blue",  ##26F7FD
	"morning glory [2]",  ##9EDEE0
	"half kumutoto",  ##9CC8CA
	"PMS3105",  ##7FD6DB
	"PMS2975",  ##BAE0E2
	"neptune [2]",  ##7CB7BB
	"sea serpent",  ##4BC7CF
	"aquamarine [9]",  ##78DBE2
	"aquamarine blue",  ##71D9E2
	"PMS304",  ##A5DDE2
	"turquoise blue [5]",  ##77DDE7
	"cadet blue [10]",  ##7AC5CD
	"PMS630",  ##8CCCD3
	"aqua blue",  ##02D8E9
	"PMS636",  ##99D6DD
	"PMS3115",  ##2DC6D6
	"half baked [2]",  ##85C4CC
	"PMS310",  ##72D1DD
	"powder blue [4]",  ##B0E0E6
	"light arctic blue",  ##8BDCE7
	"robin's egg",  ##6DEDFD
	"spray [2]",  ##79DEEC
	"viking [2]",  ##64CCDB
	"brilliant arctic blue",  ##51D5E7
	"blizzard blue [2]",  ##A3E3ED
	"blue lagoon [3]",  ##ACE5EE
	"light brilliant arctic blue",  ##65ECFF
	"PMS311",  ##28C4D8
	"charlotte [2]",  ##A4DCE6
	"scooter [2]",  ##2EBFD4
	"PMS637",  ##6BC9DB
	"medium sky blue",  ##80DAEB
	"sky blue [9]",  ##80DAEB
	"PMS305",  ##70CEE2
	"vivid arctic blue",  ##00CAE7
	"luminous vivid arctic blue",  ##00DFFF
	"refresh",  ##71B8CA
	"non photo blue",  ##A4DDED
	"meltwater",  ##6EAEC0
	"hippie blue [2]",  ##49889A
	"PMS313",  ##0099B5
	"PMS549",  ##5E99AA
	"awash",  ##739CA9
	"PMS638",  ##00B5D6
	"PMS314",  ##00829B
	"blue [12]",  ##0093AF
	"smalt blue [2]",  ##51808F
	"tax break [2]",  ##51808F
	"horizon [2]",  ##648894
	"pacific blue [2]",  ##1CA9C9
	"blue green [3]",  ##199EBD
	"aquarius",  ##43A8C5
	"PMS801",  ##00AACC
	"ball blue",  ##21ABCD
	"moderate cerulean",  ##4A91A8
	"bondi blue [3]",  ##0095B6
	"PMS639",  ##00A0C4
	"gothic [2]",  ##6D92A1
	"endorphin",  ##4190AD
	"bright cerulean",  ##1DACD6
	"cerulean [5]",  ##1DACD6
	"dirty blue",  ##3F829D
	"PMS640",  ##008CB2
	"boston blue [2]",  ##438EAC
	"greyblue",  ##77A1B5
	"bluegrey",  ##85A3B2
	"PMS801 2X",  ##0089AF
	"abacus",  ##768993
	"peacock",  ##33A1C9
	"moonstone blue",  ##73A9C2
	"PMS306 2X",  ##00A3D1
	"shakespeare [2]",  ##4EABD1
	"summer sky",  ##38B0DE
	"cyan [5]",  ##00B7EB
	"bowie",  ##0084AC
	"waterfront",  ##3E7F9D
	"blue moon",  ##7296AB
	"deep sky blue [5]",  ##009ACD
	"PMS2995",  ##00A5DB
	"vivid cerulean [2]",  ##00AEE7
	"tsunami",  ##6B8393
	"hemisphere",  ##4E93BA
	"PMS299",  ##00A3DD
	"moderate cornflower blue",  ##4A85A8
	"freefall",  ##1D95C9
	"PMS2915",  ##60AFDD
	"picton blue [2]",  ##45B1E8
	"wedgewood [2]",  ##4E7F9E
	"air force blue [2]",  ##5D8AA8
	"rackley",  ##5D8AA8
	"sky blue [9]",  ##3299CC
	"air superiority blue",  ##72A0C1
	"brilliant cornflower blue",  ##51AFE7
	"ocean",  ##017B92
	"allports [2]",  ##1F6A7D
	"PMS633",  ##007F99
	"teal blue [4]",  ##367588
	"norwester",  ##48798A
	"bismark [2]",  ##486C7A
	"marathon",  ##305563
	"PMS634",  ##00667F
	"jelly bean [3]",  ##44798E
	"PMS3025",  ##00546B
	"undercurrent",  ##365C6C
	"PMS308",  ##00607C
	"calypso [2]",  ##3D7188
	"sea blue [2]",  ##047495
	"lucifer",  ##2E5060
	"blumine [2]",  ##305C71
	"astral [2]",  ##376F89
	"blue sapphire",  ##126180
	"deep sky blue [5]",  ##00688B
	"strong cerulean",  ##007EA8
	"arapawa [2]",  ##274A5D
	"chathams blue [2]",  ##2C5971
	"PMS307",  ##007AA5
	"PMS641",  ##007AA5
	"cg blue",  ##007AA5
	"PMS302",  ##004F6D
	"orient [2]",  ##255B77
	"PMS5405",  ##3F6075
	"steel blue [8]",  ##236B8E
	"celadon blue",  ##007BA7
	"cerulean [5]",  ##007BA7
	"deep cerulean [2]",  ##007BA7
	"ocean blue [2]",  ##03719C
	"yeehaa",  ##006C98
	"sky blue [9]",  ##4A708B
	"wavelength",  ##3C6886
	"PMS3015",  ##00709E
	"metallic blue",  ##4F738E
	"neon blue [2]",  ##04D9FF
	"french pass [2]",  ##A4D2E0
	"glacier [2]",  ##80B3C4
	"parachute",  ##65B8D1
	"very light cerulean",  ##9EE7FF
	"PMS306",  ##00BCE2
	"light cerulean",  ##8BD0E7
	"winter wizard",  ##A0E6FF
	"anakiwa [2]",  ##9DE5FF
	"fresh air",  ##A6E7FF
	"PMS2985",  ##51BFE2
	"brilliant cerulean",  ##51C2E7
	"light brilliant cerulean",  ##65D8FF
	"seagull [2]",  ##77B7D0
	"PMS297",  ##82C6E2
	"sky blue [9]",  ##87CEEB
	"bright sky blue",  ##02CCFE
	"vivid sky blue",  ##00CCFF
	"dark sky blue [2]",  ##8CBED6
	"PMS2905",  ##93C6E0
	"baby blue [3]",  ##89CFF0
	"pale cyan [3]",  ##87D3F8
	"PMS298",  ##51B5E0
	"cornflower [6]",  ##93CCEA
	"light cornflower blue [2]",  ##93CCEA
	"light sky blue [6]",  ##8DB6CD
	"malibu [2]",  ##66B7E1
	"spiro disco ball",  ##0FC0FC
	"very light cornflower blue",  ##9EDBFF
	"capri [3]",  ##00BFFF
	"deep sky blue [5]",  ##00BFFF
	"luminous vivid cerulean",  ##00BFFF
	"sky blue deep",  ##00BFFF
	"lightblue",  ##7BC8F6
	"sky blue light",  ##87CEFA
	"light brilliant cornflower blue",  ##65C5FF
	"PMS292",  ##75B2DD
	"sky",  ##82CAFC
]

prius_full = [
	"surfie green [2]",  ##0C7A79
	"bluegreen",  ##017A79
	"deep cyan",  ##005959
	"PMS322",  ##007272
	"skobeloff",  ##007474
	"stormcloud [2]",  ##008080
	"teal [2]",  ##008080
	"dark cyan [3]",  ##275959
	"caesar",  ##1B6767
	"dark slate grey [5]",  ##2F4F4F
	"slate grey dark",  ##2F4F4F
	"deep sea green [2]",  ##095859
	"sea green [9]",  ##095859
	"oracle [2]",  ##395555
	"dark grayish cyan",  ##425959
	"blue stone [2]",  ##016162
	"deep turquoise [2]",  ##017374
	"atoll [2]",  ##2B797A
	"elm [2]",  ##1C7C7D
	#"mako [2]",  ##505555
	"dark cyanish grey",  ##535959
	"PMS445",  ##565959
	"dark aqua",  ##05696B
	"nevada [2]",  ##666F6F
	"PMS320 2X",  ##007F82
	"free spirit",  ##00777B
	"ming [2]",  ##407577
	"timekeeper",  ##41595A
	"mosque [2]",  ##036A6E
	"muse",  ##2E696D
	"paradiso [2]",  ##317D82
	"retro",  ##1B5256
	"deep teal [3]",  ##00555A
	"PMS3165",  ##00565B
	"PMS5473",  ##26686D
	"deep aqua",  ##08787F
	"william [2]",  ##3A686C
	#"river bed [2]",  ##556061
	"PMS3155",  ##006D75
	"espirit",  ##354E51
	"beatnik",  ##1F4F54
	"obelisk",  ##617274
	"sabbatical",  ##547377
	"tax break [2]",  ##496569
	"dark arctic blue",  ##275359
	"casal [2]",  ##2F6168
	#"PMS432",  ##444F51
	"petrol",  ##005F6A
	"PMS315",  ##006B77
	"blue lagoon [3]",  ##017987
	"metallic seaweed",  ##0A7E8C
	"balderdash",  ##465659
	"smalt blue [2]",  ##496267
	"deep arctic blue",  ##004E59
	"gateway",  ##5E7175
	"outer space [4]",  ##414A4C
	"dauntless",  ##166F7F
	"maestro",  ##005E6D
	"boomtown",  ##346672
	"dark grayish cerulean",  ##425459
	"kitsch",  ##006C7F
	#"pale sky [2]",  ##636D70
	"breaker bay [2]",  ##5DA19F
	"destiny",  ##889E9D
	"cyan [5]",  ##008B8B
	"dark cyan [3]",  ##008B8B
	"strong cyan",  ##00A8A8
	"patriot",  ##4F9292
	"java [2]",  ##259797
	"moderate cyan",  ##4AA8A8
	"dark slate grey [5]",  ##528B8B
	"pale turquoise [7]",  ##668B8B
	"blue chill [2]",  ##408F90
	"viridian green [3]",  ##009698
	"PMS5483",  ##609191
	"juniper [2]",  ##6D9292
	"PMS320",  ##009EA0
	"desaturated cyan",  ##669999
	"cadet blue [10]",  ##5F9F9F
	"grayish cyan",  ##7DA8A8
	"trojan",  ##757878
	"PMS321",  ##008789
	#"sirocco [2]",  ##718080
	"azure [8]",  ##838B8B
	"light cyan [6]",  ##7A8B8B
	"submarine [2]",  ##8C9C9C
	"granny smith [2]",  ##84A0A0
	"cyanish grey",  ##9CA8A8
	"meridian",  ##7C7E7E
	"steel teal",  ##5F8A8B
	"bounce",  ##679394
	"cadet [2]",  ##5F9EA0
	"turquoise [8]",  ##00868B
	"kumutoto",  ##78AFB2
	"aurometalsaurus",  ##6E7F80
	"half baked [2]",  ##558F93
	"wishlist",  ##659295
	"crescent",  ##839596
	"neptune [2]",  ##77A8AB
	"fountain blue [2]",  ##65ADB2
	"paradiso [2]",  ##488084
	#"bounty",  ##919FA0
	"PMS3145",  ##00848E
	"ziggurat [2]",  ##81A6AA
	"half smalt blue",  ##5E7C80
	"hullabaloo",  ##008B97
	"yabbadabbadoo",  ##008B97
	"such fun",  ##489EA8
	"PMS3125",  ##00B7C6
	"retreat",  ##39909B
	"gumbo [2]",  ##7CA1A6
	"PMS3135",  ##009BAA
	"moderate arctic blue",  ##4A9CA8
	"turquoise blue [5]",  ##06B1C4
	"PMS631",  ##54B7C6
	"cerulean [5]",  ##05B8CC
	"seeker",  ##0092A5
	"wot eva",  ##4495A4
	#"cool grey [2]",  ##95A3A6
	#"dusted blue",  ##929FA2
	"powder blue [4]",  ##929FA2
	"pelorous [2]",  ##3EABBF
	"strong arctic blue",  ##0093A8
	"scooter [2]",  ##308EA0
	"eastern blue [2]",  ##1E9AB0
	"PMS312",  ##00ADC6
	"gothic [2]",  ##698890
	"glacier [2]",  ##78B1BF
	"toto",  ##519DAF
	"PMS632",  ##00A0BA
	"teal blue [4]",  ##01889F
	"viking [2]",  ##4DB1C8
	"light blue [9]",  ##68838B
	"onepoto",  ##81D3D1
	"PMS318",  ##93DDDB
	"java [2]",  ##1FC2C2
	"robin's egg blue [2]",  ##00CCCC
	"robin egg blue [4]",  ##00CCCC
	"cyan [5]",  ##00CDCD
	"vivid cyan",  ##00E7E7
	"dark slate grey [5]",  ##79CDCD
	"medium turquoise [3]",  ##70DBDB
	"brilliant cyan",  ##51E7E7
	"pale turquoise [7]",  ##96CDCD
	"PMS629",  ##B2D8D8
	"light cyan [6]",  ##8BE7E7
	"pale light grayish cyan",  ##B8E7E7
	"turquoise [8]",  ##ADEAEA
	"dark turquoise [5]",  ##00CED1
	"turquoise dark",  ##00CED1
	"PMS319",  ##4CCED1
	"jungle mist [2]",  ##B0C4C4
	"PMS5445",  ##C4CCCC
	"azure [8]",  ##C1CDCD
	"tiara [2]",  ##C3D1D1
	"PMS552",  ##C4D6D6
	"zumthor [2]",  ##CDD5D5
	"light blue [9]",  ##C0D9D9
	"PMS642",  ##D1D8D8
	"PMS635",  ##BAE0E0
	"surrender",  ##B5B7B7
	"half surrender",  ##CACCCC
	"iron [2]",  ##CBCDCD
	"midwinter mist",  ##CCCECE
	"quarter surrender",  ##D6D8D8
	"athens grey [2]",  ##DCDDDD
	"bright light blue",  ##26F7FD
	"morning glory [2]",  ##9EDEE0
	"half kumutoto",  ##9CC8CA
	"PMS3105",  ##7FD6DB
	"PMS2975",  ##BAE0E2
	"neptune [2]",  ##7CB7BB
	"sea serpent",  ##4BC7CF
	"cut glass",  ##C9DBDC
	"aquamarine [9]",  ##78DBE2
	"aquamarine blue",  ##71D9E2
	"PMS304",  ##A5DDE2
	"tower grey [2]",  ##A9BDBF
	"chi",  ##9FBFC2
	"turquoise blue [5]",  ##77DDE7
	"cadet blue [10]",  ##7AC5CD
	"PMS630",  ##8CCCD3
	"aqua blue",  ##02D8E9
	"PMS636",  ##99D6DD
	"PMS3115",  ##2DC6D6
	"half baked [2]",  ##85C4CC
	"PMS310",  ##72D1DD
	"powder blue [4]",  ##B0E0E6
	"light arctic blue",  ##8BDCE7
	"robin's egg",  ##6DEDFD
	"submarine [2]",  ##BAC7C9
	"spray [2]",  ##79DEEC
	"viking [2]",  ##64CCDB
	"brilliant arctic blue",  ##51D5E7
	"blizzard blue [2]",  ##A3E3ED
	"blue lagoon [3]",  ##ACE5EE
	"light brilliant arctic blue",  ##65ECFF
	"onahau [2]",  ##C2E6EC
	"PMS311",  ##28C4D8
	"charlotte [2]",  ##A4DCE6
	"scooter [2]",  ##2EBFD4
	"PMS637",  ##6BC9DB
	"medium sky blue",  ##80DAEB
	"sky blue [9]",  ##80DAEB
	"PMS5435",  ##AFBCBF
	"PMS305",  ##70CEE2
	"vivid arctic blue",  ##00CAE7
	"quarter powder blue",  ##BEC6C8
	"luminous vivid arctic blue",  ##00DFFF
	"ziggurat [2]",  ##BFDBE2
	"casper [2]",  ##AAB5B8
	"PMS551",  ##A3C1C9
	"geyser [2]",  ##D4DFE2
	"refresh",  ##71B8CA
	"regent st blue [2]",  ##A0CDD9
	"coastal blue",  ##AAB7BB
	"escape",  ##9EC0CA
	"non photo blue",  ##A4DDED
	"grayish cerulean",  ##7D9EA8
	"meltwater",  ##6EAEC0
	"botticelli [2]",  ##92ACB4
	"hippie blue [2]",  ##49889A
	"PMS313",  ##0099B5
	"PMS549",  ##5E99AA
	"awash",  ##739CA9
	#"triple surrender",  ##999C9D
	"PMS638",  ##00B5D6
	"PMS314",  ##00829B
	"blue [12]",  ##0093AF
	"smalt blue [2]",  ##51808F
	"tax break [2]",  ##51808F
	#"regent grey [2]",  ##798488
	"horizon [2]",  ##648894
	"pacific blue [2]",  ##1CA9C9
	"hoki [2]",  ##647D86
	"bluff",  ##718187
	"blue green [3]",  ##199EBD
	"aquarius",  ##43A8C5
	"PMS801",  ##00AACC
	#"half regent grey",  ##939DA1
	"ball blue",  ##21ABCD
	"moderate cerulean",  ##4A91A8
	"bondi blue [3]",  ##0095B6
	"PMS639",  ##00A0C4
	"PMS550",  ##87AFBF
	"steel grey [3]",  ##6F828A
	"gothic [2]",  ##6D92A1
	#"clouded blue",  ##899296
	"endorphin",  ##4190AD
	#"amber grey",  ##929698
	"bright cerulean",  ##1DACD6
	"cerulean [5]",  ##1DACD6
	#"silver aluminium",  ##9EA0A1
	"so cool",  ##9EA0A1
	#"double surrender",  ##A8AAAB
	"dirty blue",  ##3F829D
	"PMS640",  ##008CB2
	"boston blue [2]",  ##438EAC
	"greyblue",  ##77A1B5
	"bluegrey",  ##85A3B2
	"battleship grey [4]",  ##6B7C85
	"PMS801 2X",  ##0089AF
	"abacus",  ##768993
	"bluish grey [2]",  ##748B97
	#"instinct",  ##8C979D
	"peacock",  ##33A1C9
	"moonstone blue",  ##73A9C2
	"pewter blue",  ##8BA8B7
	"PMS306 2X",  ##00A3D1
	"bali hai [2]",  ##849CA9
	"shakespeare [2]",  ##4EABD1
	"summer sky",  ##38B0DE
	"cyan [5]",  ##00B7EB
	"bowie",  ##0084AC
	"PMS5425",  ##8499A5
	"PMS5415",  ##607C8C
	"waterfront",  ##3E7F9D
	"light sky blue [6]",  ##607B8B
	"lynch [2]",  ##697D89
	#"aluminium [3]",  ##848789
	"blue moon",  ##7296AB
	#"rolling stone [2]",  ##747D83
	#"oslo grey [2]",  ##878D91
	"deep sky blue [5]",  ##009ACD
	"PMS2995",  ##00A5DB
	"vivid cerulean [2]",  ##00AEE7
	"nepal [2]",  ##93AAB9
	"blue grey [2]",  ##607C8E
	"tsunami",  ##6B8393
	"bermuda grey [2]",  ##6F8C9F
	"blue",  ##647D8E
	"weldon blue",  ##7C98AB
	"bluey grey",  ##89A0B0
	"cadet grey",  ##91A3B0
	"hemisphere",  ##4E93BA
	"PMS299",  ##00A3DD
	"moderate cornflower blue",  ##4A85A8
	"pigeon post [2]",  ##77848E
	"freefall",  ##1D95C9
	"PMS2915",  ##60AFDD
	"picton blue [2]",  ##45B1E8
	"wedgewood [2]",  ##4E7F9E
	"air force blue [2]",  ##5D8AA8
	"rackley",  ##5D8AA8
	"sky blue [9]",  ##3299CC
	"air superiority blue",  ##72A0C1
	#"el nino",  ##92A0AC
	"brilliant cornflower blue",  ##51AFE7
	#"forecast",  ##A2AAB1
	"grey blue",  ##6B8BA4
	"streetwise",  ##4F6971
	"deep space sparkle",  ##4A646C
	"ocean",  ##017B92
	#"infinity",  ##6A7376
	#"gunmetal [4]",  ##536267
	"allports [2]",  ##1F6A7D
	"blue bayoux [2]",  ##62777E
	"PMS633",  ##007F99
	"dark cerulean [2]",  ##274D59
	#"abbey [2]",  ##495154
	"teal blue [4]",  ##367588
	"norwester",  ##48798A
	#"trout [2]",  ##4C5356
	#"PMS431",  ##666D70
	"bismark [2]",  ##486C7A
	"marathon",  ##305563
	"pickled bluewood [2]",  ##4F5A5F
	"PMS634",  ##00667F
	"jelly bean [3]",  ##44798E
	"PMS3025",  ##00546B
	"undercurrent",  ##365C6C
	"cadet [2]",  ##536872
	"atomic [2]",  ##3D4B52
	"deep cove [2]",  ##3A4E58
	"explorer",  ##374E59
	"san juan [2]",  ##445761
	"fiord [2]",  ##4B5A62
	"PMS308",  ##00607C
	"snapshot",  ##505F67
	"calypso [2]",  ##3D7188
	"sea blue [2]",  ##047495
	"lucifer",  ##2E5060
	"blumine [2]",  ##305C71
	"astral [2]",  ##376F89
	"dark slate [2]",  ##394851
	"limed spruce [2]",  ##394851
	"navigate",  ##304E5E
	#"mid grey [3]",  ##717476
	"jetsetter",  ##40505A
	#"quarter foundry",  ##525557
	"blue sapphire",  ##126180
	"deep sky blue [5]",  ##00688B
	"slate grey [6]",  ##59656D
	"seachange",  ##4E6E81
	"strong cerulean",  ##007EA8
	"arapawa [2]",  ##274A5D
	"style pasifika turquoise sea",  ##274A5D
	"guru",  ##154C65
	"ivanhoe",  ##3B4C57
	"chathams blue [2]",  ##2C5971
	"slate [2]",  ##516572
	#"nevada [2]",  ##646E75
	"PMS307",  ##007AA5
	"PMS641",  ##007AA5
	"cg blue",  ##007AA5
	"PMS302",  ##004F6D
	"orient [2]",  ##255B77
	"PMS5405",  ##3F6075
	"steel blue [8]",  ##236B8E
	"celadon blue",  ##007BA7
	"cerulean [5]",  ##007BA7
	"deep cerulean [2]",  ##007BA7
	"ocean blue [2]",  ##03719C
	"dark electric blue",  ##536878
	"payne's grey",  ##536878
	"payne grey",  ##536878
	"compass",  ##4B5761
	"yeehaa",  ##006C98
	"sky blue [9]",  ##4A708B
	"half new denim blue",  ##535C64
	"wavelength",  ##3C6886
	"PMS3015",  ##00709E
	"metallic blue",  ##4F738E
	"neon blue [2]",  ##04D9FF
	"french pass [2]",  ##A4D2E0
	"pale light grayish cerulean",  ##B8DCE7
	#"gull grey [2]",  ##A4ADB0
	"eskimo",  ##A2B4BA
	#"triple athens grey",  ##C2C5C6
	"moby",  ##8EB2BE
	"belgion",  ##ADD8E6
	"blue light",  ##ADD8E6
	"light blue [9]",  ##ADD8E6
	"half escape",  ##BADAE5
	"glacier [2]",  ##80B3C4
	"parachute",  ##65B8D1
	"botticelli [2]",  ##C7DDE5
	"very light cerulean",  ##9EE7FF
	"PMS306",  ##00BCE2
	"pastel blue [2]",  ##AEC6CF
	"light cerulean",  ##8BD0E7
	"regent st blue [2]",  ##AAD6E6
	"winter wizard",  ##A0E6FF
	"loblolly [2]",  ##BDC9CE
	"anakiwa [2]",  ##9DE5FF
	"fresh air",  ##A6E7FF
	"PMS2985",  ##51BFE2
	"brilliant cerulean",  ##51C2E7
	"light brilliant cerulean",  ##65D8FF
	"seagull [2]",  ##77B7D0
	"heather [3]",  ##AEBBC1
	"PMS643",  ##C6D1D6
	"PMS297",  ##82C6E2
	"sky blue [9]",  ##87CEEB
	"bright sky blue",  ##02CCFE
	"vivid sky blue",  ##00CCFF
	#"silver sand [2]",  ##BFC1C2
	"half iron",  ##D5D7D8
	"white thunder",  ##D5D7D8
	"PMS290",  ##C4D8E2
	"columbia blue [2]",  ##C4D8E2
	"dark sky blue [2]",  ##8CBED6
	"PMS2905",  ##93C6E0
	"baby blue [3]",  ##89CFF0
	"PMS291",  ##A8CEE2
	"pale cyan [3]",  ##87D3F8
	"PMS298",  ##51B5E0
	"cornflower [6]",  ##93CCEA
	"light cornflower blue [2]",  ##93CCEA
	"altitude",  ##BCD4E2
	"light sky blue [6]",  ##8DB6CD
	#"metro",  ##ACAFB1
	"iron [2]",  ##D4D7D9
	"hit grey [2]",  ##A1ADB5
	"malibu [2]",  ##66B7E1
	"frozen",  ##9DB6C6
	"spiro disco ball",  ##0FC0FC
	"PMS545",  ##C4D3DD
	"sail [2]",  ##B8E0F9
	"very light cornflower blue",  ##9EDBFF
	"capri [3]",  ##00BFFF
	"deep sky blue [5]",  ##00BFFF
	"luminous vivid cerulean",  ##00BFFF
	"sky blue deep",  ##00BFFF
	"PMS544",  ##B7CCDB
	"lightblue",  ##7BC8F6
	"sky blue light",  ##87CEFA
	"PMS543",  ##93B7D1
	"beau blue",  ##BCD4E6
	"pale aqua [2]",  ##BCD4E6
	"PMS283",  ##9BC4E2
	"pale cerulean [2]",  ##9BC4E2
	"splat",  ##A5CEEC
	"light grey blue",  ##9DBCD4
	"light brilliant cornflower blue",  ##65C5FF
	"PMS277",  ##B5D1E8
	"PMS292",  ##75B2DD
	"sky"  ##82CAFC
]

prius_colors = frozenset(prius_full)
'''
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=False,
                help="path to the input image")
ap.add_argument("-i", "--image", required=False,
                help="path to the input image")
ap.add_argument("-t", "--type", required=False,
                help="Prius or Vehicle")
args = vars(ap.parse_args())

path = args['path']
arr = os.listdir(path)
'''


def write_json(data, filename="color_counts.json"):
	with open(filename, "w") as f:
		json.dump(data, f, indent=4)


def find_significant_contour(img):
	Image, contours, hierarchy = cv2.findContours(
		img,
		cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE
	)

	# Find level 1 contours
	level1Meta = []
	for contourIndex, tupl in enumerate(hierarchy[0]):
		# Each array is in format (Next, Prev, First child, Parent)
		# Filter the ones without parent
		if tupl[3] == -1:
			tupl = np.insert(tupl.copy(), 0, [contourIndex])
			level1Meta.append(tupl)

	# From among them, find the contours with large surface area.
	contoursWithArea = []
	for tupl in level1Meta:
		contourIndex = tupl[0]
		contour = contours[contourIndex]
		area = cv2.contourArea(contour)
		contoursWithArea.append([contour, area, contourIndex])

	contoursWithArea.sort(key=lambda meta: meta[1], reverse=True)
	topThree = []
	for i in range(0, min(len(contoursWithArea) - 1, 2)):
		topThree.append(contoursWithArea[i][0])

	return topThree


def get_contour_colors(image_src):
	# load the image and resize it to a smaller factor so that
	# the shapes can be approximated better
	# image = cv2.imread(os.path.join(path,image))
	return get_contour_colors_from_array(cv2.imread(image_src))


def get_contour_colors_from_array(image):
	# load the image and resize it to a smaller factor so that
	# the shapes can be approximated better
	# image = cv2.imread(os.path.join(path,image))
	try:
		resized = imutils.resize(image, width=300)
		ratio = image.shape[0] / float(resized.shape[0])
		# blur the resized image slightly, then convert it to both
		# grayscale and the L*a*b* color spaces
		blurred = cv2.GaussianBlur(resized, (5, 5), 0)
		gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
		lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
		thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]

		# find contours in the thresholded image
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		                        cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)

		cl = ColorLabeler()

		contours = []
		# loop over the contours
		for c in cnts:
			color = cl.label(lab, c)
			contours.append(color)
		return contours
	except Exception as e:
		print("While getting contours: " + str(e))


def has_prius_contour(image):
	colors = get_contour_colors(image)

	for color in colors:
		if color in prius_colors:
			return True
	return False


def has_prius_contour_from_array(arr):
	colors = get_contour_colors_from_array(arr)
	for color in colors:
		if color in prius_colors:
			return True
	return False


def detect_color(image_src):
	return detect_color_from_array(cv2.imread(image_src))


def detect_color_from_array(image):
	# load the image and resize it to a smaller factor so that
	# the shapes can be approximated better
	# image = cv2.imread(os.path.join(path,image))
	try:
		resized = imutils.resize(image, width=300)
		ratio = image.shape[0] / float(resized.shape[0])
		# blur the resized image slightly, then convert it to both
		# grayscale and the L*a*b* color spaces
		blurred = cv2.GaussianBlur(resized, (5, 5), 0)
		gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
		lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
		thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]

		# find contours in the thresholded image
		cnts = find_significant_contour(thresh.copy())
		cl = ColorLabeler()

		contours = []

		for c in cnts:
			color = cl.label(lab, c)
			#print("Color: " + color)
			contours.append(color)

		return contours
	except Exception as e:
		print("While detecting color: " + str(e))


def has_prius_color(image):
	detected_color = detect_color(image)
	if detected_color in prius_colors:
		return True
	return False


def has_prius_color_from_array(image):
	# Top 3 contours
	detected_colors = detect_color_from_array(image)
	for color in detected_colors:
		if color in prius_colors:
			return color
	return None


def load_counts():
	with open("color_counts.json", "r") as read_file:
		return json.load(read_file)


def detect_colors():
	results_dict = load_counts()
	results = {}
	if args['type'] == "prius":
		results = results_dict['prius']
	else:
		results = results_dict['vehicle']

	for image in arr:
		color = detect_color(os.path.join(path, image), image)
		if color is not None:
			if color in results:
				count = results[color]
				results[color] = count + 1
			else:
				results[color] = 1

# print(str(results_dict))
# for key, value in results_dict['prius']:
#	print("Color: " str(key) "  Count: " str(value))

# write_json(results_dict)
