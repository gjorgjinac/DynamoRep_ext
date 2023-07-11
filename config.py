import matplotlib

light_blue='#87cefa'
dark_cyan='#008b8b'
lime='#9acd32'
dark_blue='#056098'
grey='#575757'
black='#000000'
color_palette=[light_blue,dark_cyan,lime,dark_blue, grey]


color_palette_4=color_palette[:4]
stat_color_mapping={s:c for s,c in zip (['mean','min','max','std'], color_palette_4)}


my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['black','#008b8b','#9acd32','#e3f8b7'])